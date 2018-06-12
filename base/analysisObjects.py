#!/usr/bin/env python

# Utilities for the jet-hadron anaylsis
#
# Author: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
# Date: 13 Jan 2017

# From the future package
from builtins import super
from future.utils import iteritems
from future.utils import itervalues

import os
import aenum
import copy
import re
import numpy as np
import logging
# Setup logger
logger = logging.getLogger(__name__)

import rootpy.ROOT as ROOT

import jetH.base.utils as utils

class jetHCorrelationType(aenum.Enum):
    """ 1D correlation projection type """
    fullRange = 0
    # dPhi specialized
    signalDominated = 1
    backgroundDominated = 2
    # dEta specialized
    nearSide = 3
    awaySide = 4

    def __str__(self):
        """ Returns the name of the correlation type. """
        return self.name

    def str(self):
        """ Helper for __str__ to allow it to be accessed the same as the other str functions. """
        return self.__str__()

    def displayStr(self):
        """ Turns "signalDominated" into "Signal Dominated". """
        # Convert to display name by splitting on camel case
        # For the regex, see: https://stackoverflow.com/a/43898219
        splitString = re.sub('([a-z])([A-Z])', r'\1 \2', self.name)
        # Capitalize the first letter of every word
        return splitString.title()

    def filenameStr(self):
        """ Name to use in a filename. """
        return self.__str__()

class observable(object):
    """ Base observable object. Intended to store a histContainer.

    Args:
        hist (histContainer): The hist we are interested in.
    """
    def __init__(self, hist = None):
        self.hist = hist

class correlationObservable(observable):
    """ General correlation observable object. Usually used for 2D correlations.

    Args:
        jetPtBin (int): Bin of the jet pt of the observable.
        trackPtBin (int): Bin of the track pt of the observable.
        hist (histContainer): Associated histogram of the observable. Optional.
    """
    def __init__(self, jetPtBin, trackPtBin, *args, **kwargs):
        """ Initialize the observable """
        super().__init__(*args, **kwargs)

        self.jetPtBin = jetPtBin
        self.trackPtBin = trackPtBin

class correlationObservable1D(correlationObservable):
    """ For 1D correlation observable object. Can be either dPhi or dEta.

    Args:
        axis (jetHCorrelationAxis): Axis of the 1D observable.
        correlationType (jetHCorrelationType): Type of the 1D observable.
        jetPtBin (int): Bin of the jet pt of the observable.
        trackPtBin (int): Bin of the track pt of the observable.
        hist (histContainer): Associated histogram of the observable. Optional.
    """
    def __init__(self, axis, correlationType, *args, **kwargs):
        # Initialize the base class
        super().__init__(*args, **kwargs)

        self.axis = axis
        self.correlationType = correlationType

class extractedObservable(object):
    """ For extracted observable such as widths or yields.

   Args:
        jetPtBin (int): Bin of the jet pt of the observable.
        trackPtBin (int): Bin of the track pt of the observable.
        value (number): Extracted value.
        error (number): Error associated with the extracted value.
    """
    def __init__(self, jetPtBin, trackPtBin, value, error):
        self.jetPtBin = jetPtBin
        self.trackPtBin = trackPtBin
        self.value = value
        self.error = error

class histContainer(object):
    """ Container for a histogram to allow for normal function access except for those that
    we choose to overload.

    Args:
        hist (ROOT.TH1, ROOT.THnBase, or similar): Histogram to be stored in the histogram container.
    """
    def __init__(self, hist):
        self.hist = hist

    # Inspired by: https://stackoverflow.com/q/14612442
    def __getattr__(self, attr):
        """ Forwards all requested functions or attributes to the histogram. However, other
        functions or attributes defined in this class are still accessible!

        This function is usually called implicitly by calling the attribute on the histContainer.

        Args:
            attr (str): Desired attribute of the stored histogram.
        Returns:
            property or function: Requested attrbiute of the stored hist.
        """
        # Uncomment for debugging, as this is far too verbose otherwise
        #logger.debug("attr: {0}".format(attr))

        # Execute the attribute on the hist
        return getattr(self.hist, attr)

    def createScaledByBinWidthHist(self, additionalScaleFactor = 1.0):
        """ Create a new histogram scaled by the bin width. The return histogram is a clone
        of the histogram inside of the container with the scale factor(s) applied.

        One can always assign the result to replace the existing hist (although take care to avoid memory leaks!)

        Args:
            additionalScaleFactor (float): Additional scale factor to apply to the scaled hist. Default: 1.0
        Returns:
            ROOT.TH1: Cloned histogram scaled by the calculated scale factor.
        """

        finalScaleFactor = self.calculateFinalScaleFactor(additionalScaleFactor = additionalScaleFactor)

        # Clone hist and scale
        scaledHist = self.hist.Clone(self.hist.GetName() + "_Scaled")
        scaledHist.Scale(finalScaleFactor)

        return scaledHist

    def calculateFinalScaleFactor(self, additionalScaleFactor = 1.0):
        """ Calculate the scale factor to be applied to a hist before plotting by multiplying all of the
        bin widths together. Assumes a uniformly binned histogram.

        Args:
            additionalScaleFactor (float): Additional scale factor to apply to the scaled hist. Default: 1.0
        Returns:
            float: The factor by which the hist should be scaled.
        """
        # The first bin should always exist!
        binWidthScaleFactor = self.hist.GetXaxis().GetBinWidth(1)

        if self.hist.InheritsFrom(ROOT.TH2.Class()):
            binWidthScaleFactor *= self.hist.GetYaxis().GetBinWidth(1)
        if self.hist.InheritsFrom(ROOT.TH3.Class()):
            binWidthScaleFactor *= self.hist.GetZaxis().GetBinWidth(1)

        finalScaleFactor = additionalScaleFactor/binWidthScaleFactor

        return finalScaleFactor

class histArray(object):
    """ Represents a histogram's binned data.

    Histograms stored in this class make a number of (soft) assumptions
    - This histogram _doesn't_ include the under-flow and over-flow bins.
    - The binning of the histogram is uniform.

    NOTE: The members of this class are stored with leading underscores because we access them
          through properties. It would be cleaner to remove the  "_" before the variables, but for now,
          it is very convenient, since it allows for automatic initialization from YAML. Perhaps
          revise later the names later while maintaining those properties.

    Args:
        _binCenters (np.ndarray): The location of the center of each bin.
        _array (np.ndarray): The values in each bin.
        _errors (np.ndarray): The associated each of the value in each bin.
    """
    # Format of the filename used for storing the hist array.
    outputFilename = "hist_{tag}_jetPt{jetPtBin}_trackPt{trackPtBin}.yaml"
    def __init__(self, _binCenters, _array, _errors):
        # Store elements
        self._binCenters = _binCenters
        self._array = _array
        self._errors = _errors

    @classmethod
    def initFromRootHist(cls, hist):
        """ Initialize the hist array from an existing ROOT histogram.

        Args:
            hist (ROOT.TH1): ROOT histogram to be converted.
        Returns:
            histArray: histArray created from the passed TH1.
        """
        arr = utils.getArrayFromHist(hist)
        return cls(_binCenters = arr["binCenters"], _array = arr["y"], _errors = arr["errors"])

    @property
    def array(self):
        """ Return array of the histogram data. """
        return self._array

    @property
    def histData(self):
        """ Synonym of array. """
        return self.array

    @property
    def binCenters(self):
        """ Return array of the histogram bin centers. """
        return self._binCenters

    @property
    def x(self):
        """ Synonym of binCenters. """
        return self.binCenters

    @property
    def errors(self):
        """ Return array of the histogram errors. """
        return self._errors

    @classmethod
    def yamlFilename(cls, prefix, histType, jetPtBin, trackPtBin):
        """ Determinte the YAML filename given the parameters.

        Args:
            prefix (str): Path to the diretory in which the YAML file is stored.
            histType (jetHCorrelationType): Histogram type.
            jetPtBin (int): Bin of the jet pt.
            trackPtBin (int): Bin of the track pt.
        Returns:
            str: The filename of the YAML file which corresponds to the given parameters.
        """
        # Handle arguments
        if isinstance(histType, str):
            histType = jetHCorrelationType[histType]
        return os.path.join(prefix, cls.outputFilename.format(tag = histType.filenameStr(),
                                                                    jetPtBin = jetPtBin,
                                                                    trackPtBin = trackPtBin))

    @classmethod
    def initFromYAML(cls, inputPrefix, histType, jetPtBin, trackPtBin):
        """ Initialize the hist information from a YAML file.

        Args:
            inputPrefix (str): Path to the directory where the hist is stored.
            histType (jetHCorrelationType or str): Histogram type.
            jetPtBin (int): Bin of the jet pt.
            trackPtBin (int): Bin of the track pt.
        Returns:
            histArray: Hist array initialized from the YAML file.
        """
        # Handle arguments
        if isinstance(histType, str):
            histType = jetHCorrelationType[histType]

        filename = cls.yamlFilename(prefix = inputPrefix,
                histType = histType,
                jetPtBin = jetPtBin,
                trackPtBin = trackPtBin)

        # Check for file.
        if not os.path.exists(filename):
            logger.warning("Requested {} hist ({}, {}) from file {} does not exist! This container will not be initialized".format(histType, jetPtBin, trackPtBin, filename))
            return None

        logger.debug("Loading {} hist ({}, {}) from file {}".format(histType.str(), jetPtBin, trackPtBin, filename))
        parameters = utils.readYAML(filename = filename)

        histArray = cls(**parameters)

        # Handle custom data type conversion
        # This must be performed after defining the object because we want to iterate over
        # the __dict__ values, which is only meaningfully possible after object defiintion
        # Convert arrays to numpy arrays
        for key, val in iteritems(histArray.__dict__):
            if isinstance(val, list):
                setattr(histArray, key, np.array(val))

        return histArray

    def saveToYAML(self, outputPrefix, histType, jetPtBin, trackPtBin, fileAccessMode = "wb"):
        """ Write the hist array properties to a YAML file.

        Args:
            outputPrefix (str): Path to the directory where the hist should be stored.
            histType (jetHCorrelationType or str): Histogram type.
            jetPtBin (int): Bin of the jet pt.
            trackPtBin (int): Bin of the track pt.
            fileAccessMode (str): Mode under which the file should be opened. Defualt: "wb"
        """
        # Handle arguments
        if isinstance(histType, str):
            histType = jetHCorrelationType[histType]

        # Determine filename
        filename = histArray.yamlFilename(prefix = outputPrefix,
                histType = histType,
                jetPtBin = jetPtBin,
                trackPtBin = trackPtBin)

        # Use __dict__ so we don't have to explicitly define each field (which is easier to keep up to date!)
        outputDict = copy.deepcopy(self.__dict__)

        # Handle custom data type conversion
        # Convert values for storage
        for key, val in iteritems(outputDict):
            # Handle numpy arrays to lists
            #logger.debug("Processing key {} with type {} and value {}".format(key, type(val), val))
            if isinstance(val, np.ndarray):
                #logger.debug("Converting list {}".format(val))
                outputDict[key] = val.tolist()

        #logger.debug("outputDict: {}".format(outputDict))
        logger.debug("Saving {} hist ({}, {}) to file {}".format(histType.str(), jetPtBin, trackPtBin, filename))
        utils.writeYAML(filename = filename,
                        fileAccessMode = fileAccessMode,
                        outputDict = outputDict)

class fitContainer(object):
    """ Contains information about a particular fit.

    Fits should only be stored if they are valid!

    Args:
        jetPtBin (int): Jet pt bin
        trackPtBin (int): Track pt bin
        fitType (jetHCorrelationType): Type of fit being stored. Usually signalDominated or backgroundDominated
        values (dict): Dictionary from minuit.values storing parameter name to value. This is useful to have separately.
        params (dict): Dictionary from minuit.fitarg storing all relevant fit parameters (value, limits, errors,
            etc). This contains the values in values, but it useful to have them available separately, and it would
            take some work to separate the values from the other params.
        covarianceMatrix (dict): Dictionary from minuit.covariance storing the covariance matrix. It is a dict
            from ("param1", "param2") -> value
        errors (dict): Errors associated with the signal dominated and background dominated fit functions. Each
            error has one point for each point in the data.
    """
    # Format of the filename used for storing the fit container.
    # Make the filename accessible even if we don't have a class instance
    outputFilename = "fit_{}_jetPt{}_trackPt{}.yaml"

    def __init__(self, jetPtBin, trackPtBin, fitType, values, params, covarianceMatrix, errors = None):
        # Store elements
        self.jetPtBin = jetPtBin
        self.trackPtBin = trackPtBin
        self.fitType = fitType
        self.values = values
        self.params = params
        self.covarianceMatrix = covarianceMatrix
        if errors is None:
            errors = {}
        self.errors = errors

    @staticmethod
    def yamlFilename(prefix, fitType, jetPtBin, trackPtBin):
        """ Determinte the YAML filename given the parameters.

        Args:
            prefix (str): Path to the diretory in which the YAML file is stored.
            fitType (jetHCorrelationType): Fit type.
            jetPtBin (int): Bin of the jet pt.
            trackPtBin (int): Bin of the track pt.
        Returns:
            str: The filename of the YAML file which corresponds to the given parameters.
        """
        if isinstance(fitType, str):
            fitType = jetHCorrelationType[fitType]
        return os.path.join(prefix, fitContainer.outputFilename.format(fitType.filenameStr(), jetPtBin, trackPtBin))

    @classmethod
    def initFromYAML(cls, inputPrefix, fitType, jetPtBin, trackPtBin):
        """ Initial the fit information from a YAML file.

        Args:
            inputPrefix (str): Path to the directory where the fit is stored.
            fitType (jetHCorrelationType or str): fitType type.
            jetPtBin (int): Bin of the jet pt.
            trackPtBin (int): Bin of the track pt.
        Returns:
            fitContainer: Fit container initialized from the YAML file.
        """
        # Handle arguments
        if isinstance(fitType, str):
            fitType = jetHCorrelationType[fitType]

        filename = fitContainer.yamlFilename(prefix = inputPrefix,
                fitType = fitType,
                jetPtBin = jetPtBin,
                trackPtBin = trackPtBin)

        # Check for file. It may not exist if the fit was not successful
        if not os.path.exists(filename):
            logger.warning("Requested fit container ({}, {}) from file {} does not exist! This container will not be initialized".format(jetPtBin, trackPtBin, filename))
            return None

        logger.debug("Loading fit container ({}, {}) from file {}".format(jetPtBin, trackPtBin, filename))
        parameters = utils.readYAML(filename = filename)

        # Handle custom data type conversion
        # Convert errors to numpy array
        if "errors" in parameters:
            for k,v in iteritems(parameters["errors"]):
                parameters["errors"][k] = np.array(v)
        # Fit type to enum
        parameters["fitType"] = jetHCorrelationType[parameters["fitType"]]

        #logger.debug("parameters[errors]: {}".format(parameters["errors"]))

        # NOTE: The limits in the params dict will be restored as a list instead of a tuple.
        # However, this is fine, as the Minuit object interprets it properly anyway
        return cls(**parameters)

    def saveToYAML(self, outputPrefix, fileAccessMode = "wb", *args, **kwargs):
        """ Write the fit container properties to a YAML file.

        Args:
            outputPrefix (str): Path to the directory where the fit should be stored.
            fileAccessMode (str): Mode under which the file should be opened. Defualt: "wb"
            args (list): Additional arguments. They will be ignored. This is only so it can
                called with the same API as the other saveToYAML()
            kwargs (dict): Additional named arguments. They will be ignored.This is only so it can
                called with the same API as the other saveToYAML()
        """
        # Determine filename
        filename = fitContainer.yamlFilename(prefix = outputPrefix,
                fitType = self.fitType,
                jetPtBin = self.jetPtBin,
                trackPtBin = self.trackPtBin)

        # Convert the np array as necessary
        errors = {}
        logger.debug("self.errors: {}".format(self.errors))
        for identifier, data in iteritems(self.errors):
            if isinstance(data, np.ndarray):
                # Convert to a normal list so it can be stored
                errors[identifier] = data.tolist()
            else:
                errors[identifier] = data
        logger.debug("errors: {}".format(errors))
        # Fit type enum to str
        fitType = self.fitType.str()

        # Use __dict__ so we don't have to explicitly define each field (which is easier to keep up to date!)
        # TODO: This appears to be quite slow. Can we speed this up somehow?
        outputDict = copy.deepcopy(self.__dict__)
        # Reassign values based on the above
        outputDict["errors"] = errors
        outputDict["fitType"] = fitType

        logger.debug("Saving fit container ({}, {}) to file {}".format(self.jetPtBin, self.trackPtBin, filename))
        utils.writeYAML(filename = filename,
                        fileAccessMode = fileAccessMode,
                        outputDict = outputDict)

###############
# Additional experimental code
###############
# Then, for THn's like Salvatore createHists function
# Salvatore recommends binning by hand, as in his BinMultiSet class
# To use this function, much more work would be required!
def createHist(axes): # pragma: no cover
    hist = None
    for (axis, func) in (axes, [hist.GetXaxis, hist.GetYaxis, hist.GetZaxis]):
        func().SetTitle(axis.name)
        # ...

