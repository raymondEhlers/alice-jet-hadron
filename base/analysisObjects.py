#!/usr/bin/env python

# Utilities for the jet-hadron anaylsis
#
# Author: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
# Date: 13 Jan 2017

# From the future package
from builtins import super

import os
import enum
import copy
import re
import ruamel.yaml as yaml
import logging
# Setup logger
logger = logging.getLogger(__name__)

import rootpy.ROOT as ROOT

import jetH.base.utils as utils

##############
# Enumerations
##############
class TriggerClass(enum.Enum):
    """ Define the possible trigger classes """
    kMB = 0
    kGA = 1
    kGAL = 2
    kGAH = 3
    kJE = 4
    kJEL = 5
    kJEH = 6

    def __str__(self):
        """ Return the name of the value with the appended "k". This is just a convenience function """
        return str(self.name.replace("k", "", 1))

class JetHCorrelationType(enum.Enum):
    """ 1D correlation projection type """
    fullRange = 0
    # dPhi specialized
    signalDominated = 1
    backgroundDominated = 2
    # dEta specialized
    nearSide = 3
    awaySide = 4

    def __str__(self):
        return self.name

    def str(self):
        return self.__str__()

    def displayStr(self):
        # Convert to display name by splitting on camel case
        # For the regex, see: https://stackoverflow.com/a/43898219
        splitString = re.sub('([a-z])([A-Z])', r'\1 \2', self.name)
        # Capitalize the first letter of every word
        return splitString.title()

class Observable(object):
    """
    Base observable object. Intended to store a HistContainer.

    """
    def __init__(self, hist = None):
        self.hist = hist

class CorrelationObservable(Observable):
    """

    """
    def __init__(self, jetPtBin, trackPtBin, hist = None):
        """ Initialize the observable """
        super().__init__(hist = hist)

        self.jetPtBin = jetPtBin
        self.trackPtBin = trackPtBin

class CorrelationObservable1D(CorrelationObservable):
    """

    """
    def __init__(self, jetPtBin, trackPtBin, axis, correlationType, hist = None):
        # Initialize the base class
        super().__init__(jetPtBin = jetPtBin, trackPtBin = trackPtBin, hist = hist)

        self.axis = axis
        self.correlationType = correlationType

class ExtractedObservable(object):
    """ For extracted observable such as widths or yields. """
    def __init__(self, jetPtBin, trackPtBin, value, error):
        self.jetPtBin = jetPtBin
        self.trackPtBin = trackPtBin
        self.value = value
        self.error = error

class HistContainer(object):
    """
    Container for a histogram.
    """
    def __init__(self, hist):
        self.hist = hist

    # Inspired by: https://stackoverflow.com/questions/14612442/how-to-handle-return-both-properties-and-functions-missing-in-a-python-class-u
    def __getattr__(self, attr):
        """ Forwards all requested functions or attributes to the histogram. However, other
        functions or attributes defined in this class are still accessible!
        """
        # Uncomment for debugging, as this is far too verbose otherwise
        #logger.debug("attr: {0}".format(attr))

        # Execute the attribute on the hist
        return getattr(self.hist, attr)

    def createScaledByBinWidthHist(self, additionalScaleFactor = 1.0):
        """ Create a new histogram scaled by the bin width. The return histogram is a clone
        of the histogram inside of the container with the scale factor(s) applied.

        One can always assign the result to replace the existing hist (although take care to avoid memory leaks!)
        """

        finalScaleFactor = self.calculateFinalScaleFactor(additionalScaleFactor = additionalScaleFactor)

        # Clone hist and scale
        scaledHist = self.hist.Clone(self.hist.GetName() + "_Scaled")
        scaledHist.Scale(finalScaleFactor)

        return scaledHist

    def calculateFinalScaleFactor(self, additionalScaleFactor = 1.0):
        # The first bin should always exist!
        binWidthScaleFactor = self.hist.GetXaxis().GetBinWidth(1)

        if self.hist.InheritsFrom(ROOT.TH2.Class()):
            binWidthScaleFactor *= self.hist.GetYaxis().GetBinWidth(1)
        if self.hist.InheritsFrom(ROOT.TH3.Class()):
            binWidthScaleFactor *= self.hist.GetZaxis().GetBinWidth(1)

        finalScaleFactor = additionalScaleFactor/binWidthScaleFactor

        return finalScaleFactor

    def testFunc(self):
        """ Denomstrates the other functions than the hists can still be called! """
        return "testFunc"

class HistArray(object):
    outputFilename = "hist_{tag}_jetPt{jetPtBin}_trackPt{trackPtBin}.yaml"
    def __init__(self, _binCenters, _array, _errors):
        """ Represents a histogram's binned data.

        NOTE: It would be cleaner to remove the  "_" before the variables, but for now, it is very convenient,
              since it allows for automatic initialization from YAML. Perhaps revise later the names later
              while maintaining those properties. """
        self._binCenters = _binCenters
        self._array = _array
        self._errors = _errors

    @staticmethod
    def initFromRootHist(hist):
        arr = utils.getArrayFromHist(hist)
        return HistArray(_binCenters = arr["binCenters"], _array = arr["y"], _errors = arr["errors"])

    @property
    def array(self):
        return self._array

    @property
    def histData(self):
        """ Synonym of array. """
        return self.array

    @property
    def binCenters(self):
        return self._binCenters

    @property
    def x(self):
        """ Synonym of binCenters. """
        return self.binCenters

    @property
    def errors(self):
        return self._errors

    @staticmethod
    def initFromYAML(inputPrefix, histType, jetPtBin, trackPtBin):
        """ Initial the hist information from a YAML file """
        # Load configuration
        filename = os.path.join(inputPrefix, HistArray.outputFilename.format(tag = histType,
                                                                             jetPtBin = jetPtBin,
                                                                             trackPtBin = trackPtBin))
        logger.debug("Loading {} hist ({}, {}) from file {}".format(histType.str(), jetPtBin, trackPtBin, filename))

        # Check for file.
        if not os.path.exists(filename):
            logger.warning("Requested {} hist ({}, {}) from file {} does not exist! This container will not be initialized".format(histType, jetPtBin, trackPtBin, filename))
            return None

        parameters = None
        with open(filename, "r") as f:
            parameters = yaml.safe_load(f)

        histArray = HistArray(**parameters)

        # Handle custom data type conversion
        # This must be performed after defining the object because we want to iterate over
        # the __dict__ values, which is only meaningfully possible after object defiintion
        # Convert arrays to numpy arrays
        for key, val in histArray.__dict__.iteritems():
            if isinstance(val, list):
                setattr(histArray, key, np.array(val))

        return histArray

    def saveToYAML(self, outputPrefix, histType, jetPtBin, trackPtBin, fileAccessMode = "wb"):
        # Determine filename
        filename = os.path.join(outputPrefix, self.outputFilename.format(tag = histType.str(), jetPtBin = jetPtBin, trackPtBin = trackPtBin))

        # Use __dict__ so we don't have to explicitly define each field (which is easier to keep up to date!)
        outputDict = copy.deepcopy(self.__dict__)

        # Handle custom data type conversion
        # Convert values for storage
        for key, val in outputDict.iteritems():
            # Handle numpy arrays to lists
            #logger.debug("Processing key {} with type {} and value {}".format(key, type(val), val))
            if isinstance(val, np.ndarray):
                #logger.debug("Converting list {}".format(val))
                outputDict[key] = val.tolist()
            # Handle histogram type
            if isinstance(val, JetHCorrelationType):
                #logger.debug("Converting hist type {}".format(val))
                outputDict[key] = val.str()

        logger.debug("outputDict: {}".format(outputDict))
        logger.debug("Saving {} hist ({}, {}) to file {}".format(histType.str(), jetPtBin, trackPtBin, filename))
        with open(filename, fileAccessMode) as f:
            yaml.safe_dump(outputDict, f, default_flow_style = False)

class FitContainer(object):
    """

    Fits should only be stored if they are valid!
    """
    # Make the filename accessible even if we don't have a class instance
    outputFilename = "fit_{}_jetPt{}_trackPt{}.yaml"

    def __init__(self, jetPtBin, trackPtBin, fitType, values, params, covarianceMatrix, errors = None):
        """

        Args:
            jetPtBin (int): Jet pt bin
            trackPtBin (int): Track pt bin
            fitType (JetHCorrelationType): Type of fit being stored. Usually signalDominated or backgroundDominated
            values (dict): Dictionary from minuit.values storing parameter name to value. This is useful to have separately.
            params (dict): Dictionary from minuit.fitarg storing all relevant fit parameters (value, limits, errors, etc). This contains the values in values, but it useful to have them available separately, and it would take some work to separate the values from the other params.
            covarianceMatrix (dict): Dictionary from minuit.covariance storing the covariance matrix. It is a dict from ("param1", "param2") -> value
            errors (dict): Errors associated with the signal dominated and background dominated fit functions. Each error has one point for each point in the data.
        """
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
    def initFromYAML(inputPrefix, fitType, jetPtBin, trackPtBin):
        """ Initial the fit information from a YAML file """
        # Load configuration
        filename = os.path.join(inputPrefix, FitContainer.outputFilename.format(fitType.str(), jetPtBin, trackPtBin))
        logger.debug("Loading fit container ({}, {}) from file {}".format(jetPtBin, trackPtBin, filename))

        # Check for file. It may not exist if the fit was not successful
        if not os.path.exists(filename):
            logger.warning("Requested fit container ({}, {}) from file {} does not exist! This container will not be initialized".format(jetPtBin, trackPtBin, filename))
            return None

        parameters = None
        with open(filename, "r") as f:
            parameters = yaml.safe_load(f)

        # Handle custom data type conversion
        # Convert errors to numpy array
        if "errors" in parameters.iterkeys():
            for data in parameters["errors"].itervalues():
                data = np.array(data)
        # Fit type to enum
        parameters["fitType"] = JetHCorrelationType[parameters["fitType"]]

        #logger.debug("parameters[errors]: {}".format(parameters["errors"]))

        # NOTE: The limits in the params dict will be restored as a list instead of a tuple.
        # However, this is fine, as the Minuit object interprets it properly anyway
        return FitContainer(**parameters)

    def saveToYAML(self, outputPrefix, fileAccessMode = "wb"):
        # Determine filename
        filename = os.path.join(outputPrefix, self.outputFilename.format(self.fitType.str(), self.jetPtBin, self.trackPtBin))

        # Convert the np array as necessary
        errors = {}
        logger.debug("self.errors: {}".format(self.errors))
        for identifier, data in self.errors.iteritems():
            if isinstance(data, np.ndarray):
                # Convert to a normal list so it can be stored
                errors[identifier] = data.tolist()
            else:
                errors[identifier] = data
        logger.debug("errors: {}".format(errors))
        # Fit type enum to str
        fitType = self.fitType.str()

        # Use __dict__ so we don't have to explicitly define each field (which is easier to keep up to date!)
        outputDict = copy.deepcopy(self.__dict__)
        # Reassign values based on the above
        outputDict["errors"] = errors
        outputDict["fitType"] = fitType

        logger.debug("Saving fit container ({}, {}) to file {}".format(self.jetPtBin, self.trackPtBin, filename))
        with open(filename, fileAccessMode) as f:
            yaml.safe_dump(outputDict, f, default_flow_style = False)

###############
# Additional experimental code
###############
# Then, for THn's like Salvatore createHists function
# Salvatore recommends binning by hand, as in his BinMultiSet class
# To use this function, much more work would be required!
def createHist(axes):
    hist = None
    for (axis, func) in (axes, [hist.GetXaxis, hist.GetYaxis, hist.GetZaxis]):
        func().SetTitle(axis.name)
        # ...

