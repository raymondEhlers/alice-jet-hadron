#!/usr/bin/env python

""" Utilities for the jet-hadron anaylsis

.. code-author: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

# From the future package
from builtins import super
from future.utils import iteritems

import aenum
import copy
import logging
import numpy as np
import os
import re

from jet_hadron.base import utils

# Setup logger
logger = logging.getLogger(__name__)

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

class Observable(object):
    """ Base observable object. Intended to store a HistContainer.

    Args:
        hist (HistContainer): The hist we are interested in.
    """
    def __init__(self, hist = None):
        self.hist = hist

class CorrelationObservable(Observable):
    """ General correlation observable object. Usually used for 2D correlations.

    Args:
        jetPtBin (int): Bin of the jet pt of the observable.
        trackPtBin (int): Bin of the track pt of the observable.
        hist (HistContainer): Associated histogram of the observable. Optional.
    """
    def __init__(self, jetPtBin, trackPtBin, *args, **kwargs):
        """ Initialize the observable """
        super().__init__(*args, **kwargs)

        self.jetPtBin = jetPtBin
        self.trackPtBin = trackPtBin

class CorrelationObservable1D(CorrelationObservable):
    """ For 1D correlation observable object. Can be either dPhi or dEta.

    Args:
        axis (jetHCorrelationAxis): Axis of the 1D observable.
        correlationType (jetHCorrelationType): Type of the 1D observable.
        jetPtBin (int): Bin of the jet pt of the observable.
        trackPtBin (int): Bin of the track pt of the observable.
        hist (HistContainer): Associated histogram of the observable. Optional.
    """
    def __init__(self, axis, correlationType, *args, **kwargs):
        # Initialize the base class
        super().__init__(*args, **kwargs)

        self.axis = axis
        self.correlationType = correlationType

class ExtractedObservable(object):
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

class HistContainer(object):
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

        This function is usually called implicitly by calling the attribute on the HistContainer.

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
        # Because of a ROOT quirk, even a TH1* hist has a Y and Z axis, with 1 bin
        # each. This bin has bin width 1, so it doesn't change anything if we multiply
        # by that bin width. So we just do it for all histograms.
        # This has the benefit that we don't need explicit dependence on an imported
        # ROOT package.
        binWidthScaleFactor *= self.hist.GetYaxis().GetBinWidth(1)
        binWidthScaleFactor *= self.hist.GetZaxis().GetBinWidth(1)

        finalScaleFactor = additionalScaleFactor / binWidthScaleFactor

        return finalScaleFactor

class YAMLStorableObject(object):
    """ Base class for objects which can be represented and stored in YAML. """
    outputFilename = "yamlObject.yaml"

    def __init__(self):
        pass

    @classmethod
    def yamlFilename(cls, prefix, **kwargs):
        """ Determinte the YAML filename given the parameters.

        Args:
            prefix (str): Path to the diretory in which the YAML file is stored.
            kwargs (dict): Formatting dictionary for outputFilename.
        Returns:
            str: The filename of the YAML file which corresponds to the given parameters.
        """
        # Handle arguments
        if isinstance(kwargs["objType"], str):
            kwargs["objType"] = jetHCorrelationType[kwargs["objType"]]

        # Convert to proper name for formatting the string
        formattingArgs = {}
        formattingArgs.update(kwargs)
        formattingArgs["type"] = formattingArgs.pop("objType")

        return os.path.join(prefix, cls.outputFilename.format(**formattingArgs))

    @staticmethod
    def initializeSpecificProcessing(parameters):  # pragma: no cover
        """ Initialization specific processing plugin. Applied to obj immediately after creation from YAML.

        NOTE: This is called on `cls`, but we don't want to modify the object, so
              we define it as a `staticmethod`.

        Args:
            parameters (dict): Parameters which will be used to construct the object.
        Returns:
            dict: Parameters with the initialization specific processing applied.
        """
        return parameters

    @classmethod
    def initFromYAML(cls, *args, **kwargs):
        """ Initialize the object using information from a YAML file.

        Args:
            args (list): Positional arguments to use for intialization. They currently aren't used.
            kwargs (dict): Named arguments to use for initialization. Must contain:
                prefix (str): Path to the diretory in which the YAML file is stored.
                objType (jetHCorrelationType or str): Type of the object.
                jetPtBin (int): Bin of the jet pt of the object.
                trackPtBin (int): Bin of the track pt of the object.
        Returns:
            obj: The object constructed from the YAML file.
        """
        # Handle arguments
        if isinstance(kwargs["objType"], str):
            kwargs["objType"] = jetHCorrelationType[kwargs["objType"]]

        filename = cls.yamlFilename(**kwargs)

        # Check for file.
        if not os.path.exists(filename):
            logger.warning("Requested {objType} {className} ({jetPtBin}, {trackPtBin}) from file {filename}"
                           " does not exist! This container will not be"
                           " initialized".format(objType = kwargs["objType"].str(),
                                                 className = type(cls).__name__,
                                                 jetPtBin = kwargs["jetPtBin"],
                                                 trackPtBin = kwargs["trackPtBin"],
                                                 filename = filename))
            return None

        logger.debug("Loading {objType} {className} ({jetPtBin}, {trackPtBin}) from file"
                     " {filename}".format(objType = kwargs["objType"].str(),
                                          className = type(cls).__name__,
                                          jetPtBin = kwargs["jetPtBin"],
                                          trackPtBin = kwargs["trackPtBin"],
                                          filename = filename))
        parameters = utils.readYAML(filename = filename)

        # Handle custom data type conversion
        parameters = cls.initializeSpecificProcessing(parameters)

        obj = cls(**parameters)

        return obj

    @staticmethod
    def saveSpecificProcessing(parameters):  # pragma: no cover
        """ Save specific processing plugin. Applied to obj immediately before saving to YAML.

        NOTE: This is called on `self` since we don't have a `cls` instance, but
              we don't want to modify the object, so we define it as a `staticmethod`.

        Args:
            parameters (dict): Parameters which will be saved to the YAML file.
        Returns:
            dict: Parameters with the save specific processing applied.
        """
        return parameters

    def saveToYAML(self, fileAccessMode = "w", *args, **kwargs):
        """ Write the object properties to a YAML file.

        Args:
            fileAccessMode (str): Mode under which the file should be opened. Defualt: "w"
            args (list): Positional arguments to use for intialization. They currently aren't used.
            kwargs (dict): Named arguments to use for initialization. Must contain:
                prefix (str): Path to the diretory in which the YAML file is stored.
                objType (jetHCorrelationType or str): Type of the object.
                jetPtBin (int): Bin of the jet pt of the object.
                trackPtBin (int): Bin of the track pt of the object.
        """
        # Handle arguments
        if isinstance(kwargs["objType"], str):
            kwargs["objType"] = jetHCorrelationType[kwargs["objType"]]

        # Determine filename
        filename = self.yamlFilename(**kwargs)

        # Use __dict__ so we don't have to explicitly define each field (which is easier to keep up to date!)
        # TODO: This appears to be quite slow. Can we speed this up somehow?
        parameters = copy.deepcopy(self.__dict__)

        # Handle custom data type conversion
        parameters = self.saveSpecificProcessing(parameters)

        #logger.debug("parameters: {}".format(parameters))
        logger.debug("Saving {objType} {className} ({jetPtBin}, {trackPtBin}) to file"
                     " {filename}".format(objType = kwargs["objType"],
                                          className = type(self).__name__,
                                          jetPtBin = kwargs["jetPtBin"],
                                          trackPtBin = kwargs["trackPtBin"],
                                          filename = filename))
        utils.writeYAML(filename = filename,
                        fileAccessMode = fileAccessMode,
                        parameters = parameters)

class HistArray(YAMLStorableObject):
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
    outputFilename = "hist_{type}_jetPt{jetPtBin}_trackPt{trackPtBin}.yaml"

    def __init__(self, _binCenters, _array, _errors):
        # Init base class
        super().__init__()

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
            HistArray: HistArray created from the passed TH1.
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

    @staticmethod
    def initializeSpecificProcessing(parameters):
        """ Converts all lists to numpy arrays.

        Args:
            parameters (dict): Parameters which will be used to construct the object.
        Returns:
            dict: Parameters with the initialization specific processing applied.
        """
        # Convert arrays to numpy arrays
        for key, val in iteritems(parameters):
            if isinstance(val, list):
                parameters[key] = np.array(val)

        return parameters

    @staticmethod
    def saveSpecificProcessing(parameters):
        """ Converts all numpy arrays to lists for saving to YAML.

        Args:
            parameters (dict): Parameters which will be saved to the YAML file.
        Returns:
            dict: Parameters with the save specific processing applied.
        """
        # Convert values for storage
        for key, val in iteritems(parameters):
            # Handle numpy arrays to lists
            #logger.debug("Processing key {} with type {} and value {}".format(key, type(val), val))
            if isinstance(val, np.ndarray):
                #logger.debug("Converting list {}".format(val))
                parameters[key] = val.tolist()

        return parameters

class FitContainer(YAMLStorableObject):
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
    outputFilename = "fit_{type}_jetPt{jetPtBin}_trackPt{trackPtBin}.yaml"

    def __init__(self, jetPtBin, trackPtBin, fitType, values, params, covarianceMatrix, errors = None):
        # Init base class
        super().__init__()

        # Handle arguments
        if isinstance(fitType, str):
            fitType = jetHCorrelationType[fitType]

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
    def initializeSpecificProcessing(parameters):
        """ Converts lists in the errors dict into numpy arrays.

        Args:
            parameters (dict): Parameters which will be used to construct the object.
        Returns:
            dict: Parameters with the initialization specific processing applied.
        """
        if "errors" in parameters:
            for k, v in iteritems(parameters["errors"]):
                parameters["errors"][k] = np.array(v)
        # NOTE: The enum will be converted in the constructor, so we don't need to handle it here.

        return parameters

    @staticmethod
    def saveSpecificProcessing(parameters):
        """ Convert numpy arrays in the errors dict into lists.

        Args:
            parameters (dict): Parameters which will be saved to the YAML file.
        Returns:
            dict: Parameters with the save specific processing applied.
        """
        for identifier, data in iteritems(parameters["errors"]):
            if isinstance(data, np.ndarray):
                # Convert to a normal list so it can be stored
                parameters["errors"][identifier] = data.tolist()
        # Fit type enum to str
        parameters["fitType"] = parameters["fitType"].str()

        return parameters

    def saveToYAML(self, prefix, fileAccessMode = "w", *args, **kwargs):
        """ Write the fit container properties to a YAML file.

        Args:
            prefix (str): Path to the directory where the fit should be stored.
            fileAccessMode (str): Mode under which the file should be opened. Defualt: "w"
            args (list): Additional arguments. They will be forwarded to saveToYAML().
            kwargs (dict): Additional named arguments. `objType`, `jetPtBin`, and `trackPtBin`
                will be overwritten by values stored in the class.
        """
        # Fill in the values from the object.
        kwargs["objType"] = self.fitType
        kwargs["jetPtBin"] = self.jetPtBin
        kwargs["trackPtBin"] = self.trackPtBin

        # Call the base class to handle the actual work.
        super().saveToYAML(prefix = prefix,
                           fileAccessMode = fileAccessMode,
                           *args, **kwargs)

###############
# Additional experimental code
###############
# Then, for THn's like Salvatore createHists function
# Salvatore recommends binning by hand, as in his BinMultiSet class
# To use this function, much more work would be required!
def createHist(axes):  # pragma: no cover
    hist = None
    for (axis, func) in (axes, [hist.GetXaxis, hist.GetYaxis, hist.GetZaxis]):
        func().SetTitle(axis.name)
        # ...

