#!/usr/bin/env python

# Contains class to handle generic TH1 and THn projections

# From the future package
from future.utils import iteritems

import aenum
import copy
import sys
import logging
# Setup logger
logger = logging.getLogger(__name__)

import jetH.base.genericClass as genericClass

import rootpy.ROOT as ROOT

class TH1AxisType(aenum.Enum):
    """ Map from (x,y,z) axis to the axis number.

    Other enumerations that refer to this enum should refer to the _values_ to ensure
    consistnecy in `.value` pointing to the axis value.
    """
    xAxis = 0
    yAxis = 1
    zAxis = 2

# TODO: Rename functions to lower case for consistency.

class HistAxisRange(genericClass.EqualityMixin):
    """ Represents the restriction of a range of an axis of a histogram.

    An axis can be restricted by multiple HistAxisRange elements (although separate projections are
    needed to apply more than one. This would be accomplished with separate entries to the
    HistProjector.projectionDependentCutAxes).

    NOTE:
        A single axis which has multiple ranges could be represented by multiple HistAxis objects!

    Args:
        axisRangeName (str): Name of the axis range. Usually some combination of the axis name and
            some sort of description of the range.
        axisType (aenum.Enum): Enumeration corresponding to the axis to be restricted. The numerical
            value of the enum should be axis number (for a THnBase).
        minVal (function): Minimum range value for the axis. Usually set via ApplyFuncToFindBin().
        minVal (function): Maximum range value for the axis. Usually set via ApplyFuncToFindBin().
    """
    def __init__(self, axisRangeName, axisType, minVal, maxVal):
        self.name = axisRangeName
        self.axisType = axisType
        self.minVal = minVal
        self.maxVal = maxVal

    def __repr__(self):
        """ Representation of the object. """
        # The axis type is an enumeration of some type. In such a case, we want the repr to represent it using the str method instead
        return "{}(name = {name!r}, axisType = {axisType}, minVal = {minVal!r}, maxVal = {maxVal!r})".format(self.__class__.__name__, **self.__dict__)

    def __str__(self):
        """ Print the elements of the object. """
        return "{}: name: {name}, axisType: {axisType}, minVal: {minVal}, maxVal: {maxVal}".format(self.__class__.__name__, **self.__dict__)

    @property
    def axis(self):
        """ Wrapper to determine the axis to return based on the hist type. """
        def axisFunc(hist):
            """ Retrieve the axis associated with the HistAxisRange object for a given hist.

            Args:
                hist (ROOT.TH1, ROOT.THnBase, or similar): Histogram from which the selected axis should be retrieved.
            Returns:
                ROOT.TAxis: The axis associated with the HistAxisRange object.
            """
            # Determine the axisType value
            # Use try here instead of checking for a particular type to protect against type changes (say in the enum)
            try:
                # Try to extract the value from an enum
                axisType = self.axisType.value
            except AttributeError:
                # Seems that we received an int, so just use that value
                axisType = self.axisType

            if isinstance(hist, ROOT.THnBase):
                # Return the proper THn access
                #logger.debug("From hist: {0}, axisType: {1}, axis: {2}".format(hist, self.axisType, ROOT.THnBase.GetAxis(hist, self.axisType.value)))
                return ROOT.THnBase.GetAxis(hist, axisType)
            else:
                # If it's not a THn, then it must be a TH1 derived
                # This effectively emuates THnBase.GetAxis(...)
                axisFunctionMap = {TH1AxisType.xAxis.value : ROOT.TH1.GetXaxis,
                        TH1AxisType.yAxis.value : ROOT.TH1.GetYaxis,
                        TH1AxisType.zAxis.value : ROOT.TH1.GetZaxis}

                # Retrieve the axis function and execute it. It is done separately to
                # clarify any possible errors.
                returnFunc = axisFunctionMap[axisType]
                return returnFunc(hist)

        return axisFunc

    def ApplyRangeSet(self, hist):
        """ Apply the associated range set to the axis of a given hist.

        Note:
            The min and max values should be bins, not user ranges! For more, see the binning
            explanation in `ApplyFuncToFindBin(...)`.

        Args:
            hist (ROOT.TH1 or similar): Histogram to which the axis range restriction should be applied.
        """
        # Do inidividual assignments to clarify which particular value is causing an error here.
        axis = self.axis(hist)
        #logger.debug("axis: {}, axis(): {}".format(axis, axis.GetName()))
        minVal = self.minVal(axis)
        maxVal = self.maxVal(axis)
        # NOTE: Using SetRangeUser() here was a bug, since I've been passing bin values! In general,
        #       passing bin values is more flexible, but requires the values to be passed to ApplyFuncToFindBin()
        #       to be shifted by some small epsilon to get the desired bin.
        self.axis(hist).SetRange(minVal, maxVal)

    @staticmethod
    def ApplyFuncToFindBin(func, values = None):
        """ Closure to determine the bin associated with a value on an axis.

        It can apply a function to an axis if necessary to determine the proper bin.  Otherwise,
        it can just return a stored value.

        Note:
            To properly determine the value, carefully note the information below. In many cases,
            such as when we want values [2, 5), the values need to be shifted by a small epsilon
            to retrieve the proper bin. This is done automatically in `SetRangeUser()`.

            >>> hist = ROOT.TH1D("test", "test", 10, 0, 10)
            >>> x = 2, y = 5
            >>> hist.FindBin(x)
            2
            >>> hist.FindBin(x+epsilon)
            2
            >>> hist.FindBin(y)
            6
            >>> hist.FindBin(y-epsilon)
            5

            Note that the bin + epsilon on the lower bin is not strictly necessary, but it is
            used for consistency with the upper bound.

        Args:
            func (function): Function to apply to the histogram axis. If it is None, the value
                will be returned.
            values (int or float): Value to pass to the function. Default: None (in which case,
                it won't be passed).
        Returns:
            function: Function to be called with an axis to determine the desired bin on that axis.
        """
        def returnFunc(axis):
            """ Apply the stored function and value to a the given axis.

            Args:
                axis (TAxis or similar): Axis to which the function should be applied.
            Returns:
                any: The value returned by the function. Often a float or int, but not necessarily.
            """
            #logger.debug("func: {0}, values: {1}".format(func, values))
            if func:
                if values is not None:
                    return func(axis, values)
                else:
                    return func(axis)
            else:
                return values

        return returnFunc

class HistProjector(object):
    """ Handles generic ROOT THn and TH1 projections.

    There are three types of cuts which can be specified:
     - additionalAxisCuts: List of axis cuts which are neither projected nor depend on the axis being projected.
     - projectionDependentCutAxes: List of list of axis cuts which depend on the projected axis. For example, if
        we want to project non-continuous ranges of a non-projection axis (say, dEta when projecting dPhi). It is
        a list of list to allow for groups of cuts to be specified together if necessary.
     - projectionAxes: List of axes which should be projected.
    
    Note:
        The TH1 projections have not been tested as extensively as the THn projections.

    Note:
        "inputKey", "inputHist", "inputObservable", "projectionName", anad "outputHist" are all reserved
        keys, such they will be overwritten by predefined information when passed to the various functions.
        Thus, they should be avoided by the user when storing projection information

    Args:
        observableList (dict): Where the projected hists will be stored. They will be stored under the dict key
            determined by OutputKeyName(...).
        observableToProjectFrom (dict): The observables which should be used to project from. The dict key is
            passed to ProjectionName(...) as "inputKey".
        projectionNameFormat (str): Format string to determine the projected hist name.
        projectionInformation (dict): Keyword arguments to be passed to ProjectionName(...) to determine
            the name of the projected histogram.
    """
    def __init__(self, observableList, observableToProjectFrom, projectionNameFormat, projectionInformation = None):
        # Input and output lists
        # TODO: Rename observableList to observableDict
        self.observableList = observableList
        # TODO: Rename observableToProjectFrom to observablesToProjectFrom
        self.observableToProjectFrom = observableToProjectFrom
        # Output hist name format
        self.projectionNameFormat = projectionNameFormat
        # Additional projection information to help create names, input/output objects, etc
        # NOTE: See reserved keys enumerated above.
        if projectionInformation is None:
            projectionInformation = {}
        # Ensure that the dict is copied successfully
        self.projectionInformation = copy.deepcopy(projectionInformation)

        # Axes
        # Cuts for axes which are not projected
        self.additionalAxisCuts = []
        # Axes cuts which depend on the projection axes
        # ie. If we want to change the range of the axis that we are projecting
        # For example, we may want to project an axis non-continuously (say, -1 - 0.5, 0.5 - 1)
        self.projectionDependentCutAxes = []
        # Axes to actually project
        self.projectionAxes = []

    # Printing functions
    def __str__(self):
        """ Prints the properties of the projector.

        This will only show up properly when printed - otherwise the tabs and newlines won't be printed.
        """
        retVal = "{}: Projection Information:\n".format(self.__class__.__name__)
        retVal += "\tProjectionNameFormat: \"{projectionNameFormat}\"\n"
        retVal += "\tProjectionInformation:\n"
        retVal += "\n".join(["\t\t- " + str("Arg: ") + str(val) for arg, val in self.projectionInformation])
        retVal += "\tadditionalAxisCuts:\n"
        retVal += "\n".join(["\t\t- " + str(axis) for axis in self.additionalAxisCuts])
        retVal += "\tprojectionDependentCutAxes:\n"
        retVal += "\n".join(["\t\t- " + str([",".join(axis.axisName for axis in axisList)]) for axisList in self.projectionDependentCutAxes])
        retVal += "\tprojectionAxes:\n"
        retVal += "\n".join(["\t\t- " + str(axis) for axis in self.projectionAxes])

        return retVal.format(**self.__dict__)

    def CallProjectionFunction(self, hist):
        """ Calls the actual projection function for the hist.

        Args:
            hist (ROOT.TH1 or ROOT.THnBase): Histogram from which the projections should be performed.
        Returns:
            ROOT.TH1 or ROOT.THnBase derived: The projected histogram.
        """
        # Restrict projection axis ranges
        for axis in self.projectionAxes:
            logger.debug("Apply projection axes hist range: {0}".format(axis.name))
            axis.ApplyRangeSet(hist)

        projectedHist = None
        if isinstance(hist, ROOT.THnBase):
            # THnBase projections args are given as a list of axes, followed by any possible options.
            projectionAxes = [axis.axisType.value for axis in self.projectionAxes]

            # Handle ROOT THnBase quirk...
            # 2D projection are called as (y, x, options), so we should reverse the order so it performs as expected
            if len(projectionAxes) == 2:
                # Reverses in place
                projectionAxes.reverse()

            # Test calculating errors
            # Add "E" to ensure that errors will be calculated
            args = projectionAxes + ["E"]
            # Do the actual projection
            logger.debug("hist: {0} args: {1}".format(hist.GetName(), args))

            if len(projectionAxes) > 3:
                # Project into a THnBase object.
                projectedHist = ROOT.THnBase.ProjectionND(hist, *args)
            else:
                # Project a TH1 derived object.
                projectedHist = ROOT.THnBase.Projection(hist, *args)
        elif isinstance(hist, ROOT.TH3):
            # Axis length validation
            if len(self.projectionAxes) < 1 or len(self.projectionAxes) > 2:
                raise ValueError(len(self.projectionAxes), "Invalid number of axes")

            projectionAxisMap = {
                TH1AxisType.xAxis.value : "x",
                TH1AxisType.yAxis.value : "y",
                TH1AxisType.zAxis.value : "z",
            }

            # Need to concatenate the names of the axes together
            projectionAxisName = ""
            for axis in self.projectionAxes:
                # [:1] returns just the first letter. For example, we could get "xy" if the first axis as xAxis and the second was yAxis
                # NOTE: Careful. This depends on the name of the enumerated values!!!
                # TODO: Remove this dependency. This is not very safe.
                projectionAxisName += axis.name[:1]

            # Handle ROOT Project3D quirk...
            # 2D projection are called as (y, x, options), so we should reverse the order so it performs as expected
            # NOTE: This isn't well documented in TH3. It is instead described in THnBase.Projection(...)
            if len(self.projectionAxes) == 2:
                # Reverse the axes
                projectionAxisName = projectionAxisName[::-1]

            # Do the actual projection
            logger.info("Projecting onto axes \"{0}\" from hist {1}".format(projectionAxisName, hist.GetName()))
            projectedHist = ROOT.TH3.Project3D(hist, projectionAxisName)
        elif isinstance(hist, ROOT.TH2):
            if len(self.projectionAxes) != 1:
                raise ValueError(len(self.projectionAxes), "Invalid number of axes")

            #logger.debug("self.projectionAxes[0].axis: {}, axis range name: {}, axisType: {}".format(self.projectionAxes[0].axis, self.projectionAxes[0].name , self.projectionAxes[0].axisType))
            # NOTE: We cannot use TH3.ProjectionZ(...) because it has different sematnics than ProjectionX and ProjectionY.
            #       In particular, it doesn't respect the axis limits of axis onto which it is projected.
            #       So we have to separate the projection by histogram type as opposed to axis length.
            projectionFuncMap = {
                TH1AxisType.xAxis.value : ROOT.TH2.ProjectionX,
                TH1AxisType.yAxis.value : ROOT.TH2.ProjectionY
            }

            # Determine the axisType value
            # Use try here instead of checking for a particular type to protect against type changes (say in the enum)
            try:
                # Try to extract the value from an enum
                axisType = self.projectionAxes[0].axisType.value
            except:
                # Seems that we received an int, so just use that value
                axisType = self.axisType

            #if not isinstance(axisType, TH1AxisType):
            #    axisType = axisType.value
            projectionFunc = projectionFuncMap[axisType]

            # Do the actual projection
            logger.info("Projecting onto axis range {} from hist {}".format(self.projectionAxes[0].name, hist.GetName()))
            projectedHist = projectionFunc(hist)
        else:
            raise TypeError(type(hist), "Could not recognize hist {0} of type {1}".format(hist, hist.GetClass().GetName()))

        # Cleanup restricted axes
        self.CleanupCuts(hist, cutAxes = self.projectionAxes)

        return projectedHist

    def Project(self, *args, **kwargs):
        """ Perform the requested projections.

        Note:
            All cuts on the original histograms will be reset when this function is completed.

        Args:
            args (list): Additional args to be passed to ProjectionName(...) and OutputKeyName(...)
            kwargs (dict): Additional named args to be passed to ProjectionName(...) and OutputKeyName(...)
        """
        for key, inputObservable in iteritems(self.observableToProjectFrom):
            # Retrieve histogram
            hist = self.GetHist(observable = inputObservable, *args, **kwargs)

            # Define projection name
            projectionNameArgs = {}
            projectionNameArgs.update(self.projectionInformation)
            projectionNameArgs.update(kwargs)
            # Put the values included by default last to ensure nothing overwrites these values
            projectionNameArgs.update({"inputKey": key, "inputObservable": inputObservable, "inputHist": hist})
            projectionName = self.ProjectionName(*args, **projectionNameArgs)

            # First apply the cuts
            # Restricting the range with SetRangeUser Works properly for both THn and TH1.
            logger.info("hist: {0}".format(hist))
            for axis in self.additionalAxisCuts:
                logger.debug("Apply additional axis hist range: {0}".format(axis.name))
                axis.ApplyRangeSet(hist)

            # We need to ensure that it isn't empty so at least one project occurs
            if self.projectionDependentCutAxes == []:
                self.projectionDependentCutAxes.append([])

            # Perform the projections
            hists = []
            for (i, axes) in enumerate(self.projectionDependentCutAxes):
                # Projection dependent range set
                for axis in axes:
                    logger.debug("Apply projection dependent hist range: {0}".format(axis.name))
                    axis.ApplyRangeSet(hist)

                # Do the projection
                projectedHist = self.CallProjectionFunction(hist)
                projectedHist.SetName("{0}_{1}".format(projectionName, i))

                hists.append(projectedHist)

                # Cleanup projection dependent cuts (although they should be set again on the next iteration of the loop)
                self.CleanupCuts(hist, cutAxes = axes)

            # Add all projections together
            outputHist = hists[0]
            for tempHist in hists[1:]:
                outputHist.Add(tempHist)

            # Ensure that the hist doesn't get deleted by ROOT
            # A reference to the histogram within python may not be enough
            outputHist.SetDirectory(0)

            outputHist.SetName(projectionName)
            outputHistArgs = projectionNameArgs
            outputHistArgs.update({"outputHist" : outputHist, "projectionName" : projectionName})
            outputKeyName = self.OutputKeyName(*args, **outputHistArgs)
            self.observableList[outputKeyName] = self.OutputHist(*args, **outputHistArgs)

            # Cleanup cuts
            self.CleanupCuts(hist, cutAxes = self.additionalAxisCuts)

    def CleanupCuts(self, hist, cutAxes):
        """ Cleanup applied cuts by resetting the axis to the full range.

        Inspired by: https://github.com/matplo/rootutils/blob/master/python/2.7/THnSparseWrapper.py

        Args:
            hist (ROOT.TH1 or ROOT.THnBase): Histogram for which the axes should be reset.
            cutAxes (list): List of axis cuts, which correspond to axes that should be reset.
        """
        for axis in cutAxes:
            # According to the function TAxis::SetRange(first, last), the widest possible range is
            # (1, Nbins). Anything beyond that will be reset to (1, Nbins)
            axis.axis(hist).SetRange(1, axis.axis(hist).GetNbins())

    #############################
    # Functions to be overridden!
    #############################
    def ProjectionName(self, *args, **kwargs):
        """ Define the projection name for this projector.

        Note:
            This function is just a basic placeholder and likely should be overridden.

        Args:
            args (list): Additional arguments passed to the projection function
            kwargs (dict): Projection information dict combined with additional arguments passed to the projection function
        Returns:
            str: Projection name string formatted with the passed options. By default, it returns projectionNameFormat formatted
                with the arguments to this function.
        """
        return self.projectionNameFormat.format(**kwargs)

    def GetHist(self, observable, *args, **kwargs):
        """ Get the histogram that may be stored in some object. This histogram is used
        to project from.

        Note:
            The output object could just be the raw histogram.

        Note:
            This function is just a basic placeholder and likely should be overridden.

        Args:
            observable (object): The input object. It could be a histogram or something more complex
            args (list): Additional arguments passed to the projection function
            kwargs (dict): Additional arguments passed to the projection function
        Return:
            A ROOT.TH1 or ROOT.THnBase histogram which should be projected. By default, it returns the observable (input object).
        """
        return observable

    def OutputKeyName(self, inputKey, outputHist, projectionName, *args, **kwargs):
        """ Returns the key under which the output object should be stored.

        Note:
            This function is just a basic placeholder which returns the projection name
            and likely should be overridden.

        Args:
            inputKey (str): Key of the input hist in the input dict
            outputHist (ROOT.TH1 or ROOT.THnBase): The output histogram
            projectionName (str): Projection name for the output histogram
            args (list): Additional arguments passed to the projection function
            kwargs (dict): Projection information dict combined with additional arguments passed to the projection function
        Returns:
            str: Key under which the output object should be stored. By default, it returns the projection name.
        """
        return projectionName

    def OutputHist(self, outputHist, inputObservable, *args, **kwargs):
        """ Return an output object. It should store the outputHist.

        Note:
            The output object could just be the raw histogram.

        Note:
            This function is just a basic placeholder which returns the given output object (a histogram)
            and likely should be overridden.

        Args:
            outputHist (ROOT.TH1 or ROOT.THnBase): The output histogram
            inputObservable (object): The corresponding input object. It could be a histogram or something more complex.
            args (list): Additional arguments passed to the projection function
            kwargs (dict): Projection information dict combined with additional arguments passed to the projection function
        Return:
            str: The output object which should be stored in the output dict. By default, it returns the output hist.
        """
        return outputHist

