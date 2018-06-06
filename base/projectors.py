#!/usr/bin/env python

# Contains class to handle generic TH1 and THn projections

import aenum
import copy
import sys
import logging
# Setup logger
logger = logging.getLogger(__name__)

import rootpy.ROOT as ROOT

class TH1AxisType(aenum.Enum):
    xAxis = 0
    yAxis = 1
    zAxis = 2

class HistAxisRange(object):
    """ Represents the restriction of a range of an axis of a histogram. An axis can be restricted by multiple HistAxisRange elements
    (although separate projections are needed to apply more than one. This would be accomplished with separate entries to the 
    HistProjector.projectionDependentCutAxes. See below.)

    NOTE:
        A single axis which has multiple ranges could be represented by multiple HistAxis objects!

    Args:
        axisRangeName (str): Name of the axis range. Usually some combination of the axis name and some sort of description of the range
        axisType (aenum.Enum): Enumeration corresponding to the axis to be restricted. The numerical value of the enum should be axis number (for a THnBase)
        minVal (float or function): Minimum range value for the axis
        minVal (float or function): Maximum range value for the axis
    """
    def __init__(self, axisRangeName, axisType, minVal, maxVal):
        self.name = axisRangeName
        self.axisType = axisType
        self.minVal = minVal
        self.maxVal = maxVal

    def __repr__(self):
        # The axis type is an enumeration of some type. In such a case, we want the repr to represent it using the str method instead
        return "{}(name = {name!r}, axisType = {axisType}, minVal = {minVal!r}, maxVal = {maxVal!r})".format(self.__class__.__name__, **self.__dict__)

    def __str__(self):
        return "{}: name: {name}, axisType: {axisType}, axisRange: {axisRange}".format(self.__class__.__name__, **self.__dict__)

    def __eq__(self, other):
        if other is None:
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self == other

    @property
    def axis(self):
        """ Determine the axis to return based on the hist type. """
        def axisFunc(hist):
            if hist.InheritsFrom(ROOT.THnBase.Class()):
                # Return the proper THn access
                if self.axisType == None:
                    logger.critical("Must define axis type for range \"{name}\" on axis \"{axisType}\" and hist \"{histName}\".".format(name = self.name, axisType = self.axisType, histName = hist.GetName()))
                    sys.exit(1)
                #logger.debug("From hist: {0}, axisType: {1}, axis: {2}".format(hist, self.axisType, ROOT.THnBase.GetAxis(hist, self.axisType.value)))
                return ROOT.THnBase.GetAxis(hist, self.axisType.value)
            else:
                # If it's not a THn, then it must be a TH1 derived
                if self.axisType == None:
                    logger.critical("Must define axis type for range \"{name}\" with range \"{axisType}\" and hist \"{histName}\".".format(name = self.name, axisType = self.axisType, histName = hist.GetName()))
                    sys.exit(1)

                # If not of the type TH1AxisType, it is probably an enumeration alias for a TH1AxisType (for example,
                # JetHCorrelationAxis.kDeltaPhi), so should use the value of that enum, which will be a TH1AxisType
                axisType = self.axisType
                if not isinstance(axisType, TH1AxisType):
                    axisType = axisType.value

                if axisType == TH1AxisType.xAxis:
                    returnFunc = ROOT.TH1.GetXaxis
                elif axisType == TH1AxisType.yAxis:
                    returnFunc = ROOT.TH1.GetYaxis
                elif axisType == TH1AxisType.zAxis:
                    returnFunc = ROOT.TH1.GetZaxis
                else:
                    logger.critical("Unrecognized axis type. name: {}, value: {}".format(axisType.name, axisType.value))
                    sys.exit(1)
                return returnFunc(hist)

        return axisFunc

    def ApplyRangeSet(self, hist):
        """ Apply the associated range set to the axis.

        NOTE: The min and max values should be bins, not user ranges!"""
        # Do inidividual assignments to clarify which particular value is causing an error here.
        axis = self.axis(hist)
        #logger.debug("axis: {}, axis(): {}".format(axis, axis.GetName()))
        minVal = self.minVal(axis)
        maxVal = self.maxVal(axis)
        # NOTE: Using SetRangeUser() was a bug, since I've been passing bin values!
        #self.axis(hist).SetRangeUser(minVal, maxVal)
        self.axis(hist).SetRange(minVal, maxVal)

    @staticmethod
    def ApplyFuncToFindBin(func, values = None):
        """ Closure to apply histogram functions if necessary to determine the proper bin for projections. """
        def returnFunc(axis):
            #logger.debug("func: {0}, values: {1}".format(func, values))
            if func:
                if values != None:
                    return func(axis, values)
                else:
                    return func(axis)
            else:
                return values

        return returnFunc

class HistProjector(object):
    """ Handles generic ROOT projections.
    
    Should handle both THn and TH1 projections.

    NOTE: The TH1 projections have not been tested as extensively as the THn projections. """
    def __init__(self, observableList, observableToProjectFrom, projectionNameFormat, projectionInformation = None):
        # Input and output lists
        self.observableList = observableList
        self.observableToProjectFrom = observableToProjectFrom
        # Output hist name format
        self.projectionNameFormat = projectionNameFormat
        # Additional projection information to help create names, input/output objects, etc
        # NOTE: "inputKey", "inputHist", "inputObservable", "projectionName", anad "outputHist" are all reserved
        #       keys, such they will be overwritten by predefined information when passed to the various functions.
        #       Thus, they should be avoided by the user when storing projection information
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
        """ Prints the properties of the projector. This will only show up properly when printed - otherwise the
        tabs and newlines won't be printed.
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
        """ Calls the actual projection function for the hist """
        # Restrict projection axis ranges
        for axis in self.projectionAxes:
            logger.debug("Apply projection axes hist range: {0}".format(axis.name))
            axis.ApplyRangeSet(hist)

        projectedHist = None
        if hist.InheritsFrom(ROOT.THnBase.Class()):
            projectionAxes = [axis.axisType.value for axis in self.projectionAxes]
            if len(projectionAxes) > 3:
                logger.critical("Does not currently support projecting higher than a 3D hist. Given axes: {0}".format(projectionAxes))
                sys.exit(1)

            # Handle ROOT quirk...
            # 2D projection are called as (y, x, options), so we should reverse the order so it performs as expected
            if len(projectionAxes) == 2:
                # Reverses in place
                projectionAxes.reverse()

            # Test calculating errors
            # Add "E" to ensure that errors will be calculated
            args = projectionAxes + ["E"]
            # Do the actual projection
            logger.debug("hist: {0} args: {1}".format(hist.GetName(), args))
            projectedHist = ROOT.THnBase.Projection(hist, *args)
        elif hist.InheritsFrom(ROOT.TH1.Class()):
            if len(self.projectionAxes) == 1:
                #logger.debug("self.projectionAxes[0].axis: {}, axis range name: {}, axisType: {}".format(self.projectionAxes[0].axis, self.projectionAxes[0].name , self.projectionAxes[0].axisType))
                projectionFuncMap = { TH1AxisType.xAxis : ROOT.TH2.ProjectionX,
                                      TH1AxisType.yAxis : ROOT.TH2.ProjectionY,
                                      TH1AxisType.zAxis : ROOT.TH3.ProjectionZ }
                # If not of the type TH1AxisType, it is probably an enumeration alias for a TH1AxisType (for example,
                # JetHCorrelationAxis.kDeltaPhi), so should use the value of that enum, which will be a TH1AxisType
                axisType = self.projectionAxes[0].axisType
                if not isinstance(axisType, TH1AxisType):
                    axisType = axisType.value
                projectionFunc = projectionFuncMap[axisType]

                # Do the actual projection
                logger.info("Projecting onto axis range {} from hist {}".format(self.projectionAxes[0].name, hist.GetName()))
                projectedHist = projectionFunc(hist)
            elif len(self.projectionAxes) == 2:
                # Need to concatenate the names of the axes together
                projectionAxisName = ""
                for axis in self.projectionAxes:
                    # [:1] returns just the first letter. For example, we could get "xy" if the first axis as xAxis and the second was yAxis
                    projectionAxisName += axis.name[:1]

                # Do the actual projection
                logger.info("Projecting onto axes \"{0}\" from hist {1}".format(projectionAxisName, hist.GetName()))
                projectedHist = ROOT.TH3.Project3D(hist, projectionAxisName)
            else:
                logger.critical("Invalid number of axes: {0}".format(len(self.projectionAxes)))
                sys.exit(1)
        else:
            logger.critical("Could not recognize hist {0} of type {1}".format(hist, hist.GetClass().GetName()))
            sys.exit(1)

        # Cleanup restricted axes
        self.CleanupCuts(hist, cutAxes = self.projectionAxes)

        return projectedHist

    def Project(self, *args, **kwargs):
        """ Perform the requested projections. """
        for key, inputObservable in self.observableToProjectFrom.iteritems():
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

        Inspired by: https://github.com/matplo/rootutils/blob/master/python/2.7/THnSparseWrapper.py """
        for axis in cutAxes:
            # According to the function TAxis::SetRange(first, last), the widest possible range is
            # (1, Nbins). Anything beyond that will be reset to (1, Nbins)
            axis.axis(hist).SetRange(1, axis.axis(hist).GetNbins())

    #############################
    # Functions to be overridden!
    #############################
    def ProjectionName(self, *args, **kwargs):
        """ Define the projection name for this projector.

        This function is just a basic placeholder and likely should be overridden.

        Args:
            args (list): Additional arguments passed to the projection function
            kwargs (dict): Projection information dict combined with additional arguments passed to the projection function
        """
        return self.projectionNameFormat.format(**kwargs)

    def GetHist(self, observable, *args, **kwargs):
        """ Return the histogram that may be stored in some object.

        This function is just a basic placeholder which returns the given object (a histogram)
        and likely should be overridden.

        Args:
            observable (object): The input object. It could be a histogram or something more complex
            args (list): Additional arguments passed to the projection function
            kwargs (dict): Additional arguments passed to the projection function

        Return:
            A ROOT.TH1 or ROOT.THnBase histogram which should be projected
        """
        return observable

    def OutputKeyName(self, inputKey, outputHist, projectionName, *args, **kwargs):
        """ Returns the key under which the output object should be stored.

        Args:
            inputKey (str): Key of the input hist in the input dict
            outputHist (ROOT.TH1 or ROOT.THnBase): The output histogram
            projectionName (str): Projection name for the output histogram
            args (list): Additional arguments passed to the projection function
            kwargs (dict): Projection information dict combined with additional arguments passed to the projection function

        This function is just a basic placeholder which returns the projection name
        and likely should be overridden.
        """
        return projectionName

    def OutputHist(self, outputHist, inputObservable, *args, **kwargs):
        """ Return an output object. It should store the outputHist.

        This function is just a basic placeholder which returns the given output object (a histogram)
        and likely should be overridden.

        Args:
            outputHist (ROOT.TH1 or ROOT.THnBase): The output histogram
            inputObservable (object): The corresponding input object. It could be a histogram or something more complex.
            args (list): Additional arguments passed to the projection function
            kwargs (dict): Projection information dict combined with additional arguments passed to the projection function

        Return:
            The output object which should be stored in the output dict
        """
        return outputHist

