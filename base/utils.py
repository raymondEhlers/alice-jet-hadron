#!/usr/bin/env python

# (Mainly histogram) utilities

# From the future package
from builtins import range

import collections
import logging
# Setup logger
logger = logging.getLogger(__name__)

import root_numpy
import rootpy
import rootpy.io
import rootpy.ROOT as ROOT

# Small value - epsilon
# For use to offset from bin edges when finding bins for use with SetRange()
# NOTE: sys.float_info.epsilon is too small in some cases and thus should be avoided
epsilon = 1e-5

###################
# Utility functions
###################
def getHistogramsInList(filename, listTaskName = "AliAnalysisTaskJetH_tracks_caloClusters_clusbias5R2GA"):
    """ Get histograms from the file and make them available in a dict """
    hists = {}
    with rootpy.io.root_open(filename, "READ") as fIn:
        taskOutputList = fIn.Get(listTaskName)
        if not taskOutputList:
            logger.critical("Could not find list \"{0}\" with name \"{1}\". Possible names include:".format(taskOutputList, listTaskName))
            fIn.ls()
            return None

        for obj in taskOutputList:
            retrieveObject(hists, obj)

    return hists

def retrieveObject(outputDict, obj):
    """ Function to recusrively retrieve histograms from a list in a ROOT file. """
    # Store TH1 or THn
    if obj.InheritsFrom(ROOT.TH1.Class()) or obj.InheritsFrom(ROOT.THnBase.Class()):
        # Ensure that it is not lost after the file is closed
        # Only works for TH1
        if obj.InheritsFrom(ROOT.TH1.Class()):
            obj.SetDirectory(0)

        # Explictily note that python owns the object
        # From more on memory management with ROOT and python, see:
        # https://root.cern.ch/root/html/guides/users-guide/PythonRuby.html#memory-handling
        ROOT.SetOwnership(obj, True)
        
        # Store the objects
        outputDict[obj.GetName()] = obj

    # Recurse over lists
    if obj.InheritsFrom(ROOT.TCollection.Class()):
        # Keeping it in order simply makes it easier to follow
        outputDict[obj.GetName()] = collections.OrderedDict()
        for objTemp in list(obj):
            retrieveObject(outputDict[obj.GetName()], objTemp)

# From: https://stackoverflow.com/a/14314054
def movingAverage(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def getArrayFromHist(observable):
    """ Return array of data from a histogram.

    Args:
        observable (JetHUtils.Observable or ROOT.TH1): Histogram from which the array should be extracted.
    Returns:
        dict: "y": hist data, "errors" : y errors, "binCenters" : x bin centers
    """
    try:
        hist = observable.hist.hist
    except AttributeError as e:
        hist = observable
    #logger.debug("hist: {}".format(hist))
    arrayFromHist = root_numpy.hist2array(hist)
    xAxis = hist.GetXaxis()
    # Don't include overflow
    xBins = range(1, xAxis.GetNbins() + 1)
    # NOTE: The bin error is stored with the hist, not the axis.
    errors = np.array([hist.GetBinError(i) for i in xBins])
    binCenters = np.array([xAxis.GetBinCenter(i) for i in xBins])
    return {"y" : arrayFromHist, "errors" : errors, "binCenters" : binCenters}

def getArrayFromHist2D(hist):
    """ Extract the necessary data from the hist.

    Converts the histogram into a numpy array, and suitably processes it for a surface plot
    by removing 0s (which can cause problems when taking logs), and returning the bin centers
    for (X,Y).

    NOTE: This is a different format than the 1D version!

    Args:
        hist (ROOT.TH2): Histogram to be highlighted.
    Returns:
        tuple: Contains (x bin centers, y bin centers, numpy array of hist data)
    """
    # Process the hist into a suitable state
    (histArray, binEdges) = root_numpy.hist2array(hist, return_edges=True)
    # Set all 0s to nan to get similar behavior to ROOT. In ROOT, it will basically ignore 0s. This is especially important
    # for log plots. Matplotlib doesn't handle 0s as well, since it attempts to plot them and then will throw exceptions
    # when the log is taken.
    # By setting to nan, matplotlib basically ignores them similar to ROOT
    # NOTE: This requires a few special functions later which ignore nan when calculating min and max.
    histArray[histArray == 0] = np.nan

    # We want an array of bin centers
    xRange = np.array([hist.GetXaxis().GetBinCenter(i) for i in range(1, hist.GetXaxis().GetNbins()+1)])
    yRange = np.array([hist.GetYaxis().GetBinCenter(i) for i in range(1, hist.GetYaxis().GetNbins()+1)])
    X, Y = np.meshgrid(xRange, yRange)

    return (X, Y, histArray)

def getArrayForFit(observables, trackPtBin, jetPtBin):
    """ Return array of data from histogram """
    for name, observable in observables.iteritems():
        if observable.trackPtBin == trackPtBin and observable.jetPtBin == jetPtBin:
            return getArrayFromHist(observable)

