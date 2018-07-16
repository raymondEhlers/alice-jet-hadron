#!/usr/bin/env python

# From the future package
from builtins import range
from future.utils import itervalues

import pytest
import os
import numpy as np
import collections
import logging
# Setup logger
logger = logging.getLogger(__name__)

import jetH.base.utils as utils

@pytest.fixture
def retrieveRootList(testRootHists):
    """ Create an set of lists to load for a ROOT file.

    NOTE: Not using a mock since I'd like to the real objects and storing
          a ROOT file is just as easy here.

    The expected should look like:
    ```
    {'mainList': OrderedDict([('test', Hist('test_1')),
                             ('test2', Hist('test_2')),
                             ('test3', Hist('test_3')),
                             ('innerList',
                              OrderedDict([('test', Hist('test_1')),
                                           ('test', Hist('test_2')),
                                           ('test', Hist('test_3'))]))])}
    ```
    """
    import rootpy.ROOT as ROOT
    import rootpy.io

    # Create values for the test
    # We only use 1D hists so we can do the comparison effectively.
    # This is difficult because root hists don't handle operator==
    # very well. Identical hists will be not equal in smoe cases...
    hists = []
    h = testRootHists.hist1D
    for i in range(3):
        hists.append(h.Clone("{}_{}".format(h.GetName(), i)))
    l = ROOT.TList()
    l.SetName("mainList")
    l2 = ROOT.TList()
    l2.SetName("innerList")
    for h in hists:
        l.Add(h)
        l2.Add(h)
    l.Add(l2)

    # File for comparison.
    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), "testFiles", "testOpeningList.root")
    # Create the file if needed.
    if not os.path.exists(filename):
        lCopy = l.Clone("tempMainList")
        # The objects will be destroyed when l is written.
        # However, we write it under the l name to ensure it is read corectly later
        with rootpy.io.root_open(filename, "RECREATE") as f:
            lCopy.Write(l.GetName(), ROOT.TObject.kSingleKey)

    # Create expected values
    # See the docstring for an explanation of the format.
    expected = {}
    innerDict = collections.OrderedDict()
    mainList = collections.OrderedDict()
    for h in hists:
        innerDict[h.GetName()] = h
        mainList[h.GetName()] = h
    mainList["innerList"] = innerDict
    expected["mainList"] = mainList

    return (filename, l, expected)

def testGetHistogramsInList(loggingMixin, retrieveRootList):
    """ Test for retrieving a list of histograms from a ROOT file. """
    (filename, rootList, expected) = retrieveRootList

    output = utils.getHistogramsInList(filename, "mainList")

    # The first level of the output is removed by `getHistogramsInList()`
    expected = expected["mainList"]

    # This isn't the most sophisticated way of comparsion, but bin-by-bin is sufficient for here.
    # We take advantage that we know the structure of the file so we don't need to handle recursion
    # or higher dimensional hists.
    outputInnerList = output.pop("innerList")
    expectedInnerList = expected.pop("innerList")
    for (o, e) in [(output, expected), (outputInnerList, expectedInnerList)]:
        for oHist, eHist in zip(itervalues(o), itervalues(e)):
            oValues = [oHist.GetBinContent(i) for i in range(0, oHist.GetXaxis().GetNbins()+2)]
            eValues = [eHist.GetBinContent(i) for i in range(0, eHist.GetXaxis().GetNbins()+2)]
            assert np.allclose(oValues, eValues)

def testGetNonExistentList(loggingMixin, retrieveRootList):
    """ Test for retrieving a list which doesn't exist from a ROOT file. """
    (filename, rootList, expected) = retrieveRootList

    output = utils.getHistogramsInList(filename, "nonExistent")
    assert output is None

def testRetrieveObject(loggingMixin, retrieveRootList):
    """ Test for retrieving a list of histograms from a ROOT file.

    NOTE: One would normally expect to have the hists in the first level of the dict, but
          this is actually taken care of in `getHistogramsInList()`, so we need to avoid
          doing it in the tests here.
    """
    (filename, rootList, expected) = retrieveRootList

    output = {}
    utils.retrieveObject(output, rootList)

    assert output == expected

@pytest.mark.parametrize("inputs, expected", [
        ((3, np.array([1,2,3,4,5,4,3,2,1])),
            np.array([6, 9, 12, 13, 12, 9, 6])),
        ((4, np.array([1,2,3,4,5,4,3,2,1])),
            np.array([10, 14, 16, 16, 14, 10])),
        ((3, np.array([1,2,3,4,5,6,7,8,9,10])),
            np.array([6, 9, 12, 15, 18, 21, 24, 27])),
        ((3, np.array([10,9,8,7,6,5,4,3,2,1])),
            np.array([27, 24, 21, 18, 15, 12, 9, 6]))
    ], ids = ["n = 3 trianglur values", "n = 4 triangular values", "n = 3 increasing values", "n = 3 decreasing values"])
def testMovingAverage(loggingMixin, inputs, expected):
    """ Test the moving average calculation. """
    (n, arr) = inputs
    expected = expected/n
    assert np.array_equal(utils.movingAverage(arr = arr, n = n), expected)

def testGetArrayFromHist(loggingMixin, testRootHists):
    """ Test getting numpy arrays from a 1D hist. """
    hist = testRootHists.hist1D
    histArray = utils.getArrayFromHist(hist)

    # Determine expected values
    xBins = range(1, hist.GetXaxis().GetNbins() + 1)
    expectedHistArray = {"y" : np.array([hist.GetBinContent(i) for i in xBins]),
            "errors" : np.array([hist.GetBinError(i) for i in xBins]),
            "binCenters" : np.array([hist.GetXaxis().GetBinCenter(i) for i in xBins])}

    assert np.array_equal(histArray["y"], expectedHistArray["y"])
    assert np.array_equal(histArray["errors"], expectedHistArray["errors"])
    assert np.array_equal(histArray["binCenters"], expectedHistArray["binCenters"])

@pytest.mark.parametrize("setZeroToNaN", [
        False, True
    ], ids = ["Keep zeroes as zeroes", "Set zeroes to NaN"])
def testGetArrayFromHist2D(loggingMixin, setZeroToNaN, testRootHists):
    """ Test getting numpy arrays from a 2D hist. """
    hist = testRootHists.hist2D
    histArray = utils.getArrayFromHist2D(hist = hist, setZeroToNaN = setZeroToNaN)

    # Determine expected values
    xRange = np.array([hist.GetXaxis().GetBinCenter(i) for i in range(1, hist.GetXaxis().GetNbins()+1)])
    yRange = np.array([hist.GetYaxis().GetBinCenter(i) for i in range(1, hist.GetYaxis().GetNbins()+1)])
    expectedX, expectedY = np.meshgrid(xRange, yRange)
    expectedHistArray = np.array([hist.GetBinContent(x, y) for x in range(1, hist.GetXaxis().GetNbins()+1) for y in range(1, hist.GetYaxis().GetNbins()+1)], dtype=np.float32).reshape(hist.GetXaxis().GetNbins(), hist.GetYaxis().GetNbins())
    if setZeroToNaN:
        expectedHistArray[expectedHistArray == 0] = np.nan

    assert np.array_equal(histArray[0], expectedX)
    assert np.array_equal(histArray[1], expectedY)
    # Need to use the special `np.testing.assert_array_equal()` to properly
    # handle comparing NaN in the array. It returns _None_ if it is successful,
    # so we compare against that. It will raise an exception if they disagree
    assert np.testing.assert_array_equal(histArray[2], expectedHistArray) is None

def testGetArrayForFit(loggingMixin, mocker, testRootHists):
    """ Test getting an array from a hist in a dict of observables. """
    observables = {}
    for i in range(5):
        observables[i] = mocker.MagicMock(spec = ["jetPtBin", "trackPtBin", "hist"],
                jetPtBin = i, trackPtBin = i + 2,
                hist = None)
    # We only want one to work. All others shouldn't have a hist to ensure that the test
    # will fail if something has gone awry.
    observables[3].hist = mocker.MagicMock(spec = ["hist"], hist = testRootHists.hist1D)
    histArray = utils.getArrayForFit(observables, jetPtBin = 3, trackPtBin = 5)

    # Expected values
    expectedHistArray = utils.getArrayFromHist(testRootHists.hist1D)

    assert np.array_equal(histArray["y"], expectedHistArray["y"])
    assert np.array_equal(histArray["errors"], expectedHistArray["errors"])
    assert np.array_equal(histArray["binCenters"], expectedHistArray["binCenters"])

