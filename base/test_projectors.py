#!/usr/bin/env python

# Test projector functionality
#
# author: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
# date: 6 June 2018

import pytest
import aenum
import collections
import logging
logger = logging.getLogger(__name__)

import jetH.base.utils as utils
import jetH.base.projectors as projectors

import numpy as np
import rootpy.ROOT as ROOT

# Set logging level as a global variable to simplify configuration.
# This is not ideal, but fine for simple tests.
loggingLevel = logging.DEBUG

@pytest.fixture
def createHistAxisRange():
    """ Create a HistAxisRange object to use for testing. """
    #axisType, axis = request.param
    objectArgs = {
            "axisRangeName" : "zAxisTestProjector",
            "axisType" : projectors.TH1AxisType.yAxis,
            "minVal" : lambda x : x,
            "maxVal" : lambda y : y
        }
    obj = projectors.HistAxisRange(**objectArgs)
    # axisRangeName is referred to as name internally, so we rename to that
    objectArgs["name"] = objectArgs.pop("axisRangeName")

    return (obj, objectArgs)

def testHistAxisRange(caplog, createHistAxisRange):
    """ Tests for creating a HistAxisRange object. """
    caplog.set_level(loggingLevel)
    obj, objectArgs = createHistAxisRange

    assert obj.name == objectArgs["name"]
    assert obj.axisType == objectArgs["axisType"]
    assert obj.minVal == objectArgs["minVal"] 
    assert obj.maxVal == objectArgs["maxVal"]

    # Test repr and str to esnure that they are up to date.
    assert repr(obj) == "HistAxisRange(name = {name!r}, axisType = {axisType}, minVal = {minVal!r}, maxVal = {maxVal!r})".format(**objectArgs)
    assert str(obj) == "HistAxisRange: name: {name}, axisType: {axisType}, minVal: {minVal}, maxVal: {maxVal}".format(**objectArgs)
    # Assert that the dict is equal so we don't miss anything in the repr or str representations.
    assert obj.__dict__ == objectArgs

@pytest.mark.parametrize("axisType, axis", [
        (projectors.TH1AxisType.xAxis, ROOT.TH1.GetXaxis),
        (projectors.TH1AxisType.yAxis, ROOT.TH1.GetYaxis),
        (projectors.TH1AxisType.zAxis, ROOT.TH1.GetZaxis),
        (0, ROOT.TH1.GetXaxis),
        (1, ROOT.TH1.GetYaxis),
        (2, ROOT.TH1.GetZaxis),
    ], ids = ["xAxis", "yAxis", "zAxis", "number for x axis", "number for y axis", "number for z axis"])
@pytest.mark.parametrize("histToTest", range(0, 3), ids = ["1D", "2D", "3D"])
def testTH1AxisDetermination(caplog, createHistAxisRange, axisType, axis, histToTest, testRootHists):
    """ Test TH1 axis determination in the HistAxisRange object. """
    caplog.set_level(loggingLevel)
    # Get the HistAxisRange object
    obj, objectArgs = createHistAxisRange
    # Insert the proepr axis type
    obj.axisType = axisType
    # Determine the test hist
    hist = testRootHists[histToTest]

    # Check that the axis retrieved by the specified function is the same
    # as that retrieved by the HistAxisRange object.
    # NOTE: GetZaxis() (for example) is still valid for a TH1. It is a minimal axis
    #       object with 1 bin. So it is fine to check for equivalnce for axes that
    #       don't really make sense in terms of a hist's dimensions.
    assert axis(hist) == obj.axis(hist)

class selectedTestAxis(aenum.Enum):
    """ Enum to map from our selected axes to their axis values. Goes along with the sparse created in testTHnSparse. """
    axisOne = 2
    axisTwo = 4
    axisThree = 5

@pytest.fixture
def testTHnSparse():
    """ Create a THnSparse for use in testing. """
    # namedtuple is just for convenience
    sparseAxis = collections.namedtuple("sparseAxis", ["nBins", "min", "max"])
    ignoredAxis   = sparseAxis(nBins =  1, min =   0.0, max =  1.0)
    selectedAxis1 = sparseAxis(nBins = 10, min =   0.0, max = 20.0)
    selectedAxis2 = sparseAxis(nBins = 20, min = -10.0, max = 10.0)
    selectedAxis3 = sparseAxis(nBins = 30, min =   0.0, max = 30.0)
    # We want to select axes 2, 4, 5
    axes = [ignoredAxis, ignoredAxis, selectedAxis1, ignoredAxis, selectedAxis2, selectedAxis3, ignoredAxis]
    

    # Create the actual sparse
    bins = np.array([el.nBins for el in axes], dtype=np.int32)
    mins = np.array([el.min for el in axes])
    maxes = np.array([el.max for el in axes])
    sparse = ROOT.THnSparseF("testSparse", "testSparse", len(axes),
                bins, mins, maxes)

    # Fill in some strategic values.
    # Wrapper function is for convenience.
    def fillSparse(one, two, three):
        sparse.Fill(np.array([0., 0., one, 0., two, three, 0.]))
    fillValues = [
            (2., 0., 10.),
            (4., 0., 10.)
        ]
    for values in fillValues:
        fillSparse(*values)

    return (sparse, fillValues)

@pytest.mark.parametrize("axisSelection", [
        selectedTestAxis.axisOne,
        selectedTestAxis.axisTwo,
        selectedTestAxis.axisThree,
        2, 4, 5
    ], ids = ["axisOne", "axisTwo", "axisThree", "number for axis one", "number for axis two", "number for axis three"])
def testTHnAxisDetermination(caplog, axisSelection, createHistAxisRange, testTHnSparse):
    """ Test THn axis determination in the HistAxisRange object. """
    caplog.set_level(loggingLevel)
    # Retrieve sparse.
    sparse, _ = testTHnSparse
    # Retrieve object and setup.
    obj, objectArgs = createHistAxisRange
    obj.axisType = axisSelection

    axisValue = axisSelection.value if isinstance(axisSelection, aenum.Enum) else axisSelection
    assert sparse.GetAxis(axisValue) == obj.axis(sparse)

@pytest.mark.parametrize("minVal, maxVal, minValFunc, maxValFunc, expectedFunc", [
        (0, 10,
            lambda x : projectors.HistAxisRange.ApplyFuncToFindBin(ROOT.TAxis.FindBin, x + utils.epsilon),
            lambda x : projectors.HistAxisRange.ApplyFuncToFindBin(ROOT.TAxis.FindBin, x - utils.epsilon),
            lambda axis, x, y : axis.SetRangeUser(x, y) ),
        (1, None,
            lambda x : projectors.HistAxisRange.ApplyFuncToFindBin(None, x),
            lambda x : projectors.HistAxisRange.ApplyFuncToFindBin(ROOT.TAxis.GetNbins),
            lambda axis, x, y : True), # This is just a no-op. We don't want to restrict the range.
        (0, 7,
            lambda x : projectors.HistAxisRange.ApplyFuncToFindBin(None, x),
            lambda x : projectors.HistAxisRange.ApplyFuncToFindBin(None, x),
            lambda axis, x, y : axis.SetRange(x, y) )
    ], ids = ["0 - 10 with ApplyFuncToFindBin with FindBin", "1 - Nbins with ApplyFuncToFindBin (no under/overflow)", "0 - 10 with raw bin value passed ApplyFuncToFindBin"])
def testApplyRangeSet(caplog, minVal, maxVal, minValFunc, maxValFunc, expectedFunc, testTHnSparse):
    """ Test apply a range set to an axis via a HistAxisRange object. 

    This is intentionally tested against SetRangeUser, so we can be certain that it reproduces
    that selection as expected.
    
    Note:
        It doens't matter whether we operate on TH1 or THn, since they both set ranges on TAxis.

    Note:
        This implicity tests ApplyFuncToFindBin, which is fine given how often the two are used
        together (almost always).
    """
    caplog.set_level(loggingLevel)
    selectedAxis = selectedTestAxis.axisOne
    sparse, _ = testTHnSparse
    expectedAxis = sparse.GetAxis(selectedAxis.value).Clone("axis2")
    expectedFunc(expectedAxis, minVal, maxVal)

    obj = projectors.HistAxisRange(axisRangeName = "axisOneTest",
            axisType = selectedAxis,
            minVal = minValFunc(minVal),
            maxVal = maxValFunc(maxVal))
    # Applys the restriction to the sparse.
    obj.ApplyRangeSet(sparse)
    ax = sparse.GetAxis(selectedAxis.value)

    # Unfortunately, equality comparison doesn't work...
    # GetXmin() and GetXmax() aren't restircted by SetRange(), so instead use GetFirst() and GetLast()
    assert ax.GetFirst() == expectedAxis.GetFirst()
    assert ax.GetLast() == expectedAxis.GetLast()

@pytest.mark.parametrize("func, value, expected", [
        (None, 3, 3),
        (ROOT.TAxis.GetNbins, None, 10),
        (ROOT.TAxis.FindBin, 10 - utils.epsilon, 5)
    ], ids = ["Only value", "Func only", "Func with value"])
def testRetrieveAxisValue(caplog, func, value, expected, testTHnSparse):
    """ Test retrieving axis values using ApplyFuncToFindBin(). """
    caplog.set_level(loggingLevel)
    selectedAxis = selectedTestAxis.axisOne
    sparse, _ = testTHnSparse
    expectedAxis = sparse.GetAxis(selectedAxis.value)

    assert projectors.HistAxisRange.ApplyFuncToFindBin(func, value)(expectedAxis) == expected

def testProjectors(caplog, testRootHists):
    """ Test creation and basic methods of the projection class. """
    caplog.set_level(loggingLevel)
    # Args
    projectionNameFormat = "{test} world"
    # Create object
    obj = projectors.HistProjector(observableList = {},
            observableToProjectFrom = {},
            projectionNameFormat = projectionNameFormat,
            projectionInformation = {})

    # These objects should be overridden so they aren't super meaningful, but we can still
    # test to ensure that they provide the basic functionality that is expected.
    assert obj.ProjectionName(test = "Hello") == projectionNameFormat.format(test = "Hello")
    assert obj.GetHist(observable = testRootHists.hist2D) == testRootHists.hist2D
    assert obj.OutputKeyName(inputKey = "inputKey",
            outputHist = testRootHists.hist2D,
            projectionName = projectionNameFormat.format(test = "Hello")) == projectionNameFormat.format(test = "Hello")
    assert obj.OutputHist(outputHist = testRootHists.hist1D,
            inputObservable = testRootHists.hist2D) == testRootHists.hist1D

def testTH1Projection(caplog, testRootHists):
    """ Test projection of a TH1 derived class. """
    caplog.set_level(loggingLevel)

    assert False

def testTHnProjection(caplog, testRootHists):
    """ Test projection of a THnSparse. """
    caplog.set_level(loggingLevel)

    assert False

