#!/usr/bin/env python

# Tests for the analysisObjects module.
#
# author: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
# date: 8 May 2018

import builtins
from future.utils import iteritems

import pytest
import os
import copy
import ruamel.yaml
import numpy as np
import logging
# Setup logger
logger = logging.getLogger(__name__)

import jetH.base.analysisObjects as analysisObjects

# For reproducibility
np.random.seed(1234)

@pytest.mark.parametrize("corrType, expected", [
        ("fullRange",
            {"str" : "fullRange",
            "displayStr" : "Full Range"}),
        ("signalDominated",
            {"str" : "signalDominated",
            "displayStr" : "Signal Dominated"}),
        ("nearSide",
            {"str" : "nearSide",
            "displayStr" : "Near Side"})
    ], ids = ["full range", "dPhi signal dominated", "dEta near side"])
def testCorrelationTypes(loggingMixin, corrType, expected):
    """ Test jet-hadron correlation types. """
    obj = analysisObjects.jetHCorrelationType[corrType]

    assert str(obj) == expected["str"]
    assert obj.str() == expected["str"]
    assert obj.displayStr() == expected["displayStr"]

def testCorrelationObservable1D(loggingMixin, mocker):
    """ Tests for CorrelationObservable1D. Implicitly tests Observable and CorrelationObservable. """
    # Arguments are not selected for any particular reason
    values = {"hist" : mocker.MagicMock(), "jetPtBin" : 2, "trackPtBin" : 3,
            "axis" : mocker.MagicMock(),
            "correlationType" : analysisObjects.jetHCorrelationType.signalDominated}

    obj = analysisObjects.CorrelationObservable1D(**values)
    assert obj.hist == values["hist"]
    assert obj.jetPtBin == values["jetPtBin"]
    assert obj.trackPtBin == values["trackPtBin"]
    assert obj.axis == values["axis"]
    assert obj.correlationType == values["correlationType"]

def testExtractedObservable(loggingMixin):
    """ Tests for ExtractedObservable. """
    # Arguments are not selected for any particular reason
    values = {"jetPtBin" : 2, "trackPtBin" : 3,
              "value" : 1.5, "error" : 0.3}

    obj = analysisObjects.ExtractedObservable(**values)
    assert obj.jetPtBin == values["jetPtBin"]
    assert obj.trackPtBin == values["trackPtBin"]
    assert obj.value == values["value"]
    assert obj.error == values["error"]

def testHistContainer(loggingMixin, testRootHists):
    """ Test the hist container class function override. """
    (hist, hist2D, hist3D) = testRootHists

    obj = analysisObjects.HistContainer(hist)
    # Test the basic properties.
    assert obj.GetName() == "test"
    # Defined functions are called directly.
    assert obj.calculateFinalScaleFactor() == 10.0

@pytest.mark.parametrize("histIndex, expected", [
        (0, {"scaleFactor" : 10.0}),
        (1, {"scaleFactor" :  5.0}),
        (2, {"scaleFactor" :  0.5}),
    ], ids = ["hist1D", "hist2D", "hist3D"])
def testHistContainerScaleFactor(loggingMixin, histIndex, expected, testRootHists):
    """ Test hist container scale factor calculation. """
    obj = analysisObjects.HistContainer(testRootHists[histIndex])
    assert obj.calculateFinalScaleFactor() == expected["scaleFactor"]
    additionalScaleFactor = 0.5
    assert obj.calculateFinalScaleFactor(additionalScaleFactor = additionalScaleFactor) == expected["scaleFactor"]*additionalScaleFactor

def testHistContainerCloneAndScale(loggingMixin, testRootHists):
    """ Test hist container cloning and scaling by bin width. """
    # Test only a 1D hist because we test scaling of different hist dimensions elsewhere.
    hist = testRootHists.hist1D
    # Add an additional entry so there is a bit more to test.
    hist.Fill(.2)

    obj = analysisObjects.HistContainer(hist)
    scaledHist = obj.createScaledByBinWidthHist()
    obj.Scale(1/obj.GetXaxis().GetBinWidth(1))

    assert scaledHist.GetEntries() == obj.GetEntries()
    # Compare all bins
    for bin in range(1, obj.GetXaxis().GetNbins()+1):
        assert scaledHist.GetBinContent(bin) == obj.GetBinContent(bin)

@pytest.fixture
def createHistArray():
    """ Create a basic HistArray for testing.

    Returns:
        tuple: (HistArray, dict of args used to create the hist array)
    """
    args = {"_binCenters" : np.arange(1, 11),
            "_array" : np.random.random_sample(10),
            "_errors" : np.random.random_sample(10)/10.0}

    obj = analysisObjects.HistArray(**args)

    return (obj, args)

def testHistArray(loggingMixin, createHistArray):
    """ Test basic HistArray functionality. """
    (obj, args) = createHistArray

    assert np.array_equal(obj.array, args["_array"])
    assert np.array_equal(obj.histData, args["_array"])
    assert np.array_equal(obj.binCenters, args["_binCenters"])
    assert np.array_equal(obj.x, args["_binCenters"])
    assert np.array_equal(obj.errors, args["_errors"])

def testHistArrayFromRootHist(loggingMixin, testRootHists):
    """ Test creating a HistArray from a ROOT hist. """
    hist = testRootHists.hist1D
    obj = analysisObjects.HistArray.initFromRootHist(hist)

    xAxis = hist.GetXaxis()
    xBins = range(1, xAxis.GetNbins() + 1)
    histData = np.array([hist.GetBinContent(i) for i in xBins])
    binCenters = np.array([xAxis.GetBinCenter(i) for i in xBins])
    errors = np.array([hist.GetBinError(i) for i in xBins])

    assert np.array_equal(obj.array, histData)
    assert np.array_equal(obj.binCenters, binCenters)
    assert np.array_equal(obj.errors, errors)

@pytest.fixture
def createFitContainer(mocker):
    """ Create a basic fit conatiner for testing.

    Returns:
        tuple: (FitContainer, dict of args used to create the hist array)
    """
    values = {"jetPtBin" : 1, "trackPtBin" : 3,
            "fitType" : analysisObjects.jetHCorrelationType.signalDominated,
            "values" : {"B" : 1, "BG" : 2},
            "params" : {"B" : 1, "limit_v3" : [-0.1, 0.5]},
            "covarianceMatrix" : {("a", "b") : 1.234},
            "errors" : {("all", "signalDominated") : [1, 2, 3]}}
    obj = analysisObjects.FitContainer(**values)

    return (obj, values)

def testFitContainer(loggingMixin, createFitContainer):
    """ Test FitContainer initialization. """
    (obj, values) = createFitContainer

    assert obj.jetPtBin == values["jetPtBin"]
    assert obj.trackPtBin == values["trackPtBin"]
    assert obj.fitType == values["fitType"]
    assert obj.values == values["values"]
    assert obj.params == values["params"]
    assert obj.covarianceMatrix == values["covarianceMatrix"]
    assert obj.errors == values["errors"]

@pytest.mark.parametrize("objType", [
        "signalDominated",
        analysisObjects.jetHCorrelationType.signalDominated
    ], ids = ["str obj type", "enum obj type"])
@pytest.mark.parametrize("obj, objArgs", [
    (analysisObjects.HistArray, {"jetPtBin" : 1, "trackPtBin" : 3}),
    (analysisObjects.FitContainer, {"jetPtBin" : 1, "trackPtBin" : 4})
    ], ids = ["HistArray", "FitContainer"])
def testAnalysisObjectsWithYAMLReadAndWrite(loggingMixin, obj, objArgs, objType, mocker):
    """ Test initializing and writing objects to/from YAML files. Tests both HistArray and FitContainer objects.

    NOTE: This test uses real files, which are opened through mocked open() objects. The mocking is
          to avoid any read/write side effects, while the real files are used to ensure the data
          for the test accurately represents real world usage.
    """
    objArgs["prefix"] = os.path.join(os.path.dirname(os.path.realpath(__file__)), "testFiles")
    objArgs["objType"] = objType

    # Make sure the file will be read
    mocker.patch("os.path.exists")

    # Determine the filename and get the test data
    # `inputData` contains the data for the test
    dataFilename = obj.yamlFilename(**objArgs)
    inputData = ""
    with open(dataFilename, "r") as f:
        inputData = f.read()

    # Read
    mRead = mocker.mock_open(read_data = inputData)
    mocker.patch("builtins.open", mRead)
    testObj = obj.initFromYAML(**objArgs)
    calls = mRead.mock_calls
    mRead.assert_called_once_with(dataFilename, "r")

    # Test a few object specific details
    # It isn't exactly ideal to check like this, but I think it's fine
    if isinstance(testObj, analysisObjects.HistArray):
        assert type(testObj.binCenters) is np.ndarray
        assert type(testObj.histData) is np.ndarray
        assert type(testObj.errors) is np.ndarray
    if isinstance(testObj, analysisObjects.FitContainer):
        # This should be converted on import.
        for k, v in iteritems(testObj.errors):
            assert type(v) is np.ndarray

    # Write
    mWrite = mocker.mock_open(read_data = inputData)
    mocker.patch("builtins.open", mWrite)
    testObj.saveToYAML(**objArgs)
    mWrite.assert_called_with(dataFilename, "wb")
    calls = mWrite.mock_calls
    fileHandle = mWrite()
    writtenData = fileHandle.read()
    # NOTE: Cannot use assert_called_once_with() here because yaml.dump() calls
    #       write many times. Perhaps once per line? In any case, far too often
    #       to mock easily.
    assert inputData == writtenData

    # Necessary to ensure that profiling works (it seems that it runs before all mocks are cleared)
    # Probably something to do with mocking open
    mocker.stopall()

