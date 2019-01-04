#!/usr/bin/env python

""" Tests for the analysis_objects module.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import dataclasses
import logging
import numpy as np
import os
import pytest
import ruamel.yaml

from jet_hadron.base import analysis_config
from jet_hadron.base import analysis_objects
from jet_hadron.base import params

# Setup logger
logger = logging.getLogger(__name__)
# For reproducibility
np.random.seed(1234)

@pytest.mark.parametrize("corr_type, expected", [
    ("full_range",
        {"str": "full_range",
            "display_str": "Full Range"}),
    ("signal_dominated",
        {"str": "signal_dominated",
            "display_str": "Signal Dominated"}),
    ("near_side",
        {"str": "near_side",
            "display_str": "Near Side"})
], ids = ["full range", "dPhi signal dominated", "dEta near side"])
def test_correlation_types(logging_mixin, corr_type, expected):
    """ Test jet-hadron correlation types. """
    obj = analysis_objects.JetHCorrelationType[corr_type]

    assert str(obj) == expected["str"]
    assert obj.display_str() == expected["display_str"]

@pytest.mark.parametrize("leading_hadron_bias", [
    (params.LeadingHadronBiasType.track),
    (params.LeadingHadronBias(type = params.LeadingHadronBiasType.track, value = 5))
], ids = ["leadingHadronEnum", "leadingHadronClass"])
def test_JetHBase_object_construction(logging_mixin, leading_hadron_bias, object_yaml_config, override_options_helper, check_JetHBase_object, mocker):
    """ Test construction of the JetHBase object. """
    object_config, task_name = object_yaml_config
    (config, selected_analysis_options) = override_options_helper(
        object_config,
        config_containing_override = object_config[task_name]
    )

    # Avoid os.makedirs actually making directories
    mocker.patch("os.makedirs")

    config_filename = "configFilename.yaml"
    task_config = config[task_name]
    reaction_plane_orientation = params.ReactionPlaneOrientation.all
    config_base = analysis_objects.JetHBase(
        task_name = task_name,
        config_filename = config_filename,
        config = config,
        task_config = task_config,
        collision_energy = selected_analysis_options.collision_energy,
        collision_system = selected_analysis_options.collision_system,
        event_activity = selected_analysis_options.event_activity,
        leading_hadron_bias = selected_analysis_options.leading_hadron_bias,
        reaction_plane_orientation = reaction_plane_orientation,
    )

    # We need values to compare against. However, namedtuples are immutable,
    # so we have to create a new one with the proper value.
    temp_selected_options = selected_analysis_options.asdict()
    temp_selected_options["leading_hadron_bias"] = leading_hadron_bias
    selected_analysis_options = params.SelectedAnalysisOptions(**temp_selected_options)
    # Only need for the case of LeadingHadronBiasType!
    if isinstance(leading_hadron_bias, params.LeadingHadronBiasType):
        selected_analysis_options = analysis_config.determine_leading_hadron_bias(config, selected_analysis_options)

    # Assertions are performed in this function
    res = check_JetHBase_object(
        obj = config_base,
        config = config,
        selected_analysis_options = selected_analysis_options,
        reaction_plane_orientation = reaction_plane_orientation
    )
    assert res is True

    # Just to be safe
    mocker.stopall()

def test_correlation_observable1D(logging_mixin, mocker):
    """ Tests for CorrelationObservable1D. Implicitly tests Observable and CorrelationObservable. """
    # Arguments are not selected for any particular reason
    values = {
        "hist": mocker.MagicMock(), "jet_pt_bin": 2, "track_pt_bin": 3,
        "axis": mocker.MagicMock(),
        "correlation_type": analysis_objects.JetHCorrelationType.signal_dominated
    }

    obj = analysis_objects.CorrelationObservable1D(**values)
    assert obj.hist == values["hist"]
    assert obj.jet_pt_bin == values["jet_pt_bin"]
    assert obj.track_pt_bin == values["track_pt_bin"]
    assert obj.axis == values["axis"]
    assert obj.correlation_type == values["correlation_type"]

def test_extracted_observable(logging_mixin):
    """ Tests for ExtractedObservable. """
    # Arguments are not selected for any particular reason
    values = {"jet_pt_bin": 2, "track_pt_bin": 3,
              "value": 1.5, "error": 0.3}

    obj = analysis_objects.ExtractedObservable(**values)
    assert obj.jet_pt_bin == values["jet_pt_bin"]
    assert obj.track_pt_bin == values["track_pt_bin"]
    assert obj.value == values["value"]
    assert obj.error == values["error"]

def test_hist_container(logging_mixin, test_root_hists):
    """ Test the hist container class function override. """
    (hist, hist2D, hist3D) = dataclasses.astuple(test_root_hists)

    obj = analysis_objects.HistContainer(hist)
    # Test the basic properties.
    assert obj.GetName() == "test"
    # Defined functions are called directly.
    assert obj.calculate_final_scale_factor() == 10.0

@pytest.mark.parametrize("hist_index, expected", [
    (0, {"scale_factor": 10.0}),
    (1, {"scale_factor": 5.0}),
    (2, {"scale_factor": 0.5}),
], ids = ["hist1D", "hist2D", "hist3D"])
def test_hist_container_scale_factor(logging_mixin, hist_index, expected, test_root_hists):
    """ Test hist container scale factor calculation. """
    obj = analysis_objects.HistContainer(dataclasses.astuple(test_root_hists)[hist_index])
    assert obj.calculate_final_scale_factor() == expected["scale_factor"]
    additional_scale_factor = 0.5
    assert obj.calculate_final_scale_factor(additional_scale_factor = additional_scale_factor) == expected["scale_factor"] * additional_scale_factor

def test_hist_container_clone_and_scale(logging_mixin, test_root_hists):
    """ Test hist container cloning and scaling by bin width. """
    # Test only a 1D hist because we test scaling of different hist dimensions elsewhere.
    hist = test_root_hists.hist1D
    # Add an additional entry so there is a bit more to test.
    hist.Fill(.2)

    obj = analysis_objects.HistContainer(hist)
    scaled_hist = obj.create_scaled_by_bin_width_hist()
    obj.Scale(1 / obj.GetXaxis().GetBinWidth(1))

    assert scaled_hist.GetEntries() == obj.GetEntries()
    # Compare all bins
    for bin in range(1, obj.GetXaxis().GetNbins() + 1):
        assert scaled_hist.GetBinContent(bin) == obj.GetBinContent(bin)

@pytest.fixture
def createHistArray():
    """ Create a basic HistArray for testing.

    Returns:
        tuple: (HistArray, dict of args used to create the hist array)
    """
    args = {"_binCenters": np.arange(1, 11),
            "_array": np.random.random_sample(10),
            "_errors": np.random.random_sample(10) / 10.0}

    obj = analysis_objects.HistArray(**args)

    return (obj, args)

def testHistArray(logging_mixin, createHistArray):
    """ Test basic HistArray functionality. """
    (obj, args) = createHistArray

    assert np.array_equal(obj.array, args["_array"])
    assert np.array_equal(obj.histData, args["_array"])
    assert np.array_equal(obj.binCenters, args["_binCenters"])
    assert np.array_equal(obj.x, args["_binCenters"])
    assert np.array_equal(obj.errors, args["_errors"])

def testHistArrayFromRootHist(logging_mixin, test_root_hists):
    """ Test creating a HistArray from a ROOT hist. """
    hist = test_root_hists.hist1D
    obj = analysis_objects.HistArray.initFromRootHist(hist)

    xAxis = hist.GetXaxis()
    xBins = range(1, xAxis.GetNbins() + 1)
    histData = np.array([hist.GetBinContent(i) for i in xBins])
    binCenters = np.array([xAxis.GetBinCenter(i) for i in xBins])
    errors = np.array([hist.GetBinError(i) for i in xBins])

    assert np.allclose(obj.array, histData)
    assert np.allclose(obj.binCenters, binCenters)
    assert np.allclose(obj.errors, errors)

@pytest.fixture
def createFitContainer(mocker):
    """ Create a basic fit conatiner for testing.

    Returns:
        tuple: (FitContainer, dict of args used to create the hist array)
    """
    values = {"jetPtBin": 1, "trackPtBin": 3,
              "fitType": analysis_objects.JetHCorrelationType.signal_dominated,
              "values": {"B": 1, "BG": 2},
              "params": {"B": 1, "limit_v3": [-0.1, 0.5]},
              "covarianceMatrix": {("a", "b"): 1.234},
              "errors": {("all", "signal_dominated"): [1, 2, 3]}}
    obj = analysis_objects.FitContainer(**values)

    return (obj, values)

def testFitContainer(logging_mixin, createFitContainer):
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
    "signal_dominated",
    analysis_objects.JetHCorrelationType.signal_dominated
], ids = ["str obj type", "enum obj type"])
@pytest.mark.parametrize("obj, objArgs", [
    (analysis_objects.HistArray, {"jetPtBin": 1, "trackPtBin": 3}),
    (analysis_objects.FitContainer, {"jetPtBin": 1, "trackPtBin": 4})
], ids = ["HistArray", "FitContainer"])
def testAnalysisObjectsWithYAMLReadAndWrite(logging_mixin, obj, objArgs, objType, mocker):
    """ Test initializing and writing objects to/from YAML files.

    This tests both HistArray and FitContainer objects.

    Note:
        This test uses real files, which are opened through mocked open() objects. The mocking is
        to avoid any read/write side effects, while the real files are used to ensure the data
        for the test accurately represents real world usage.
    """
    objArgs["prefix"] = os.path.join(os.path.dirname(os.path.realpath(__file__)), "testFiles")
    objArgs["objType"] = objType

    # Make sure the file will be read
    mExists = mocker.MagicMock(return_value = True)
    mocker.patch("jet_hadron.base.analysis_objects.os.path.exists", mExists)

    # Determine the filename and get the test data.
    # `inputData` contains the data for the test.
    dataFilename = obj.yamlFilename(**objArgs)
    with open(dataFilename, "r") as f:
        inputData = f.read()
        f.seek(0)
        yaml = ruamel.yaml.YAML(typ = "rt")
        yaml.default_flow_style = False
        expectedYAMLParameters = yaml.load(f)

    # Read
    mRead = mocker.mock_open(read_data = inputData)
    mocker.patch("pachyderm.utils.open", mRead)
    testObj = obj.initFromYAML(**objArgs)
    # Check the expected read call.
    mRead.assert_called_once_with(dataFilename, "r")

    # Test a few object specific details
    # It isn't exactly ideal to check like this, but I think it's fine
    if isinstance(testObj, analysis_objects.HistArray):
        assert type(testObj.binCenters) is np.ndarray
        assert type(testObj.histData) is np.ndarray
        assert type(testObj.errors) is np.ndarray
    if isinstance(testObj, analysis_objects.FitContainer):
        # This should be converted on import.
        for k, v in testObj.errors.items():
            assert type(v) is np.ndarray

    # Write
    mWrite = mocker.mock_open()
    mocker.patch("pachyderm.utils.open", mWrite)
    mYaml = mocker.MagicMock()
    mocker.patch("pachyderm.utils.ruamel.yaml.YAML.dump", mYaml)
    testObj.saveToYAML(**objArgs)
    # Check the expected write and YAML calls.
    mWrite.assert_called_once_with(dataFilename, "w")
    mYaml.assert_called_once_with(expectedYAMLParameters, mWrite())

