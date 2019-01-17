#!/usr/bin/env python

""" Tests for the JetH configuration functionality defined in the analysis_config module.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import copy
import logging
import pytest

from pachyderm import generic_config
from pachyderm import yaml

from jet_hadron.base import analysis_config
from jet_hadron.base import analysis_objects
from jet_hadron.base import params

# Setup logger
logger = logging.getLogger(__name__)

@pytest.fixture
def leading_hadron_bias_config():
    """ Basic configuration for testing the leading hadron bias determination. """
    test_yaml = """
leadingHadronBiasValues:
    track:
        value: 5
    2.76:
        central:
            cluster:
                value: 10
        semi_central:
            cluster:
                value: 6
aliceLabel: "thesis"
override:
    # Just need a trivial override value, since "override" is a required field.
    aliceLabel: "final"
    """
    yml = yaml.yaml()
    data = yml.load(test_yaml)
    return data

@pytest.mark.parametrize("bias_type, event_activity, expected_leading_hadron_bias_value", [
    ("track", None, 5),
    ("cluster", None, 10),
    ("cluster", "semi_central", 6),
], ids = ["track5", "cluster10", "cluster6"])
def test_determine_leading_hadron_bias(logging_mixin, bias_type, event_activity, expected_leading_hadron_bias_value, leading_hadron_bias_config, override_options_helper):
    """ Test determination of the leading hadron bias. """
    (config, selected_analysis_options) = override_options_helper(leading_hadron_bias_config)

    # Add in the different selected options
    if bias_type:
        # Both options will lead here. Doing this with "track" doesn't change the values, but
        # also doesn't hurt anything, so it's fine.
        kwargs = selected_analysis_options.asdict()
        kwargs["leading_hadron_bias"] = params.LeadingHadronBiasType[bias_type]
        selected_analysis_options = params.SelectedAnalysisOptions(**kwargs)
    if event_activity:
        kwargs = selected_analysis_options.asdict()
        kwargs["event_activity"] = params.EventActivity[event_activity]
        selected_analysis_options = params.SelectedAnalysisOptions(**kwargs)

    returned_options = analysis_config.determine_leading_hadron_bias(config = config, selected_analysis_options = selected_analysis_options)
    # Check that we still got these right.
    assert returned_options.collision_energy == selected_analysis_options.collision_energy
    assert returned_options.collision_system == selected_analysis_options.collision_system
    assert returned_options.event_activity == selected_analysis_options.event_activity
    # Check the bias value and type
    logger.debug(f"type(leading_hadron_bias): {type(returned_options.leading_hadron_bias)}")
    assert returned_options.leading_hadron_bias.value == expected_leading_hadron_bias_value
    assert returned_options.leading_hadron_bias.type == params.LeadingHadronBiasType[bias_type]

@pytest.fixture
def basic_config():
    """ Basic YAML configuration to test overriding the configuration.

    See the config for which selected options are implemented.

    Args:
        None
    Returns:
        CommentedMap: dict-like object from ruamel.yaml containing the configuration.
    """

    test_yaml = """
inputFilename: "inputFilenameValue"
inputListName: "inputListNameValue"
outputPrefix: "outputPrefixValue"
outputFilename: "outputFilenameValue"
printingExtensions: ["pdf"]
aliceLabel: "thesis"
trainNumber: 1234

responseTaskName: &responseTaskName ["baseName"]
intVal: 1
halfwayValue: 3.1
directOverride: 1
additionalValue: 1
mergeValue: 1
# Cannot perform a merge at the "track" level because otherwise
# it will just be overritten by the existing dict
# Instead, we need to explicitly add it to the "track" dict.
mergeKeyOverride: &mergeKeyOverride
    mergeValue: 2
override:
    track:
        directOverride: 2
        << : *mergeKeyOverride
    2.76:
        halfwayValue: 3.14
        central:
            responseTaskName: "jetHPerformance"
            intVal: 2
            track:
                additionalValue: 2
        semi_central:
            responseTaskName: "ignoreThisValue"
"""

    yml = yaml.yaml()
    data = yml.load(test_yaml)
    return data

def test_basic_selected_overrides(logging_mixin, basic_config, override_options_helper):
    """ Test that override works for the selected options. """
    (config, selected_analysis_options) = override_options_helper(basic_config)

    assert config["responseTaskName"] == "jetHPerformance"
    assert config["intVal"] == 2
    assert config["halfwayValue"] == 3.14
    # This doesn't need to follow the entire set of selected options to retireve the value.
    assert config["directOverride"] == 2
    # However, this one does follow the entire path
    assert config["additionalValue"] == 2
    # Ensures that merge keys also work
    assert config["mergeValue"] == 2

def test_ignore_unselected_options(logging_mixin, basic_config, override_options_helper):
    """ Test ignoring unselected values. """
    # Delete the central values, thereby removing any values to override.
    # Thus, the configuration values should not change!
    del basic_config["override"][2.76]["central"]

    (config, selected_analysis_options) = override_options_helper(basic_config)

    # NOTE: It should be compared against "baseName" because it also converts single entry
    #       lists to just the single entry.
    assert config["responseTaskName"] == "baseName"

def test_argument_parsing():
    """ Test argument parsing.  """
    test_args = ["-c", "analysis_configArg.yaml",
                 "-e", "5.02",
                 "-s", "embedPP",
                 "-a", "semi_central",
                 "-b", "track"]
    reversed_iterator = iter(reversed(test_args))
    reversed_test_args = []
    # Need to reverse by twos
    for x in reversed_iterator:
        reversed_test_args.extend([next(reversed_iterator), x])

    # Test both in the defined order and in a different order (just for completeness).
    for args in [test_args, reversed_test_args]:
        (config_filename, selected_analysis_options, returned_args) = analysis_config.determine_selected_options_from_kwargs(args = args)

        assert config_filename == "analysis_configArg.yaml"
        assert selected_analysis_options.collision_energy == 5.02
        assert selected_analysis_options.collision_system == "embedPP"
        assert selected_analysis_options.event_activity == "semi_central"
        assert selected_analysis_options.leading_hadron_bias == "track"

        # Strictly speaking, this is adding some complication, but it will consistently be used with this option,
        # so it's worth doing the integration test.
        validated_analysis_options, _ = analysis_config.validate_arguments(selected_analysis_options)

        assert validated_analysis_options.collision_energy == params.CollisionEnergy.five_zero_two
        assert validated_analysis_options.collision_system == params.CollisionSystem.embedPP
        assert validated_analysis_options.event_activity == params.EventActivity.semi_central
        assert validated_analysis_options.leading_hadron_bias == params.LeadingHadronBiasType.track

@pytest.mark.parametrize("args, expected", [
    ((2.76, "PbPb", "central", "track"), None),
    ((None, "PbPb", "central", "track"), None),
    ((2.76, None, "central", "track"), None),
    ((2.76, "PbPb", None, "track"), None),
    ((2.76, "PbPb", "central", None), None),
    ((params.CollisionEnergy.two_seven_six,
      params.CollisionSystem.PbPb,
      params.EventActivity.central,
      params.LeadingHadronBiasType.track), None),
    ((5.02, "embedPP", "semi_central", "cluster"),
     (params.CollisionEnergy.five_zero_two,
      params.CollisionSystem.embedPP,
      params.EventActivity.semi_central,
      params.LeadingHadronBiasType.cluster))
], ids = [
    "Standard 2.76",
    "Missing collision energy",
    "Missing collision system",
    "Missing event activity",
    "Missing leading hadron bias",
    "Standard 2.76 with enums",
    "5.02 semi-central embedPP with cluster bias"])
def test_validate_arguments(logging_mixin, args, expected):
    """ Test argument validation. """
    if expected is None:
        expected = (params.CollisionEnergy.two_seven_six,
                    params.CollisionSystem.PbPb,
                    params.EventActivity.central,
                    params.LeadingHadronBiasType.track)

    args = params.SelectedAnalysisOptions(*args)
    args, _ = analysis_config.validate_arguments(args)
    expected = params.SelectedAnalysisOptions(*expected)
    assert args.collision_energy == expected.collision_energy
    assert args.collision_system == expected.collision_system
    assert args.event_activity == expected.event_activity
    assert args.leading_hadron_bias == expected.leading_hadron_bias

@pytest.mark.parametrize("additional_iterables", [
    None,
    {"iterable1": params.CollisionEnergy, "iterable2": params.CollisionSystem}
], ids = ["No additional iterables", "Two additional iterables"])
def test_construct_object_from_config(logging_mixin, additional_iterables, object_yaml_config, override_options_helper, check_JetHBase_object, mocker):
    """ Test construction of objects through a configuration file.

    Note:
        This is an integration test.
    """
    # Basic setup
    # We need both the input and the expected out.
    # NOTE: We only want to override the options of the expected config because
    #       construct_from_configuration_file() applies the overriding itself.
    config, task_name = object_yaml_config
    expected_names = ["reaction_plane_orientation", "qvector"]
    if additional_iterables:
        for iterable in additional_iterables:
            expected_names.extend([iterable])
            config[task_name]["override"]["iterables"][iterable] = True
    expected_config = copy.deepcopy(config)
    (expected_config, selected_analysis_options) = override_options_helper(
        expected_config,
        config_containing_override = expected_config[task_name]
    )
    expected_analysis_options = analysis_config.determine_leading_hadron_bias(config = expected_config, selected_analysis_options = selected_analysis_options)

    # Task arguments
    config_filename = "configFilename.yaml"
    obj = analysis_objects.JetHBase

    # Mock reading the config
    # Needs the full path to the module.
    mocker.patch("jet_hadron.base.analysis_config.generic_config.load_configuration", return_value = config)
    # Avoid os.makedirs actually making directories
    mocker.patch("os.makedirs")

    (_, returned_iterables, objects) = analysis_config.construct_from_configuration_file(
        task_name = task_name,
        config_filename = config_filename,
        selected_analysis_options = selected_analysis_options,
        obj = obj,
        additional_possible_iterables = additional_iterables,
    )

    assert list(returned_iterables) == expected_names
    for values, obj in generic_config.iterate_with_selected_objects(objects):
        res = check_JetHBase_object(
            obj = obj,
            config = expected_config,
            selected_analysis_options = expected_analysis_options,
            reaction_plane_orientation = values.reaction_plane_orientation
        )
        assert res is True

    # Just to be safe
    mocker.stopall()
