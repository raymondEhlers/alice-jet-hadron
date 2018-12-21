#!/usr/bin/env python

""" Tests for the JetH configuration functionality defined in the analysis_config module.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

# Py2/3
from future.utils import iteritems

import copy
import inspect
import logging
import pytest
import ruamel.yaml
from io import StringIO

from pachyderm import generic_config

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
        semiCentral:
            cluster:
                value: 6
aliceLabel: "thesis"
override:
    # Just need a trivial override value, since "override" is a required field.
    aliceLabel: "final"
    """
    yaml = ruamel.yaml.YAML()
    data = yaml.load(test_yaml)
    return data

@pytest.mark.parametrize("bias_type, event_activity, expected_leading_hadron_bias_value", [
    ("track", None, 5),
    ("cluster", None, 10),
    ("cluster", "semiCentral", 6),
], ids = ["track5", "cluster10", "cluster6"])
def test_determine_leading_hadron_bias(logging_mixin, bias_type, event_activity, expected_leading_hadron_bias_value, leading_hadron_bias_config):
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

def log_yaml_dump(yaml, config):
    """ Helper function to log the YAML config. """
    s = StringIO()
    yaml.dump(config, s)
    s.seek(0)
    logger.debug(s)

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
        semiCentral:
            responseTaskName: "ignoreThisValue"
"""

    yaml = ruamel.yaml.YAML()
    data = yaml.load(test_yaml)
    return data

def override_options_helper(config, selected_options = None, config_containing_override = None):
    """ Helper function to override the configuration.

    It can print the configuration before and after overridding the options if enabled.

    NOTE: If selected_options is not specified, it defaults to (2.76, "PbPb", "central", "track")

    Args:
        config (CommentedMap): dict-like object containing the configuration to be overridden.
        selected_options (params.SelectedAnalysisOptions): The options selected for this analysis, in
            the order defined used with analysis_config.override_options() and in the configuration file.
        config_containing_override (CommentedMap): dict-like object containing the override options.
    Returns:
        tuple: (dict-like CommentedMap object containing the overridden configuration, selected analysis
                    options used with the config)
    """
    if selected_options is None:
        selected_options = params.SelectedAnalysisOptions(
            collision_energy = params.CollisionEnergy.twoSevenSix,
            collision_system = params.CollisionSystem.PbPb,
            event_activity = params.EventActivity.central,
            leading_hadron_bias = params.LeadingHadronBiasType.track
        )

    yaml = ruamel.yaml.YAML()
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Before override:")
        log_yaml_dump(yaml, config)

    config = analysis_config.override_options(config = config,
                                              selected_options = selected_options,
                                              config_containing_override = config_containing_override)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("After override:")
        log_yaml_dump(yaml, config)

    return (config, selected_options)

def test_basic_selected_overrides(logging_mixin, basic_config):
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

def test_ignore_unselected_options(logging_mixin, basic_config):
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
                 "-a", "semiCentral",
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
        assert selected_analysis_options.event_activity == "semiCentral"
        assert selected_analysis_options.leading_hadron_bias == "track"

        # Strictly speaking, this is adding some complication, but it will consistently be used with this option,
        # so it's worth doing the integration test.
        validated_analysis_options, _ = analysis_config.validate_arguments(selected_analysis_options)

        assert validated_analysis_options.collision_energy == params.CollisionEnergy.fiveZeroTwo
        assert validated_analysis_options.collision_system == params.CollisionSystem.embedPP
        assert validated_analysis_options.event_activity == params.EventActivity.semiCentral
        assert validated_analysis_options.leading_hadron_bias == params.LeadingHadronBiasType.track

@pytest.mark.parametrize("args, expected", [
    ((2.76, "PbPb", "central", "track"), None),
    ((None, "PbPb", "central", "track"), None),
    ((2.76, None, "central", "track"), None),
    ((2.76, "PbPb", None, "track"), None),
    ((2.76, "PbPb", "central", None), None),
    ((params.CollisionEnergy.twoSevenSix,
      params.CollisionSystem.PbPb,
      params.EventActivity.central,
      params.LeadingHadronBiasType.track), None),
    ((5.02, "embedPP", "semiCentral", "cluster"),
     (params.CollisionEnergy.fiveZeroTwo,
      params.CollisionSystem.embedPP,
      params.EventActivity.semiCentral,
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
        expected = (params.CollisionEnergy.twoSevenSix,
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

def check_jetH_base_object(obj, config, selected_analysis_options, event_plane_angle, **kwargs):
    """ Helper function to check JetHBase properties.

    The values are asserted in this function.

    Note:
        The default values correspond to those in the object_config config, so they don't
        need to be specified in each function.

    Args:
        obj (analysis_config.JetHBase): JetHBase object to compare values against.
        config (CommentedMap): dict-like configuration file.
        selected_analysis_options (params.SelectedAnalysisOptions): Selected analysis options.
        event_plane_angle (params.EventPlaneAngle): Selected event plane angle.
        kwargs (dict): All other values to compare against for which the default value defined
            in this function is not sufficient.
    Returns:
        bool: True if it successfully make it to the end of the assertions.
    """
    # Determine default values
    task_name = "taskName"
    default_values = {
        "task_name": task_name,
        "config_filename": "configFilename.yaml",
        "config": config,
        "task_config": config[task_name],
        "event_plane_angle": event_plane_angle
    }
    # Add these afterwards so we don't have to do each value by hand.
    default_values.update(selected_analysis_options.asdict())
    # NOTE: All other values will be taken from the config when constructing the object.
    for k, v in iteritems(default_values):
        if k not in kwargs:
            kwargs[k] = v

    # Creating thie object is something of a tautology, because in both cases we use the
    # constructed object, so we also have other tests, which are performed below.
    # However, we keep the test to provide a check for the equality operators.
    # We perform the actual test later because it is not very verbose.
    comparison = analysis_config.JetHBase(**kwargs)

    # We need to retrieve these values so we can test them directly.
    # `default_values` will now be used as the set of reference values.
    value_names = ["inputFilename", "inputListName", "outputPrefix", "outputFilename", "printingExtensions", "aliceLabel"]
    for k in value_names:
        # The AliceLabel gets converted to the enum in the object, so we need to do the conversion here.
        if k == "aliceLabel":
            val = params.AliceLabel[config[k]]
        else:
            val = config[k]
        default_values[k] = val
    default_values.update(kwargs)

    # Directly compare against the available values
    # NOTE: This isn't wholly independent because the object comparison relies on comparing values
    #       in obj.__dict__, but it is still an improvement on the above.
    # Use we this map to allow us to translate between the attribute names and the config names.
    default_values_key_map = {
        "input_filename": "inputFilename",
        "input_list_name": "inputListName",
        "output_prefix": "outputPrefix",
        "output_filename": "outputFilename",
        "printing_extensions": "printingExtensions",
        "alice_label": "aliceLabel",
    }
    for prop, val in iteritems(obj.__dict__):
        assert val == default_values[default_values_key_map.get(prop, prop)]

    # Perform the comparison test.
    assert obj == comparison

    return True

@pytest.fixture
def object_config():
    """ Object YAML configuration to test object args, validation, and construction.

    Args:
        None
    Returns:
        CommentedMap: dict-like object from ruamel.yaml containing the configuration.
    """

    test_yaml = """
iterables:
    event_plane_angle: False
    QVector: False
leadingHadronBiasValues:
    track:
        value: 5
    2.76:
        central:
            cluster:
                value: 10
        semiCentral:
            cluster:
                value: 6
inputFilename: "inputFilenameValue"
inputListName: "inputListNameValue"
outputPrefix: "outputPrefixValue"
outputFilename: "outputFilenameValue"
printingExtensions: ["png", "pdf"]
aliceLabel: "thesis"
taskName:
    test: "val"
    override:
        # Just need a trivial override value, since "override" is a required field.
        aliceLabel: "final"
        iterables:
            event_plane_angle: True
            qVector:
                - "all"
"""
    yaml = ruamel.yaml.YAML()
    data = yaml.load(test_yaml)
    return (data, "taskName")

@pytest.mark.parametrize("leading_hadron_bias", [
    (params.LeadingHadronBiasType.track),
    (params.LeadingHadronBias(type = params.LeadingHadronBiasType.track, value = 5))
], ids = ["leadingHadronEnum", "leadingHadronClass"])
def test_jetH_base_object_construction(logging_mixin, leading_hadron_bias, object_config, mocker):
    """ Test construction of the JetHBase object. """
    object_config, task_name = object_config
    (config, selected_analysis_options) = override_options_helper(
        object_config,
        config_containing_override = object_config[task_name]
    )

    # Avoid os.makedirs actually making directories
    mocker.patch("os.makedirs")

    config_filename = "configFilename.yaml"
    task_config = config[task_name]
    event_plane_angle = params.EventPlaneAngle.all
    config_base = analysis_config.JetHBase(
        task_name = task_name,
        config_filename = config_filename,
        config = config,
        task_config = task_config,
        collision_energy = selected_analysis_options.collision_energy,
        collision_system = selected_analysis_options.collision_system,
        event_activity = selected_analysis_options.event_activity,
        leading_hadron_bias = selected_analysis_options.leading_hadron_bias,
        event_plane_angle = event_plane_angle,
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
    res = check_jetH_base_object(
        obj = config_base,
        config = config,
        selected_analysis_options = selected_analysis_options,
        event_plane_angle = event_plane_angle
    )
    assert res is True

    # Just to be safe
    mocker.stopall()

@pytest.mark.parametrize("additional_iterables", [
    None,
    {"iterable1": params.CollisionEnergy, "iterable2": params.CollisionSystem}
], ids = ["No additional iterables", "Two additional iterables"])
def test_construct_object_from_config(logging_mixin, additional_iterables, object_config, mocker):
    """ Test construction of objects through a configuration file.

    NOTE: This is an integration test. """
    # Basic setup
    # We need both the input and the expected out.
    # NOTE: We only want to override the options of the expected config because
    #       construct_from_configuration_file() applies the overriding itself.
    config, task_name = object_config
    expected_names = ["event_plane_angle", "qVector"]
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
    obj = analysis_config.JetHBase

    # Mock reading the config
    #load_configuration_mock = mocker.MagicMock(spec_set = ["filename"], return_value = config)
    # Needs the full path to the module.
    load_configuration_mock = mocker.patch("jet_hadron.base.analysis_config.generic_config.load_configuration", return_value = config)
    # Avoid os.makedirs actually making directories
    mocker.patch("os.makedirs")

    (_, names, objects) = analysis_config.construct_from_configuration_file(
        task_name = task_name,
        config_filename = config_filename,
        selected_analysis_options = selected_analysis_options,
        obj = obj,
        additional_possible_iterables = additional_iterables,
    )

    # Check the opening the config file was called properly.
    # Need to collect the classes from the params and analysis_objects modules to check it.
    classes_to_register = set([])
    for module in [params, analysis_objects]:
        classes_to_register.update([member[1] for member in inspect.getmembers(module, inspect.isclass)])
    load_configuration_mock.assert_called_once_with(
        filename = config_filename,
        classes_to_register = classes_to_register
    )

    assert names == expected_names
    for values, obj in generic_config.iterate_with_selected_objects(objects):
        res = check_jetH_base_object(obj = obj,
                                     config = expected_config,
                                     selected_analysis_options = expected_analysis_options,
                                     event_plane_angle = values.event_plane_angle)
        assert res is True

    # Just to be safe
    mocker.stopall()
