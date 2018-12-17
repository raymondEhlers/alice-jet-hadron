#!/usr/bin/env python

""" Tests for the JetH configuration functionality defined in the analysis_config module.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

# Py2/3
from future.utils import iteritems

import copy
import logging
import pytest
import ruamel.yaml
from io import StringIO

from pachyderm import generic_config

from jet_hadron.base import analysis_config
from jet_hadron.base import params

# Setup logger
logger = logging.getLogger(__name__)

@pytest.fixture
def leadingHadronBiasConfig():
    """ Basic configuration for testing the leading hadron bias determination. """
    testYaml = """
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
    data = yaml.load(testYaml)
    return data

@pytest.mark.parametrize("biasType, eventActivity, expectedLeadingHadronBiasValue", [
    ("track", None, 5),
    ("cluster", None, 10),
    ("cluster", "semiCentral", 6),
], ids = ["track5", "cluster10", "cluster6"])
def testDetermineLeadingHadronBias(loggingMixin, biasType, eventActivity, expectedLeadingHadronBiasValue, leadingHadronBiasConfig):
    """ Test determination of the leading hadron bias. """
    (config, selectedAnalysisOptions) = overrideOptionsHelper(leadingHadronBiasConfig)

    # Add in the different selected options
    if biasType:
        # Both options will lead here. Doing this with "track" doesn't change the values, but
        # also doesn't hurt anything, so it's fine.
        kwargs = selectedAnalysisOptions.asdict()
        kwargs["leadingHadronBias"] = params.leadingHadronBiasType[biasType]
        selectedAnalysisOptions = params.SelectedAnalysisOptions(**kwargs)
    if eventActivity:
        kwargs = selectedAnalysisOptions.asdict()
        kwargs["eventActivity"] = params.eventActivity[eventActivity]
        selectedAnalysisOptions = params.SelectedAnalysisOptions(**kwargs)

    returnedOptions = analysis_config.determineLeadingHadronBias(config = config, selectedAnalysisOptions = selectedAnalysisOptions)
    # Check that we still got these right.
    assert returnedOptions.collisionEnergy == selectedAnalysisOptions.collisionEnergy
    assert returnedOptions.collisionSystem == selectedAnalysisOptions.collisionSystem
    assert returnedOptions.eventActivity == selectedAnalysisOptions.eventActivity
    # Check the bias value and type
    logger.debug("type(leadingHadronBias): {}".format(type(returnedOptions.leadingHadronBias)))
    assert returnedOptions.leadingHadronBias.value == expectedLeadingHadronBiasValue
    assert returnedOptions.leadingHadronBias.type == params.leadingHadronBiasType[biasType]

def log_yaml_dump(yaml, config):
    """ Helper function to log the YAML config. """
    s = StringIO()
    yaml.dump(config, s)
    s.seek(0)
    logger.debug(s)

@pytest.fixture
def basicConfig():
    """ Basic YAML configuration to test overriding the configuration.

    See the config for which selected options are implemented.

    Args:
        None
    Returns:
        CommentedMap: dict-like object from ruamel.yaml containing the configuration.
    """

    testYaml = """
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
    data = yaml.load(testYaml)
    return data

def overrideOptionsHelper(config, selectedOptions = None, configContainingOverride = None):
    """ Helper function to override the configuration.

    It can print the configuration before and after overridding the options if enabled.

    NOTE: If selectedOptions is not specified, it defaults to (2.76, "PbPb", "central", "track")

    Args:
        config (CommentedMap): dict-like object containing the configuration to be overridden.
        selectedOptions (params.SelectedAnalysisOptions): The options selected for this analysis, in
            the order defined used with analysis_config.overrideOptions() and in the configuration file.
        configContainingOverride (CommentedMap): dict-like object containing the override options.
    Returns:
        tuple: (dict-like CommentedMap object containing the overridden configuration, selected analysis
                    options used with the config)
    """
    if selectedOptions is None:
        selectedOptions = params.SelectedAnalysisOptions(collisionEnergy = params.collisionEnergy.twoSevenSix,
                                                         collisionSystem = params.collisionSystem.PbPb,
                                                         eventActivity = params.eventActivity.central,
                                                         leadingHadronBias = params.leadingHadronBiasType.track)

    yaml = ruamel.yaml.YAML()
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Before override:")
        log_yaml_dump(yaml, config)

    config = analysis_config.overrideOptions(config = config,
                                             selectedOptions = selectedOptions,
                                             configContainingOverride = configContainingOverride)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("After override:")
        log_yaml_dump(yaml, config)

    return (config, selectedOptions)

def testBasicSelectedOverrides(loggingMixin, basicConfig):
    """ Test that override works for the selected options. """
    (config, selectedAnalysisOptions) = overrideOptionsHelper(basicConfig)

    assert config["responseTaskName"] == "jetHPerformance"
    assert config["intVal"] == 2
    assert config["halfwayValue"] == 3.14
    # This doesn't need to follow the entire set of selected options to retireve the value.
    assert config["directOverride"] == 2
    # However, this one does follow the entire path
    assert config["additionalValue"] == 2
    # Ensures that merge keys also work
    assert config["mergeValue"] == 2

def testIgnoreUnselectedOptions(loggingMixin, basicConfig):
    """ Test ignoring unselected values. """
    # Delete the central values, thereby removing any values to override.
    # Thus, the configuration values should not change!
    del basicConfig["override"][2.76]["central"]

    (config, selectedAnalysisOptions) = overrideOptionsHelper(basicConfig)

    # NOTE: It should be compared against "baseName" because it also converts single entry
    #       lists to just the single entry.
    assert config["responseTaskName"] == "baseName"

def test_argument_parsing():
    """ Test argument parsing.  """
    testArgs = ["-c", "analysis_configArg.yaml",
                "-e", "5.02",
                "-s", "embedPP",
                "-a", "semiCentral",
                "-b", "track"]
    reversedIterator = iter(reversed(testArgs))
    reversedTestArgs = []
    # Need to reverse by twos
    for x in reversedIterator:
        reversedTestArgs.extend([next(reversedIterator), x])

    # Test both in the defined order and in a different order (just for completeness).
    for args in [testArgs, reversedTestArgs]:
        (configFilename, selectedAnalysisOptions, returnedArgs) = analysis_config.determine_selected_options_from_kwargs(args = args)

        assert configFilename == "analysis_configArg.yaml"
        assert selectedAnalysisOptions.collisionEnergy == 5.02
        assert selectedAnalysisOptions.collisionSystem == "embedPP"
        assert selectedAnalysisOptions.eventActivity == "semiCentral"
        assert selectedAnalysisOptions.leadingHadronBias == "track"

        # Strictly speaking, this is adding some complication, but it will consistently be used with this option,
        # so it's worth doing the integration test.
        validatedAnalysisOptions, _ = analysis_config.validateArguments(selectedAnalysisOptions)

        assert validatedAnalysisOptions.collisionEnergy == params.collisionEnergy.fiveZeroTwo
        assert validatedAnalysisOptions.collisionSystem == params.collisionSystem.embedPP
        assert validatedAnalysisOptions.eventActivity == params.eventActivity.semiCentral
        assert validatedAnalysisOptions.leadingHadronBias == params.leadingHadronBiasType.track

@pytest.mark.parametrize("args, expected", [
    ((2.76, "PbPb", "central", "track"), None),
    ((None, "PbPb", "central", "track"), None),
    ((2.76, None, "central", "track"), None),
    ((2.76, "PbPb", None, "track"), None),
    ((2.76, "PbPb", "central", None), None),
    ((params.collisionEnergy.twoSevenSix,
      params.collisionSystem.PbPb,
      params.eventActivity.central,
      params.leadingHadronBiasType.track), None),
    ((5.02, "embedPP", "semiCentral", "cluster"),
     (params.collisionEnergy.fiveZeroTwo,
      params.collisionSystem.embedPP,
      params.eventActivity.semiCentral,
      params.leadingHadronBiasType.cluster))
], ids = [
    "Standard 2.76",
    "Missing collision energy",
    "Missing collision system",
    "Missing event activity",
    "Missing leading hadron bias",
    "Standard 2.76 with enums",
    "5.02 semi-central embedPP with cluster bias"])
def testValidateArguments(loggingMixin, args, expected):
    """ Test argument validation. """
    if expected is None:
        expected = (params.collisionEnergy.twoSevenSix,
                    params.collisionSystem.PbPb,
                    params.eventActivity.central,
                    params.leadingHadronBiasType.track)

    args = params.SelectedAnalysisOptions(*args)
    args, _ = analysis_config.validateArguments(args)
    expected = params.SelectedAnalysisOptions(*expected)
    assert args.collisionEnergy == expected.collisionEnergy
    assert args.collisionSystem == expected.collisionSystem
    assert args.eventActivity == expected.eventActivity
    assert args.leadingHadronBias == expected.leadingHadronBias

def check_jetH_base_object(obj, config, selected_analysis_options, event_plane_angle, **kwargs):
    """ Helper function to check JetHBase properties.

    The values are asserted in this function.

    Note:
        The default values correspond to those in the objectConfig config, so they don't
        need to be specified in each function.

    Args:
        obj (analysis_config.JetHBase): JetHBase object to compare values against.
        config (CommentedMap): dict-like configuration file.
        selected_analysis_options (params.SelectedAnalysisOptions): Selected analysis options.
        event_plane_angle (params.eventPlaneAngle): Selected event plane angle.
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
        "eventPlaneAngle": event_plane_angle
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
    valueNames = ["inputFilename", "inputListName", "outputPrefix", "outputFilename", "printingExtensions", "aliceLabel"]
    for k in valueNames:
        # The aliceLabel gets converted to the enum in the object, so we need to do the conversion here.
        if k == "aliceLabel":
            val = params.aliceLabel[config[k]]
        else:
            val = config[k]
        default_values[k] = val
    default_values.update(kwargs)

    # Directly compare against the available values
    # NOTE: This isn't wholly independent because the object comparison relies on comparing values
    #       in obj.__dict__, but it is still an improvement on the above.
    for prop, val in iteritems(obj.__dict__):
        assert val == default_values[prop]

    # Perform the comparison test.
    assert obj == comparison

    return True

@pytest.fixture
def objectConfig():
    """ Object YAML configuration to test object args, validation, and construction.

    Args:
        None
    Returns:
        CommentedMap: dict-like object from ruamel.yaml containing the configuration.
    """

    testYaml = """
iterables:
    eventPlaneAngle: False
    qVector: False
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
            eventPlaneAngle: True
            qVector:
                - "all"
"""
    yaml = ruamel.yaml.YAML()
    data = yaml.load(testYaml)
    return (data, "taskName")

@pytest.mark.parametrize("leadingHadronBias", [
    (params.leadingHadronBiasType.track),
    (params.leadingHadronBias(type = params.leadingHadronBiasType.track, value = 5))
], ids = ["leadingHadronEnum", "leadingHadronClass"])
def testJetHBaseObjectConstruction(loggingMixin, leadingHadronBias, objectConfig, mocker):
    """ Test construction of the JetHBase object. """
    objectConfig, taskName = objectConfig
    (config, selected_analysis_options) = overrideOptionsHelper(
        objectConfig,
        configContainingOverride = objectConfig[taskName]
    )

    # Avoid os.makedirs actually making directories
    mocker.patch("os.makedirs")

    configFilename = "configFilename.yaml"
    taskConfig = config[taskName]
    event_plane_angle = params.eventPlaneAngle.all
    config_base = analysis_config.JetHBase(
        task_name = taskName,
        config_filename = configFilename,
        config = config,
        task_config = taskConfig,
        collisionEnergy = selected_analysis_options.collisionEnergy,
        collisionSystem = selected_analysis_options.collisionSystem,
        eventActivity = selected_analysis_options.eventActivity,
        leadingHadronBias = selected_analysis_options.leadingHadronBias,
        eventPlaneAngle = event_plane_angle
    )

    # We need values to compare against. However, namedtuples are immutable,
    # so we have to create a new one with the proper value.
    temp_selected_options = selected_analysis_options.asdict()
    temp_selected_options["leadingHadronBias"] = leadingHadronBias
    selected_analysis_options = params.SelectedAnalysisOptions(**temp_selected_options)
    # Only need for the case of leadingHadronBiasType!
    if isinstance(leadingHadronBias, params.leadingHadronBiasType):
        selected_analysis_options = analysis_config.determineLeadingHadronBias(config, selected_analysis_options)

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
    {"iterable1": params.collisionEnergy, "iterable2": params.collisionSystem}
], ids = ["No additional iterables", "Two additional iterables"])
def test_construct_object_from_config(loggingMixin, additional_iterables, objectConfig, mocker):
    """ Test construction of objects through a configuration file.

    NOTE: This is an integration test. """
    # Basic setup
    # We need both the input and the expected out.
    # NOTE: We only want to override the options of the expected config because
    #       construct_from_configuration_file() applies the overriding itself.
    config, task_name = objectConfig
    expectedNames = ["eventPlaneAngle", "qVector"]
    if additional_iterables:
        for iterable in additional_iterables:
            expectedNames.extend([iterable])
            config[task_name]["override"]["iterables"][iterable] = True
    expectedConfig = copy.deepcopy(config)
    (expectedConfig, selected_analysis_options) = overrideOptionsHelper(
        expectedConfig,
        configContainingOverride = expectedConfig[task_name]
    )
    expected_analysis_options = analysis_config.determineLeadingHadronBias(config = expectedConfig, selectedAnalysisOptions = selected_analysis_options)

    # Task arguments
    config_filename = "configFilename.yaml"
    obj = analysis_config.JetHBase

    # Mock reading the config
    #loadConfigurationMock = mocker.MagicMock(spec_set = ["filename"], return_value = config)
    # Needs the full path to the module.
    loadConfigurationMock = mocker.patch("jet_hadron.base.analysis_config.generic_config.loadConfiguration", return_value = config)
    # Avoid os.makedirs actually making directories
    mocker.patch("os.makedirs")

    (_, names, objects) = analysis_config.construct_from_configuration_file(
        task_name = task_name,
        config_filename = config_filename,
        selected_analysis_options = selected_analysis_options,
        obj = obj,
        additional_possible_iterables = additional_iterables
    )
    # Check the opening the config file was called properly.
    loadConfigurationMock.assert_called_once_with(config_filename)

    assert names == expectedNames
    for values, obj in generic_config.iterate_with_selected_objects(objects):
        res = check_jetH_base_object(obj = obj,
                                     config = expectedConfig,
                                     selected_analysis_options = expected_analysis_options,
                                     event_plane_angle = values.eventPlaneAngle)
        assert res is True

    # Just to be safe
    mocker.stopall()
