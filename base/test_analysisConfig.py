#!/usr/bin/env python

# Tests for the JetH configuration functionality defined in the analysisConfig module.
#
# date: 8 May 2018

# Py2/3
from future.utils import iteritems

import pytest
import os
import copy
import ruamel.yaml
import logging
# Setup logger
logger = logging.getLogger(__name__)

import jetH.base.params as params
import jetH.base.genericConfig as genericConfig
import jetH.base.analysisConfig as analysisConfig

# Set logging level as a global variable to simplify configuration.
# This is not ideal, but fine for simple tests.
loggingLevel = logging.DEBUG

def testUnrollNestedDict(caplog):
    """ Test unrolling the analysis dictionary. """
    caplog.set_level(loggingLevel)

    cDict = {"c1" : "obj", "c2" : "obj2", "c3": "obj3"}
    bDict = {"b" : cDict.copy()}
    print("bDict: {}".format(bDict))
    testDict = {"a1" : bDict.copy(), "a2" : bDict.copy()}
    unroll = analysisConfig.unrollNestedDict(testDict)

    assert next(unroll) == (["a1", "b", "c1"], "obj")
    assert next(unroll) == (["a1", "b", "c2"], "obj2")
    assert next(unroll) == (["a1", "b", "c3"], "obj3")
    assert next(unroll) == (["a2", "b", "c1"], "obj")
    assert next(unroll) == (["a2", "b", "c2"], "obj2")
    assert next(unroll) == (["a2", "b", "c3"], "obj3")

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
def testDetermineLeadingHadronBias(biasType, eventActivity, expectedLeadingHadronBiasValue, caplog, leadingHadronBiasConfig):
    """ Test determination of the leading hadron bias. """
    caplog.set_level(loggingLevel)
    (config, selectedAnalysisOptions) = overrideOptionsHelper(leadingHadronBiasConfig)

    # Add in the different selected options
    if biasType:
        # Both options will lead here. Doing this with "track" doesn't change the values, but
        # also doesn't hurt anything, so it's fine.
        kwargs = selectedAnalysisOptions._asdict()
        kwargs["leadingHadronBias"] = params.leadingHadronBiasType[biasType]
        selectedAnalysisOptions =  params.selectedAnalysisOptions(**kwargs)
    if eventActivity:
        kwargs = selectedAnalysisOptions._asdict()
        kwargs["eventActivity"] = params.eventActivity[eventActivity]
        selectedAnalysisOptions =  params.selectedAnalysisOptions(**kwargs)

    returnedOptions = analysisConfig.determineLeadingHadronBias(config = config, selectedAnalysisOptions = selectedAnalysisOptions)
    # Check that we still got these right.
    assert returnedOptions.collisionEnergy == selectedAnalysisOptions.collisionEnergy
    assert returnedOptions.collisionSystem == selectedAnalysisOptions.collisionSystem
    assert returnedOptions.eventActivity == selectedAnalysisOptions.eventActivity
    # Check the bias value and type
    logger.debug("type(leadingHadronBias): {}".format(type(returnedOptions.leadingHadronBias)))
    assert returnedOptions.leadingHadronBias.value == expectedLeadingHadronBiasValue
    assert returnedOptions.leadingHadronBias.type == params.leadingHadronBiasType[biasType]

def logYAMLDump(s):
    """ Simple function that transforms the yaml.dump() call to a stream
    and redirects it to the logger.

    Inspired by: https://stackoverflow.com/a/47617341
    """
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
        selectedOptions (params.selectedAnalysisOptions): The options selected for this analysis, in
            the order defined used with analysisConfig.overrideOptions() and in the configuration file.
        configContainingOverride (CommentedMap): dict-like object containing the override options.
    Returns:
        tuple: (dict-like CommentedMap object containing the overridden configuration, selected analysis
                    options used with the config)
    """
    if selectedOptions is None:
        selectedOptions = params.selectedAnalysisOptions(collisionEnergy = params.collisionEnergy.twoSevenSix,
                           collisionSystem = params.collisionSystem.PbPb,
                           eventActivity = params.eventActivity.central,
                           leadingHadronBias = params.leadingHadronBiasType.track)

    yaml = ruamel.yaml.YAML()
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Before override:")
        yaml.dump(config, None, transform = logYAMLDump)

    config = analysisConfig.overrideOptions(config = config,
            selectedOptions = selectedOptions,
            configContainingOverride = configContainingOverride)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("After override:")
        yaml.dump(config, None, transform = logYAMLDump)

    return (config, selectedOptions)

def testBasicSelectedOverrides(caplog, basicConfig):
    """ Test that override works for the selected options. """
    caplog.set_level(loggingLevel)
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

def testIgnoreUnselectedOptions(caplog, basicConfig):
    """ Test ignoring unselected values. """
    caplog.set_level(loggingLevel)

    # Delete the central values, thereby removing any values to override.
    # Thus, the configuration values should not change!
    del basicConfig["override"][2.76]["central"]

    (config, selectedAnalysisOptions) = overrideOptionsHelper(basicConfig)

    # NOTE: It should be compared against "baseName" because it also converts single entry
    #       lists to just the single entry.
    assert config["responseTaskName"] == "baseName"

def testArgumentParsing():
    """ Test argument parsing.  """
    testArgs = ["-c", "analysisConfigArg.yaml",
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
        (configFilename, selectedAnalysisOptions, returnedArgs) = analysisConfig.determineSelectedOptionsFromKwargs(args = args)

        assert configFilename == "analysisConfigArg.yaml"
        assert selectedAnalysisOptions.collisionEnergy == 5.02
        assert selectedAnalysisOptions.collisionSystem == "embedPP"
        assert selectedAnalysisOptions.eventActivity == "semiCentral"
        assert selectedAnalysisOptions.leadingHadronBias == "track"

        # Strictly speaking, this is adding some complication, but it will consistently be used with this option,
        # so it's worth doing the integration test.
        validatedAnalysisOptions, _ = analysisConfig.validateArguments(selectedAnalysisOptions)

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
def testValidateArguments(args, expected, caplog):
    """ Test argument validation. """
    caplog.set_level(loggingLevel)
    if expected is None:
        expected = (params.collisionEnergy.twoSevenSix,
                 params.collisionSystem.PbPb,
                 params.eventActivity.central,
                 params.leadingHadronBiasType.track)

    args = params.selectedAnalysisOptions(*args)
    args, _ = analysisConfig.validateArguments(args)
    expected = params.selectedAnalysisOptions(*expected)
    assert args.collisionEnergy == expected.collisionEnergy
    assert args.collisionSystem == expected.collisionSystem
    assert args.eventActivity == expected.eventActivity
    assert args.leadingHadronBias == expected.leadingHadronBias

def checkJetHBaseObject(obj, config, selectedAnalysisOptions, eventPlaneAngle, **kwargs):
    """ Helper function to check JetHBase properties. The values are asserted
    in this function.

    NOTE: The default values correspond to those in the objectConfig config, so
          they don't need to be specified in each function.

    Args:
        obj (analysisConfig.JetHBase): JetHBase object to compare values against.
        config (CommentedMap): dict-like configuration file.
        selectedAnalysisOptions (params.selectedAnalysisOptions): Selected analysis options.
        eventPlaneAngle (params.eventPlaneAngle): Selected event plane angle.
        kwargs (dict): All other values to compare against for which the default
            value defined in this function is not sufficient.
    Returns:
        bool: True if it successfully make it to the end of the assertions.
    """

    # Determine default values
    taskName = "taskName"
    defaultValues = {
            "taskName" : taskName,
            "configFilename" : "configFilename.yaml",
            "config" : config,
            "taskConfig" : config[taskName],
            "eventPlaneAngle" : eventPlaneAngle
    }
    # Add these afterwards so we don't have to do each value by hand.
    defaultValues.update(selectedAnalysisOptions._asdict())
    # NOTE: All other values will be taken from the config when constructing the object.
    for k, v in iteritems(defaultValues):
        if not k in kwargs:
            kwargs[k] = v

    # Creating thie object is something of a tautology, because in both cases we use the
    # constructed object, so we also have other tests, which are performed below.
    # However, we keep the test to provide a check for the equality operators.
    # We perform the actual test later because it is not very verbose.
    comparison = analysisConfig.JetHBase(**kwargs)

    # We need to retrieve these values so we can test them directly.
    # `defaultValues` will now be used as the set of reference values.
    valueNames = ["inputFilename", "inputListName", "outputPrefix", "outputFilename", "printingExtensions", "aliceLabel"]
    for k in valueNames:
        # The aliceLabel gets converted to the enum in the object, so we need to do the conversion here.
        if k == "aliceLabel":
            val = params.aliceLabel[config[k]]
        else:
            val = config[k]
        defaultValues[k] = val
    defaultValues.update(kwargs)

    # Directly compare against the available values
    # NOTE: This isn't wholly independent because the object comparison relies on comparing values
    #       in obj.__dict__, but it is still an improvement on the above.
    for prop, val in iteritems(obj.__dict__):
        assert val == defaultValues[prop]

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
def testJetHBaseObjectConstruction(leadingHadronBias, caplog, objectConfig, mocker):
    """ Test construction of the JetHBase object. """
    caplog.set_level(loggingLevel)
    objectConfig, taskName = objectConfig
    (config, selectedAnalysisOptions) = overrideOptionsHelper(objectConfig,
            configContainingOverride = objectConfig[taskName])

    # Avoid os.makedirs actually making directories
    mocker.patch("os.makedirs")

    configFilename = "configFilename.yaml"
    taskConfig = config[taskName]
    eventPlaneAngle = params.eventPlaneAngle.all
    configBase = analysisConfig.JetHBase(taskName = taskName,
            configFilename = configFilename,
            config = config,
            taskConfig = taskConfig,
            collisionEnergy = selectedAnalysisOptions.collisionEnergy,
            collisionSystem = selectedAnalysisOptions.collisionSystem,
            eventActivity = selectedAnalysisOptions.eventActivity,
            leadingHadronBias = selectedAnalysisOptions.leadingHadronBias,
            eventPlaneAngle = eventPlaneAngle)

    # We need values to compare against. However, namedtuples are immutable,
    # so we have to create a new one with the proper value.
    tempSelectedOptions = selectedAnalysisOptions._asdict()
    tempSelectedOptions["leadingHadronBias"] = leadingHadronBias
    selectedAnalysisOptions = params.selectedAnalysisOptions(**tempSelectedOptions)
    # Only need for the case of leadingHadronBiasType!
    if isinstance(leadingHadronBias, params.leadingHadronBiasType):
        selectedAnalysisOptions = analysisConfig.determineLeadingHadronBias(config, selectedAnalysisOptions)

    # Assertions are performed in this function
    res = checkJetHBaseObject(obj = configBase,
            config = config,
            selectedAnalysisOptions = selectedAnalysisOptions,
            eventPlaneAngle = eventPlaneAngle)
    assert res == True

    # Just to be safe
    mocker.stopall()

@pytest.mark.parametrize("additionalIterables", [
        None,
        {"iterable1" : params.collisionEnergy, "iterable2" : params.collisionSystem}
    ], ids = ["No additional iterables", "Two additional iterables"])
def testConstructObjectFromConfig(additionalIterables, caplog, objectConfig, mocker):
    """ Test construction of objects through a configuration file.

    NOTE: This is an integration test. """
    caplog.set_level(loggingLevel)
    # Basic setup
    # We need both the input and the expected out.
    # NOTE: We only want to override the options of the expected config because
    #       constructFromConfigurationFile() applies the overriding itself.
    config, taskName = objectConfig
    expectedNames = ["eventPlaneAngle", "qVector"]
    if additionalIterables:
        for iterable in additionalIterables:
            expectedNames.extend([iterable])
            config[taskName]["override"]["iterables"][iterable] = True
    expectedConfig = copy.deepcopy(config)
    (expectedConfig, selectedAnalysisOptions) = overrideOptionsHelper(expectedConfig,
            configContainingOverride = expectedConfig[taskName])
    expectedAnalysisOptions = analysisConfig.determineLeadingHadronBias(config = expectedConfig, selectedAnalysisOptions = selectedAnalysisOptions)

    # Task arguments
    configFilename = "configFilename.yaml"
    obj = analysisConfig.JetHBase

    # Mock reading the config
    #loadConfigurationMock = mocker.MagicMock(spec_set = ["filename"], return_value = config)
    # Needs the full path to the module.
    loadConfigurationMock = mocker.patch("jetH.base.analysisConfig.genericConfig.loadConfiguration", return_value = config)
    # Avoid os.makedirs actually making directories
    mocker.patch("os.makedirs")

    (names, objects) = analysisConfig.constructFromConfigurationFile(taskName = taskName,
            configFilename = configFilename,
            selectedAnalysisOptions = selectedAnalysisOptions,
            obj = obj,
            additionalPossibleIterables = additionalIterables)
    # Check the opening the config file was called properly.
    loadConfigurationMock.assert_called_once_with(configFilename)

    assert names == expectedNames
    for unrolled  in analysisConfig.unrollNestedDict(objects):
        (values, obj) = unrolled
        res = checkJetHBaseObject(obj = obj,
                config = expectedConfig,
                selectedAnalysisOptions = expectedAnalysisOptions,
                eventPlaneAngle = values[0])
        assert res == True

    # Just to be safe
    mocker.stopall()
