#!/usr/bin/env python

# Tests for the JetH configuration functionality defined in the analysisConfig module.
#
# author: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
# date: 8 May 2018

import pytest
import os
import ruamel.yaml
import logging
# Setup logger
logger = logging.getLogger(__name__)

import jetH.base.params as params
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
aliceLabelType: "thesis"

responseTaskName: &responseTaskName ["baseName"]
intVal: 1
halfwayValue: 3.1
override:
    2.76:
        halfwayValue: 3.14
        central:
            responseTaskName: "jetHPerformance"
            intVal: 2
        semiCentral:
            responseTaskName: "ignoreThisValue"
"""

    yaml = ruamel.yaml.YAML()
    data = yaml.load(testYaml)
    return data

def overrideOptionsHelper(basicConfig, selectedOptions = None):
    """ Helper function to override the configuration.

    It can print the configuration before and after overridding the options if enabled.

    NOTE: If selectedOptions is not specified, it defaults to (2.76, "PbPb", "central", "track")

    Args:
        basicConfig (CommentedMap): dict-like object containing the configuration to be overridden.
        selectedOptions (params.selectedAnalysisOptions): The options selected for this analysis, in
            the order defined used with analysisConfig.overrideOptions() and in the configuration file.
    Returns:
        tuple: (dict-like CommentedMap object containing the overridden configuration, selected analysis
                    options used with the config)
    """
    if selectedOptions is None:
        selectedOptions = params.selectedAnalysisOptions(collisionEnergy = params.collisionEnergy.twoSevenSix,
                           collisionSystem = params.collisionSystem.PbPb,
                           eventActivity = params.eventActivity.central,
                           leadingHadronBiasType = params.leadingHadronBiasType.track)

    yaml = ruamel.yaml.YAML()
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Before override:")
        yaml.dump(basicConfig, None, transform = logYAMLDump)

    basicConfig = analysisConfig.overrideOptions(basicConfig, selectedOptions)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("After override:")
        yaml.dump(basicConfig, None, transform = logYAMLDump)

    return (basicConfig, selectedOptions)

def testBasicSelectedOverrides(caplog, basicConfig):
    """ Test that override works for the selected options. """
    caplog.set_level(loggingLevel)
    (config, selectedAnalysisOptions) = overrideOptionsHelper(basicConfig)

    assert config["responseTaskName"] == "jetHPerformance"
    assert config["intVal"] == 2
    assert config["halfwayValue"] == 3.14

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
    """

    """
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
        assert selectedAnalysisOptions.leadingHadronBiasType == "track"

        # Strictly speaking, this is adding some complication, but it will consistently be used with this option,
        # so it's worth doing the integration test.
        validatedAnalysisOptions, _ = analysisConfig.validateArguments(selectedAnalysisOptions)

        assert validatedAnalysisOptions.collisionEnergy == params.collisionEnergy.fiveZeroTwo
        assert validatedAnalysisOptions.collisionSystem == params.collisionSystem.embedPP
        assert validatedAnalysisOptions.eventActivity == params.eventActivity.semiCentral
        assert validatedAnalysisOptions.leadingHadronBiasType == params.leadingHadronBiasType.track

def testValidateArguments(caplog):
    """ Test argument validation. """
    standardArgs  = (params.collisionEnergy.twoSevenSix,
             params.collisionSystem.PbPb,
             params.eventActivity.central,
             params.leadingHadronBiasType.track)
    testParams = [params.selectedAnalysisOptions(2.76, "PbPb", "central", "track")]
    testParams = [
            # Testing default values
            ((None, "PbPb", "central", "track"),
             standardArgs),
            ((2.76, None, "central", "track"),
             standardArgs),
            ((2.76, "PbPb", None, "track"),
             standardArgs),
            ((2.76, "PbPb", "central", None),
             standardArgs),
            ((2.76, "PbPb", "central", "track"),
             standardArgs),
            ((5.02, "embedPP", "semiCentral", "cluster"),
             (params.collisionEnergy.fiveZeroTwo,
             params.collisionSystem.embedPP,
             params.eventActivity.semiCentral,
             params.leadingHadronBiasType.cluster))
            ]

    for opts, expected in testParams:
        opts = params.selectedAnalysisOptions(*opts)
        opts, _ = analysisConfig.validateArguments(opts)
        expected = params.selectedAnalysisOptions(*expected)
        assert opts.collisionEnergy == expected.collisionEnergy
        assert opts.collisionSystem == expected.collisionSystem
        assert opts.eventActivity == expected.eventActivity 
        assert opts.leadingHadronBiasType == expected.leadingHadronBiasType

@pytest.fixture
def objectConfig():
    """ Object YAML configuration to test object args, validation, and construction.

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
printingExtensions: ["png", "pdf"]
aliceLabelType: "thesis"
taskName:
    test: "val"
override:
    # Just need a trivial override value, since "override" is a required field.
    aliceLabelType: "thesis"
"""
    yaml = ruamel.yaml.YAML()
    data = yaml.load(testYaml)
    return data

def testJetHBaseObjectConstruction(caplog, objectConfig, mocker):
    """ Test construction of the JetHBase object. """
    (config, selectedAnalysisOptions) = overrideOptionsHelper(objectConfig)

    # Avoid os.makedirs actually making directories
    mocker.patch("os.makedirs")

    taskName = "taskName"
    configFilename = "configFilename"
    taskConfig = config[taskName]
    eventPlaneAngle = params.eventPlaneAngle.all
    configBase = analysisConfig.JetHBase(taskName = taskName,
            configFilename = configFilename,
            config = config,
            taskConfig = taskConfig,
            collisionEnergy = selectedAnalysisOptions.collisionEnergy,
            collisionSystem = selectedAnalysisOptions.collisionSystem,
            eventActivity = selectedAnalysisOptions.eventActivity,
            leadingHadronBiasType = selectedAnalysisOptions.leadingHadronBiasType,
            eventPlaneAngle = eventPlaneAngle,
            createOutputFolder = False)

    assert configBase.taskName == taskName
    assert configBase.configFilename == configFilename
    assert configBase.config == config
    assert configBase.taskConfig == taskConfig
    assert configBase.collisionEnergy == selectedAnalysisOptions.collisionEnergy
    assert configBase.collisionSystem == selectedAnalysisOptions.collisionSystem
    assert configBase.eventActivity == selectedAnalysisOptions.eventActivity
    assert configBase.leadingHadronBiasType == selectedAnalysisOptions.leadingHadronBiasType
    assert configBase.eventPlaneAngle == eventPlaneAngle
    assert configBase.inputFilename == "inputFilenameValue"
    assert configBase.inputListName == "inputListNameValue"
    assert configBase.outputPrefix == "outputPrefixValue"
    assert configBase.outputFilename == "outputFilenameValue"
    assert configBase.printingExtensions == ["png", "pdf"]
    assert configBase.aliceLabelType == "thesis"
