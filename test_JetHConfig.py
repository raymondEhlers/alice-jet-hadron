#!/usr/bin/env python

# Tests for the JetH configuration functionality defined in the JetHConfig module.
#
# author: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
# date: 8 May 2018

import pytest
import ruamel.yaml
import logging
# Setup logger
logger = logging.getLogger(__name__)

import JetHConfig

# Set logging level as a global variable to simplify configuration.
# This is not ideal, but fine for simple tests.
loggingLevel = logging.DEBUG

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
responseTaskName: &responseTaskName ["baseName"]
intVal: 1
override:
    2.76:
        central:
            responseTaskName: "jetHPerformance"
            intVal: 2
        semiCentral:
            responseTaskName: "ignoreThisValue"
    """

    yaml = ruamel.yaml.YAML()
    data = yaml.load(testYaml)

    return data

def overrideOptions(basicConfig, selectedOptions = None):
    """ Helper function to override the configuration.

    It can print the configuration before and after overridding the options if enabled.

    NOTE: If selectedOptions is not specified, it defaults to (2.76, "PbPb", "central", "track")

    Args:
        basicConfig (CommentedMap): dict-like object containing the configuration to be overridden.
        selectedOptions (tuple): The options selected for this analysis, in the order defined used
            with overrideOptions() and in the configuration file.
    Returns:
        CommentedMap: dict-like object containing the overridden configuration
    """
    if selectedOptions is None:
        selectedOptions = (2.76, "PbPb", "central", "track")

    yaml = ruamel.yaml.YAML()
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Before override:")
        yaml.dump(basicConfig, None, transform = logYAMLDump)

    basicConfig = JetHConfig.overrideOptions(basicConfig, selectedOptions)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("After override:")
        yaml.dump(basicConfig, None, transform = logYAMLDump)

    return basicConfig

def testBasicSelectedOverrides(caplog, basicConfig):
    """ Test that override works for the selected options. """
    caplog.set_level(loggingLevel)
    config = overrideOptions(basicConfig)

    assert config["responseTaskName"] == "jetHPerformance"
    assert config["intVal"] == 2

def testIgnoreUnselectedOptions(caplog, basicConfig):
    """ Test ignoring unselected values. """
    caplog.set_level(loggingLevel)

    # Delete the central values, thereby removing any values to override.
    # Thus, the configuration values should not change!
    del basicConfig["override"][2.76]["central"]

    config = overrideOptions(basicConfig)

    # NOTE: It should be compared against "baseName" because it also converts single entry
    #       lists to just the single entry.
    assert config["responseTaskName"] == "baseName"
