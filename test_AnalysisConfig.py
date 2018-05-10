#!/usr/bin/env python

# Tests for analysis configuration.
#
# NOTE: These are more like integration tests.
# 
# author: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
# date: 8 May 2018

import pytest
import ruamel.yaml
import logging
# Setup logger
logger = logging.getLogger(__name__)

import AnalysisConfig

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
responseTasks: &responseTasks
    responseMaker: &responseMakerTaskName "AliJetResponseMaker_{cent}histos"
    jetHPerformance: &jetHPerformanceTaskName ""
responseTaskName: &responseTaskName [""]
pythiaInfoAfterEventSelectionTaskName: *responseTaskName
# Demonstrate that anchors are preserved
test1: &test1
- val1
- val2
test2: *test1
# Test overrid values
test3: &test3 ["test3"]
test4: *test3
testList: [1, 2]
testDict:
    1: 2
override:
    responseTaskName: *responseMakerTaskName
    test3: "test6"
    testList: [3, 4]
    testDict:
        3: 4
    """

    yaml = ruamel.yaml.YAML()
    data = yaml.load(testYaml)

    return data

def basicConfigException(data):
    """ Add an unmatched key (ie does not exist in the main config) to the override
    map to cause an exception.

    Note that this assumes that "testException" does not exist in the main configuration!

    Args:
        data (CommentedMap): dict-like object containing the configuration
    Returns:
        CommentedMap: dict-like object containing an unmatched entry in the override map.
    """
    data["override"]["testException"] = "value"
    return data

def overrideData(basicConfig):
    """ Helper function to override the configuration.

    It can print the configuration before and after overridding the options if enabled.

    Args:
        basicConfig (CommentedMap): dict-like object containing the configuration to be overridden.
    Returns:
        CommentedMap: dict-like object containing the overridden configuration
    """
    yaml = ruamel.yaml.YAML()

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Before override:")
        yaml.dump(basicConfig, None, transform = logYAMLDump)

    # Override and simplify the values
    basicConfig = AnalysisConfig.overrideOptions(basicConfig, (), ())
    basicConfig = AnalysisConfig.simplifyDataRepresentations(basicConfig)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("After override:")
        yaml.dump(basicConfig, None, transform = logYAMLDump)

    return basicConfig

def testOverrideRetrieveUnrelatedValue(caplog, basicConfig):
    """ Test retrieving a basic value unrelated to the overridden data. """
    caplog.set_level(loggingLevel)
    valueName = "test1"
    valueBeforeOverride = basicConfig[valueName]
    basicConfig = overrideData(basicConfig)
    
    assert basicConfig[valueName] == valueBeforeOverride

def testOverrideWithBasicConfig(caplog, basicConfig):
    """ Test override with the basic config.
    """
    caplog.set_level(loggingLevel)
    basicConfig = overrideData(basicConfig)

    # This value is overridden directly
    assert basicConfig["test3"] == "test6"

def testBasicAnchorOverride(caplog, basicConfig):
    """ Test overriding with an anchor.

    When an anchor refernce is overridden, we expect that the anchor value is updated.
    """
    caplog.set_level(loggingLevel)
    basicConfig = overrideData(basicConfig)
 
    # The two conditions below are redundant, but each are useful for visualizing
    # different configuration circumstances, so both are kept.
    assert basicConfig["responseTaskName"] == "AliJetResponseMaker_{cent}histos"
    assert basicConfig["test4"] == "test6"

def testAdvancedAnchorOverride(caplog, basicConfig):
    """ Test overriding a anchored value with another anchor.
    
    When an override value is using an anchor value, we expect that value to propagate fully.
    """
    caplog.set_level(loggingLevel)
    basicConfig = overrideData(basicConfig)

    # This value is overridden indirectly, from another referenced value.
    assert basicConfig["responseTaskName"] == basicConfig["pythiaInfoAfterEventSelectionTaskName"]

def testForUnmatchedKeys(caplog, basicConfig):
    """ Test for an unmatched key in the override field (ie without a match in the config).

    Such an unmatched key should cause a `KeyError` exception, which we catch.
    """
    caplog.set_level(loggingLevel)
    # Add entry that will cause the exception.
    basicConfig = basicConfigException(basicConfig)

    # Process and note the exception
    caughtExpectedException = False
    exceptionValue = None
    try:
        basicConfig = overrideData(basicConfig)
    except KeyError as e:
        caughtExpectedException = True
        # The first arg is the key which caused the KeyError.
        exceptionValue = e.args[0]

    assert caughtExpectedException == True

def testComplexObjectOverride(caplog, basicConfig):
    """ Test override with complex objects.
    
    In particular, test with lists, dicts.
    """
    caplog.set_level(loggingLevel)
    basicConfig = overrideData(basicConfig)

    assert basicConfig["testList"] == [3, 4]
    assert basicConfig["testDict"] == {3: 4}

@pytest.fixture
def dataSimplificationConfig():
    """ Simple YAML config to test the data simplification functionality of the AnalysisConfig module.
    
    It povides example configurations entries for numbers, str, list, and dict.

    Args:
        None
    Returns:
        CommentedMap: dict-like object from ruamel.yaml containing the configuration.
    """

    testYaml = """
int: 3
float: 3.14
str: "hello"
singleEntryList: [ "hello" ]
multiEntryList: [ "hello", "world" ]
singleEntryDict:
    hello: "world"
multiEntryDict:
    hello: "world"
    foo: "bar"
"""
    yaml = ruamel.yaml.YAML()
    data = yaml.load(testYaml)

    return data

def testDataSimplificationOnBaseTypes(caplog, dataSimplificationConfig):
    """ Test the data simplification function on base types.
    
    Here we tests int, float, and str.  They should always stay the same.
    """
    caplog.set_level(loggingLevel)
    config = AnalysisConfig.simplifyDataRepresentations(dataSimplificationConfig)

    assert config["int"] == 3
    assert config["float"] == 3.14
    assert config["str"] == "hello"

def testDataSimplificationOnLists(caplog, dataSimplificationConfig):
    """ Test the data simplification function on lists.
    
    A single entry list should be returned as a string, while a multiple entry list should be
    preserved as is.
    """
    caplog.set_level(loggingLevel)
    config = AnalysisConfig.simplifyDataRepresentations(dataSimplificationConfig)

    assert config["singleEntryList"] == "hello"
    assert config["multiEntryList"] == ["hello", "world"]

def testDictDataSimplification(caplog, dataSimplificationConfig):
    """ Test the data simplification function on dicts.
    
    Dicts should always maintain their structure.
    """
    caplog.set_level(loggingLevel)
    config = AnalysisConfig.simplifyDataRepresentations(dataSimplificationConfig)

    assert config["singleEntryDict"] == {"hello" : "world"}
    assert config["multiEntryDict"] == {"hello" : "world", "foo" : "bar"}

