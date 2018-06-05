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

import jetH.base.genericConfig as genericConfig

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
        tuple: (dict-like CommentedMap object from ruamel.yaml containing the configuration, str containing
            a string representation of the YAML configuration)
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

    return (data, testYaml)

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

def overrideData(config):
    """ Helper function to override the configuration.

    It can print the configuration before and after overridding the options if enabled.

    Args:
        config (CommentedMap): dict-like object containing the configuration to be overridden.
    Returns:
        CommentedMap: dict-like object containing the overridden configuration
    """
    yaml = ruamel.yaml.YAML()

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Before override:")
        yaml.dump(config, None, transform = logYAMLDump)

    # Override and simplify the values
    config = genericConfig.overrideOptions(config, (), ())
    config = genericConfig.simplifyDataRepresentations(config)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("After override:")
        yaml.dump(config, None, transform = logYAMLDump)

    return config

def testOverrideRetrieveUnrelatedValue(caplog, basicConfig):
    """ Test retrieving a basic value unrelated to the overridden data. """
    caplog.set_level(loggingLevel)
    (basicConfig, yamlString) = basicConfig

    valueName = "test1"
    valueBeforeOverride = basicConfig[valueName]
    basicConfig = overrideData(basicConfig)
    
    assert basicConfig[valueName] == valueBeforeOverride

def testOverrideWithBasicConfig(caplog, basicConfig):
    """ Test override with the basic config.
    """
    caplog.set_level(loggingLevel)
    (basicConfig, yamlString) = basicConfig
    basicConfig = overrideData(basicConfig)

    # This value is overridden directly
    assert basicConfig["test3"] == "test6"

def testBasicAnchorOverride(caplog, basicConfig):
    """ Test overriding with an anchor.

    When an anchor refernce is overridden, we expect that the anchor value is updated.
    """
    caplog.set_level(loggingLevel)
    (basicConfig, yamlString) = basicConfig
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
    (basicConfig, yamlString) = basicConfig
    basicConfig = overrideData(basicConfig)

    # This value is overridden indirectly, from another referenced value.
    assert basicConfig["responseTaskName"] == basicConfig["pythiaInfoAfterEventSelectionTaskName"]

def testForUnmatchedKeys(caplog, basicConfig):
    """ Test for an unmatched key in the override field (ie without a match in the config).

    Such an unmatched key should cause a `KeyError` exception, which we catch.
    """
    caplog.set_level(loggingLevel)
    (basicConfig, yamlString) = basicConfig
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
    (basicConfig, yamlString) = basicConfig
    basicConfig = overrideData(basicConfig)

    assert basicConfig["testList"] == [3, 4]
    assert basicConfig["testDict"] == {3: 4}

def testLoadConfiguration(caplog, basicConfig):
    """ Test that loading yaml goes according to expectations. This may be somewhat trivial, but it
    is still important to check in case ruamel.yaml changes APIs or defaults.

    NOTE: We can only compare at the YAML level because the dumped string does not preserve anchors that
          are not actually referenced, as well as some trivial variation in quote types and other similarly
          trivial formatting issues.
    """
    caplog.set_level(loggingLevel)
    (basicConfig, yamlString) = basicConfig

    import tempfile
    with tempfile.NamedTemporaryFile() as f:
        # Write and move back to the start of the file
        f.write(yamlString.encode())
        f.seek(0)
        # Then get the config from the file
        retrievedConfig = genericConfig.loadConfiguration(f.name)

    assert retrievedConfig == basicConfig

    # NOTE: Not utilized due to the note above
    # Use yaml.dump() to dump the configuration to a string.
    #yaml = ruamel.yaml.YAML(typ = "rt")
    #with tempfile.NamedTemporaryFile() as f:
    #    yaml.dump(retrievedConfig, f)
    #    f.seek(0)
    #    # Save as a standard string. Need to decode from bytes
    #    retrievedString = f.read().decode()
    #assert retrievedString == yamlString

@pytest.fixture
def dataSimplificationConfig():
    """ Simple YAML config to test the data simplification functionality of the genericConfig module.
    
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
    config = genericConfig.simplifyDataRepresentations(dataSimplificationConfig)

    assert config["int"] == 3
    assert config["float"] == 3.14
    assert config["str"] == "hello"

def testDataSimplificationOnLists(caplog, dataSimplificationConfig):
    """ Test the data simplification function on lists.
    
    A single entry list should be returned as a string, while a multiple entry list should be
    preserved as is.
    """
    caplog.set_level(loggingLevel)
    config = genericConfig.simplifyDataRepresentations(dataSimplificationConfig)

    assert config["singleEntryList"] == "hello"
    assert config["multiEntryList"] == ["hello", "world"]

def testDictDataSimplification(caplog, dataSimplificationConfig):
    """ Test the data simplification function on dicts.
    
    Dicts should always maintain their structure.
    """
    caplog.set_level(loggingLevel)
    config = genericConfig.simplifyDataRepresentations(dataSimplificationConfig)

    assert config["singleEntryDict"] == {"hello" : "world"}
    assert config["multiEntryDict"] == {"hello" : "world", "foo" : "bar"}

@pytest.fixture
def formattingConfig():
    config = """
int: 3
float: 3.14
noFormat: "test"
format: "{a}"
noFormatBecauseNoFormatter: "{noFormatHere}"
list:
    - "noFormat"
    - 2
    - "{a}{c}"
dict:
    noFormat: "hello"
    format: "{a}{c}"
dict2:
    dict:
        str: "do nothing"
        format: "{c}"
latexLike: $latex_{like \mathrm{x}}$
noneExample: null
"""
    yaml = ruamel.yaml.YAML()
    config = yaml.load(config)

    formatting = {"a" : "b", "c": 1}

    return genericConfig.applyFormattingDict(config, formatting)

def testApplyFormattingToBasicTypes(caplog, formattingConfig):
    """ Test applying formatting to basic types. """
    caplog.set_level(loggingLevel)
    config = formattingConfig

    assert config["int"] == 3
    assert config["float"] == 3.14
    assert config["noFormat"] == "test"
    assert config["format"] == "b"
    assert config["noFormatBecauseNoFormatter"] == "{noFormatHere}"

def testApplyFormattingToIterableTypes(caplog, formattingConfig):
    """ Test applying formatting to iterable types. """
    caplog.set_level(loggingLevel)
    config = formattingConfig

    assert config["list"] == ["noFormat", 2, "b1"]
    assert config["dict"] == {"noFormat" : "hello", "format" : "b1"}
    assert config["dict2"]["dict"] == { "str" : "do nothing", "format" : "1" }

def testApplyFormattingSkipLatex(caplog, formattingConfig):
    """ Test skipping the application of the formatting to strings which look like latex. """
    caplog.set_level(loggingLevel)
    config = formattingConfig

    assert config["latexLike"] == "$latex_{like \mathrm{x}}$"


