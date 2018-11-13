#!/usr/bin/env python

# Tests for generic analysis configuration.
#
# author: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
# date: 8 May 2018

import pytest
import collections
import ruamel.yaml
import logging
logger = logging.getLogger(__name__)

import jetH.base.params as params
import jetH.base.genericConfig as genericConfig

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

def testOverrideRetrieveUnrelatedValue(loggingMixin, basicConfig):
    """ Test retrieving a basic value unrelated to the overridden data. """
    (basicConfig, yamlString) = basicConfig

    valueName = "test1"
    valueBeforeOverride = basicConfig[valueName]
    basicConfig = overrideData(basicConfig)

    assert basicConfig[valueName] == valueBeforeOverride

def testOverrideWithBasicConfig(loggingMixin, basicConfig):
    """ Test override with the basic config.  """
    (basicConfig, yamlString) = basicConfig
    basicConfig = overrideData(basicConfig)

    # This value is overridden directly
    assert basicConfig["test3"] == "test6"

def testBasicAnchorOverride(loggingMixin, basicConfig):
    """ Test overriding with an anchor.

    When an anchor refernce is overridden, we expect that the anchor value is updated.
    """
    (basicConfig, yamlString) = basicConfig
    basicConfig = overrideData(basicConfig)

    # The two conditions below are redundant, but each are useful for visualizing
    # different configuration circumstances, so both are kept.
    assert basicConfig["responseTaskName"] == "AliJetResponseMaker_{cent}histos"
    assert basicConfig["test4"] == "test6"

def testAdvancedAnchorOverride(loggingMixin, basicConfig):
    """ Test overriding a anchored value with another anchor.

    When an override value is using an anchor value, we expect that value to propagate fully.
    """
    (basicConfig, yamlString) = basicConfig
    basicConfig = overrideData(basicConfig)

    # This value is overridden indirectly, from another referenced value.
    assert basicConfig["responseTaskName"] == basicConfig["pythiaInfoAfterEventSelectionTaskName"]

def testForUnmatchedKeys(loggingMixin, basicConfig):
    """ Test for an unmatched key in the override field (ie without a match in the config).

    Such an unmatched key should cause a `KeyError` exception, which we catch.
    """
    (basicConfig, yamlString) = basicConfig
    # Add entry that will cause the exception.
    basicConfig = basicConfigException(basicConfig)

    # Test fails if it _doesn't_ throw an exception.
    with pytest.raises(KeyError) as exceptionInfo:
        basicConfig = overrideData(basicConfig)
    # This is the value that we expected to fail.
    assert exceptionInfo.value.args[0] == "testException"

def testComplexObjectOverride(loggingMixin, basicConfig):
    """ Test override with complex objects.

    In particular, test with lists, dicts.
    """
    (basicConfig, yamlString) = basicConfig
    basicConfig = overrideData(basicConfig)

    assert basicConfig["testList"] == [3, 4]
    assert basicConfig["testDict"] == {3: 4}

def testLoadConfiguration(loggingMixin, basicConfig):
    """ Test that loading yaml goes according to expectations. This may be somewhat trivial, but it
    is still important to check in case ruamel.yaml changes APIs or defaults.

    NOTE: We can only compare at the YAML level because the dumped string does not preserve anchors that
          are not actually referenced, as well as some trivial variation in quote types and other similarly
          trivial formatting issues.
    """
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

def testDataSimplificationOnBaseTypes(loggingMixin, dataSimplificationConfig):
    """ Test the data simplification function on base types.

    Here we tests int, float, and str.  They should always stay the same.
    """
    config = genericConfig.simplifyDataRepresentations(dataSimplificationConfig)

    assert config["int"] == 3
    assert config["float"] == 3.14
    assert config["str"] == "hello"

def testDataSimplificationOnLists(loggingMixin, dataSimplificationConfig):
    """ Test the data simplification function on lists.

    A single entry list should be returned as a string, while a multiple entry list should be
    preserved as is.
    """
    config = genericConfig.simplifyDataRepresentations(dataSimplificationConfig)

    assert config["singleEntryList"] == "hello"
    assert config["multiEntryList"] == ["hello", "world"]

def testDictDataSimplification(loggingMixin, dataSimplificationConfig):
    """ Test the data simplification function on dicts.

    Dicts should always maintain their structure.
    """
    config = genericConfig.simplifyDataRepresentations(dataSimplificationConfig)

    assert config["singleEntryDict"] == {"hello": "world"}
    assert config["multiEntryDict"] == {"hello": "world", "foo": "bar"}

@pytest.fixture
def objectCreationConfig():
    """ Configuration to test creating objects based on the stored values. """
    config = """
iterables:
    eventPlaneAngle:
        - inPlane
        - midPlane
    qVector: True
    collisionEnergy: False
"""
    yaml = ruamel.yaml.YAML()
    config = yaml.load(config)

    possibleIterables = collections.OrderedDict()
    possibleIterables["eventPlaneAngle"] = params.eventPlaneAngle
    possibleIterables["qVector"] = params.qVector
    possibleIterables["collisionEnergy"] = params.collisionEnergy

    return (config, possibleIterables, ([params.eventPlaneAngle.inPlane, params.eventPlaneAngle.midPlane], list(params.qVector)))

def testDetermineSelectionOfIterableValuesFromConfig(loggingMixin, objectCreationConfig):
    """ Test determining which values of an iterable to use. """
    (config, possibleIterables, (eventPlaneAngles, qVectors)) = objectCreationConfig
    iterables = genericConfig.determineSelectionOfIterableValuesFromConfig(config = config,
                                                                           possibleIterables = possibleIterables)

    assert iterables["eventPlaneAngle"] == eventPlaneAngles
    assert iterables["qVector"] == qVectors
    # Collision Energy should _not_ be included! It was only a possible iterator.
    # Check in two ways.
    assert "collisionEnergy" not in iterables
    assert len(iterables) == 2

def testDetermineSelectionOfIterableValuesWithUndefinedIterable(loggingMixin, objectCreationConfig):
    """ Test determining which values of an iterable to use when an iterable is not defined. """
    (config, possibleIterables, (eventPlaneAngles, qVectors)) = objectCreationConfig

    del possibleIterables["qVector"]
    with pytest.raises(KeyError) as exceptionInfo:
        genericConfig.determineSelectionOfIterableValuesFromConfig(config = config,
                                                                   possibleIterables = possibleIterables)
    assert exceptionInfo.value.args[0] == "qVector"

def testDetermineSelectionOfIterableValuesWithStringSelection(loggingMixin, objectCreationConfig):
    """ Test trying to determine values with a string. This is not allowed, so it should raise an exception. """
    (config, possibleIterables, (eventPlaneAngles, qVectors)) = objectCreationConfig

    config["iterables"]["qVector"] = "True"
    with pytest.raises(TypeError) as exceptionInfo:
        genericConfig.determineSelectionOfIterableValuesFromConfig(config = config,
                                                                   possibleIterables = possibleIterables)
    assert exceptionInfo.value.args[0] is str

@pytest.fixture
def objectAndCreationArgs():
    """ Create the object and args for object creation. """
    # Define fake object. We don't use a mock because we need to instantiate the object
    # in the function that is being tested. This is not super straightforward with mock,
    # so instead we create a test object by hand.
    obj = collections.namedtuple("testObj", ["eventPlaneAngle", "qVector", "a", "b", "optionsFmt"])
    # Include args that depend on the iterable values to ensure that they are varied properly!
    args = {"a": 1, "b": "{fmt}", "optionsFmt": "{eventPlaneAngle}_{qVector}"}
    formattingOptions = {"fmt": "formatted", "optionsFmt": "{eventPlaneAngle}_{qVector}"}

    return (obj, args, formattingOptions)

def testCreateObjectsFromIterables(loggingMixin, objectCreationConfig, objectAndCreationArgs):
    """ Test object creation from a set of iterables. """
    # Collect variables
    (config, possibleIterables, (eventPlaneAngles, qVectors)) = objectCreationConfig
    (obj, args, formattingOptions) = objectAndCreationArgs

    # Get iterables
    iterables = genericConfig.determineSelectionOfIterableValuesFromConfig(config = config,
                                                                           possibleIterables = possibleIterables)

    # Create the objects.
    (names, objects) = genericConfig.createObjectsFromIterables(obj = obj,
                                                                args = args,
                                                                iterables = iterables,
                                                                formattingOptions = formattingOptions)

    # Check the names of the iterables.
    assert names == list(iterables)
    # Check the precise values passed to the object.
    for epAngle in eventPlaneAngles:
        for qVector in qVectors:
            createdObject = objects[epAngle][qVector]
            assert createdObject.eventPlaneAngle == epAngle
            assert createdObject.qVector == qVector
            assert createdObject.a == args["a"]
            assert createdObject.b == formattingOptions["fmt"]
            assert createdObject.optionsFmt == formattingOptions["optionsFmt"].format(eventPlaneAngle = epAngle, qVector = qVector)

def testMissingIterableForObjectCreation(loggingMixin, objectAndCreationArgs):
    """ Test object creation when the iterables are missing. """
    (obj, args, formattingOptions) = objectAndCreationArgs
    # Create empty iterables for this test.
    iterables = {}

    # Create the objects.
    with pytest.raises(ValueError) as exceptionInfo:
        (names, objects) = genericConfig.createObjectsFromIterables(obj = obj,
                                                                    args = args,
                                                                    iterables = iterables,
                                                                    formattingOptions = formattingOptions)
    assert exceptionInfo.value.args[0] == iterables

@pytest.fixture
def formattingConfig():
    """ Config for testing the formatting of strings after loading them.

    Returns:
        tuple: (Config with formatting applied, formatting dict)
    """
    config = r"""
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

    formatting = {"a": "b", "c": 1}

    return (genericConfig.applyFormattingDict(config, formatting), formatting)

def testApplyFormattingToBasicTypes(loggingMixin, formattingConfig):
    """ Test applying formatting to basic types. """
    config, formattingDict = formattingConfig

    assert config["int"] == 3
    assert config["float"] == 3.14
    assert config["noFormat"] == "test"
    assert config["format"] == formattingDict["a"]
    assert config["noFormatBecauseNoFormatter"] == "{noFormatHere}"

def testApplyFormattingToIterableTypes(loggingMixin, formattingConfig):
    """ Test applying formatting to iterable types. """
    config, formattingDict = formattingConfig

    assert config["list"] == ["noFormat", 2, "b1"]
    assert config["dict"] == {"noFormat": "hello", "format": "{}{}".format(formattingDict["a"], formattingDict["c"])}
    # NOTE: The extra str() call is because the formated string needs to be compared against a str.
    assert config["dict2"]["dict"] == {"str": "do nothing", "format": str(formattingDict["c"])}

def testApplyFormattingSkipLatex(loggingMixin, formattingConfig):
    """ Test skipping the application of the formatting to strings which look like latex. """
    config, formattingDict = formattingConfig

    assert config["latexLike"] == r"$latex_{like \mathrm{x}}$"

