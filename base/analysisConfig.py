#!/usr/bin/env python

# Manages configuration of the jet-hadron analysis

# Py2/3 compatibility
from builtins import super
from future.utils import iteritems
from future.utils import itervalues

import os
import argparse
import collections
import itertools
import enum
import logging
logger = logging.getLogger(__name__)

import jetH.base.genericConfig as genericConfig
import jetH.base.params as params

def overrideOptions(config, selectedOptions, configContainingOverride = None):
    """ Override options for the jet-hadron analysis.

    Selected options include: (energy, collisionSystem, eventActivity, leadingHadronBiasType). Note that the order
    is extremely important! If one of them is not specified in the override, then it will be skipped.

    Args:
        config (CommentedMap): The dict-like configuration from ruamel.yaml which should be overridden.
        selectedOptions (tuple): The selected analysis options. They will be checked in the order with which
            they are passed, so make certain that it matches the order in the configuration file!
        configContainingOverride (CommentedMap): The dict-like config containing the override options in a map called
            "override". If it is not specified, it will look for it in the main config.
    Returns:
        dict: The updated configuration
    """
    setOfPossibleOptions = (params.collisionEnergy,
            params.collisionSystem,
            params.eventActivity,
            params.leadingHadronBiasType)

    config = genericConfig.overrideOptions(config, selectedOptions, setOfPossibleOptions,
            configContainingOverride = configContainingOverride)
    config = genericConfig.simplifyDataRepresentations(config)

    return config

def determineSelectedOptionsFromKwargs(description = "Jet-hadron {taskName}", addOptionsFunction = None, **kwargs):
    """ Determine the selected analysis options from the command line arguments.

    Defaults are equivalent to None or False so values can be added in the validation
    function if argument values are not specified.

    Args:
        description (str): Help description for arguments
        addOptionsFunction (func): Function which takes the ArgumentParser() object, adds
            arguments, and returns the object.
        kwargs (dict): Additional arguments to format the description
    Returns:
        tuple: (configFilename, energy, collisionSystem, eventActivity, biasType, argparse.namespace). The args
            are return for handling custom arguments added with addOptionsFunction.
    """
    # Make sure there is always a task name
    if "taskName" not in kwargs:
        kwargs["taskName"] = "analysis"

    # Setup parser
    parser = argparse.ArgumentParser(description = description.format(**kwargs))
    # General options
    parser.add_argument("-c", "--configFilename", metavar="configFilename",
            type = str, default = "config/analysisConfig.yaml",
            help="Path to config filename")
    parser.add_argument("-e", "--energy", metavar = "energy",
            type = float, default = 0.0,
            help = "Collision energy")
    parser.add_argument("-s", "--collisionSystem", metavar = "collisionSystem",
            type = str, default = "",
            help = "Collision system")
    parser.add_argument("-a", "--eventActivity", metavar = "eventActivity",
            type = str, default = "",
            help = "Event activity")
    parser.add_argument("-b", "--biasType", metavar="biasType",
            type = str, default = "",
            help = "Leading hadron bias type")

    # Extension for additional arguments
    if addOptionsFunction:
        args = addOptionsFunction(parser)

    # Parse arguments
    args = parser.parse_args()

    # Even though we will need to create a new selected analysis options tuple, we store the
    # return values in one for convenience.
    selectedAnalysisOptions = params.selectedAnalysisOptions(energy = args.energy,
            collisionSystem = args.collisionSystem,
            eventActivity = args.eventActivity,
            leadingHadronBiasType = args.biasType)
    return (args.configFilename, selectedAnalysisOptions, args)

def validateArguments(selectedArgs, validateExtraArgsFunc = None):
    """ Validate arguments passed to the analysis task.

    Args:
        selectedArgs (params.selectedAnalysisOptions): Selected analysis options from args or otherwise.
        validateExtraArgsFunc (func): Function to validate additional args that were added using addOptionsFunction().
            It should be a closure with the args returned from initial parsing and return a dict containing
            the validated args.
    Returns:
        tuple: (validatedSelectedOptions, additionalValidatedArgs)
    """

    # Energy. Default: 2.76
    energy = selectedArgs.energy if selectedArgs.energy else 2.76
    # Retrieves the enum by value
    energy = params.collisionEnergy(energy)
    # Collision system. Default: PbPb
    collisionSystem = selectedArgs.collisionSystem if selectedArgs.collisionSystem else "PbPb"
    collisionSystem = params.collisionSystem[collisionSystem]
    # Event activity. Default: central
    eventActivity = selectedArgs.eventActivity if selectedArgs.eventActivity else "central"
    eventActivity = params.eventActivity[eventActivity]
    # Leading hadron bias type. Default: track
    leadingHadronBiasType = selectedArgs.leadingHadronBiasType if selectedArgs.leadingHadronBiasType else "track"
    leadingHadronBiasType = params.leadingHadronBiasType[leadingHadronBiasType]

    # Handle additional arguments
    additionalValidatedArgs = {}
    if validateExtraArgsFunc:
        additionalValidatedArgs.update(validateExtraArgsFunc())

    selectedAnalysisOptions = params.selectedAnalysisOptions(energy =energy,
            collisionSystem = collisionSystem,
            eventActivity = eventActivity,
            leadingHadronBiasType = leadingHadronBiasType)
    return (selectedAnalysisOptions, additionalValidatedArgs)

def constructFromConfigurationFile(taskName, configFilename, selectedAnalysisOptions, obj, additionalIterators = None):
    """ This is the main driver function to create an analysis object from a configuration.
    
    Args:
        taskName (str): Name of the analysis task.
        configFilename (str): Filename of the yaml config.
        selectedAnalysisOptions (params.selectedAnalysisOptions): Selected analysis options.
        obj (object): The object to be constructed.
        additionalIterators (dict): Additional iterators to use when creating the objects, in the
            form of "name" : list(values). Default: None.
    Returns:
        OrderedDict: Arguments which can be used to construct the analysis task.
    """
    if additionalIterators is None:
        additionalIterators = {}
    # Validate arguments
    (selectedAnalysisOptions, additionalValidatedArgs) = validateArguments(selectedAnalysisOptions)

    # Load configuration
    config = genericConfig.loadConfiguration(configFilename)
    config = overrideOptions(config, selectedAnalysisOptions,
            configContainingOverride = config[taskName])
    # We (re)define the task config here after we have overridden the relevant values.
    taskConfig = config[taskName]

    # Iteration options
    # Selected on event plane and q vector are required since they are included in
    # the output prefix for consistency (event if a task doesn't select in one or
    # both of them)
    iteratorsConfig = config["iterators"]
    iterators = collections.OrderedDict()
    if iteratorsConfig.get("eventPlaneDependentAnalysis", False):
        # Explicitly call list to ensure that the possible values are enumerated.
        eventPlaneIterator = list(params.eventPlaneAngle)
    else:
        eventPlaneIterator = [params.eventPlaneAngle.kAll]
    iterators["eventPlaneAngle"] = eventPlaneIterator
    if iteratorsConfig.get("qVectorDependentAnalysis", False):
        # Explicitly call list to ensure that the possible values are enumerated.
        qVectorIterator = list(params.qVector)
    else:
        qVectorIterator = [params.qVector.all]
    iterators["qVector"] = qVectorIterator
    # Additional iterators defined in the configuration. Note that these should
    additionalIteratorsInConfig = iteratorsConfig.get("additional", {})
    for k, v in iteritems(additionalIteratorsInConfig):
        logger.debug("k: {}, v: {}".format(k, v))
        additionalIterator = []
        enum = additionalIterators[k]
        for el in v:
            additionalIterator.append(enum[el])
        iterators[k] = additionalIterator

    # Determine formatting options
    # NOTE: `_asdict()` is a public method - it has an underscore to avoid namespace conflicts.
    #       See: https://stackoverflow.com/a/26180604
    logger.debug("selectedAnalysisOptions: {}".format(selectedAnalysisOptions._asdict()))
    formattingOptions = {k : v.str() for k, v in iteritems(selectedAnalysisOptions._asdict())}
    formattingOptions["taskName"] = taskName
    formattingOptions["trainNumber"] = config.get("trainNumber", "trainNo")

    # Determine task arguments
    args = collections.OrderedDict()
    args.update(formattingOptions)
    args["config"] = config
    args["taskConfig"] = taskConfig

    # Iterate over the iterators defined above to create the objects.
    objects = collections.OrderedDict()
    names = list(iterators)
    logger.debug("iterators: {iterators}".format(iterators = iterators))
    for values in itertools.product(*itervalues(iterators)):
        logger.debug("Values: {values}".format(values = values))
        tempDict = objects
        for i, val in enumerate(values):
            args[names[i]] = val
            logger.debug("i: {i}, val: {val}".format(i = i, val = val))
            formattingOptions[names[i]] = val.filenameStr()
            # We should construct the object once we get to the last value
            if i != len(values) - 1:
                tempDict = tempDict.setdefault(val, collections.OrderedDict())
            else:
                # Apply formatting options
                objectArgs = genericConfig.applyFormattingDict(args, formattingOptions)
                # Skip printing the config because it is quite long
                printArgs = {k : v for k, v in iteritems(objectArgs) if k != "config"}
                printArgs["config"] = "..."
                logger.debug("Constructing obj \"{obj}\" with args: \"{printArgs}\"".format(obj = obj, printArgs = printArgs))

                # Create and store the object
                tempDict[val] = obj(**objectArgs)

    # If nothing has been created at this point, then we are didn't iterating over anything and something
    # has gone wrong.
    if not objects:
        logger.critical("Failed to create any objects using args: {args}".format(args = args))

    logger.debug("objects: {objects}".format(objects = objects))

    return (names, objects)

class JetHBase(object):
    """ Base class for shared jet-hadron configuration values.

    Args:
        taskName (str): Name of the task.
        config (dict-like object): Contains the analysis configuration. Note that it must already be
            fully configured and overridden.
        taskConfig (dict-like object): Contains the task specific configuration. Note that it must already be
            fully configured and overridden. Also note that by convention it is also available at `config[taskName]`.
        energy (params.collisionEnergy): Selected collision energy.
        collisionSystem (params.collisionSystem): Selected collision system.
        eventActivity (params.eventActivity): Selected event activity.
        leadingHadronBiasType (params.leadingHadronBiasType): Selected leading hadron bias.
        eventPlaneAngle (params.eventPlaneAngle): Selected event plane angle.
        args (list): Absorb extra arguments. They will be ignored.
        kwargs (dict): Absorb extra named arguments. They will be ignored.
    """
    def __init__(self, taskName, config, taskConfig, energy, collisionSystem, eventActivity, leadingHadronBiasType, eventPlaneAngle, *args, **kwargs):
        # Store the configuration
        self.taskName = taskName
        self.config = config
        self.taskConfig = taskConfig
        self.energy = energy
        self.collisionSystem = collisionSystem
        self.eventActivity = eventActivity
        self.leadingHadronBiasType = leadingHadronBiasType
        self.eventPlaneAngle = eventPlaneAngle

        # File I/O
        # If in kwargs, use that value (which inherited class may use to override the config)
        # otherwise, use the value from the value from the config
        self.inputFilename = config["inputFilename"]
        ## TODO: "outputListName" -> "inputListName"
        self.inputListName = config["inputListName"]
        self.outputPrefix = config["outputPrefix"]
        self.outputFilename = config["outputFilename"]
        #self.inputFilename = getValueFromConfigIfNotPassed("inputFilename", config, kwargs)
        ## TODO: "outputListName" -> "inputListName"
        #self.inputListName = getValueFromConfigIfNotPassed("inputListName", config, kwargs)
        #self.outputPrefix = getValueFromConfigIfNotPassed("outputPrefix", config, kwargs)
        #self.outputFilename = getValueFromConfigIfNotPassed("outputFilename", config, kwargs)

        # Setup output area
        # TODO: Uncomment
        #if not os.path.exists(outputPrefix):
        #    os.makedirs(outputPrefix)

        self.printingExtensions = config["printingExtensions"]
        self.aliceLabelType = config["aliceLabelType"]

    #@staticmethod
    #def getValueFromConfigIfNotPassed(name, config, kwargs):
    #    if name in kwargs:
    #        return kwargs[name]
    #    else:
    #        return config[name]

def createFromTerminal(obj, taskName, additionalIterators = None):
    """ Main function to create an object from the terminal

    Args:

    Returns:
    """
    (configFilename, terminalArgs, additionalArgs) = determineSelectedOptionsFromKwargs(taskName = taskName)
    return constructFromConfigurationFile(taskName = taskName,
            configFilename = configFilename,
            selectedAnalysisOptions = terminalArgs,
            obj = obj,
            additionalIterators = additionalIterators)

def unrollNestedDict(d, keys = None):
    """

    Args:

    Returns:
    """
    if keys is None:
        keys = []
    #logger.debug("d: {}".format(d))
    for k, v in iteritems(d):
        #logger.debug("k: {}, v: {}".format(k, v))
        #logger.debug("keys: {}".format(keys))
        if isinstance(v, dict):
            keys.append(k)
            #logger.debug("v is a dict!")
            # Could be `yield from`, but then it wouldn't work in python 2.
            # We take a small performance hit here, but it's fine.
            # See: https://stackoverflow.com/a/38254338
            for val in unrollNestedDict(d = v, keys = keys):
                yield val
        else:
            # We need a copy of keys before we append to ensure that we don't
            # have the final keys build up (ie. first yield [a], next [a, b], then [a, b, c], etc...)
            copyOfKeys = keys[:]
            copyOfKeys.append(k)
            #logger.debug("Yielding {}".format(v))
            yield (copyOfKeys, v)

