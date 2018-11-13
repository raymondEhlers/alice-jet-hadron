#!/usr/bin/env python

# Manages configuration of the jet-hadron analysis
#
# author: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
# date: 8 May 2018

# Py2/3 compatibility
from future.utils import iteritems

import os
import argparse
import collections
import logging
logger = logging.getLogger(__name__)

import jetH.base.genericConfig as genericConfig
import jetH.base.genericClass as genericClass
import jetH.base.params as params

def determineLeadingHadronBias(config, selectedAnalysisOptions):
    """ Determines the leading hadron bias based on the analysis options. It is then created and
    stored in the updated analysis options which are returned.

    Args:
        config (dict-like object): Contains the analysis configuration. Note that it must already be
            fully configured and overridden.
        selectedAnalysisOptions (params.selectedAnalysisOptions): Selected analysis options.
    Returns:
        params.selectedAnalysisOptions: Selected analysis options with the determined leading hadron
            bias object.
    """
    overrideOptions = genericConfig.determineOverrideOptions(selectedOptions = selectedAnalysisOptions,
                                                             overrideOptions = config["leadingHadronBiasValues"],
                                                             setOfPossibleOptions = params.setOfPossibleOptions)
    leadingHadronBiasValue = overrideOptions["value"]

    # Namedtuple is immutable, so we need to return a new one with the proper parameters
    returnOptions = selectedAnalysisOptions._asdict()
    returnOptions["leadingHadronBias"] = params.leadingHadronBias(type = selectedAnalysisOptions.leadingHadronBias, value = leadingHadronBiasValue)
    return params.selectedAnalysisOptions(**returnOptions)

def overrideOptions(config, selectedOptions, configContainingOverride = None):
    """ Override options for the jet-hadron analysis.

    Selected options include: (energy, collisionSystem, eventActivity, leadingHadronBias). Note that the order
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
    config = genericConfig.overrideOptions(config, selectedOptions,
                                           setOfPossibleOptions = params.setOfPossibleOptions,
                                           configContainingOverride = configContainingOverride)
    config = genericConfig.simplifyDataRepresentations(config)

    return config

def determineSelectedOptionsFromKwargs(args = None, description = "Jet-hadron {taskName}.", addOptionsFunction = None, **kwargs):
    """ Determine the selected analysis options from the command line arguments.

    Defaults are equivalent to None or False so values can be added in the validation
    function if argument values are not specified.

    Args:
        args (list): Arguments to parse. Default: None (which will then use sys.argv)
        description (str): Help description for arguments
        addOptionsFunction (func): Function which takes the ArgumentParser() object, adds
            arguments, and returns the object.
        kwargs (dict): Additional arguments to format the help description. Often contains "taskName" to specify
            the task name.
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
    args = parser.parse_args(args)

    # Even though we will need to create a new selected analysis options tuple, we store the
    # return values in one for convenience.
    selectedAnalysisOptions = params.selectedAnalysisOptions(collisionEnergy = args.energy,
                                                             collisionSystem = args.collisionSystem,
                                                             eventActivity = args.eventActivity,
                                                             leadingHadronBias = args.biasType)
    return (args.configFilename, selectedAnalysisOptions, args)

def validateArguments(selectedArgs, validateExtraArgsFunc = None):
    """ Validate arguments passed to the analysis task. Converts str and float types to enumerations.

    Note:
        If the selections are not specified, it will define to 2.76 TeV central PbPb collisions with a track bias!

    Args:
        selectedArgs (params.selectedAnalysisOptions): Selected analysis options from args or otherwise.
        validateExtraArgsFunc (func): Function to validate additional args that were added using addOptionsFunction().
            It should be a closure with the args returned from initial parsing and return a dict containing
            the validated args.
    Returns:
        tuple: (validatedSelectedOptions, additionalValidatedArgs)
    """

    # Validate the given arguments.
    # The general strategy is as follows:
    #   Input:
    #   - If the value is None, use the default.
    #   - If the value is given, then use the given value.
    #   Enum object creation:
    #   - Check if the input is already of the enum type. If so, use it.
    #   - If not, initialize the enum value using the given value.

    # Energy. Default: 2.76
    energy = selectedArgs.collisionEnergy if selectedArgs.collisionEnergy else 2.76
    # Retrieves the enum by value
    energy = energy if type(energy) is params.collisionEnergy else params.collisionEnergy(energy)
    # Collision system. Default: PbPb
    collisionSystem = selectedArgs.collisionSystem if selectedArgs.collisionSystem else "PbPb"
    collisionSystem = collisionSystem if type(collisionSystem) is params.collisionSystem else params.collisionSystem[collisionSystem]
    # Event activity. Default: central
    eventActivity = selectedArgs.eventActivity if selectedArgs.eventActivity else "central"
    eventActivity = eventActivity if type(eventActivity) is params.eventActivity else params.eventActivity[eventActivity]
    # Leading hadron bias type. Default: track
    leadingHadronBiasType = selectedArgs.leadingHadronBias if selectedArgs.leadingHadronBias else "track"
    leadingHadronBiasType = leadingHadronBiasType if type(leadingHadronBiasType) is params.leadingHadronBiasType else params.leadingHadronBiasType[leadingHadronBiasType]

    # Handle additional arguments
    additionalValidatedArgs = {}
    if validateExtraArgsFunc:
        additionalValidatedArgs.update(validateExtraArgsFunc())

    selectedAnalysisOptions = params.selectedAnalysisOptions(collisionEnergy = energy,
                                                             collisionSystem = collisionSystem,
                                                             eventActivity = eventActivity,
                                                             leadingHadronBias = leadingHadronBiasType)
    return (selectedAnalysisOptions, additionalValidatedArgs)

def constructFromConfigurationFile(taskName, configFilename, selectedAnalysisOptions, obj, additionalPossibleIterables = None):
    """ This is the main driver function to create an analysis object from a configuration.

    Args:
        taskName (str): Name of the analysis task.
        configFilename (str): Filename of the yaml config.
        selectedAnalysisOptions (params.selectedAnalysisOptions): Selected analysis options.
        obj (object): The object to be constructed.
        additionalPossibleIterables(collections.OrderedDict): Additional iterators to use when creating
            the objects, in the form of "name" : list(values). Default: None.
    Returns:
        (list, collections.OrderedDict): Roughly, (names, objects). Specifically, the list is the names
            of the iterables used. The ordered dict entries are of the form of a nested dict, with each
            object available at the iterable values used to constructed it. For example,
            output["a"]["b"] == obj(a = "a", b = "b", ...). For a full example, see above.
    """
    # Validate arguments
    (selectedAnalysisOptions, additionalValidatedArgs) = validateArguments(selectedAnalysisOptions)
    if additionalPossibleIterables is None:
        additionalPossibleIterables = collections.OrderedDict()

    # Load configuration
    config = genericConfig.loadConfiguration(configFilename)
    config = overrideOptions(config, selectedAnalysisOptions,
                             configContainingOverride = config[taskName])
    # We (re)define the task config here after we have overridden the relevant values.
    taskConfig = config[taskName]

    # Now that the values have been overrideen, we can determine the full leading hadron bias
    selectedAnalysisOptions = determineLeadingHadronBias(config = config, selectedAnalysisOptions = selectedAnalysisOptions)
    logger.debug("Selected analysis options: {}".format(selectedAnalysisOptions))

    # Iteration options
    # Selected on event plane and q vector are required since they are included in
    # the output prefix for consistency (event if a task doesn't select in one or
    # both of them)
    possibleIterables = collections.OrderedDict()
    possibleIterables["eventPlaneAngle"] = params.eventPlaneAngle
    possibleIterables["qVector"] = params.qVector
    # NOTE: Careful here - in principle, this could overwrite the EP or qVector iterators. However,
    #       it is unlikely.
    possibleIterables.update(additionalPossibleIterables)
    # NOTE: These requested iterators should be passed by the task,
    #       but then the values should be selected in the YAML config.
    iterables = genericConfig.determineSelectionOfIterableValuesFromConfig(config = config,
                                                                           possibleIterables = possibleIterables)

    # Determine formatting options
    logger.debug("selectedAnalysisOptions: {}".format(selectedAnalysisOptions._asdict()))
    # TODO: Do we want to modify str() to something like recreationString() or something in conjunction
    #       with renaming filenameStr()?
    formattingOptions = {}
    formattingOptions["taskName"] = taskName
    formattingOptions["trainNumber"] = config.get("trainNumber", "trainNo")

    # Determine task arguments
    args = collections.OrderedDict()
    args.update(formattingOptions)
    args["configFilename"] = configFilename
    args["config"] = config
    args["taskConfig"] = taskConfig

    # Add the selected analysis options into the args and formatting options
    # NOTE: We don't want to update the formattingOptions and then use that to update the args
    #       because otherwise we will have strings for the selected analysis options instead
    #       of the actual enumeration values.
    # NOTE: `_asdict()` is a public method - it has an underscore to avoid namespace conflicts.
    #       See: https://stackoverflow.com/a/26180604
    args.update(selectedAnalysisOptions._asdict())
    # We want to convert the enum values into strs for formatting. Performed with a dict comprehension.
    formattingOptions.update({k: v.str() for k, v in iteritems(selectedAnalysisOptions._asdict())})

    # Iterate over the iterables defined above to create the objects.
    (names, objects) = genericConfig.createObjectsFromIterables(obj = obj,
                                                                args = args,
                                                                iterables = iterables,
                                                                formattingOptions = formattingOptions)

    #logger.debug("objects: {objects}".format(objects = objects))

    return (names, objects)

class JetHBase(genericClass.EqualityMixin):
    """ Base class for shared jet-hadron configuration values.

    Args:
        taskName (str): Name of the task.
        configFilename (str): Filename of the YAML configuration.
        config (dict-like object): Contains the analysis configuration. Note that it must already be
            fully configured and overridden.
        taskConfig (dict-like object): Contains the task specific configuration. Note that it must already be
            fully configured and overridden. Also note that by convention it is also available at `config[taskName]`.
        collisionEnergy (params.collisionEnergy): Selected collision energy.
        collisionSystem (params.collisionSystem): Selected collision system.
        eventActivity (params.eventActivity): Selected event activity.
        leadingHadronBias (params.leadingHadronBias or params.leadingHadronBiasType): Selected leading hadron bias. The
            class member will contain both the type and the value.
        eventPlaneAngle (params.eventPlaneAngle): Selected event plane angle.
        args (list): Absorb extra arguments. They will be ignored.
        kwargs (dict): Absorb extra named arguments. They will be ignored.
    """
    def __init__(self, taskName, configFilename, config, taskConfig, collisionEnergy, collisionSystem, eventActivity, leadingHadronBias, eventPlaneAngle, *args, **kwargs):
        # Store the configuration
        self.taskName = taskName
        self.configFilename = configFilename
        self.config = config
        self.taskConfig = taskConfig
        self.collisionEnergy = collisionEnergy
        self.collisionSystem = collisionSystem
        self.eventActivity = eventActivity
        self.eventPlaneAngle = eventPlaneAngle

        # Handle leading hadron bias depending on the type.
        if isinstance(leadingHadronBias, params.leadingHadronBiasType):
            leadingHadronBias = determineLeadingHadronBias(config = self.config,
                                                           selectedAnalysisOptions = params.selectedAnalysisOptions(
                                                               collisionEnergy = self.collisionEnergy,
                                                               collisionSystem = self.collisionSystem,
                                                               eventActivity = self.eventActivity,
                                                               leadingHadronBias = leadingHadronBias)
                                                           ).leadingHadronBias
        # The type of leadingHadronBias should now be params.leadingHadronBias, regardless of whether that type was passed.
        self.leadingHadronBias = leadingHadronBias

        # File I/O
        # If in kwargs, use that value (which inherited class may use to override the config)
        # otherwise, use the value from the value from the config
        self.inputFilename = config["inputFilename"]
        self.inputListName = config["inputListName"]
        self.outputPrefix = config["outputPrefix"]
        self.outputFilename = config["outputFilename"]
        # Setup output area
        if not os.path.exists(self.outputPrefix):
            os.makedirs(self.outputPrefix)

        self.printingExtensions = config["printingExtensions"]
        # Convert the ALICE label if necessary
        aliceLabel = config["aliceLabel"]
        if isinstance(aliceLabel, str):
            aliceLabel = params.aliceLabel[aliceLabel]
        self.aliceLabel = aliceLabel

    def writeConfig(self):
        """ Write the properties of the analysis to a YAML configuration file for future reference. """
        logger.info("{name} Properties:".format(name = self.__class__.__name__))
        # See: https://stackoverflow.com/a/1398059
        properties = {attr: getattr(self, attr) for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")}

        # TODO: Implement writing this out

def createFromTerminal(obj, taskName, additionalPossibleIterables = None):
    """ Main function to create an object from the terminal.

    Args:
        obj (object): Object to be created.
        taskName (str): Name of the task to be created.
        additionalPossibleIterables(collections.OrderedDict): Additional iterators to use when creating
            the objects, in the form of "name" : list(values). Default: None.
    Returns:
        (list, collections.OrderedDict): Roughly, (names, objects). Specifically, the list is the names
            of the iterables used. The ordered dict entries are of the form of a nested dict, with each
            object available at the iterable values used to constructed it. For example,
            output["a"]["b"] == obj(a = "a", b = "b", ...). For a full example, see above.
    """
    (configFilename, terminalArgs, additionalArgs) = determineSelectedOptionsFromKwargs(taskName = taskName)
    return constructFromConfigurationFile(taskName = taskName,
                                          configFilename = configFilename,
                                          selectedAnalysisOptions = terminalArgs,
                                          obj = obj,
                                          additionalPossibleIterables = additionalPossibleIterables)

# TODO: Create a list of pt hard objects based on the YAML config.
#       Loop over that list to create objects.
#       Implement the various string functions to match up with the enums
#       However, where is the list going to be stored? It's not really
#       natural in an analysis task...
class ptHard(object):
    def __init__(self):
        pass

