#!/usr/bin/env python

""" Manages configuration of the jet-hadron analysis

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

# Py2/3 compatibility
from future.utils import iteritems

import argparse
import logging
import os

from pachyderm import generic_class
from pachyderm import generic_config
from jet_hadron.base import params

logger = logging.getLogger(__name__)

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
    override_options = generic_config.determineOverrideOptions(
        selectedOptions = selectedAnalysisOptions,
        override_opts = config["leadingHadronBiasValues"],
        setOfPossibleOptions = params.setOfPossibleOptions
    )
    leadingHadronBiasValue = override_options["value"]

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
    config = generic_config.overrideOptions(config, selectedOptions,
                                            setOfPossibleOptions = params.setOfPossibleOptions,
                                            configContainingOverride = configContainingOverride)
    config = generic_config.simplifyDataRepresentations(config)

    return config

def determine_selected_options_from_kwargs(args = None, description = "Jet-hadron {task_name}.", add_options_function = None, **kwargs):
    """ Determine the selected analysis options from the command line arguments.

    Defaults are equivalent to None or False so values can be added in the validation
    function if argument values are not specified.

    Args:
        args (list): Arguments to parse. Default: None (which will then use sys.argv)
        description (str): Help description for arguments
        add_options_function (func): Function which takes the ArgumentParser() object, adds
            arguments, and returns the object.
        kwargs (dict): Additional arguments to format the help description. Often contains ``task_name``
            to specify the task name.
    Returns:
        tuple: (configFilename, energy, collisionSystem, eventActivity, biasType, argparse.namespace). The args
            are return for handling custom arguments added with addOptionsFunction.
    """
    # Make sure there is always a task name
    if "task_name" not in kwargs:
        kwargs["task_name"] = "analysis"

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
    if add_options_function:
        args = add_options_function(parser)

    # Parse arguments
    args = parser.parse_args(args)

    # Even though we will need to create a new selected analysis options tuple, we store the
    # return values in one for convenience.
    selected_analysis_options = params.selectedAnalysisOptions(collisionEnergy = args.energy,
                                                               collisionSystem = args.collisionSystem,
                                                               eventActivity = args.eventActivity,
                                                               leadingHadronBias = args.biasType)
    return (args.configFilename, selected_analysis_options, args)

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

def construct_from_configuration_file(task_name, config_filename, selected_analysis_options, obj, additional_possible_iterables = None):
    """ This is the main driver function to create an analysis object from a configuration.

    Args:
        task_name (str): Name of the analysis task.
        config_filename (str): Filename of the yaml config.
        selected_analysis_options (params.selectedAnalysisOptions): Selected analysis options.
        obj (object): The object to be constructed.
        additional_possible_iterables(dict): Additional iterators to use when creating the objects,
            in the form of "name" : list(values). Default: None.
    Returns:
        (object, list, dict): Roughly, (KeyIndex, names, objects). Specifically, the key_index is a
            new dataclass which defines the parameters used to create the object, names is the names
            of the iterables used. The dictionary keys are KeyIndex objects which describe the iterable
            arguments passed to the object, while the values are the newly constructed arguments. See
            ``pachyderm.generic_config.create_objects_from_iterables(...)`` for more.
    """
    # Validate arguments
    (selected_analysis_options, additional_validated_args) = validateArguments(selected_analysis_options)
    if additional_possible_iterables is None:
        additional_possible_iterables = {}

    # Load configuration
    config = generic_config.loadConfiguration(config_filename)
    config = overrideOptions(config, selected_analysis_options,
                             configContainingOverride = config[task_name])
    # We (re)define the task config here after we have overridden the relevant values.
    task_config = config[task_name]

    # Now that the values have been overrideen, we can determine the full leading hadron bias
    selected_analysis_options = determineLeadingHadronBias(config = config, selectedAnalysisOptions = selected_analysis_options)
    logger.debug(f"Selected analysis options: {selected_analysis_options}")

    # Iteration options
    # Selected on event plane and q vector are required since they are included in
    # the output prefix for consistency (event if a task doesn't select in one or
    # both of them)
    possible_iterables = {}
    possible_iterables["eventPlaneAngle"] = params.eventPlaneAngle
    possible_iterables["qVector"] = params.qVector
    # NOTE: Careful here - in principle, this could overwrite the EP or qVector iterators. However,
    #       it is unlikely.
    possible_iterables.update(additional_possible_iterables)
    # NOTE: These requested iterators should be passed by the task,
    #       but then the values should be selected in the YAML config.
    iterables = generic_config.determineSelectionOfIterableValuesFromConfig(config = config,
                                                                            possibleIterables = possible_iterables)

    # Determine formatting options
    logger.debug(f"selectedAnalysisOptions: {selected_analysis_options._asdict()}")
    # TODO: Do we want to modify str() to something like recreationString() or something in conjunction
    #       with renaming filenameStr()?
    formatting_options = {}
    formatting_options["task_name"] = task_name
    formatting_options["trainNumber"] = config.get("trainNumber", "trainNo")

    # Determine task arguments
    args = {}
    args.update(formatting_options)
    args["config_filename"] = config_filename
    args["config"] = config
    args["task_config"] = task_config

    # Add the selected analysis options into the args and formatting options
    # NOTE: We don't want to update the formatting_options and then use that to update the args
    #       because otherwise we will have strings for the selected analysis options instead
    #       of the actual enumeration values.
    # NOTE: `_asdict()` is a public method - it has an underscore to avoid namespace conflicts.
    #       See: https://stackoverflow.com/a/26180604
    args.update(selected_analysis_options._asdict())
    # We want to convert the enum values into strs for formatting. Performed with a dict comprehension.
    formatting_options.update({k: v.str() for k, v in iteritems(selected_analysis_options._asdict())})

    # Iterate over the iterables defined above to create the objects.
    (KeyIndex, names, objects) = generic_config.create_objects_from_iterables(
        obj = obj,
        args = args,
        iterables = iterables,
        formatting_options = formatting_options,
    )

    logger.debug(f"KeyIndex: {KeyIndex}, objects: {objects}")

    return (KeyIndex, names, objects)

class JetHBase(generic_class.EqualityMixin):
    """ Base class for shared jet-hadron configuration values.

    Args:
        task_name (str): Name of the task.
        config_filename (str): Filename of the YAML configuration.
        config (dict-like object): Contains the analysis configuration. Note that it must already be
            fully configured and overridden.
        task_config (dict-like object): Contains the task specific configuration. Note that it must already be
            fully configured and overridden. Also note that by convention it is also available at
            ``config[task_name]``.
        collisionEnergy (params.collisionEnergy): Selected collision energy.
        collisionSystem (params.collisionSystem): Selected collision system.
        eventActivity (params.eventActivity): Selected event activity.
        leadingHadronBias (params.leadingHadronBias or params.leadingHadronBiasType): Selected leading hadron
            bias. The class member will contain both the type and the value.
        eventPlaneAngle (params.eventPlaneAngle): Selected event plane angle.
        args (list): Absorb extra arguments. They will be ignored.
        kwargs (dict): Absorb extra named arguments. They will be ignored.
    """
    def __init__(self, task_name, config_filename, config, task_config, collisionEnergy, collisionSystem, eventActivity, leadingHadronBias, eventPlaneAngle, *args, **kwargs):
        # Store the configuration
        self.task_name = task_name
        self.config_filename = config_filename
        self.config = config
        self.task_config = task_config
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

def create_from_terminal(obj, task_name, additional_possible_iterables = None):
    """ Main function to create an object from the terminal.

    Args:
        obj (object): Object to be created.
        task_name (str): Name of the task to be created.
        additional_possible_iterables(dict): Additional iterators to use when creating
            the objects, in the form of "name" : list(values). Default: None.
    Returns:
        (object, list, dict): Roughly, (KeyIndex, names, objects). Specifically, the key_index is a
            new dataclass which defines the parameters used to create the object, names is the names
            of the iterables used. The dictionary keys are KeyIndex objects which describe the iterable
            arguments passed to the object, while the values are the newly constructed arguments. See
            ``pachyderm.generic_config.create_objects_from_iterables(...)`` for more.
    """
    (config_filename, terminal_args, additional_args) = determine_selected_options_from_kwargs(task_name = task_name)
    return construct_from_configuration_file(
        task_name = task_name,
        config_filename = config_filename,
        selected_analysis_options = terminal_args,
        obj = obj,
        additional_possible_iterables = additional_possible_iterables
    )

# TODO: Create a list of pt hard objects based on the YAML config.
#       Loop over that list to create objects.
#       Implement the various string functions to match up with the enums
#       However, where is the list going to be stored? It's not really
#       natural in an analysis task...
class ptHard(object):
    def __init__(self):
        pass

