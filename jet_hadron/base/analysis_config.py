#!/usr/bin/env python

""" Manages configuration of the jet-hadron analysis

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import argparse
import logging
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from pachyderm import generic_config
# Provided for convenience of analysis classes
from pachyderm.generic_config import create_key_index_object  # noqa: F401
from pachyderm.generic_config import iterate_with_selected_objects  # noqa: F401
from pachyderm.generic_config import iterate_with_selected_objects_in_order  # noqa: F401
from pachyderm import yaml

from jet_hadron.base import analysis_objects
from jet_hadron.base import params

logger = logging.getLogger(__name__)

# Typing
ConstructedObjects = Tuple[Any, Mapping[str, Any], Mapping[Any, Any]]

def determine_leading_hadron_bias(config: generic_config.DictLike, selected_analysis_options: params.SelectedAnalysisOptions) -> params.SelectedAnalysisOptions:
    """ Determines the leading hadron bias based on the analysis options.

    The determined leading hadron bias object is then created and stored in an updated selected
    analysis options object which is returned.

    Args:
        config: Contains the dict-like analysis configuration. Note that it must already be
            fully configured and overridden.
        selected_analysis_options: Selected analysis options.
    Returns:
        Selected analysis options with the determined leading hadron bias object.
    """
    override_options = generic_config.determine_override_options(
        selected_options = selected_analysis_options.astuple(),
        override_opts = config["leadingHadronBiasValues"],
        set_of_possible_options = params.SetOfPossibleOptions.astuple(),
    )
    leading_hadron_bias_type = selected_analysis_options.leading_hadron_bias
    leading_hadron_bias_value = override_options["value"]
    # Help out mypy
    assert isinstance(leading_hadron_bias_type, params.LeadingHadronBiasType)

    # Namedtuple is immutable, so we need to return a new one with the proper parameters
    return_options = dict(selected_analysis_options)
    return_options["leading_hadron_bias"] = params.LeadingHadronBias(type = leading_hadron_bias_type,
                                                                     value = leading_hadron_bias_value)
    return params.SelectedAnalysisOptions(**return_options)

def override_options(config: generic_config.DictLike, selected_options: params.SelectedAnalysisOptions, config_containing_override: generic_config.DictLike = None) -> generic_config.DictLike:
    """ Override options for the jet-hadron analysis.

    Selected options include: (energy, collision_system, event_activity, leading_hadron_bias). Note
    that the order is extremely important! If one of them is not specified in the override, then it
    will be skipped.

    Args:
        config: The dict-like configuration from ruamel.yaml which should be overridden.
        selected_options: The selected analysis options. They will be checked in the order with which
            they are passed, so make certain that it matches the order in the configuration file!
        config_containing_override (CommentedMap): The dict-like config containing the override options in
            a map called "override". If it is not specified, it will look for it in the main config.
    Returns:
        dict: The updated configuration
    """
    config = generic_config.override_options(
        config, selected_options.astuple(),
        set_of_possible_options = params.SetOfPossibleOptions.astuple(),
        config_containing_override = config_containing_override,
    )
    config = generic_config.simplify_data_representations(config)

    return config

def determine_selected_options_from_kwargs(
        args: Optional[List[Any]] = None,
        description: str = "Jet-hadron {task_name}.",
        add_options_function: Optional[Callable[[argparse.ArgumentParser], Any]] = None,
        **kwargs: str) -> Tuple[str, params.SelectedAnalysisOptions, argparse.Namespace]:
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
        tuple: (config_filename, energy, collision_system, event_activity, bias_type, argparse.Namespace).
            The args are return for handling custom arguments added with add_options_function.
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
    parsed_args = parser.parse_args(args)

    # Even though we will need to create a new selected analysis options tuple, we store the
    # return values in one for convenience.
    selected_analysis_options = params.SelectedAnalysisOptions(collision_energy = parsed_args.energy,
                                                               collision_system = parsed_args.collisionSystem,
                                                               event_activity = parsed_args.eventActivity,
                                                               leading_hadron_bias = parsed_args.biasType)
    return (parsed_args.configFilename, selected_analysis_options, parsed_args)

def validate_arguments(selected_args: params.SelectedAnalysisOptions, validate_extra_args_func: Any = None) -> Tuple[params.SelectedAnalysisOptions, Dict[str, Any]]:
    """ Validate arguments passed to the analysis task. Converts str and float types to enumerations.

    Note:
        If the selections are not specified, it will define to 2.76 TeV central PbPb collisions with a
        track bias!

    Args:
        selected_args: Selected analysis options from args or otherwise.
        validate_extra_args_func (func): Function to validate additional args that were added using
            ``add_options_function()``. It should be a closure with the args returned from initial parsing
            and return a dict containing the validated args.
    Returns:
        tuple: (validated_selected_options, additional_validated_args)
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
    energy = selected_args.collision_energy if selected_args.collision_energy else 2.76
    # Retrieves the enum by value
    energy = energy if type(energy) is params.CollisionEnergy else params.CollisionEnergy(energy)
    # Collision system. Default: PbPb
    collision_system = selected_args.collision_system if selected_args.collision_system else "PbPb"
    collision_system = collision_system if type(collision_system) is params.CollisionSystem else params.CollisionSystem[collision_system]  # type: ignore
    # Event activity. Default: central
    event_activity = selected_args.event_activity if selected_args.event_activity else "central"
    event_activity = event_activity if type(event_activity) is params.EventActivity else params.EventActivity[event_activity]  # type: ignore
    # Leading hadron bias type. Default: track
    leading_hadron_bias_type = selected_args.leading_hadron_bias if selected_args.leading_hadron_bias else "track"
    leading_hadron_bias_type = leading_hadron_bias_type if type(leading_hadron_bias_type) is params.LeadingHadronBiasType else params.LeadingHadronBiasType[leading_hadron_bias_type]  # type: ignore

    # Handle additional arguments
    additional_validated_args: Dict[str, Any] = {}
    if validate_extra_args_func:
        additional_validated_args.update(validate_extra_args_func())

    selected_analysis_options = params.SelectedAnalysisOptions(  # type: ignore
        collision_energy = energy,
        collision_system = collision_system,
        event_activity = event_activity,
        leading_hadron_bias = leading_hadron_bias_type,
    )
    return (selected_analysis_options, additional_validated_args)

def read_config_using_selected_options(task_name: str, config_filename: str,
                                       selected_analysis_options: params.SelectedAnalysisOptions,
                                       additional_classes_to_register: Optional[Sequence[Any]] = None) -> Tuple[generic_config.DictLike, params.SelectedAnalysisOptions]:
    """ Read the YAML config and override the values using the task config.

    This function provides separate functionality for a class to determine its
    associated configuration without creating itself via iterators. This is particularly
    useful for manager classes.

    We combine these steps together because we never want to use a configuration
    without overriding the values based on the selected analysis options.

    Args:
        task_name: Name of the analysis task.
        config_filename: Filename of the YAML config.
        selected_analysis_options: Selected analysis options.
        additional_classes_to_register: Additional classes to register from YAML object
            construction. Default: None.
    Returns:
        (The YAML configuration, the overridden selected analysis options).
    """
    # Validation
    if additional_classes_to_register is None:
        additional_classes_to_register = []

    # Classes to register for reconstruction within YAML
    classes_to_register: Set[Any] = set([
        # We can add any additional classes that are needed here.
        # These should be additional classes which aren't defined in params or analysis_objects.
    ])
    # Add requested iterables
    classes_to_register.update(additional_classes_to_register)
    logger.debug(f"classes_to_register: {classes_to_register}")
    # Add in all classes defined in the params and analysis_objects module
    yml = yaml.yaml(modules_to_register = [params, analysis_objects], classes_to_register = classes_to_register)

    # Load and override the configuration
    config = generic_config.load_configuration(
        yaml = yml,
        filename = config_filename,
    )
    config = override_options(
        config = config,
        selected_options = selected_analysis_options,
        config_containing_override = config[task_name]
    )

    # Now that the values have been overridden, we can determine the full leading hadron bias
    selected_analysis_options = determine_leading_hadron_bias(
        config = config,
        selected_analysis_options = selected_analysis_options
    )

    return config, selected_analysis_options

def determine_formatting_options(task_name: str, config: generic_config.DictLike,
                                 selected_analysis_options: params.SelectedAnalysisOptions) -> Dict[str, Any]:
    """ Determine the formatting dict with the selected analysis options.

    Beyond the selected analysis options, it is provides additional information, including task name,
    train number, etc.  This is just a simpler helper function which can also be used outside of
    configuring JetHBase objects (for example, configuring analysis managers).

    Args:
        task_name: Name of the analysis task.
        config: Contains the dict-like analysis configuration. Note that it must already be
            fully configured and overridden.
        selected_analysis_options: Selected analysis options.
    Returns:
        Dict containing the formatting options.
    """
    formatting_options = {}
    formatting_options["task_name"] = task_name
    formatting_options["trainNumber"] = config.get("trainNumber", "trainNo")

    # We want to convert the enum values into strings for formatting. Performed with a dict comprehension.
    formatting_options.update({k: str(v) for k, v in selected_analysis_options})

    return formatting_options

def construct_from_configuration_file(task_name: str, config_filename: str,
                                      selected_analysis_options: params.SelectedAnalysisOptions,
                                      obj: Any,
                                      additional_possible_iterables: Optional[Dict[str, Any]] = None,
                                      additional_classes_to_register: Optional[List[Any]] = None) -> ConstructedObjects:
    """ This is the main driver function to create an analysis object from a configuration.

    Args:
        task_name: Name of the analysis task.
        config_filename: Filename of the YAML config.
        selected_analysis_options: Selected analysis options.
        obj (object): The object to be constructed.
        additional_possible_iterables: Additional iterators to use when creating the objects,
            in the form of "name": list(values). Default: None.
        additional_classes_to_register: Additional classes to register from YAML object
            construction.
    Returns:
        (object, list, dict): Roughly, (KeyIndex, names, objects). Specifically, the key_index is a
            new dataclass which defines the parameters used to create the object, names is the names
            of the iterables used. The dictionary keys are KeyIndex objects which describe the iterable
            arguments passed to the object, while the values are the newly constructed arguments. See
            ``pachyderm.generic_config.create_objects_from_iterables(...)`` for more.
    """
    # Validate and setup arguments
    (selected_analysis_options, additional_validated_args) = validate_arguments(selected_analysis_options)
    if additional_possible_iterables is None:
        additional_possible_iterables = {}
    if additional_classes_to_register is None:
        additional_classes_to_register = []

    # Setup iterables
    # Selected on event plane and q vector are required since they are included in
    # the output prefix for consistency (event if a task doesn't select in one or
    # both of them)
    possible_iterables = {}
    # These names map the config/attribute names to the iterable.
    possible_iterables["reaction_plane_orientation"] = params.ReactionPlaneOrientation
    possible_iterables["qvector"] = params.QVector  # type: ignore
    possible_iterables.update({k: v for k, v in additional_possible_iterables.items() if k not in possible_iterables})

    # We also want all possible iterables, but we have to skip None iterables (whose values are
    # defined in the YAML), and thus must already included in the additional_classes_to_register list.
    additional_classes_to_register.extend([v for v in possible_iterables.values() if v])

    # Now we actually read the config, overriding the config values as appropriate based
    # on the selected analysis options.
    config, selected_analysis_options = read_config_using_selected_options(
        task_name = task_name, config_filename = config_filename,
        selected_analysis_options = selected_analysis_options,
        additional_classes_to_register = additional_classes_to_register,
    )
    logger.debug(f"Selected analysis options: {selected_analysis_options}")
    # We (re)define the task config here after we have overridden the relevant values.
    task_config = config[task_name]

    # Iteration options
    # NOTE: These requested iterators should be passed by the task,
    #       but then the values should be selected in the YAML config.
    iterables = generic_config.determine_selection_of_iterable_values_from_config(
        config = config,
        possible_iterables = possible_iterables
    )

    # Determine formatting options
    logger.debug(f"selected_analysis_options: {dict(selected_analysis_options)}")
    formatting_options = determine_formatting_options(
        task_name = task_name, config = config,
        selected_analysis_options = selected_analysis_options,
    )

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
    args.update(dict(selected_analysis_options))

    # Iterate over the iterables defined above to create the objects.
    (KeyIndex, returned_iterables, objects) = generic_config.create_objects_from_iterables(
        obj = obj,
        args = args,
        iterables = iterables,
        formatting_options = formatting_options,
    )

    #logger.debug(f"KeyIndex: {KeyIndex}, objects: {objects}")

    return (KeyIndex, returned_iterables, objects)

def create_from_terminal(obj: Any, task_name: str, additional_possible_iterables: Optional[Dict[str, Any]] = None) -> ConstructedObjects:
    """ Main function to create an object from the terminal.

    Note:
        This frequently isn't used directly because we want to use an analysis manager approach.
        However, it is kept to illustrate how directly creating an object might look.

    Args:
        obj: Object to be created.
        task_name: Name of the task to be created.
        additional_possible_iterables: Additional iterators to use when creating
            the objects, in the form of "name" : list(values). Default: None.
    Returns:
        (object, list, dict): Roughly, (KeyIndex, names, objects). Specifically, the key_index is a
            new dataclass which defines the parameters used to create the object, names is the names
            of the iterables used. The dictionary keys are KeyIndex objects which describe the iterable
            arguments passed to the object, while the values are the newly constructed arguments. See
            ``pachyderm.generic_config.create_objects_from_iterables(...)`` for more.
    """
    (config_filename, terminal_args, additional_args) = determine_selected_options_from_kwargs(task_name = task_name)
    selected_analysis_options, _ = validate_arguments(selected_args = terminal_args)
    return construct_from_configuration_file(
        task_name = task_name,
        config_filename = config_filename,
        selected_analysis_options = selected_analysis_options,
        obj = obj,
        additional_possible_iterables = additional_possible_iterables
    )

