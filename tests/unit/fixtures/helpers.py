#!/usr/bin/env python

""" Helper functions wrapped in fixtures to aid testing.

The idea for this approach is inspired by https://stackoverflow.com/a/51389067,
which is really quite clever!

.. codeauthor:: Raymond Ehlers <raymond.ehlers@yale.edu>, Yale University
"""

from io import StringIO
import pytest
import ruamel.yaml

@pytest.fixture
def check_JetHBase_object():
    """ Provides checking for the JetHBase object.

    This function is provided via fixture to allow for use in multiple tests modules.
    """
    def func(obj, config: "ruamel.yaml.comments.CommentedMap", selected_analysis_options, reaction_plane_orientation, **kwargs):
        """ Helper function to check JetHBase properties.

        The values are asserted in this function.

        Note:
            The default values correspond to those in the object_config config, so they don't
            need to be specified in each function.

        Args:
            obj (analysis_config.JetHBase): JetHBase object to compare values against.
            config (CommentedMap): dict-like configuration file.
            selected_analysis_options (params.SelectedAnalysisOptions): Selected analysis options.
            reaction_plane_orientation (params.ReactionPlaneOrientation): Selected event plane angle.
            kwargs (dict): All other values to compare against for which the default value defined
                in this function is not sufficient.
        Returns:
            bool: True if it successfully make it to the end of the assertions.
        """
        # Determine default values
        task_name = "taskName"
        default_values = {
            "task_name": task_name,
            "config_filename": "configFilename.yaml",
            "config": config,
            "task_config": config[task_name],
            "reaction_plane_orientation": reaction_plane_orientation
        }
        # Add these afterwards so we don't have to do each value by hand.
        default_values.update(selected_analysis_options.asdict())
        # We need to modify the name from "leading_hadron_bias" -> "_leading_hadron_bias"
        default_values["_leading_hadron_bias"] = selected_analysis_options.leading_hadron_bias
        default_values["_leading_hadron_bias_type"] = selected_analysis_options.leading_hadron_bias.type
        # NOTE: All other values will be taken from the config when constructing the object.
        for k, v in default_values.items():
            if k not in kwargs:
                kwargs[k] = v

        # Creating thie object is something of a tautology, because in both cases we use the
        # constructed object, so we also have other tests, which are performed below.
        # However, we keep the test to provide a check for the equality operators.
        # We perform the actual test later because it is not very verbose.
        # We use type(obj) to determine the type so that we don't have to explicitly import the module.
        comparison = type(obj)(**kwargs)

        # We need to retrieve these values so we can test them directly.
        # `default_values` will now be used as the set of reference values.
        value_names = ["inputFilename", "inputListName", "outputPrefix", "outputFilename", "printingExtensions", "aliceLabel"]
        for k in value_names:
            # The AliceLabel gets converted to the enum in the object, so we need to do the conversion here.
            if k == "aliceLabel":
                from jet_hadron.base import params
                val = params.AliceLabel[config[k]]
            else:
                val = config[k]
            default_values[k] = val
        default_values.update(kwargs)

        # Directly compare against the available values
        # NOTE: This isn't wholly independent because the object comparison relies on comparing values
        #       in obj.__dict__, but it is still an improvement on the above.
        # Use we this map to allow us to translate between the attribute names and the config names.
        default_values_key_map = {
            "input_filename": "inputFilename",
            "input_list_name": "inputListName",
            "output_prefix": "outputPrefix",
            "output_filename": "outputFilename",
            "printing_extensions": "printingExtensions",
            "alice_label": "aliceLabel",
        }
        for prop, val in obj.__dict__.items():
            # Skip "_" fields
            if prop.startswith("_"):
                continue
            assert val == default_values[default_values_key_map.get(prop, prop)]

        # Perform the comparison test.
        assert obj == comparison
        # Check property
        assert obj.leading_hadron_bias == comparison.leading_hadron_bias

        return True

    return func

@pytest.fixture
def log_yaml_dump():
    """ Helper function to log the YAML config.

    This function is provided via fixture to allow for use in multiple tests modules.
    """
    def func(yaml, config):
        """ Helper function to log the YAML config. """
        s = StringIO()
        yaml.dump(config, s)
        s.seek(0)

        return s.read()

    return func

@pytest.fixture
def override_options_helper(log_yaml_dump):
    """ Helper function to override the configuration.

    This function is provided via fixture to allow for use in multiple tests modules.
    """
    def func(config, selected_options = None, config_containing_override = None):
        """ Helper function to override the configuration.

        It can print the configuration before and after overridding the options if enabled.

        Note:
            If selected_options is not specified, it defaults to (2.76, "PbPb", "central", "track")

        Args:
            config (CommentedMap): dict-like object containing the configuration to be overridden.
            selected_options (params.SelectedAnalysisOptions): The options selected for this analysis, in
                the order defined used with analysis_config.override_options() and in the configuration file.
            config_containing_override (CommentedMap): dict-like object containing the override options.
        Returns:
            tuple: (dict-like CommentedMap object containing the overridden configuration, selected analysis
                        options used with the config)
        """
        # Import modules here so we can delay it until they are actually needed.
        from jet_hadron.base import analysis_config
        from jet_hadron.base import params
        import logging
        logger = logging.getLogger(__name__)

        if selected_options is None:
            selected_options = params.SelectedAnalysisOptions(
                collision_energy = params.CollisionEnergy.two_seven_six,
                collision_system = params.CollisionSystem.PbPb,
                event_activity = params.EventActivity.central,
                leading_hadron_bias = params.LeadingHadronBiasType.track
            )

        yaml = ruamel.yaml.YAML()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Before override:")
            logger.debug(log_yaml_dump(yaml, config))

        config = analysis_config.override_options(
            config = config,
            selected_options = selected_options,
            config_containing_override = config_containing_override
        )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("After override:")
            logger.debug(log_yaml_dump(yaml, config))

        return (config, selected_options)

    return func

@pytest.fixture
def object_yaml_config():
    """ Object YAML configuration to test object args, validation, and construction.

    Args:
        None
    Returns:
        CommentedMap: dict-like object from ruamel.yaml containing the configuration.
    """
    test_yaml = """
iterables:
    reaction_plane_orientation: False
    QVector: False
leadingHadronBiasValues:
    track:
        value: 5
    2.76:
        central:
            cluster:
                value: 10
        semi_central:
            cluster:
                value: 6
inputFilename: "inputFilenameValue"
inputListName: "inputListNameValue"
outputPrefix: "outputPrefixValue"
outputFilename: "outputFilenameValue"
printingExtensions: ["png", "pdf"]
aliceLabel: "thesis"
# This is the configuration for a test task of the name "taskName".
taskName:
    test: "val"
    override:
        # Just need a trivial override value, since "override" is a required field.
        aliceLabel: "final"
        iterables:
            reaction_plane_orientation: True
            qVector:
                - "all"
"""
    yaml = ruamel.yaml.YAML()
    data = yaml.load(test_yaml)
    return (data, "taskName")

