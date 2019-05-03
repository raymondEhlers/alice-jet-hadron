#!/usr/bin/env python

""" Base functionality for analysis managers.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import abc
import coloredlogs
import enlighten
import logging
from typing import Any, Type, TypeVar

from pachyderm import generic_class

from jet_hadron.base import analysis_config
from jet_hadron.base import analysis_objects
from jet_hadron.base import params

class Manager(generic_class.EqualityMixin, abc.ABC):
    """ Analysis manager for creating and directing analysis tasks.

    It is expected that the inheriting class will actually create and store the analysis tasks
    (often under ``self.analyses``).

    Args:
        config_filename: Path to the configuration filename.
        selected_analysis_options: Selected analysis options.
        manager_task_name: Name of the analysis manager task name in the config.

    Attributes:
        config_filename: Path to the configuration filename.
        selected_analysis_options: Selected analysis options.
        task_name: Name of the analysis manager task name in the config.
        config: Overall YAML configuration, formatted using the parameters for the analysis manager.
        task_config: Task YAML configuration, formatted using the parameters for the analysis manager.
            This is equivalent to ``self.config[self.task_name]``.
        output_info: Output information for storing data, histograms, etc.
        processing_options: Processing options specified in the task config. Created for convenience.
        _progress_manager: Keep track of the analysis progress using status bars.
    """
    def __init__(self, config_filename: str,
                 selected_analysis_options: params.SelectedAnalysisOptions,
                 manager_task_name: str,
                 **kwargs: Any):
        self.config_filename = config_filename
        self.selected_analysis_options = selected_analysis_options
        self.task_name = manager_task_name

        # Retrieve YAML config for manager configuration
        # NOTE: We don't store the overridden selected_analysis_options because in principle they depend
        #       on the selected task. In practice, such options are unlikely to vary between the manager
        #       and the analysis tasks. However, the validation cannot handle the overridden options
        #       (because the leading hadron bias enum is converting into the object). So we just use
        #       the overridden option in formatting the output prefix (where it is required to determine
        #       the right path), and then passed the non-overridden values to the analysis objects.
        self.config, overridden_selected_analysis_options = analysis_config.read_config_using_selected_options(
            task_name = self.task_name,
            config_filename = self.config_filename,
            selected_analysis_options = self.selected_analysis_options
        )
        # Determine the formatting options needed for the output prefix
        formatting_options = analysis_config.determine_formatting_options(
            task_name = self.task_name, config = self.config,
            selected_analysis_options = overridden_selected_analysis_options
        )
        # Additional helper variables
        self.task_config = self.config[self.task_name]
        self.output_info = analysis_objects.PlottingOutputWrapper(
            # Format to ensure that the selected analysis options are filled in.
            output_prefix = self.config["outputPrefix"].format(**formatting_options),
            printing_extensions = self.config["printingExtensions"],
        )
        # For convenience since it is frequently accessed.
        self.processing_options = self.task_config["processing_options"]

        # Monitor the progress of the analysis.
        self._progress_manager = enlighten.get_manager()

    @abc.abstractmethod
    def run(self) -> bool:
        """ Run the analyses contained in the manager.

        Returns:
            True if the analyses were run successfully.
        """
        ...

    def _run(self) -> bool:
        """ Wrapper around the actual call to run to restore a normal output.

        Here we disable enlighten so that it won't mess with any later steps (such as exploration
        with IPython). Otherwise, IPython will act very strangely and is basically impossible to use.

        Returns:
            True if the analyses were run successfully
        """
        result = self.run()

        # Disable enlighten so that it won't mess with any later steps (such as exploration with IPython).
        # Otherwise, IPython will act very strangely and is basically impossible to use.
        self._progress_manager.stop()

        return result

_T = TypeVar("_T", bound = Manager)

def run_helper(manager_class: Type[_T], **kwargs: str) -> _T:
    """ Helper function to execute most analysis managers.

    It sets up the passed analysis manager object and then calls ``run()``. It also enables logging
    with colors in the output.

    Note:
        This won't pass the ``task_name`` to the manager class. It's expected to have that name hard coded
        in the inherited manager class.

    Args:
        manager_class: Class which will manage execution of the task.
        task_name: Name of the tasks that will be analyzed for the argument parsing help (it doesn't
            matter if it matches the YAML config).
        description: Description of the task for the argument parsing help.
    Returns:
        The created and executed task manager.
    """
    # Basic setup
    coloredlogs.install(
        level = logging.DEBUG,
        fmt = "%(asctime)s %(name)s:%(lineno)d %(levelname)s %(message)s"
    )
    # Quiet down the matplotlib logging
    logging.getLogger("matplotlib").setLevel(logging.INFO)

    # Setup the analysis
    # mypy doesn't know that we won't ever pass the `args` argument.
    (config_filename, terminal_args, additional_args) = analysis_config.determine_selected_options_from_kwargs(  # type: ignore
        **kwargs,
    )
    # We don't necessarily need to pass the kwargs, as the task_name should be specified by the
    # inheriting manager rather than passed. But it also doesn't hurt to pass the values, and it
    # makes mypy happy.
    analysis_manager = manager_class(
        config_filename = config_filename,
        selected_analysis_options = terminal_args,
        **kwargs,
    )
    # Finally run the analysis.
    analysis_manager._run()

    # Provide the final result back to the caller.
    return analysis_manager

