#!/usr/bin/env python

""" Generic task hist plotting code.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

from abc import ABC
import dataclasses
import enum
import logging
import os
import pprint  # noqa: F401
from typing import Any, Dict, Iterator, List, Mapping, Tuple, Type, Union

from pachyderm import generic_class
from pachyderm import histogram

from jet_hadron.base import analysis_config
from jet_hadron.base import analysis_objects
from jet_hadron.base import params
from jet_hadron.base.typing_helpers import Hist
from jet_hadron.plot import generic_hist as plot_generic_hist

logger = logging.getLogger(__name__)

# Typing helpers
PlotConfigurations = Dict[str, Union[plot_generic_hist.HistPlotter, Any]]
InputHists = Dict[str, Union[Hist, Any]]

def iterate_over_plot_configurations(plot_configurations: PlotConfigurations,
                                     path_to_plot_configuration: List = None,
                                     **kwargs) -> Iterator[Tuple[str, plot_generic_hist.HistPlotter, List[str]]]:
    """ Iterate over the provided plot configurations.

    This generator searches through the provided plot configurations recursively to find every
    ``HistPlotter`` object that has been defined.

    Args:
        plot_configurations: Plot configurations over which one wants to operate.
        path_to_plot_configuration: List of names necessary to get to the returned object.
        kwargs: Additional arguments to pass to the recursive function call.
    Returns:
        Iterator over all of the ``HistPlotter`` objects found recursively in the plot configurations.
    """
    if path_to_plot_configuration is None:
        path_to_plot_configuration = []

    # NOTE: Commented out the debug messages because otherwise they are way too verbose.
    #       But they are kept commented out because they are quite useful for debugging.
    #logger.debug(f"Starting with config config: {plot_configurations} at path {path_to_plot_configuration}")
    for name, config in plot_configurations.items():
        #logger.debug(f"Looking at config name: {name}, config: {config} at path {path_to_plot_configuration}")
        if isinstance(config, plot_generic_hist.HistPlotter):
            #logger.debug(f"Yielding {name}, {config} at path {path_to_plot_configuration}")
            yield name, config, path_to_plot_configuration
        else:
            #logger.debug(f"Going a level deeper for name: {name}, path: {path_to_plot_configuration}")
            # Need to copy path when we iterate...
            #path_to_plot_configuration.append(name)
            recurse_path = path_to_plot_configuration + [name]
            # We need to yield from to iterate over the result!
            yield from iterate_over_plot_configurations(
                plot_configurations = config,
                path_to_plot_configuration = recurse_path,
                **kwargs
            )

def _setup_plot_configurations(plot_configurations: PlotConfigurations) -> None:
    """ Recursively setup the HistPlotter objects. """
    config_iter = iterate_over_plot_configurations(plot_configurations = plot_configurations)

    for name, plot_config, _ in config_iter:
        # Set the hist_names based on the key name in the config.
        logger.debug(f"key name {name}, plot_config: {plot_config}, type: {type(plot_config)}")
        # Cannot just use if hist_names because ``[{}]`` evalutes to true...
        if plot_config.hist_names == [{}]:
            logger.debug(f"Assigning key name {name} to hist_names")
            plot_config.hist_names = [{name: ""}]

def _determine_hists_for_plot_configurations(plot_configurations: PlotConfigurations, input_hists: InputHists) -> None:
    """ Determine which histogram lists correspond with each plot configuration.

    The actual searching for and assignment of that histogram list to the plot configuration
    is performed in ``assign_hists_to_plot_configurations(...)``. This function is called
    recursively to determine the histograms for all plot configuration objects.

    Args:
        plot_configurations: Plot configurations.
        input_hists: Histograms to be assigned to the plot configurations.
    Returns:
        None. The plot configurations are modified in place.
    """
    logger.debug(f"plot_configurations: {plot_configurations}, input_hists: {input_hists}")
    for name, config in plot_configurations.items():
        # If we have a HistPlotter, then look for and assign hists at the given recursion level.
        if isinstance(config, plot_generic_hist.HistPlotter):
            _assign_hists_to_plot_configurations(config, input_hists)
        else:
            # If we don't have a hist plotter, then we hists at this level to see if we can find a match
            for hists_name, hists in input_hists.items():
                if name in hists_name:
                    # If we do find a match in the input hists dict, go the next level of recursion.
                    # NOTE: We ignore the typing here because mypy doesn't like the recursive redefinition of
                    #       plot_configurations.
                    _determine_hists_for_plot_configurations(  # type: ignore
                        plot_configurations = config,
                        input_hists = input_hists[hists_name]
                    )

def _assign_hists_to_plot_configurations(plotter: plot_generic_hist.HistPlotter, input_hists: InputHists) -> None:
    """ Search for and assign hists to a HistPlotter.

    Args:
        plotter: ``HistPlotter`` which defines the desired hists and will have them assigned to them.
    Returns:
        None. The ``HistPlotter`` is modified in place.
    """
    for hist_label in plotter.hist_names:
        # There should only be one key in the hist label
        if len(hist_label) > 1:
            raise ValueError(f"Too many entries in the hist_label object! Should only be 1! Object: {hist_label}")
        hist_name, hist_title = next(iter(hist_label.items()))

        for hist_key, hist in input_hists.items():
            #logger.debug(f"Considering hist_name from plotter: {hist_name}, hist.GetName(): {hist.GetName()} with exact_name_match: {plotter.exact_name_match}")

            # Check for match between the config object and the current hist
            # The name can be required to be exact, but by default, it doesn't need to be
            # NOTE: We use hist_key instead of hist.GetName() because input_hists may contain other dicts. Those other
            #       dicts won't define GetName() (obviously), but the key will always be valid.
            if (plotter.exact_name_match and hist_name == hist_key) \
                    or (not plotter.exact_name_match and hist_name in hist_key):
                logger.debug(f"Found match of hist name: {hist.GetName()} and HistPlotter hist name {hist_name}")
                # Keep the title as the hist name if we haven't specified it so we don't lose information
                hist_title = hist_title if hist_title != "" else hist.GetTitle() if hist.GetTitle() != "" else hist.GetName()

                logger.debug("Adding hist to existing config")
                # Set the hist title (for legend, plotting, etc)
                hist.SetTitle(hist_title)

                plotter.hists.append(hist)
                logger.debug(f"Hists after adding: {plotter.hists}")

    # Validation the number of histograms that we've found to ensure that we've found as many as expected.
    # NOTE: We may have configurations defined for which there are no histogramms. In this case, no hists
    #       is okay.
    if len(plotter.hists) > 0 and len(plotter.hist_names) != len(plotter.hists):
        raise ValueError(f"Found {len(plotter.hists)} hists, but expected {len(plotter.hist_names)}!"
                         f" Hist names: {plotter.hist_names}, hists: {plotter.hists}."
                         f" Input hists: {input_hists}")

def _plot_histograms(plot_configurations, task_hists_obj) -> None:
    """ Driver function to plotting the histograms contained in the object. """

class PlotTaskHists(analysis_objects.JetHBase):
    """ Generic class to plot hists in analysis task.

    Hists are selected and configured by a configuration file.

    Args:
        task_label: Enum which labels the task and can be converted into a string.
        args (list): Additional arguments to pass along to the base config class.
        kwargs (dict): Additional arguments to pass along to the base config class.
    """
    def __init__(self, task_label: enum.Enum, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Basic labeling of the task.
        self.task_label = task_label
        # These are the objects for each component stored in the YAML config
        try:
            self.plot_configurations: PlotConfigurations = self.task_config["componentsToPlot"]
        except KeyError as e:
            # Reraise with additional information. This is pretty common to forget.
            raise KeyError("Was \"componentsToPlot\" defined in the task configuration?") from e
        # Store the input histograms
        self.input_hists: Dict[str, Any] = {}

    def _retrieve_histograms(self) -> None:
        """ Retrieve hists corresponding to the task name.

        The histograms are stored in the ``hists`` attribute of the object.
        """
        self.input_hists = histogram.get_histograms_in_list(self.input_filename, self.input_list_name)

        # Don't process this line unless we are debugging because pprint may be slow
        # see: https://stackoverflow.com/a/11093247
        #if logger.isEnabledFor(logging.DEBUG):
        #    logger.debug("Hists:")
        #    logger.debug(pprint.pformat(self.hists))

    def _setup_plot_configurations(self):
        """ Fully setup the plot configuration objects. """
        _setup_plot_configurations(plot_configurations = self.plot_configurations)

    def setup(self) -> None:
        """ Setup for processing and plotting. """
        self._retrieve_histograms()

        # Complete setup by setting up the contained plot configuration objects.
        self._setup_plot_configurations()

    def _plot_configuration_processing(self) -> None:
        """ Run any additional plotting configuration processing. """
        config_iter = iterate_over_plot_configurations(plot_configurations = self.plot_configurations)

        for name, plot_config, _ in config_iter:
            if plot_config.processing:
                func_name = plot_config.processing["func_name"]
                # Retrieve the method from the class
                func = getattr(self, func_name)
                # Call the method with the plot configuration.
                # We mainly pass in the object to provide access to any necessary parameters.
                # Note that we don't pass ``self`` because we already bound to it using ``getattr(...)``.
                func(plot_config = plot_config)

    def _hist_specific_preprocessing(self) -> None:
        """ Perform processing on specific histograms in the input hists.

        Each component and histogram in the input hists are searched for particular histograms.
        When they are found, particular functions are applied to those hists, which are then
        stored in the input hists (depending on the function, it is sometimes saved as a
        replacement of the existing hist and sometimes as an additional hist).
        """
        pass

    def _determine_hists_for_plot_configurations(self):
        """ Determine which hists belong to which plot configurations. """
        _determine_hists_for_plot_configurations(
            plot_configurations = self.plot_configurations,
            input_hists = self.input_hists
        )

    def _plot_histograms(self) -> None:
        """ Plot histograms. """
        config_iter = iterate_over_plot_configurations(plot_configurations = self.plot_configurations)

        for name, plot_config, path in config_iter:
            # Only attempt to plot if we actually have underlying hists.
            if plot_config.hists:
                logger.info(f"Plotting plot configuration {name} located at {os.path.join(*path)}")
                plot_config.plot(self, output_name = os.path.join(*(path + [name])))

    def run(self, run_plotting: bool = True) -> bool:
        """ Process and plot the histograms for this task.

        Note:
            Nearly all of these functions are defined at the module level, but the calls to
            the functions are made through simple wrapping functions at the class level to
            allow for the methods to be overridden.

        Args:
            run_plotting: If true, run plotting after the processing.
        Returns:
            True if the processing was successful.
        """
        # Perform preprocessing for both the plot configurations and the input histograms.
        self._plot_configuration_processing()
        self._hist_specific_preprocessing()

        # Match and assign the histograms to the plot configurations
        self._determine_hists_for_plot_configurations()

        # Plotting
        # Usually we want to plot, but we make it an option because writing out the plots
        # takes much longer than the processing. So when we are developing the processing, it is
        # helpful to not have to wait for the plotting.
        if run_plotting:
            self._plot_histograms()

        return True

    def hist_options_specific_processing(self, hist_options_name: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """ Run a particular processing functions for some set of hist options.

        It looks for a function name specified in the configuration, so a bit of care is
        required to this safely.

        Args:
            hist_options_name: Name of the hist options.
            options: Associated set of hist options.
        Returns:
            Updated set of hist options.
        """
        return options

class TaskManager(ABC, generic_class.EqualityMixin):
    """ Manages execution of the plotting of generic tasks.

    Args:
        config_filename: Filename of the configuration
        selected_analysis_options: Options selected for this analysis.
    Attributes:
        config_filename: Filename of the configuration
        selected_analysis_options: Options selected for this analysis.
        tasks: Dictionary of ``PlotTaskHists`` objects indexed by the selected analysis
            options used for each partiuclar object.
    """
    def __init__(self, config_filename: str, selected_analysis_options: params.SelectedAnalysisOptions, **kwargs):
        self.config_filename = config_filename
        self.selected_analysis_options = selected_analysis_options

        # Create the tasks.
        self.tasks: Mapping[Any, PlotTaskHists]
        (self.key_index, self.selected_iterables, self.tasks) = self.construct_tasks_from_configuration_file()

    def construct_tasks_from_configuration_file(self) -> analysis_config.ConstructedObjects:
        """ Construct the tasks for the manager. """
        return analysis_config.construct_from_configuration_file(
            task_name = "PlotGenericTask",
            config_filename = self.config_filename,
            selected_analysis_options = self.selected_analysis_options,
            additional_possible_iterables = {"pt_hard_bin": None, "jet_pt_bin": None, "track_pt_bin": None},
            additional_classes_to_register = [plot_generic_hist.HistPlotter],
            obj = PlotTaskHists,
        )

    def run(self, run_plotting: bool = True) -> bool:
        """ Main driver function to create, process, and plot task hists.

        Args:
            run_plotting: If true, run plotting after the processing.
        Returns:
            True if the processing was successful.
        """
        logger.info("About to process")
        for keys, task in analysis_config.iterate_with_selected_objects(self.tasks):
            # Print the task selected analysis options
            opts = [f"{name}: \"{str(value)}\""for name, value in dataclasses.asdict(keys).items()]
            options = "\n\t".join(opts)
            logger.info(f"Processing plotting task {task.task_name} with options:\n\t{options}")

            # Setup task, run the processing, and plot the histograms.
            task.setup()
            task.run(run_plotting = run_plotting)

        return True

def run_helper(manager_class: Type[TaskManager], description: str) -> TaskManager:
    """ Helper function to execute most generic task plotting managers.

    It sets up the passed manager object and then calls ``run()``.

    Args:
        manager_class: Class which will manage exeuction of the task.
        description: Description of the task for the argument parsing help.
    Returns:
        The created and executed task manager.
    """
    # Basic setup
    logging.basicConfig(level = logging.DEBUG)
    # Quiet down the matplotlib logging
    logging.getLogger("matplotlib").setLevel(logging.INFO)

    # Setup the analysis
    (config_filename, terminal_args, additional_args) = analysis_config.determine_selected_options_from_kwargs(
        description = description,
    )
    analysis_manager = manager_class(
        config_filename = config_filename,
        selected_analysis_options = terminal_args
    )
    # Finally run the analysis.
    analysis_manager.run()

    # Provide the final result back to the caller.
    return analysis_manager

