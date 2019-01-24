#!/usr/bin/env python

""" Generic task hist plotting code.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

from abc import ABC
import copy
import dataclasses
import enum
import logging
import os
import pprint  # noqa: F401
from typing import Any, Dict, Mapping, Tuple, Type

from pachyderm import generic_class
from pachyderm import histogram

from jet_hadron.base import analysis_config
from jet_hadron.base import analysis_objects
from jet_hadron.base import params
from jet_hadron.base.typing_helpers import Hist
from jet_hadron.plot import generic_hist as plot_generic_hist

logger = logging.getLogger(__name__)
# Quiet down the matplotlib logging
logging.getLogger("matplotlib").setLevel(logging.INFO)

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
            self.components_from_YAML = self.task_config["componentsToPlot"]
        except KeyError as e:
            # Reraise with additional information. This is pretty common to forget.
            raise KeyError("Was \"componentsToPlot\" defined in the task configuration?") from e
        # Contain the actual components, which consist of lists of hist configurations
        self.components: Dict[str, Dict[str, plot_generic_hist.HistPlotter]] = {}
        # Store the input histograms
        self.hists: Dict[str, Any] = {}

    def _retrieve_histograms(self) -> None:
        """ Retrieve hists corresponding to the task name.

        The histograms are stored in the ``hists`` attribute of the object.
        """
        self.hists = histogram.get_histograms_in_list(self.input_filename, self.input_list_name)

        # Don't process this line unless we are debugging because pprint may be slow
        # see: https://stackoverflow.com/a/11093247
        #if logger.isEnabledFor(logging.DEBUG):
        #    logger.debug("Hists:")
        #    logger.debug(pprint.pformat(self.hists))

    def setup(self) -> None:
        """ Setup for processing and plotting. """
        self._retrieve_histograms()

    def hist_specific_processing(self) -> None:
        """ Perform processing on specific histograms in the input hists.

        Each component and histogram in the input hists are searched for particular histograms.
        When they are found, particular functions are applied to those hists, which are then
        stored in the input hists (depending on the function, it is sometimes saved as a
        replacement of the existing hist and sometimes as an additional hist).
        """
        pass

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

    def define_plot_objects_for_component(self, component_name: str,
                                          component_hists_options: Dict[str, Any]) -> Tuple[dict, bool]:
        """ Define Hist Plotter options for a particular component based on the provided YAML config.

        Args:
            component_name: Name of the component.
            component_hists_options: Hist options for a particular component.
        Returns:
            (Component hists configuration options contained in ``HistPlotter`` objects, whether other hists
                in the component that are not configured by ``HistPlotter`` objects should also be plotted).
        """
        logger.debug(f"component_name: {component_name}, component_hist_options: {component_hists_options}")
        # Make the output directory for the component.
        component_output_path = os.path.join(self.output_prefix, component_name)
        if not os.path.exists(component_output_path):
            os.makedirs(component_output_path)

        # Pop the value so all other names can be assumed to be histogram names
        plot_additional = component_hists_options.pop("plotAdditional", False)
        if plot_additional:
            logger.info(f"Plotting additional histograms in component \"{component_name}\"")

        # Create the defined component hists
        # Each histogram in a component has a corresponding set of hist_options.
        component_hists_configuration_options = {}
        for hist_options_name, options in component_hists_options.items():
            # Copy the options dict so we can add to it
            hist_options: Dict[str, Any] = {}
            logger.debug(f"hist_options_name: {hist_options_name}, options: {options}")
            hist_options.update(options)
            logger.debug(f"hist options: {hist_options}")
            hist_names = hist_options.get("histNames", None)
            if not hist_names:
                # Use the hist options name as a proxy for the hist names if not specified
                hist_options["histNames"] = [{hist_options_name: ""}]
                logger.debug(f"Using hist names from config {hist_options_name}. histNames: {hist_options['histNames']}")

            #logger.debug(f"Pre processing: hist options name: {hist_options_name}, hist_options: {hist_options}")
            hist_options = self.hist_options_specific_processing(hist_options_name, hist_options)
            logger.debug(f"Post processing: hist options name: {hist_options_name}, hist_options: {hist_options}")

            component_hists_configuration_options[hist_options_name] = plot_generic_hist.HistPlotter(**hist_options)

        return (component_hists_configuration_options, plot_additional)

    def determine_whether_hist_is_in_hist_object(self, found_match: bool,
                                                 hist: Hist, hist_object_name: str,
                                                 hist_object: plot_generic_hist.HistPlotter,
                                                 component_hists: Dict[str, plot_generic_hist.HistPlotter]) -> bool:
        """ Determine whether the given histogram belongs in the given hist object.

        Args:
            found_match: True if a match has been found.
            hist (ROOT.TH1): Histogram being added to a hist object.
            hist_object_name: The name of the current hist
            hist_object: Histogram plotting configuration.
            component_hists: Where the match hist objects that are relevant to the component are stored.
        Returns:
            The value of found_match. Note that component_hists is modified in place.
        Raises:
            ValueError: If there is more than one set of values associated with a given hist_label.
                This is invalid because each hist object should only have one configuration object
                per key.
        """
        # Iterate over the hist names stored in the config object to determine whether the current
        # hist belongs. Note that ``CommentedMapItemsView`` apparently returns items when directly
        # iterated over, so we get the value a bit further down.
        for hist_label in hist_object.hist_names:
            # There should only be one key in the hist label
            if len(hist_label) > 1:
                raise ValueError(f"Too many entries in the hist_label object! Should only be 1! Object: {hist_label}")
            hist_name, hist_title = next(iter(hist_label.items()))

            #logger.debug(f"Considering hist_name from hist_object: {hist_name}, hist.GetName(): {hist.GetName()} with exact_name_match: {hist_object.exact_name_match}")

            # Check for match between the config object and the current hist
            # The name can be required to be exact, but by default, it doesn't need to be
            if (hist_object.exact_name_match and hist_name == hist.GetName()) or (not hist_object.exact_name_match and hist_name in hist.GetName()):
                logger.debug(f"Found match of hist name: {hist.GetName()} and options config name {hist_object_name}")
                found_match = True
                # Keep the title as the hist name if we haven't specified it so we don't lose information
                hist_title = hist_title if hist_title != "" else hist.GetTitle() if hist.GetTitle() != "" else hist.GetName()

                # Check if object is already in the component hists
                obj = component_hists.get(hist_object_name, None)
                if obj:
                    # If the object already exists, we should add the hist to the list if there
                    # are multiple hists requested for the object. If not, just ignore it
                    if len(hist_object.hist_names) > 1:
                        logger.debug("Adding hist to existing config")
                        # Set the hist title (for legend, plotting, etc)
                        hist.SetTitle(hist_title)

                        obj.hists.append(hist)
                        logger.debug(f"Hists after adding: {obj.hists}")
                    else:
                        logger.critical("Skipping hist because this object is only supposed to have one hist")
                        continue
                else:
                    # Create a copy to store this particular histogram if we haven't already created
                    # an object with this type of histogram because we can have multiple hists covered
                    # by the same configurations options set
                    obj = copy.deepcopy(hist_object)

                    # Name will be from output_name if it is defined. If not, use hist_object_name if
                    # there are multiple hists, or just the hist name if it only relates to one hist.
                    name = obj.output_name if obj.output_name != "" else hist.GetName() if len(obj.hist_names) == 1 else hist_object_name
                    logger.debug(f"Creating config for hist stored under {name}")
                    component_hists[name] = obj

                    # Set the hist title (for legend, plotting, etc)
                    hist.SetTitle(hist_title)
                    # Store the hist in the object
                    obj.hists.append(hist)

        return found_match

    def assign_hists_to_plot_objects(self, component_hists_in_file: dict, hists_configuration_options: dict, plot_additional: bool) -> dict:
        """ Assign input hists retrieved from a file to the defined Hist Plotters.

        Args:
            component_hists_in_file: Hists that are in a particular component. Keys are hist names
            hists_configuration_options: ``HistPlotter`` hist configuration objects for a particular component.
            plot_additional: If true, plot additional histograms in the component that are not specified in the config.
        Returns:
            HistPlotter configuration objects with input hists assigned to each object according to
                the configuration.
        """
        component_hists: Dict[str, plot_generic_hist.HistPlotter] = {}
        # First iterate over the available hists
        for hist in component_hists_in_file.values():
            logger.debug(f"Looking for match to hist {hist.GetName()}")
            found_match = False
            # Then iterate over the Hist Plotter configurations in the particular component.
            # We are looking for the hists which belong in a particular config
            for hist_object_name, hist_object in hists_configuration_options.items():
                if logger.isEnabledFor(logging.DEBUG):
                    debug_hist_names = [next(iter(hist_label)) for hist_label in hist_object.hist_names]
                    # Only print the first five items so we aren't overwhelmed with information
                    # "..." indicates that there are more than 5
                    debug_hist_names_trimmed = ", ".join(debug_hist_names[:5]) if len(debug_hist_names) < 5 else ", ".join(debug_hist_names[:5] + ["..."])
                    logger.debug(f"Considering hist_object \"{hist_object_name}\" for exact_name_match: {hist_object.exact_name_match}, hist_names (len {len(debug_hist_names)}): {debug_hist_names_trimmed}")

                # Determine whether the hist belongs in the current hist object.
                found_match = self.determine_whether_hist_is_in_hist_object(
                    found_match = found_match,
                    hist = hist,
                    hist_object_name = hist_object_name,
                    hist_object = hist_object,
                    component_hists = component_hists
                )

            # Create a default hist object
            if not found_match:
                if plot_additional:
                    logger.debug("No hist config options match found. Using default options...")
                    component_hists[hist.GetName()] = plot_generic_hist.HistPlotter(hist = hist)
                else:
                    logger.debug(f"No match found and not plotting additional, so hist {hist.GetName()} will not be plotted.")

        return component_hists

    def match_hists_to_hist_configurations(self) -> None:
        """ Match retrieved histograms to components and their plotting options.

        This method iterates over the available hists from the file and then over the
        hist options defined in YAML to try to find a match. Once a match is found,
        the options are iterated over to create ``HistPlotter`` objects. Then the hists are assigned
        to those newly created objects.

        The results are stored in the components dict of the class.
        """
        # component_name_in_file is the name of the component_hists_in_file to which we want to compare
        for component_name_in_file, component_hists_in_file in self.hists.items():
            # component_hists_options are the config options for a component
            for component_name, component_hists_options in self.components_from_YAML.items():
                if component_name in component_name_in_file:
                    # We've now matched the component name and configurations, we can move on to dealing with
                    # the individual hists
                    (hists_configuration_options, plot_additional) = self.define_plot_objects_for_component(
                        component_name = component_name,
                        component_hists_options = component_hists_options
                    )
                    logger.debug(f"Component name: {component_name}, hists_configuration_options: {hists_configuration_options}")

                    # Assign hists from the component in the input file to a hist object
                    component_hists = self.assign_hists_to_plot_objects(
                        component_hists_in_file = component_hists_in_file,
                        hists_configuration_options = hists_configuration_options,
                        plot_additional = plot_additional
                    )

                    logger.debug("component_hists: {pprint.pformat(component_hists)}")
                    for component_hist in component_hists.values():
                        # Even though the hist names could be defined in order in the configuration,
                        # the hists will not necessarily show up in alphabetical order when they are assigned to
                        # the plot objects. So we sort them alphabetically here
                        component_hist.hists = sorted(component_hist.hists, key = lambda hist: hist.GetName())
                        logger.debug(f"component_hist: {component_hist}, hists: {component_hist.hists}, (first) hist name: {component_hist.get_first_hist().GetName()}")

                    self.components[component_name] = component_hists

    def plot_histograms(self) -> None:
        """ Driver function to plotting the histograms contained in the object. """
        for component_name, component_hists in self.components.items():
            logger.info(f"Plotting hists for component {component_name}")

            # Apply the options, draw the plot and save it
            for hist_name, hist_obj in component_hists.items():
                logger.debug(f"Processing hist obj {hist_name}")
                hist_obj.plot(self, output_name = os.path.join(component_name, hist_name))

    def run(self, run_plotting: bool = True) -> bool:
        """ Process and plot the histograms for this task.

        Args:
            run_plotting: If true, run plotting after the processing.
        Returns:
            True if the processing was successful.
        """
        # Processing
        self.hist_specific_processing()
        self.match_hists_to_hist_configurations()

        # Plotting
        # Usually we want to plot, but we make it an option because writing out the plots
        # takes much longer than the processing. So when we are developing the processing, it is
        # helpful to not have to wait for the plotting.
        if run_plotting:
            self.plot_histograms()

        return True

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

