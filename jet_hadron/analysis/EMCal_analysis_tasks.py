#!/usr/bin/env python

""" EMCal corrections and embedding plotting code

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import enum
import logging
import os

from pachyderm import yaml

from jet_hadron.base import analysis_config
from jet_hadron.base import labels
from jet_hadron.base.typing_helpers import Hist
from jet_hadron.plot import generic_hist as plot_generic_hist
from jet_hadron.analysis import generic_tasks

# Setup logger
logger = logging.getLogger(__name__)

class EMCalCorrectionsLabel(enum.Enum):
    """ Label of possible EMCal correction tasks.

    The standard case is not labeled, but this label is important for when multiple
    correction tasks are ran during embedding.
    """
    standard = ""
    embed = "Embed"
    combined = "Data + Embed"
    data = "Data"

    def __str__(self) -> str:
        """ Return a string of the value of the label for display. """
        return str(self.value)

    # Handle YAML serialization
    @classmethod
    def to_yaml(cls, representer: yaml.Representer, data: yaml.T_EnumToYAML) -> yaml.ruamel.yaml.nodes.ScalarNode:
        """ Encode the YAML representation.

        We want to write the name of the enumeration instead of the ``str()`` value.
        """
        return representer.represent_scalar(
            f"!{cls.__name__}",
            f"{data.name}"
        )
    # We can just use the standard enum_from_yaml(...) here because it expects to construct
    # based on the name of the enum value (which is what we provided above).
    from_yaml = classmethod(yaml.enum_from_yaml)

class PlotEMCalCorrections(generic_tasks.PlotTaskHists):
    """ Task to plot EMCal corrections hists.

    Args:
        task_label (EMCalCorrectionsLabel): EMCal corrections label associated with this task.
        args (list): Additional arguments to pass along to the base config class.
        kwargs (dict): Additional arguments to pass along to the base config class.
    """
    def __init__(self, *args, **kwargs):
        # Access task_label, but don't pop it, because we need to pass it to the base class for assignment.
        task_label = kwargs["task_label"]
        # Add the task label to the output prefix
        # Note that we are using camelCase here instead of snake_case because these values haven't yet
        # been assigned in the base class.
        kwargs["config"]["outputPrefix"] = os.path.join(kwargs["config"]["outputPrefix"], task_label.name)
        # Also need to add it as "_label" to the input list name so it ends up as "name_label_histos"
        # If it is the standard correction task, then we just put in an empty string
        corrections_label = f"_{task_label.name}" if task_label != EMCalCorrectionsLabel.standard else ""
        kwargs["config"]["inputListName"] = kwargs["config"]["inputListName"].format(corrections_label = corrections_label)

        # Afterwards, we can initialize the base class
        super().__init__(*args, **kwargs)

        self.track_pt_bins = self.task_config["track_pt_bins"]

    def _hist_specific_preprocessing(self) -> None:
        """ Perform processing on specific histograms in the input hists.

        Each component and histogram in the input hists are searched for particular histograms.
        When they are found, particular functions are applied to those hists, which are then
        stored in the input hists (depending on the function, it is sometimes saved as a
        replacement of the existing hist and sometimes as an additional hist).
        """
        # Loop over available components in the hists
        for component_name in self.input_hists:
            # Clusterizer
            if "Clusterizer" in component_name:
                # Only perform the scaling if the hist actually exists.
                if "hCPUTime" in self.input_hists[component_name]:
                    scale_CPU_time(self.input_hists[component_name]["hCPUTime"])

            if "ClusterExotics" in component_name:
                hist_name = "hExoticsEtaPhiRemoved"
                before_hist_name = "hEtaPhiDistBefore"
                after_hist_name = "hEtaPhiDistAfter"
                if before_hist_name in self.input_hists[component_name] and after_hist_name in self.input_hists[component_name]:
                    self.input_hists[component_name][hist_name] = eta_phi_removed(
                        hist_name = hist_name,
                        before_hist = self.input_hists[component_name][before_hist_name],
                        after_hist = self.input_hists[component_name][after_hist_name]
                    )

    def eta_phi_match_hist_names(self, plot_config: plot_generic_hist.HistPlotter) -> None:
        """ Generate eta and phi hist names based on the available options.

        This approach allows generating of hist config options using for loops while still being defined in YAML.

        Note:
            This function is called via hist_options_specific_processing(...), so it is not
            referenced directly in the source.

        Args:
            plot_config: Plot configuration to modify.
        Returns:
            None. The plot configuration is updated in place.
        """
        # Get the hist name template
        # We don't care about the hist title
        logger.debug(f"plot_config: {plot_config}, hist_names: {plot_config.hist_names}")
        hist_name = next(iter(next(iter(plot_config.hist_names))))
        # Angle name
        angles = plot_config.processing["angles"]
        # {Number: label}
        cent_bins = plot_config.processing["cent_bins"]
        # {Number: label}
        eta_directions = plot_config.processing["eta_directions"]
        # List of pt bins
        selected_pt_bins = plot_config.processing["pt_bins"]

        hist_names = []
        for angle in angles:
            for cent_dict in cent_bins:
                cent_bin, cent_label = next(iter(cent_dict.items()))
                for track_pt in self.track_pt_bins:
                    # Only process the track pt bins that are selected.
                    if track_pt.bin not in selected_pt_bins:
                        continue
                    for eta_dict in eta_directions:
                        eta_direction, eta_direction_label = next(iter(eta_dict.items()))
                        # Determine hist name
                        # NOTE: We must convert the pt bin to 0 indexed to retrieve the correct hist.
                        name = hist_name.format(
                            angle = angle, cent = cent_bin,
                            pt_bin = track_pt.bin - 1, eta_direction = eta_direction
                        )
                        # Determine label
                        # NOTE: Can't use track_pt_range_string because it includes "assoc" in
                        #       the pt label. Instead, we create the string directly.
                        pt_bin_label = labels.pt_range_string(
                            pt_bin = track_pt,
                            lower_label = r"\mathrm{T}",
                            upper_label = r"",
                            only_show_lower_value_for_last_bin = True,
                        )

                        angle_label = determine_angle_label(angle)
                        # Ex: "$\\Delta\\varphi$, Pos. tracks, $\\eta < 0$, $4 < p_{\\mathrm{T}}< 5$"
                        label = f"{angle_label}, {cent_label}, {eta_direction_label}, {pt_bin_label}"
                        # Save in the expected format
                        hist_names.append({name: label})
                        #logger.debug("name: \"{}\", label: \"{}\"".format(name, label))

        # Assign the newly created names
        logger.debug(f"Assigning hist_names {hist_names}")
        plot_config.hist_names = hist_names

def determine_angle_label(angle: str) -> str:
    """ Determine the full angle label and return the corresponding latex.

    Args:
        angle: Angle to be used in the label.
    Returns:
        Full angle label.
    """
    return_value = r"$\Delta"
    # Need to lower because the label in the hist name is upper
    angle = angle.lower()
    if angle == "phi":
        # "phi" -> "varphi"
        angle = "var" + angle
    return_value += r"\%s$" % (angle)
    return return_value

def scale_CPU_time(hist: Hist) -> None:
    """ Rebin the CPU time for improved presentation.

    Time is only reported in increments of 10 ms. So we rebin by those 10 bins (since each bin is 1 ms)
    and then scale them down to be on the same scale as the real time hist. We can perform this scaling
    in place.

    Note:
        This scaling appears to be the same as one would usually do for a rebin, but it is slightly more
        subtle, because it is as if the data was already binned. That being said, the end result is
        effectively the same.

    Args:
        hist: CPU time histogram to be scaled.
    Returns:
        None.
    """
    logger.debug("Performing CPU time hist scaling.")
    timeIncrement = 10
    hist.Rebin(timeIncrement)
    hist.Scale(1.0 / timeIncrement)

def eta_phi_removed(hist_name: str, before_hist: Hist, after_hist: Hist) -> Hist:
    """ Show the eta phi locations of clusters removed by the exotics cut.

    This is created by subtracting the after removal hist from the before removal hist.
    The before and after histograms are expected to be TH2 histograms, but in principle
    they could be anything.

    Args:
        hist_name: Name of the new hist showing the removed clusters
        before_hist: Eta-Phi histogram before exotic clusters removal
        after_hist: Eta-Phi histogram after exotic cluster removal
    Returns:
        A new hist showing the difference between the two input hists.
    """
    # Create a new hist and remove the after hist
    hist = before_hist.Clone(hist_name)
    hist.Add(after_hist, -1)

    return hist

class EMCalCorrectionsPlotManager(generic_tasks.TaskManager):
    """ Manager for plotting EMCal corrections histograms. """
    def construct_tasks_from_configuration_file(self) -> analysis_config.ConstructedObjects:
        return analysis_config.construct_from_configuration_file(
            task_name = "EMCalCorrections",
            config_filename = self.config_filename,
            selected_analysis_options = self.selected_analysis_options,
            additional_possible_iterables = {"task_label": EMCalCorrectionsLabel},
            additional_classes_to_register = [plot_generic_hist.HistPlotter],
            obj = PlotEMCalCorrections,
        )

def run_plot_EMCal_corrections_hists_from_terminal() -> EMCalCorrectionsPlotManager:
    """ Driver function for plotting the EMCal corrections hists. """
    return generic_tasks.run_helper(
        manager_class = EMCalCorrectionsPlotManager,
        description = "EMCal corrections plotting.",
    )

class PlotEMCalEmbedding(generic_tasks.PlotTaskHists):
    """ Task to plot EMCal embedding hists.

    Note:
        This current doesn't have any embedding specific functionality. It is created for clarity and to
        encourage extension in the future.
    """
    ...

class EMCalEmbeddingPlotManager(generic_tasks.TaskManager):
    """ Manager for plotting EMCal embedding histograms. """
    def construct_tasks_from_configuration_file(self) -> analysis_config.ConstructedObjects:
        """ Construct EMCal embedding plotting tasks. """
        return analysis_config.construct_from_configuration_file(
            task_name = "EMCalEmbedding",
            config_filename = self.config_filename,
            selected_analysis_options = self.selected_analysis_options,
            additional_possible_iterables = {"pt_hard_bin": None},
            additional_classes_to_register = [plot_generic_hist.HistPlotter],
            obj = PlotEMCalEmbedding,
        )

def run_plot_EMCal_embedding_hists_from_terminal() -> EMCalEmbeddingPlotManager:
    """ Driver function for plotting the EMCal embedding hists. """
    return generic_tasks.run_helper(
        manager_class = EMCalEmbeddingPlotManager,
        description = "EMCal embedding plotting.",
    )

