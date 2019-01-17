#!/usr/bin/env python

""" EMCal corrections and embedding plotting code

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import enum
import logging
import os
import sys
from typing import Any, Dict
import warnings

from pachyderm import yaml

from jet_hadron.base import analysis_config
from jet_hadron.base import analysis_objects
from jet_hadron.base import params
from jet_hadron.analysis import generic_tasks

import rootpy.ROOT as ROOT

# Tell ROOT to ignore command line options so args are passed to python
# NOTE: Must be immediately after import ROOT and sometimes must be the first ROOT related import!
ROOT.PyConfig.IgnoreCommandLineOptions = True

# Setup logger
logger = logging.getLogger(__name__)

# Handle rootpy warning
warnings.filterwarnings(action='ignore', category=RuntimeWarning, message=r'creating converter for unknown type "_Atomic\(bool\)"')
this_module = sys.modules[__name__]

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
        """ Encore YAML representation.

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

    def hist_specific_processing(self) -> None:
        """ Perform processing on specific histograms in the input hists.

        Each component and histogram in the input hists are searched for particular histograms.
        When they are found, particular functions are applied to those hists, which are then
        stored in the input hists (depending on the function, it is sometimes saved as a
        replacement of the existing hist and sometimes as an additional hist).
        """
        # Loop over available components in the hists
        for component_name in self.hists:
            # Clusterizer
            if "Clusterizer" in component_name:
                # Only perform the scaling if the hist actually exists.
                if "hCPUTime" in self.hists[component_name]:
                    scale_CPU_time(self.hists[component_name]["hCPUTime"])

            if "ClusterExotics" in component_name:
                hist_name = "hExoticsEtaPhiRemoved"
                before_hist_name = "hEtaPhiDistBefore"
                after_hist_name = "hEtaPhiDistAfter"
                if before_hist_name in self.hists[component_name] and after_hist_name in self.hists[component_name]:
                    self.hists[component_name][hist_name] = eta_phi_removed(
                        hist_name = hist_name,
                        before_hist = self.hists[component_name][before_hist_name],
                        after_hist = self.hists[component_name][after_hist_name]
                    )

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
        if "processing" in options:
            func_name = options["processing"]["func_name"]
            func = getattr(this_module, func_name)
            if func:
                logger.debug(f"Calling func_name {func_name} (func {func}) for hist options {hist_options_name}")
                options = func(hist_options_name, options)
                logger.debug(f"Options after return: {options}")
            else:
                raise ValueError(func_name, f"Requested function for hist options {hist_options_name} doesn't exist!")

        return options

def eta_phi_match_hist_names(hist_options_name, options):
    """ Generate hist names based on the available options.

    This approach allows generating of hist config options using for loops
    while still being defined in YAML.

    Note:
        This function is called via histOptionsSpecificProcessing(...), so it is not
        referenced directly in the source.

    Args:
        histOptionsName (str): Name of the hist options.
        options (dict): Associated set of hist options.
    Returns:
        dict: Updated set of hist options.
    """
    # Pop this value so it won't cause issues when creating the hist plotter later.
    processing_options = options.pop("processing")
    # Get the hist name template
    # We don't care about the hist title
    hist_name = next(iter(next(iter(options["histNames"]))))
    # Angle name
    angles = processing_options["angles"]
    # {Number: label}
    cent_bins = processing_options["centBins"]
    # {Number: label}
    eta_directions = processing_options["etaDirections"]
    # List of pt bins
    pt_bins = processing_options["ptBins"]
    # We don't load these from YAML to avoid having to frequently copy them
    _possible_pt_bin_ranges = [0.15, 0.5, 1, 1.5, 2, 3, 4, 5, 8, -1]
    # NOTE: Careful! We are using 0 indexed pt bins here, which is different than the standard when iterating with YAML.
    possible_pt_bins = [
        analysis_objects.TrackPtBin(bin = i, range = params.SelectedRange(min = low, max = high))
        for i, (low, high) in enumerate(zip(_possible_pt_bin_ranges[:-1], _possible_pt_bin_ranges[1:]))
    ]

    hist_names = []
    for angle in angles:
        for cent_dict in cent_bins:
            cent_bin, cent_label = next(iter(cent_dict.items()))
            for track_pt in possible_pt_bins:
                # Only process the track pt bins that are selected.
                if track_pt.bin not in pt_bins:
                    continue
                for eta_dict in eta_directions:
                    eta_direction, eta_direction_label = next(iter(eta_dict.items()))
                    # Determine hist name
                    name = hist_name.format(angle = angle, cent = cent_bin, pt_bin = track_pt.bin, eta_direction = eta_direction)
                    # Determine label
                    # NOTE: Can't use generate_track_pt_range_string because it includes "assoc" in
                    # the pt label. Instead, we generate the string directly.
                    pt_bin_label = params.generate_pt_range_string(
                        pt_bin = track_pt,
                        lower_label = r"\mathrm{T}",
                        upper_label = r"",
                        only_show_lower_value_for_last_bin = True,
                    )

                    angle_label = determine_angle_label(angle)
                    # Ex: "$\\Delta\\varphi$, Pos. tracks, $\\eta < 0$, $4 < \\mathit{p}_{\\mathrm{T}}< 5$"
                    label = f"{angle_label}, {cent_label}, {eta_direction_label}, {pt_bin_label}"
                    # Save in the expected format
                    hist_names.append({name: label})
                    #logger.debug("name: \"{}\", label: \"{}\"".format(name, label))

    # Assign the newly created names
    options["histNames"] = hist_names
    logger.debug(f"Assigning histNames {hist_names}")

    return options

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

def scale_CPU_time(hist: ROOT.TH1) -> None:
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

def eta_phi_removed(hist_name: str, before_hist: ROOT.TH2, after_hist: ROOT.TH2) -> ROOT.TH2:
    """ Show the eta phi locations of clusters removed by the exotics cut.

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
            obj = PlotEMCalEmbedding,
        )

def run_plot_EMCal_embedding_hists_from_terminal() -> EMCalEmbeddingPlotManager:
    """ Driver function for plotting the EMCal embedding hists. """
    return generic_tasks.run_helper(
        manager_class = EMCalEmbeddingPlotManager,
        description = "EMCal embedding plotting.",
    )

