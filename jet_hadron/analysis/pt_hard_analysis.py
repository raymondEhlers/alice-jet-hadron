#!/usr/bin/env python

""" Manages the pt hard bin analyses.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import logging
from typing import Any, Dict, Mapping

from pachyderm import histogram
from pachyderm import projectors
from pachyderm import remove_outliers
from pachyderm import utils

from jet_hadron.base import analysis_config
from jet_hadron.base import analysis_objects
from jet_hadron.base.typing_helpers import Hist

logger = logging.getLogger(__name__)

def calculate_average_n_events(pt_hard_bins: Mapping[Any, Any]) -> float:
    """ Get relative scaling for each pt hard bin and scale the scale factors by each relative value """
    n_total_events = 0.
    for key_index, pt_hard_bin in analysis_config.iterate_with_selected_objects(pt_hard_bins):
        if pt_hard_bin.setup_complete is not True:
            raise ValueError(f"Setup was not run on pt hard bin {key_index}. Please run it and try again!")
        n_total_events += pt_hard_bin.number_of_events

    return n_total_events / len(pt_hard_bins)

class PtHardAnalysis(analysis_objects.JetHBase):
    """ Create a pt hard bin analysis object.

    These analysis objects will interact with standard analysis objects to scale them.
    """
    def __init__(self, pt_hard_bin, *args, **kwargs):
        # First, initialize the base class
        super().__init__(*args, **kwargs)

        # Basic information
        self.pt_hard_bin = pt_hard_bin
        self.use_after_event_selection_information = self.task_config.get("use_after_event_selection_information", False)
        self.train_number = self.pt_hard_bin.train_number
        self.input_filename = self.input_filename.format(pt_hard_bin_train_number = self.train_number)
        # We don't update the output prefix because nothing is output by this task.
        #self.output_prefix = self.output_prefix.format_map(pt_hard_bin_train_number = self.train_number)

        # Outliers removal
        self.moving_average_threshold = self.task_config.get("moving_average_threshold", 1.0)
        self.outliers_manager: remove_outliers.OutliersRemovalManager

        # Histograms
        self.pt_hard_spectra: Hist
        self.cross_section: Hist
        self.n_trials: Hist
        self.n_events: Hist

        # Extracted values
        self.scale_factor: float = 0.0
        self.number_of_events: int = 0

    def _retrieve_histograms(self, input_hists: Mapping[str, Any] = None) -> bool:
        """ Retrieve relevant histogram information.

        Args:
            input_hists: All histograms in a file. Default: None - They will be retrieved.
        Returns:
            bool: True if histograms were retrieved successfully.
        """
        # Retrieve the histograms if they aren't passed in
        if input_hists is None:
            input_hists = histogram.get_histograms_in_list(
                filename = self.input_filename,
                input_list = self.input_list_name
            )

        # The name is different if we use the values after event selection.
        event_sel_tag = "AfterSel" if self.use_after_event_selection_information else ""

        # Retrieve hists
        self.pt_hard_spectra = input_hists[self.input_list_name]["fHistPtHard"]
        self.cross_section = input_hists[self.input_list_name][f"fHistXsection{event_sel_tag}"]
        self.n_trials = input_hists[self.input_list_name][f"fHistTrials{event_sel_tag}"]
        self.n_events = input_hists[self.input_list_name]["fHistEventCount"]

        return True

    def _extract_scale_factor(self) -> float:
        """ Extract the scale factor from the stored information. """
        # Pt hard bin 1 is stored in root indexed by 2 by convention, so we need
        # a +1 to get the proper values.
        cross_section = self.cross_section.GetBinContent(self.pt_hard_bin.bin + 1) * self.cross_section.GetEntries()
        n_trials = self.n_trials.GetBinContent(self.pt_hard_bin.bin + 1)

        # Helpful for debugging empty values, but otherwise, it's excessively verbose.
        #for i in range(0, self.n_trials.GetNcells()):
        #    logger.debug(f"i: {i}, n_trials: {self.n_trials.GetBinContent(i)}")
        #logger.debug(f"cross_section: {cross_section}, n_trials: {n_trials}")

        scale_factor = 0
        if n_trials > 0:
            scale_factor = cross_section / n_trials
        else:
            raise ValueError("nTrials is 0! Cannot calculate it, so setting to 0 (ie won't contribute)!")

        return scale_factor

    def _extract_number_of_events(self) -> int:
        """ Extract number of accepted events in the pt hard bin."""
        return self.n_events.GetBinContent(1)

    def setup(self, input_hists: Mapping[str, Any] = None) -> bool:
        """ Setup the pt hard bin objects.

        Note:
            This is separate from the run set because we needed the number of events from every object
            to be able to determine the relative scale factor scaling.

        Args:
            input_hists: All histograms in a file. Default: None - They will be retrieved.
        Returns:
            True if the pt hard bin was successfully setup.
        Raisees:
            ValueError: If the histograms could not be retrieved.
            ValueError: If it failed to extract the scale factor.
            ValueError: If it failed to extract the number of events.
        """
        result = self._retrieve_histograms(input_hists = input_hists)
        if result is False:
            raise ValueError("Could not retrieve histograms.")

        # Setup outliers removal
        self.outliers_manager = remove_outliers.OutliersRemovalManager(
            moving_average_threshold = self.moving_average_threshold
        )

        # Extract scale factors and number of events in the particular pt hard bin.
        self.scale_factor = self._extract_scale_factor()
        if not self.scale_factor > 0.:
            raise ValueError("Failed to extract scale factor.")
        self.number_of_events = self._extract_number_of_events()
        if self.number_of_events == 0:
            raise ValueError("Failed to extract number of events.")

        self.setup_complete = True
        return self.setup_complete

    def _calculate_rescale_factor(self, average_number_of_events: float) -> float:
        """ Calculate the scale factor rescaled for the different number of events in each pt hard bin. """
        return self.scale_factor * average_number_of_events / self.number_of_events

    def run(self, average_number_of_events: float,
            outliers_removal_axis: projectors.TH1AxisType,
            hists: Mapping[str, Hist] = None,
            analyses: Mapping[str, analysis_objects.JetHBase] = None,
            hist_attribute_name: str = "") -> bool:
        """ Run the pt hard analysis.

        Histograms are often grouped together if we want them to remove outliers at the same index location.
        For example, we may provide all EP orientation histograms at once so they all start removing outliers
        from the same axis.

        Args:
            average_number_of_events: Average number of events per pt hard bin. Must be calculated
                externally from all of the pt hard bin analysis objects.
            outliers_removal_axis: Projection axis to be used in outliers removal. Usually the particle level axis.
            hists: Collection of histograms to be processed according to the values in this pt hard bin.
                Default: None, in which case ``analyses`` and ``hist_attribute_name`` must be specified.
            analyses: Analysis objects to be processed according to values in this pt hard bin. Keys are the identifier,
                while the values are the objects. Must be specified in conjunction with ``hist_attribute_name``.
                Default: None, in which case, the hists must be provided directly.
            hist_attribute_name: Name of the attributes to retrieve the histogram from the analysis object. Must
                be specified in conjunction with ``analyses``.  Default: None, in which case, the hists must be
                provided directly.
        Returns:
            True if the process was successful.
        Raises:
            ValueError: If the pt hard object hasn't already been setup.
            ValueError: If the arguments provided are somehow invalid.
        """
        # Validation
        if self.setup_complete is not True:
            raise ValueError("Must complete setup of the pt hard object!")
        # Ensure we haven't passed too many arguments
        if hists and (analyses or hist_attribute_name):
            raise ValueError("Must not specify both hists and analyses or hist_attribute_name.")
        # We will use hists throughout regardless of the arguments, so we need to ensure that it is valid
        if hists is None:
            hists = {}

        # Finally, extract the histograms from the analysis objects if we specified those arguments.
        if analyses or hist_attribute_name:
            if not analyses or not hist_attribute_name:
                raise ValueError("If specifying analyses and hist_attribute_name, then you must specify both!")
            hists = _get_hists_from_analysis_objects(analyses = analyses, hist_attribute_name = hist_attribute_name)

        # Final determination of the scale factor.
        self.scale_factor = self._calculate_rescale_factor(average_number_of_events)

        # Remove outliers from the given hists.
        self.outliers_manager.run(hists = hists, outliers_removal_axis = outliers_removal_axis)

        # Scale the requested histograms.
        for h in hists.values():
            h.Scale(self.scale_factor)

        return True

def _get_hists_from_analysis_objects(analyses: Mapping[str, analysis_objects.JetHBase],
                                     hist_attribute_name: str) -> Dict[str, Hist]:
    """ Retrieve histograms from an analysis object stored under specified attribute names.

    Args:
        analyses: Analysis objects to be processed according to values in this pt hard bin.
        hist_attribute_name: Names of the attributes to retrieve the histograms.
    Returns:
        Extracted histograms.
    """
    hists: Dict[str, Hist] = {}
    for key, analysis in analyses.items():
        hists[key] = utils.recursive_getattr(analysis, hist_attribute_name)
    return hists

def merge_pt_hard_binned_analyses(analyses: Mapping[Any, analysis_objects.JetHBase],
                                  hist_attribute_name,
                                  output_analysis_object) -> None:
    """ Merge together all scaled histograms.

    Args:
        analyses: Pt hard dependent analyses which should be merged together.
        hist_attribute_name: Name of the attribute where the hist is stored.
        output_analysis_object: Object where the histogram will be stored under ``hist_attribute_name``.
    Returns:
        None
    """
    output_hist: Hist = None
    for _, analysis in analyses:
        input_hist = utils.recursive_getattr(analysis, hist_attribute_name)
        if output_hist is None:
            output_hist = input_hist.Clone(input_hist.GetName() + "_merged")
            # Reset so we can just Add() all hists without worrying which hist is being processed
            output_hist.Reset()
            # NOTE: Sumw2 is kept even after resetting.

        output_hist.Add(input_hist)

    # Save the final result
    utils.recursive_setattr(output_analysis_object, hist_attribute_name, output_hist)
