#!/usr/bin/env python

""" Manages the pt hard bin analyses.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import logging
from typing import Any, Dict, Mapping, Type

from pachyderm import histogram
from pachyderm import projectors
from pachyderm import remove_outliers

from jet_hadron.base import analysis_config
from jet_hadron.base import analysis_objects

import ROOT

# Typing helper
Hist = Type[ROOT.TH1]

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

    def _retrieve_histograms(self, input_hists: Dict[str, Any] = None) -> bool:
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

    def setup(self, input_hists: Dict[str, Any] = None) -> bool:
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
            particle_level_axis = projectors.TH1AxisType.y_axis,
            moving_average_threshold = 1.0,
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

    def outliers_removal(self, analysis: Type[analysis_objects.JetHBase]) -> bool:
        """ Remove outliers from the stored histograms. """
        # TODO: Implement
        self.outliers_manager.run()

        return False

    def scale_histograms(self, analysis: Type[analysis_objects.JetHBase]) -> bool:
        """ Scale the selected histograms by the calculated scale factors. """
        # TODO: Implement
        pass

    def run(self, analysis: Type[analysis_objects.JetHBase], average_number_of_events: float) -> bool:
        """ Run the pt hard analysis.

        Args:
            analysis: Analysis object to be scaled according to values in this pt hard bin.
        Returns:
            True if the process was successful.
        Raises:
            ValueError: If the pt hard object hasn't already been setup.
            RuntimeError: If the outlier removal fails.
            RuntimeError: If the histogram scaling was not successful.
        """
        # Validation
        if self.setup_complete is not True:
            raise ValueError("Must complete setup of the pt hard object!")

        # Final determination of the scale factor.
        self.scale_factor = self._calculate_rescale_factor(average_number_of_events)

        # Remove outliers from the given analysis object.
        result = self.outliers_removal(analysis)
        if result is False:
            raise RuntimeError("Outlier removal failed.")

        # Scale the requested histograms.
        result = self.scale_histograms(analysis)
        if result is False:
            raise RuntimeError("Histogram scaling failed.")

        return True

