#!/usr/bin/env python

""" STAR jet-hardon response matrix.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@yale.edu>, Yale University
"""

import logging
import numpy as np
import os
from typing import Any, Dict, Mapping, Optional

# NOTE: This is out of the expected order, but it must be here to prevent ROOT from stealing the command
#       line options
from jet_hadron.base.typing_helpers import Hist

from pachyderm import histogram

from jet_hadron.base import analysis_config
from jet_hadron.base import analysis_manager
from jet_hadron.base import params
from jet_hadron.plot import response_matrix as plot_response_matrix
from jet_hadron.analysis import pt_hard_analysis
from jet_hadron.analysis import response_matrix

import ROOT

logger = logging.getLogger(__name__)

class STARResponseMatrix(response_matrix.ResponseMatrixBase):
    """ Response matrix for handling the STAR response.

    It takes advantage of the ``ResponseMatrixBase`` interface, but since the data is stored it trees,
    it utilizes this interface to convert the tree into histograms.
    """
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        # Pt hard bins are optional.
        self.pt_hard_bin = kwargs.get("pt_hard_bin", None)
        if self.pt_hard_bin:
            self.train_number = self.pt_hard_bin.train_number
            self.input_filename = self.input_filename.format(pt_hard_bin_train_number = self.train_number)
            self.output_prefix = self.output_prefix.format(pt_hard_bin_train_number = self.train_number)

    def _retrieve_histograms(self, input_hists: Optional[Dict[str, Any]] = None) -> bool:
        """ We define the histograms later, so do nothing here. """
        return True

    def _setup_projectors(self) -> None:
        """ Setup for projectors.

        Here, we won't use standard projectors, but instead will convert a tree into histograms. So we setup
        those histograms here.
        """
        # Create the histograms. Unfortunately, to integrate with the other response matrix
        # code, we need to use ROOT histograms.
        self.response_matrix = ROOT.TH2D("responseMatrix", "Response matrix", 100, 0, 100, 100, 0, 100)
        self.part_level_hists: response_matrix.ResponseHistograms = response_matrix.ResponseHistograms(
            jet_spectra = ROOT.TH1D("particleLevelJets", "Particle level jets", 100, 0, 100),
            unmatched_jet_spectra = None,
        )
        self.det_level_hists: response_matrix.ResponseHistograms = response_matrix.ResponseHistograms(
            jet_spectra = ROOT.TH1D("detectorLevelJets", "Detector level jets", 100, 0, 100),
            unmatched_jet_spectra = None,
        )
        self.matched_jet_pt_difference = ROOT.TH2D("fh2PtJet2VsRelPt", "Matched particle det level jet energy scale", 40, -50, 150, 241, -2.41, 2.41)
        # ROOT memory management sux...
        self.response_matrix.SetDirectory(0)
        self.part_level_hists.jet_spectra.SetDirectory(0)
        self.det_level_hists.jet_spectra.SetDirectory(0)
        self.matched_jet_pt_difference.SetDirectory(0)
        # Ensure that sumw2 is set before filling.
        self.set_sumw2()

    def run_projectors(self) -> None:
        """ Convert the jet tree into histograms. """
        # Load in the stored jets.
        with open(self.input_filename, "rb") as f:
            jets = np.load(f)

        # Response matrix and matched spectra (basically just projections)
        for det_pt, part_pt in zip(jets["det_pT"], jets["part_pT"]):
            self.response_matrix.Fill(det_pt, part_pt)
            self.part_level_hists.jet_spectra.Fill(part_pt)
            self.det_level_hists.jet_spectra.Fill(det_pt)
            self.matched_jet_pt_difference.Fill(det_pt, (part_pt - det_pt) / det_pt)

        # Create fake unmatched spectra. They don't matter, but it makes it way easier to work with the response code
        self.part_level_hists.unmatched_jet_spectra = self.part_level_hists.jet_spectra.Clone("unmathcedParticleLevelJets")
        self.det_level_hists.unmatched_jet_spectra = self.det_level_hists.jet_spectra.Clone("unmatchedDetectorLevelJets")
        # ROOT memory management sux...
        self.part_level_hists.unmatched_jet_spectra.SetDirectory(0)
        self.det_level_hists.unmatched_jet_spectra.SetDirectory(0)

    def retrieve_non_projected_hists(self) -> bool:
        """ Retrieve histograms which don't require projectors.

        We intentionally do nothing here, but we want to continue, so we return ``True``.
        """
        return True

class STARPtHardAnalysis(pt_hard_analysis.PtHardAnalysis):
    def __init__(self, *args: Any, **kwargs: Any):
        # First, initialize the base class
        super().__init__(*args, **kwargs)

    def _create_histograms(self, n_pt_hard_bins: int) -> None:
        """ Setup histograms for converting the event tree into histograms.

        Args:
            n_pt_hard_bins: Number of pt hard bins (for defining the histograms).
        Returns:
            None
        """
        self.pt_hard_spectra: Hist = ROOT.TH1D("fHistPtHard", "p_{T} Hard Spectra;p_{T} hard;Counts", 100, 0, 100)
        self.cross_section: Hist = ROOT.TProfile("fHistXsection", "Pythia Cross Section;p_{T} hard bin; XSection", n_pt_hard_bins + 1, 0, n_pt_hard_bins + 1)
        self.n_trials: Hist = ROOT.TH1D("fHistTrials", "Number of Pythia Trials;p_{T} hard bin;Trials", n_pt_hard_bins + 1, 0, n_pt_hard_bins + 1)
        self.n_events: Hist = ROOT.TH1D("fHistEventCount", "Number of events", 2, 0, 2)
        # ROOT memory management sux...
        self.pt_hard_spectra.SetDirectory(0)
        self.cross_section.SetDirectory(0)
        self.n_trials.SetDirectory(0)
        self.n_events.SetDirectory(0)

    def _retrieve_histograms(self, input_hists: Optional[Mapping[str, Any]] = None) -> bool:
        """ Converts the stored trees into histograms for further processing.

        Args:
            input_hists: Ignored.
        Returns:
            bool: True if histograms were retrieved successfully.
        """
        # Event level properties
        with open(self.input_filename, "rb") as f:
            events = np.load(f)

        # Pt hard spectra
        for value in events["pt_hard"]:
            self.pt_hard_spectra.Fill(value)
        # Cross section
        for value in events["cross_section"]:
            self.cross_section.Fill(self.pt_hard_bin.bin, value)
        # N trials
        self.n_trials.Fill(self.pt_hard_bin.bin, len(events))
        # N events
        # Bin 1 is the bin for accepted events.
        self.n_events.SetBinContent(1, len(events))

        return True

    def setup(self, input_hists: Optional[Mapping[str, Any]] = None, n_pt_hard_bins: int = -1) -> bool:
        # Convert into histograms which we can actually use.
        self._create_histograms(n_pt_hard_bins = n_pt_hard_bins)

        return super().setup(input_hists = input_hists)

class STARResponseManager(response_matrix.ResponseManager):
    def __init__(self, config_filename: str, selected_analysis_options: params.SelectedAnalysisOptions, **kwargs: Any):
        # Just need to set the "STARResponseManager" name here.
        super().__init__(
            config_filename = config_filename, selected_analysis_options = selected_analysis_options,
            manager_task_name = "STARResponseManager", **kwargs
        )

    def construct_responses_from_configuration_file(self) -> analysis_config.ConstructedObjects:
        """ Construct ``STARResponseMatrix`` objects based on iterables in a configuration file. """
        return self._construct_responses_from_configuration_file(task_name = "STARResponse", obj = STARResponseMatrix)

    def construct_final_responses_from_configuration_file(self) -> analysis_config.ConstructedObjects:
        """ Construct final ``ResponseMatrixBase`` objects based on iterables in a configuration file. """
        return self._construct_final_responses_from_configuration_file(task_name = "STARResponseFinal")

    def construct_pt_hard_bins_from_configuration_file(self) -> analysis_config.ConstructedObjects:
        """ Construct ``STARPtHardAnalysis`` objects based on iterables in a configuration file. """
        return self._construct_pt_hard_bins_from_configuration_file(task_name = "STARPtHardBins", obj = STARPtHardAnalysis)

    def construct_final_pt_hard_object_from_configuration_file(self) -> analysis_config.ConstructedObjects:
        """ Construct final ``PtHardAnalysisBase`` objects based on iterables in a configuration file. """
        return self._construct_final_pt_hard_object_from_configuration_file(task_name = "STARPtHardFinal")

    def setup(self) -> None:
        """ Setup and prepare the analysis objects.

        We reimplement it here to disable the file caching (which doesn't mix well with the tree based approach),
        and because we need the number of pt hard bins to setup the pt hard analysis.
        """
        # Setup the response matrix analysis objects and run the response matrix projectors
        # By the time that this step is complete, we should have all histograms.
        with self._progress_manager.counter(total = len(self.analyses),
                                            desc = "Configuring and projecting:",
                                            unit = "responses") as setting_up:
            for pt_hard_bin in self.selected_iterables["pt_hard_bin"]:
                logger.debug(f"pt_hard_bin: {pt_hard_bin}")
                for key_index, analysis in \
                        analysis_config.iterate_with_selected_objects(self.analyses, pt_hard_bin = pt_hard_bin):
                    # A cache of files isn't so straightforward here because we aren't working with ROOT files.
                    # Instead, we just neglect the cache.
                    logger.debug(f"key_index: {key_index}")
                    result = analysis.setup(input_hists = None)
                    if result is not True:
                        raise ValueError(f"Setup of {key_index} analysis object failed.")
                    result = analysis.retrieve_non_projected_hists()
                    if result is not True:
                        raise ValueError(f"Retrieval of non-projected hists of {key_index} analysis object failed.")
                    analysis.run_projectors()

                    # Ensure that all hists have sumw2 enabled
                    analysis.set_sumw2()

                    # Update progress
                    setting_up.update()

        # Setup the pt hard bin analysis objects.
        with self._progress_manager.counter(total = len(self.pt_hard_bins),
                                            desc = "Setting up: ",
                                            unit = "pt hard bins") as setting_up:
            for key_index, pt_hard_bin in analysis_config.iterate_with_selected_objects(self.pt_hard_bins):
                pt_hard_bin.setup(input_hists = None, n_pt_hard_bins = len(self.pt_hard_bins))

                # Update progress
                setting_up.update()

    def _plot_results(self, histogram_info_for_processing:
                      Mapping[str, pt_hard_analysis.PtHardHistogramInformation]) -> None:
        """ Plot the results of the response matrix processing.

        Args:
            histogram_info_for_processing: Specifies which histograms to process, and how to do so.
        Returns:
            None.
        """
        # Plot the STAR/ALICE comparison
        self._plot_comparison_of_STAR_and_ALICE_particle_level_spectra()
        # Then plot the rest of the histograms.
        super()._plot_results(histogram_info_for_processing = histogram_info_for_processing)

    def _plot_comparison_of_STAR_and_ALICE_particle_level_spectra(self) -> None:
        """ Compare STAR and ALICE particle level spectra. """
        # Open ALICE 2.76 semi_central reference.
        base_path = os.path.join(
            "output", "embedPythia", "2.76", str(self.selected_analysis_options.event_activity),
            "{leading_hadron_bias}", "ResponseFinal", "final_responses.root"
        )
        alice276Path = base_path.format(leading_hadron_bias = "clusterBias6")
        particle_level_spectra: Dict[params.CollisionEnergy, Hist] = {}
        with histogram.RootOpen(alice276Path) as f:
            h = f.Get("particle_level_spectra_inclusive")
            h.SetDirectory(0)
            particle_level_spectra[params.CollisionEnergy.two_seven_six] = h

        plot_response_matrix.compare_STAR_and_ALICE(
            star_final_response_task = self.final_responses[
                self.final_responses_key_index(params.ReactionPlaneOrientation.inclusive)
            ],
            alice_particle_level_spectra = particle_level_spectra,
            output_info = self.output_info,
        )

def run_from_terminal() -> STARResponseManager:
    """ Driver function for running the STAR response matrix analysis. """
    # Basic setup
    # Quiet down some pachyderm modules
    logging.getLogger("pachyderm.generic_config").setLevel(logging.INFO)
    logging.getLogger("pachyderm.histogram").setLevel(logging.INFO)
    # Run in batch mode
    ROOT.gROOT.SetBatch(True)
    # Turn off stats box
    ROOT.gStyle.SetOptStat(0)

    # Setup and run the analysis
    manager: STARResponseManager = analysis_manager.run_helper(
        manager_class = STARResponseManager, task_name = "Response matrix",
    )

    # Return it for convenience.
    return manager

if __name__ == "__main__":
    run_from_terminal()
