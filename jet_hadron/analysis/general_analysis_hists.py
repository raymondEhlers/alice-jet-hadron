#!/usr/bin/env python3

""" Plot general analysis histograms.

The generic hist plotter is cute, but these plots are easier to handle when plotted directly.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import IPython
import logging
from typing import Any, Dict, List, Mapping, Sequence

from jet_hadron.base.typing_helpers import Hist

from pachyderm import histogram

from jet_hadron.base import analysis_config, analysis_manager, analysis_objects, params
from jet_hadron.plot import general as plot_general

import ROOT

logger = logging.getLogger(__name__)

class GeneralAnalysisHists(analysis_objects.JetHBase):
    """ Plot general analysis histograms.

    This task exists because some general analysis histograms are spread over a wide variety of output lists.
    For those plots, it's much easier to have a single task take care of all of the plotting work.

    Note:
        This task explicitly breaks a lot of the encapsulation that other tasks try to maintain.
        It's not an ideal design, but it significantly simplifies the plotting efforts here, so it's
        worth the trade off.
    """
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        # Store input hists.
        self.input_hists: Dict[str, Any] = {}

    def setup(self) -> bool:
        """ Retrieve the input histograms.

        We don't restrict to just one output list. Instead, we retrieve all of the histograms
        in the file.

        Args:
            None.
        Returns:
            None.
        """
        self.input_hists = histogram.get_histograms_in_file(self.input_filename)
        return True

    def _event_cuts(self) -> None:
        """ Plot a selection of event cuts histograms.

        Puts both the central and semi-central hists on the same plot when possible.

        Args:
            None.
        Returns:
            None.
        """
        ...

    def _plot_rho(self) -> None:
        """ Plot the rho task background.

        Args:
            None.
        Returns:
            None.
        """
        rho_hists = self.input_hists[self.task_config["rho_task_name"]]
        rho_vs_centrality = rho_hists["fHistRhovsCent"]
        plot_general.rho_centrality(
            rho_hist = rho_vs_centrality,
            output_info = self.output_info,
            includes_constituent_cut = True,
        )

    def _track_eta_phi(self) -> None:
        """ Plot the track eta phi.

        Args:
            None.
        Returns:
            None.
        """
        # We need particular formatting, so we just take care of it by hand.
        event_activity_label_map = {
            params.EventActivity.central: "Central",
            params.EventActivity.semi_central: "SemiCentral",
        }
        for event_activity in [params.EventActivity.central, params.EventActivity.semi_central]:
            # Setup
            task_name = self.task_config["jet_hadron_base_task_name"]
            task_name += f"{event_activity_label_map[event_activity]}"
            # Track pt bin values are from the jet-hadron task.
            track_pt_bin_values = [0.5, 1, 2, 3, 5, 8, 20]
            # Convert into track pt bins.
            track_pt_bins = [analysis_objects.TrackPtBin(params.SelectedRange(min_value, max_value), bin = bin_number)
                             for bin_number, (min_value, max_value)
                             in enumerate(zip(track_pt_bin_values[:-1], track_pt_bin_values[1:]))]

            # Retrieve the hists
            hists: List[Hist] = []
            for track_pt_bin in track_pt_bins:
                hists.append(self.input_hists[task_name][f"fHistTrackEtaPhi_{track_pt_bin.bin}"])

            # Merge the hists together. We don't really need the track pt dependence.
            # Skip the lowest pt hist because of some weird structure which will distract from the
            # message. Plus, the lowest pt bin isn't even used for the correlations.
            output_hist = hists[1]
            for h in hists[2:]:
                output_hist.Add(h)

            # Lastly, plot
            plot_general.track_eta_phi(
                hist = output_hist,
                event_activity = event_activity,
                output_info = self.output_info,
            )

    def run(self) -> bool:
        # Event cuts
        self._event_cuts()

        # Rho
        self._plot_rho()

        # Eta-phi
        self._track_eta_phi()

        return True

class GeneralAnalysisHistsManager(analysis_manager.Manager):
    def __init__(self, config_filename: str, selected_analysis_options: params.SelectedAnalysisOptions, **kwargs: str):
        super().__init__(
            config_filename = config_filename, selected_analysis_options = selected_analysis_options,
            manager_task_name = "GeneralAnalysisHistsManager", **kwargs,
        )

        # Create the actual analysis objects.
        self.analyses: Mapping[Any, GeneralAnalysisHists]
        self.selected_iterables: Mapping[str, Sequence[Any]]
        (self.key_index, self.selected_iterables, self.analyses) = self.construct_hist_objects_from_configuration_file()

    def construct_hist_objects_from_configuration_file(self) -> analysis_config.ConstructedObjects:
        """ Construct general analysis hists objects based on iterables in a configuration file. """
        return analysis_config.construct_from_configuration_file(
            task_name = "GeneralAnalysisHists",
            config_filename = self.config_filename,
            selected_analysis_options = self.selected_analysis_options,
            additional_possible_iterables = {"pt_hard_bin": None, "jet_pt_bin": None, "track_pt_bin": None},
            obj = GeneralAnalysisHists,
        )

    def run(self) -> bool:
        with self._progress_manager.counter(total = len(self.analyses),
                                            desc = "Setting up:",
                                            unit = "analysis objects") as setting_up:
            for key_index, analysis in analysis_config.iterate_with_selected_objects(self.analyses):
                analysis.setup()
                setting_up.update()

        with self._progress_manager.counter(total = len(self.analyses),
                                            desc = "Running:",
                                            unit = "analysis objects") as running:
            for key_index, analysis in analysis_config.iterate_with_selected_objects(self.analyses):
                analysis.run()
                running.update()

        return True

def run_from_termainl() -> GeneralAnalysisHistsManager:
    """ Driver function for running the mixed event systematics analysis. """
    # Basic setup
    # Quiet down pachyderm
    logging.getLogger("pachyderm").setLevel(logging.INFO)
    # Quiet down reaction_plane_fit
    logging.getLogger("reaction_plane_fit").setLevel(logging.INFO)
    # Turn off stats box
    ROOT.gStyle.SetOptStat(0)
    # Run in batch mode
    ROOT.gROOT.SetBatch(True)

    # Setup and run the analysis
    manager: GeneralAnalysisHistsManager = analysis_manager.run_helper(
        manager_class = GeneralAnalysisHistsManager, task_name = "GeneralAnalysisHistsManager",
    )

    # Quiet down IPython.
    logging.getLogger("parso").setLevel(logging.INFO)
    # Embed IPython to allow for some additional exploration
    IPython.embed()

    # Return the manager for convenience.
    return manager

if __name__ == "__main__":
    run_from_termainl()

