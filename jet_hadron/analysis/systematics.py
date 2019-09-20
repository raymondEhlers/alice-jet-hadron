#!/usr/bin/env python

""" Calculate systematics for the jet-hadron correlations analysis.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Type

import IPython
import ROOT

from pachyderm import histogram
from pachyderm.projectors import HistAxisRange
from pachyderm.typing_helpers import Hist
from pachyderm.utils import epsilon

from jet_hadron.base import analysis_config
from jet_hadron.base import analysis_manager
from jet_hadron.base import analysis_objects
from jet_hadron.base import params
from jet_hadron.analysis import correlations

logger = logging.getLogger(__name__)

class CorrelationsZVertex(correlations.Correlations):
    def __init__(self, z_vertex: analysis_objects.ZVertexBin, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.z_vertex = z_vertex
        # Store the Z vertex identifier for convenience
        self.z_vertex_identifier = f"zVertex_{self.z_vertex.min}_{self.z_vertex.max}"
        # Add the Z vertex value to the identifier
        self.identifier = f"{self.identifier}_{self.z_vertex_identifier}"

    def _setup_sparse_projectors(self) -> None:
        """ Setup the THnSparse projectors.

        We just add the Z vertex selection onto the existing projectors.

        Args:
            None.
        Returns:
            None. The created projectors are added to the ``sparse_projectors`` list.
        """
        # First create the projectors in the base class.
        super()._setup_sparse_projectors()

        # The sparse axis definition changed after train 4703. Later trains included the Z vertex dependence.
        if self.train_number <= 4703:
            raise ValueError(
                f"Passed train number {self.train_number}, but no trains before train 4703 will have the necessary"
                " Z vertex dependence. Use a more recent one.")
        sparse_axes: Type[correlations.JetHCorrelationSparseZVertex] = correlations.JetHCorrelationSparseZVertex

        z_vertex_axis = HistAxisRange(
            axis_type = sparse_axes.z_vertex,
            axis_range_name = f"z_vertex{self.z_vertex.min}-{self.z_vertex.max}",
            min_val = HistAxisRange.apply_func_to_find_bin(
                ROOT.TAxis.FindBin, self.z_vertex.range.min + epsilon
            ),
            max_val = HistAxisRange.apply_func_to_find_bin(
                ROOT.TAxis.FindBin, self.z_vertex.range.max - epsilon
            )
        )

        # Add to the raw signal and mixed event projectors
        for projector in self.sparse_projectors:
            # We select any projector that isn't for the number of triggers.
            if "trigger" not in projector.projection_name_format:
                projector.additional_axis_cuts.append(z_vertex_axis)

    def run_projections(self, processing_options: correlations.ProcessingOptions) -> None:
        """ Run all analysis steps through projectors.

        Args:
            processing_options: Processing options to configure the projections.
        Returns:
            None. `self.ran_projections` is set to true.
        """
        self._run_2d_projections(processing_options = processing_options)
        # Only run the 2D projections!
        #self._run_1d_projections(processing_options = processing_options)

        # Store that we've completed this step.
        self.ran_projections = True

class CorrelationsZVertexManager(analysis_manager.Manager):
    """ Analysis manager for performing the correlations analysis with Z vertex dependence. """
    def __init__(self, config_filename: str, selected_analysis_options: params.SelectedAnalysisOptions, **kwargs: str):
        super().__init__(
            config_filename = config_filename, selected_analysis_options = selected_analysis_options,
            manager_task_name = "CorrelationsZVertexManager", **kwargs,
        )
        # For convenience since it is frequently accessed.
        self.processing_options = self.task_config["processing_options"]

        # Create the actual analysis objects.
        self.analyses: Mapping[Any, CorrelationsZVertex]
        self.selected_iterables: Mapping[str, Sequence[Any]]
        (self.key_index, self.selected_iterables, self.analyses) = self.construct_correlations_from_configuration_file()

    def construct_correlations_from_configuration_file(self) -> analysis_config.ConstructedObjects:
        """ Construct Z vertex correlations objects based on iterables in a configuration file. """
        return analysis_config.construct_from_configuration_file(
            task_name = "CorrelationsZVertex",
            config_filename = self.config_filename,
            selected_analysis_options = self.selected_analysis_options,
            additional_possible_iterables = {"jet_pt_bin": None, "track_pt_bin": None, "z_vertex": None},
            obj = CorrelationsZVertex,
        )

    def setup(self) -> None:
        """ Setup the correlations manager. """
        # Retrieve input histograms (with caching).
        input_hists: Dict[str, Any] = {}
        with self._progress_manager.counter(total = len(self.analyses),
                                            desc = "Setting up:",
                                            unit = "analysis objects") as setting_up:
            for key_index, analysis in analysis_config.iterate_with_selected_objects(self.analyses):
                # We should now have all RP orientations.
                # We are effectively caching the values here.
                if not input_hists:
                    input_hists = histogram.get_histograms_in_file(filename = analysis.input_filename)
                logger.debug(f"{key_index}")
                # Setup input histograms and projectors.
                analysis.setup(input_hists = input_hists)
                # Keep track of progress
                setting_up.update()

    def run(self) -> bool:
        """ Run the Z vertex analysis. """
        steps = 3
        with self._progress_manager.counter(total = steps,
                                            desc = "Overall processing progress:",
                                            unit = "") as overall_progress:
            # First setup the correlations
            self.setup()
            overall_progress.update()

            # First analysis step
            with self._progress_manager.counter(total = len(self.analyses),
                                                desc = "Projecting:",
                                                unit = "z vertex analysis objects") as projecting:
                for key_index, analysis in analysis_config.iterate_with_selected_objects(self.analyses):
                    analysis.run_projections(processing_options = self.processing_options)
                    # Keep track of progress
                    projecting.update()
            overall_progress.update()

            self._merge_z_vertex_signal_correlations()
            overall_progress.update()

        return True

    def _merge_z_vertex_signal_correlations(self) -> None:
        """ Merge the Z vertex dependent correlations and then write them out. """
        merged_correlations: Dict[str, Hist] = {}
        output_filename: str = ""
        with self._progress_manager.counter(total = len(self.analyses),
                                            desc = "Merging:",
                                            unit = "Z vertex correlations") as merging:
            for z_vertex_correlations in \
                analysis_config.iterate_with_selected_objects_in_order(
                    analysis_objects = self.analyses,
                    analysis_iterables = self.selected_iterables,
                    #selection = ["z_vertex", "reaction_plane_orientation"],
                    selection = ["z_vertex"]):
                for key_index, analysis in z_vertex_correlations:
                    # Grab the output filename. It should be the same for every analysis object, so we only
                    # grab it once.
                    if output_filename == "":
                        output_filename = analysis.output_filename
                    # Drop the Z vertex identifier, but keep the rest of the identifier.
                    identifier = analysis.identifier.replace(f"_{analysis.z_vertex_identifier}", "")
                    # We care about the EP orientation, so we need to include it in the identifier.
                    # However, we won't include it in the hist name because that's not the standard analysis format.
                    identifier = f"{identifier}_{analysis.reaction_plane_orientation}"
                    if identifier not in merged_correlations:
                        # Retrieve the signal histogram so we can sum them together.
                        h = analysis.correlation_hists_2d.signal.hist
                        # Set the name to the expected value
                        h_name = analysis.correlation_hists_2d.signal.name.replace(f"_{analysis.z_vertex_identifier}", "")
                        h_name += "_mixed_event_systematic"
                        h.SetName(h_name)
                        logger.debug(f"Renaming first hist to: {h_name}")
                        merged_correlations[identifier] = h
                    else:
                        logger.debug(f"Merging {analysis.identifier}, {analysis.reaction_plane_orientation}")
                        merged_correlations[identifier].Add(analysis.correlation_hists_2d.signal.hist)

                    merging.update()

        logger.debug(f"merged_correlations: {merged_correlations}")

        # Write out the merged correlations.
        for ep_orientation in self.selected_iterables["reaction_plane_orientation"]:
            filename = Path(self.output_info.output_prefix) / ("RP" + str(ep_orientation)) / output_filename
            logger.debug(f"filename: {filename}")
            # Create the directory if necessary
            filename.parent.mkdir(parents = True, exist_ok = True)
            # Write out the histograms.
            with histogram.RootOpen(filename = str(filename), mode = "UPDATE"):
                for name, hist in merged_correlations.items():
                    # Only write the histogram if it's valid.
                    if hist and str(ep_orientation) in name:
                        logger.debug(f"Writing hist {hist} with name {hist.GetName()}")
                        hist.Write()

def run_mixed_event_systematics_from_terminal() -> CorrelationsZVertexManager:
    """ Driver function for running the mixed event systematics analysis. """
    # Basic setup
    # Quiet down pachyderm
    logging.getLogger("pachyderm").setLevel(logging.INFO)
    # Quiet down reaction_plane_fit
    logging.getLogger("reaction_plane_fit").setLevel(logging.INFO)
    # Turn off stats box
    ROOT.gStyle.SetOptStat(0)

    # Setup and run the analysis
    manager: CorrelationsZVertexManager = analysis_manager.run_helper(
        manager_class = CorrelationsZVertexManager, task_name = "CorrelationsZVertex",
    )

    # Quiet down IPython.
    logging.getLogger("parso").setLevel(logging.INFO)
    # Embed IPython to allow for some additional exploration
    IPython.embed()

    # Return the manager for convenience.
    return manager

if __name__ == "__main__":
    run_mixed_event_systematics_from_terminal()

