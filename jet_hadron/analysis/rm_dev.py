#!/usr/bin/env python

""" Response matrix dev.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@yale.edu>, Yale University
"""

from dataclasses import dataclass
import enum
import logging
import pprint
from typing import Any, Dict, List, Mapping, Type

from pachyderm import generic_class
from pachyderm import histogram
from pachyderm import projectors

from jet_hadron.base import analysis_config
from jet_hadron.base import analysis_objects
from jet_hadron.base import params

from jet_hadron.analysis import pt_hard_analysis

import ROOT

# Typing helper
Hist = Type[ROOT.TH1]

logger = logging.getLogger(__name__)

class ResponseMakerMatchingSparse(enum.Enum):
    """ Defines the axes in the AliJetResponseMaker fMatching THnSparse. """
    det_level_jet_pt = 0
    part_level_jet_pt = 1
    matching_distance = 4
    det_level_leading_particle = 7
    part_level_leading_particle = 8
    det_level_reaction_plane_orientation = 9
    part_level_reaction_plane_orientation = 10

class ResponseMakerJetsSparse(enum.Enum):
    """ Defines the axes in the AliJetResponseMaker fJets THnSparse """
    phi = 0
    eta = 1
    jet_pt = 2
    jet_area = 3
    # Different if the event plane is included in the output or not!!
    reaction_plane_orientation = 4
    leading_particle_PP = 4
    leading_particle_PbPb = 5

class ResponseMatrixProjector(projectors.HistProjector):
    """ Projector for the Jet-h response matrix THnSparse. """
    def ProjectionName(self, **kwargs):
        """ Define the projection name for the JetH RM projector """
        ptHardBin = kwargs["inputKey"]
        hist = kwargs["inputHist"]
        logger.debug("Projecting pt hard bin: {0}, hist: {1}, projectionName: {2}".format(ptHardBin, hist.GetName(), self.projection_name_format.format(ptHardBin = ptHardBin)))
        return self.projection_name_format.format(ptHardBin = ptHardBin)

    def OutputKeyName(self, inputKey, outputHist, *args, **kwargs):
        """ Retrun the input key, which is the pt hard bin"""
        return inputKey

class ResponseMatrixPtHardAnalysis(pt_hard_analysis.PtHardAnalysis):
    def remove_outliers(self):
        # TODO: Implement
        pass

    def scale_histograms(self):
        # TODO: Implement
        pass

@dataclass
class ResponseHistograms:
    """ The main histograms for a response matrix. """
    jet_spectra: Hist
    unmatched_jet_spectra: Hist
    sample_task_jet_spectra: Hist

class ResponseMatrix(analysis_objects.JetHReactionPlane):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Pt hard bins are optional.
        self.pt_hard_bin = kwargs.get("pt_hard_bin", None)
        if self.pt_hard_bin:
            self.train_number = self.pt_hard_bin.train_number
            self.input_filename = self.input_filename.format(pt_hard_bin_train_number = self.train_number)

        # Basic information
        self.input_hists: Dict[str, Any] = {}
        self.projectors: List[ResponseMatrixProjector] = []

        # Relevant histograms
        self.response_matrix: Hist
        self.response_matrix_errirs: Hist

        self.part_level_hists: ResponseHistograms
        self.det_level_hists: ResponseHistograms

    def _setup_projectors(self):
        # TODO: Figure out projectors with single histograms.
        # Helper range
        full_axis_range = {
            "min_val": projectors.HistAxisRange.apply_func_to_find_bin(None, 1),
            "max_val": projectors.HistAxisRange.apply_func_to_find_bin(ROOT.TAxis.GetNbins)
        }
        # Reaction plane selection
        if self.reaction_plane_orientation == params.ReactionPlaneOrientation.all:
            reaction_plane_axis_range = full_axis_range
            logger.info("Using full EP angle range")
        else:
            reaction_plane_axis_range = {
                "min_val": projectors.HistAxisRange.apply_func_to_find_bin(None, self.reaction_plane_orientation.value.bin),
                "max_val": projectors.HistAxisRange.apply_func_to_find_bin(None, self.reaction_plane_orientation.value.bin)
            }
            logger.info(f"Using selected EP angle range {self.reaction_plane_orientation.name}")
        reaction_plane_orientation_projector_axis = projectors.HistAxisRange(
            axis_type = ResponseMakerMatchingSparse.det_level_reaction_plane_orientation,
            axis_range_name = "detLevelReactionPlaneOrientation",
            **reaction_plane_axis_range
        )

        #################
        # Response matrix
        #################
        response_matrix = ResponseMatrixProjector(
            observable_dict = self.hists["responseMatrixPtHard"],
            observables_to_project_from = self.hists["responseMatrixPtHardSparse"],
            projection_name_format = "responseMatrix"
        )
        response_matrix.additional_axis_cuts.append(
            projectors.HistAxisRange(
                axis_type = ResponseMakerMatchingSparse.det_level_leading_particle,
                axis_range_name = "detLevelLeadingParticle",
                min_val = projectors.HistAxisRange.apply_func_to_find_bin(ROOT.TAxis.FindBin, self.clusterBias),
                max_val = projectors.HistAxisRange.apply_func_to_find_bin(ROOT.TAxis.GetNbins)
            )
        )
        response_matrix.additional_axis_cuts.append(reaction_plane_orientation_projector_axis)

        # No additional cuts for the projection dependent axes
        response_matrix.projection_dependent_cut_axes.append([])
        response_matrix.projection_axes.append(
            projectors.HistAxisRange(
                axis_type = ResponseMakerMatchingSparse.det_level_jet_pt,
                axis_range_name = "detLevelJetPt",
                **full_axis_range
            )
        )
        response_matrix.projection_axes.append(
            projectors.HistAxisRange(
                axis_type = ResponseMakerMatchingSparse.part_level_jet_pt,
                axis_range_name = "partLevelJetPt",
                **full_axis_range
            )
        )
        # Save the projector for later use
        self.projectors.append(response_matrix)

        ###################
        # Unmatched part level jet pt
        ###################
        unmatched_part_level_jet_spectra = ResponseMatrixProjector(
            observable_dict = self.hists["unmatchedJetSpectraPartLevelPtHard"],
            observables_to_project_from = self.hists["unmatchedPartLevelJetsPtHardSparse"],
            projection_name_format = "unmatchedJetSpectraPartLevel"
        )
        # Can't apply a leading cluster cut on part level, since we don't have clusters
        unmatched_part_level_jet_spectra.projection_dependent_cut_axes.append([])
        unmatched_part_level_jet_spectra.projection_axes.append(
            projectors.HistAxisRange(
                axis_type = ResponseMakerJetsSparse.jet_pt,
                axis_range_name = "unmatchedPartLevelJetSpectra",
                **full_axis_range
            )
        )
        # Save the projector for later use
        self.projectors.append(unmatched_part_level_jet_spectra)

        ###################
        # (Matched) Part level jet pt
        ###################
        part_level_jet_spectra = ResponseMatrixProjector(
            observable_dict = self.hists["jetSpectraPartLevelPtHard"],
            observables_to_project_from = self.hists["responseMatrixPtHardSparse"],
            projection_name_format = "jetSpectraPartLevel"
        )
        part_level_jet_spectra.additional_axis_cuts.append(reaction_plane_orientation_projector_axis)
        # Can't apply a leading cluster cut on part level, since we don't have clusters
        part_level_jet_spectra.projection_dependent_cut_axes.append([])
        part_level_jet_spectra.projection_axes.append(
            projectors.HistAxisRange(
                axis_type = ResponseMakerMatchingSparse.part_level_jet_pt,
                axis_range_name = "partLevelJetSpectra",
                **full_axis_range
            )
        )
        # Save the projector for later use
        self.projectors.append(part_level_jet_spectra)

        ##################
        # Unmatched det level jet pt
        ##################
        unmatched_det_level_jet_spectra = ResponseMatrixProjector(
            observable_dict = self.hists["unmatchedJetSpectraDetLevelPtHard"],
            observables_to_project_from = self.hists["unmatchedDetLevelJetsPtHardSparse"],
            projection_name_format = "unmatchedJetSpectraDetLevel"
        )
        unmatched_det_level_jet_spectra.additional_axis_cuts.append(
            projectors.HistAxisRange(
                axis_type = ResponseMakerJetsSparse.leading_particle_PbPb if self.collision_system == analysis_objects.CollisionSystem.kPbPb else ResponseMakerJetsSparse.leading_particle_PP,
                axis_range_name = "unmatchedDetLevelLeadingParticle",
                min_val = projectors.HistAxisRange.apply_func_to_find_bin(ROOT.TAxis.FindBin, self.clusterBias),
                max_val = projectors.HistAxisRange.apply_func_to_find_bin(ROOT.TAxis.GetNbins)
            )
        )
        unmatched_det_level_jet_spectra.projection_dependent_cut_axes.append([])
        unmatched_det_level_jet_spectra.projection_axes.append(
            projectors.HistAxisRange(
                axis_type = ResponseMakerJetsSparse.jet_pt,
                axis_range_name = "unmatchedDetLevelJetSpectra",
                **full_axis_range
            )
        )
        # Save the projector for later use
        self.projectors.append(unmatched_det_level_jet_spectra)

        ##################
        # (Matched) Det level jet pt
        ##################
        det_level_jet_spectra = ResponseMatrixProjector(
            observable_dict = self.hists["jetSpectraDetLevelPtHard"],
            observables_to_project_from = self.hists["responseMatrixPtHardSparse"],
            projection_name_format = "jetSpectraDetLevel"
        )
        det_level_jet_spectra.additional_axis_cuts.append(
            projectors.HistAxisRange(
                axis_type = ResponseMakerMatchingSparse.det_level_leading_particle,
                axis_range_name = "detLevelLeadingParticle",
                min_val = projectors.HistAxisRange.apply_func_to_find_bin(ROOT.TAxis.FindBin, self.clusterBias),
                max_val = projectors.HistAxisRange.apply_func_to_find_bin(ROOT.TAxis.GetNbins)
            )
        )
        det_level_jet_spectra.additional_axis_cuts.append(reaction_plane_orientation_projector_axis)
        det_level_jet_spectra.projection_dependent_cut_axes.append([])
        det_level_jet_spectra.projection_axes.append(
            projectors.HistAxisRange(
                axis_type = ResponseMakerMatchingSparse.det_level_jet_pt,
                axis_range_name = "detLevelJetSpectra", **full_axis_range
            )
        )
        # Save the projector for later use
        self.projectors.append(det_level_jet_spectra)

    def _retrieve_histograms(self, input_hists: Dict[str, Any] = None) -> bool:
        """ Retrieve histograms from a ROOT file.

        Args:
            input_hists: All histograms in a file. Default: None - They will be retrieved.
        Returns:
            bool: True if histograms were retrieved successfully.
        """
        logger.info(f"input_filename: {self.input_filename}")
        if input_hists is None:
            input_hists = histogram.get_histograms_in_list(
                filename = self.input_filename,
                input_list = self.input_list_name
            )
        self.input_hists = input_hists

        return len(self.input_hists) > 0

    def setup(self, input_hists: Dict[str, Any] = None):
        """ Setup the response matrix.

        Args:
            input_hists: All histograms in a file. Default: None - They will be retrieved.
        Returns:
            bool: True if histograms were retrieved successfully.
        Raises:
            ValueError: If the histograms could not be retrieved.
        """
        result = self._retrieve_histograms(input_hists = input_hists)
        if result is not True:
            raise ValueError("Could not retrieve histograms.")

        self._setup_projectors()

    def run_projectors(self):
        """ Execute the projectors to create the projected histograms. """
        # Perform the various projections
        for projector in self.projectors:
            projector.project()

class ResponseManager(generic_class.EqualityMixin):
    """ Analysis manager for creating response(s).

    Attributes:
        config_filename: Filename of the configuration
        selected_analysis_options: Options selected for this analysis.
        key_index: Key index object for the analysis.
        selected_iterables: All iterables values used to create the response matrices.
        analyses: Response matrix analysis objects.

    """
    def __init__(self, config_filename: str, selected_analysis_options: params.SelectedAnalysisOptions, **kwargs):
        self.config_filename = config_filename
        self.selected_analysis_options = selected_analysis_options

        # Create the actual analysis objects.
        self.analyses: Mapping[Any, ResponseMatrix]
        (self.key_index, self.selected_iterables, self.analyses) = self.construct_responses_from_configuration_file()

        # Create the pt hard bins
        self.pt_hard_bins: Mapping[Any, Type[pt_hard_analysis.PtHardAnalysis]]
        (_, pt_hard_iterables, self.pt_hard_bins) = self.construct_pt_hard_bins_from_configuration_file()

        # Validate that we have the same pt hard iterables.
        if not self.selected_iterables["pt_hard_bin"] == pt_hard_iterables["pt_hard_bin"]:
            raise ValueError("Selected iterables pt hard bins differ from the pt hard bins of the pt hard bin analysis objects. Selected iterables: {self.selected_iterables['pt_hard_bins']}, pt hard analysis iterables: {pt_hard_iterables}")

    def construct_responses_from_configuration_file(self) -> analysis_config.ConstructedObjects:
        """ Construct ResponseMatrix objects based on iterables in a configuration file. """
        return analysis_config.construct_from_configuration_file(
            task_name = "Response",
            config_filename = self.config_filename,
            selected_analysis_options = self.selected_analysis_options,
            additional_possible_iterables = {"pt_hard_bin": None, "jet_pt_bin": None},
            obj = ResponseMatrix,
        )

    def construct_pt_hard_bins_from_configuration_file(self) -> analysis_config.ConstructedObjects:
        """ Construct PtHardAnalysis objects based on iterables in a configuration file. """
        return analysis_config.construct_from_configuration_file(
            task_name = "PtHardBins",
            config_filename = self.config_filename,
            selected_analysis_options = self.selected_analysis_options,
            additional_possible_iterables = {"pt_hard_bin": None},
            obj = ResponseMatrixPtHardAnalysis,
        )

    def setup(self):
        for pt_hard_bin in self.selected_iterables["pt_hard_bin"]:
            logger.debug(f"{pt_hard_bin}")
            input_hists = {}
            for key_index, analysis in \
                    analysis_config.iterate_with_selected_objects(self.analyses, pt_hard_bin = pt_hard_bin):
                # We should now have all RP orientations.
                # We are effectively caching the values here.
                input_hists = histogram.get_histograms_in_file(filename = analysis.input_filename)
                logger.debug(f"{key_index}")
                analysis._retrieve_histograms(input_hists = input_hists)

    def run(self) -> bool:
        logger.debug(f"key_index: {self.key_index}, selected_option_names: {list(self.selected_iterables)}, analyses: {pprint.pformat(self.analyses)}")

        # Cache input hists so we can avoid repeatedly opening files
        input_hists: Dict[Any, Dict[str, Any]] = {}

        # Run the RM projectors
        for pt_hard_bin in self.selected_iterables["pt_hard_bin"]:
            logger.debug(f"{pt_hard_bin}")
            input_hists[pt_hard_bin] = {}
            for key_index, analysis in \
                    analysis_config.iterate_with_selected_objects(self.analyses, pt_hard_bin = pt_hard_bin):
                # We should now have all RP orientations.
                # We are caching the values here to minimize opening files.
                if not input_hists[pt_hard_bin]:
                    input_hists[pt_hard_bin] = histogram.get_histograms_in_file(filename = analysis.input_filename)
                logger.debug(f"{key_index}")
                result = analysis.setup(input_hists = input_hists)
                if result is not True:
                    raise ValueError(f"Setup of {key_index} analysis object failed.")
                analysis.run_projectors()

        # Setup the pt hard bins
        for _, pt_hard_bin in analysis_config.iterate_with_selected_objects(self.pt_hard_bins):
            pt_hard_bin.setup()

        # We have to determine the relative scale factors after the setup because they depend on the number of
        # events in all pt hard bins.
        average_number_of_events = pt_hard_analysis.calculate_average_n_events(self.pt_hard_bins)

        # Finally, scaling the projected histograms according to their pt hard bins.
        for pt_hard_bin_index in self.selected_iterables["pt_hard_bin"]:
            pt_hard_bin = self.pt_hard_bin[pt_hard_bin_index]
            for _, analysis in \
                    analysis_config.iterate_with_selected_objects(self.analyses, pt_hard_bin = pt_hard_bin_index):
                pt_hard_bin.run(analysis = analysis, average_number_of_events = average_number_of_events)

        # Now merge the scale histograms into the final response matrix results.
        #response_matrix = ResponseMatrix(...)

        for pt_hard_bin_index in self.selected_iterables["pt_hard_bin"]:
            pt_hard_bin = self.pt_hard_bin[pt_hard_bin_index]
            for _, analysis in \
                    analysis_config.iterate_with_selected_objects(self.analyses, pt_hard_bin = pt_hard_bin_index):
                        pass

        return True

        # Test
        #test_object = next(iter(self.analyses.values()))
        #logger.info(f"Attempting to dump obj of type: {type(test_object)}")
        #import ruamel.yaml
        #import sys
        #yaml = ruamel.yaml.YAML(typ = "rt")
        #yaml.register_class(ResponseMatrix)
        #yaml.dump([test_object], sys.stdout)

def run_from_terminal():
    # Basic setup
    logging.basicConfig(level = logging.DEBUG)
    # Quiet down the matplotlib logging
    logging.getLogger("matplotlib").setLevel(logging.INFO)

    # Setup the analysis
    (config_filename, terminal_args, additional_args) = analysis_config.determine_selected_options_from_kwargs(
        task_name = "Response matrix"
    )
    analysis_manager = ResponseManager(
        config_filename = config_filename,
        selected_analysis_options = terminal_args
    )
    # Finally run the analysis.
    analysis_manager.run()

    # Return it for convenience.
    return analysis_manager

if __name__ == "__main__":
    run_from_terminal()
