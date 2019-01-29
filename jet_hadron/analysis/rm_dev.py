#!/usr/bin/env python

""" Response matrix dev.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@yale.edu>, Yale University
"""

import coloredlogs
from dataclasses import dataclass
import enlighten
import enum
import logging
import pprint
from typing import Any, Dict, Iterator, List, Mapping, Tuple

from pachyderm import generic_class
from pachyderm import histogram
from pachyderm import projectors
from pachyderm import utils
from pachyderm.utils import epsilon

from jet_hadron.base import analysis_config
from jet_hadron.base import analysis_objects
from jet_hadron.base import params
from jet_hadron.base.typing_helpers import Hist
from jet_hadron.plot import base as plot_base
from jet_hadron.plot import response_matrix as plot_response_matrix

from jet_hadron.analysis import pt_hard_analysis

import ROOT

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
    """ Projector for the Jet-h response matrix THnSparse.

    Note:
        Nothing more is needed at the moment, but we keep it to simpify customization in
        the future.
    """
    ...

@dataclass
class ResponseHistograms:
    """ The main histograms for a response matrix. """
    jet_spectra: Hist
    unmatched_jet_spectra: Hist
    #sample_task_jet_spectra: Hist

    def __iter__(self) -> Iterator[Tuple[str, Hist]]:
        for k, v in vars(self).items():
            yield k, v

class ResponseMatrixBase(analysis_objects.JetHReactionPlane):
    """ Base response matrix class.

    Stores the response matrix histograms. Often used for final response matrix histograms
    after the intermediate steps are fully projected.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Relevant histograms
        self.response_matrix: Hist = None
        self.response_matrix_errors: Hist = None

        self.part_level_hists: ResponseHistograms = ResponseHistograms(None, None)
        self.det_level_hists: ResponseHistograms = ResponseHistograms(None, None)

    def set_sumw2(self) -> None:
        """ Enable sumw2 on all hists.

        It is enabled automatically in some cases, but it's better to ensure that it is always done
        if it's not enabled.
        """
        for hist in [self.response_matrix, self.response_matrix_errors]:
            if hist and not hist.GetSumw2N() > 0:
                logger.debug(f"hist: {hist.GetName()}")
                hist.Sumw2()

        for hists in [self.part_level_hists, self.det_level_hists]:
            for _, hist in hists:
                if hist and not hist.GetSumw2N() > 0:
                    logger.debug(f"hist: {hist.GetName()}")
                    hist.Sumw2()

    def create_response_matrix_errors(self) -> Hist:
        """ Create response matrix errors hist from the response matrix hist.

        Args:
            None.
        Returns:
            The newly created response matrix errors histogram.
        """
        # Validation
        if self.response_matrix_errors:
            raise ValueError("Response matrix errors has already been created!")
        if not self.response_matrix:
            raise ValueError("Must create the response matrix first.")

        # Get response matrix for convenience
        response_matrix = self.response_matrix

        # Clone response matrix so that it automatically has the same limits
        response_matrix_errors = response_matrix.Clone("responseMatrixErrors")
        # Reset so that we can fill it with the errors
        response_matrix_errors.Reset()

        # Fill response matrix errors
        # We don't fill in the overflow bins - they are rather irrelevant for the errors.
        # NOTE: Careful with GetXaxis().GetFirst() -> The range can be restricted by SetRange()
        for x in range(1, response_matrix.GetXaxis().GetNbins() + 1):
            for y in range(1, response_matrix.GetYaxis().GetNbins() + 1):
                fill_value = response_matrix.GetBinError(x, y)
                #if fill_value > 1:
                #    logger.debug(
                #        f"Error > 1 before scaling: {fill_value},"
                #        f" bin content: {response_matrix.GetBinContent(x, y)},"
                #        f" bin error: {response_matrix.GetBinError(x, y)}, ({x}, {y})"
                #    )
                if response_matrix.GetBinContent(x, y) > 0:
                    if response_matrix.GetBinContent(x, y) < response_matrix.GetBinError(x, y):
                        raise ValueError(
                            "Bin content < bin error."
                            f" Bin content: {response_matrix.GetBinContent(x, y)},"
                            f" bin error: {response_matrix.GetBinError(x, y)}, ({x}, {y})"
                        )
                    fill_value = fill_value / response_matrix.GetBinContent(x, y)
                else:
                    if response_matrix.GetBinError(x, y) > epsilon:
                        logger.warning(
                            "No bin content, but associated error is non-zero."
                            f" Content: {response_matrix.GetBinContent(x, y)},"
                            f" error: {response_matrix.GetBinError(x, y)}"
                        )
                if fill_value > 1:
                    logger.error(
                        f"Error > 1 after scaling: {fill_value},"
                        f" bin content: {response_matrix.GetBinContent(x, y)},"
                        f" bin error: {response_matrix.GetBinError(x, y)}, ({x}, {y})"
                    )

                # Fill hist
                bin_number = response_matrix_errors.Fill(
                    response_matrix_errors.GetXaxis().GetBinCenter(x),
                    response_matrix_errors.GetYaxis().GetBinCenter(y),
                    fill_value
                )

                # Check to ensure that we filled where we expected
                if bin_number != response_matrix_errors.GetBin(x, y):
                    raise ValueError(
                        f"Mismatch between fill bin number ({bin_number})"
                        f" and GetBin() ({response_matrix_errors.GetBin(x, y)})"
                    )

        return response_matrix_errors

class ResponseMatrix(ResponseMatrixBase):
    """ Main response matrix class.

    Stores the response matrix histograms, as well as the methods to process the response matrix.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Pt hard bins are optional.
        self.pt_hard_bin = kwargs.get("pt_hard_bin", None)
        if self.pt_hard_bin:
            self.train_number = self.pt_hard_bin.train_number
            self.input_filename = self.input_filename.format(pt_hard_bin_train_number = self.train_number)
            self.output_prefix = self.output_prefix.format(pt_hard_bin_train_number = self.train_number)

        # Basic information
        self.input_hists: Dict[str, Any] = {}
        self.projectors: List[ResponseMatrixProjector] = []

    def _setup_projectors(self) -> None:
        """ Setup the sparse projectors. """
        # Helper range
        full_axis_range = {
            "min_val": projectors.HistAxisRange.apply_func_to_find_bin(None, 1),
            "max_val": projectors.HistAxisRange.apply_func_to_find_bin(ROOT.TAxis.GetNbins)
        }
        # Reaction plane selection
        if self.reaction_plane_orientation == params.ReactionPlaneOrientation.all:
            reaction_plane_axis_range = full_axis_range
            logger.debug("Using full EP angle range")
        else:
            reaction_plane_axis_range = {
                "min_val": projectors.HistAxisRange.apply_func_to_find_bin(
                    None, self.reaction_plane_orientation.value.bin
                ),
                "max_val": projectors.HistAxisRange.apply_func_to_find_bin(
                    None, self.reaction_plane_orientation.value.bin
                )
            }
            logger.debug(f"Using selected EP angle range {self.reaction_plane_orientation.name}")
        reaction_plane_orientation_projector_axis = projectors.HistAxisRange(
            axis_type = ResponseMakerMatchingSparse.det_level_reaction_plane_orientation,
            axis_range_name = "detLevelReactionPlaneOrientation",
            **reaction_plane_axis_range
        )

        #################
        # Response matrix
        #################
        response_matrix = ResponseMatrixProjector(
            observable_to_project_from = self.input_hists["fHistMatching"],
            output_observable = self,
            output_attribute_name = "response_matrix",
            projection_name_format = "responseMatrix",
        )
        response_matrix.additional_axis_cuts.append(
            projectors.HistAxisRange(
                axis_type = ResponseMakerMatchingSparse.det_level_leading_particle,
                axis_range_name = "detLevelLeadingParticle",
                min_val = projectors.HistAxisRange.apply_func_to_find_bin(
                    ROOT.TAxis.FindBin, self.leading_hadron_bias.value
                ),
                max_val = projectors.HistAxisRange.apply_func_to_find_bin(
                    ROOT.TAxis.GetNbins
                )
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

        #############################
        # Unmatched part level jet pt
        #############################
        unmatched_part_level_jet_spectra = ResponseMatrixProjector(
            observable_to_project_from = self.input_hists["fHistJets2"],
            output_observable = self.part_level_hists,
            output_attribute_name = "unmatched_jet_spectra",
            projection_name_format = "unmatchedJetSpectraPartLevel",
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

        #############################
        # (Matched) Part level jet pt
        #############################
        part_level_jet_spectra = ResponseMatrixProjector(
            observable_to_project_from = self.input_hists["fHistMatching"],
            output_observable = self.part_level_hists,
            output_attribute_name = "jet_spectra",
            projection_name_format = "jetSpectraPartLevel",
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

        ############################
        # Unmatched det level jet pt
        ############################
        unmatched_det_level_jet_spectra = ResponseMatrixProjector(
            observable_to_project_from = self.input_hists["fHistJets1"],
            output_observable = self.det_level_hists,
            output_attribute_name = "unmatched_jet_spectra",
            projection_name_format = "unmatchedJetSpectraDetLevel",
        )

        # The leading particle axis varies depending on whether the event plane is included in the sparse.
        leading_particle_axis = ResponseMakerJetsSparse.leading_particle_PP
        if self.collision_system in [params.CollisionSystem.PbPb, params.CollisionSystem.embedPythia, params.CollisionSystem.embedPP]:
            leading_particle_axis = ResponseMakerJetsSparse.leading_particle_PbPb
        unmatched_det_level_jet_spectra.additional_axis_cuts.append(
            projectors.HistAxisRange(
                axis_type = leading_particle_axis,
                axis_range_name = "unmatchedDetLevelLeadingParticle",
                min_val = projectors.HistAxisRange.apply_func_to_find_bin(ROOT.TAxis.FindBin, self.leading_hadron_bias.value),
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

        ############################
        # (Matched) Det level jet pt
        ############################
        det_level_jet_spectra = ResponseMatrixProjector(
            observable_to_project_from = self.input_hists["fHistMatching"],
            output_observable = self.det_level_hists,
            output_attribute_name = "jet_spectra",
            projection_name_format = "jetSpectraDetLevel",
        )
        det_level_jet_spectra.additional_axis_cuts.append(
            projectors.HistAxisRange(
                axis_type = ResponseMakerMatchingSparse.det_level_leading_particle,
                axis_range_name = "detLevelLeadingParticle",
                min_val = projectors.HistAxisRange.apply_func_to_find_bin(ROOT.TAxis.FindBin, self.leading_hadron_bias.value),
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

    def run_projectors(self) -> None:
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
        pt_hard_bins: Pt hard analysis objects for pt hard binned analyses (optional).
    """
    def __init__(self, config_filename: str, selected_analysis_options: params.SelectedAnalysisOptions, **kwargs):
        self.config_filename = config_filename
        self.selected_analysis_options = selected_analysis_options

        # Create the actual response matrix objects.
        self.analyses: Mapping[Any, ResponseMatrix]
        (self.key_index, self.selected_iterables, self.analyses) = self.construct_responses_from_configuration_file()
        # Create the final response matrix objects.
        self.final_responses: Mapping[Any, ResponseMatrixBase]
        (self.final_responses_key_index, final_responses_selected_iterables, self.final_responses) = \
            self.construct_final_responses_from_configuration_file()

        # Validate that we have the same reaction plane iterables
        if not self.selected_iterables["reaction_plane_orientation"] == \
                final_responses_selected_iterables["reaction_plane_orientation"]:
                    raise ValueError(
                        "Selected iterables for reaction plane orientations in the final response matrix objects differ"
                        " from the reaction plane orientations for analysis response matrix objects."
                        f" Selected iterables: {self.selected_iterables['reaction_plane_orientation']},"
                        f" final responses iterables: {final_responses_selected_iterables}"
                    )

        # Create the pt hard bins
        self.pt_hard_bins: Mapping[Any, pt_hard_analysis.PtHardAnalysis]
        (self.pt_hard_bins_key_index, pt_hard_iterables, self.pt_hard_bins) = \
            self.construct_pt_hard_bins_from_configuration_file()
        # Create the final pt hard spectra
        self.pt_hard_spectra: Hist

        # Validate that we have the same pt hard iterables.
        if not self.selected_iterables["pt_hard_bin"] == pt_hard_iterables["pt_hard_bin"]:
            raise ValueError(
                "Selected iterables pt hard bins differ from the pt hard bins of the pt hard bin analysis objects."
                f" Selected iterables: {self.selected_iterables['pt_hard_bins']},"
                f" pt hard analysis iterables: {pt_hard_iterables}"
            )

        # Monitor the progress of the analysis.
        self.progress_manager = enlighten.get_manager()

    def construct_responses_from_configuration_file(self) -> analysis_config.ConstructedObjects:
        """ Construct ResponseMatrix objects based on iterables in a configuration file. """
        return analysis_config.construct_from_configuration_file(
            task_name = "Response",
            config_filename = self.config_filename,
            selected_analysis_options = self.selected_analysis_options,
            additional_possible_iterables = {"pt_hard_bin": None, "jet_pt_bin": None},
            obj = ResponseMatrix,
        )

    def construct_final_responses_from_configuration_file(self) -> analysis_config.ConstructedObjects:
        """ Construct final ResponseMatrixBase objects based on iterables in a configuration file. """
        return analysis_config.construct_from_configuration_file(
            task_name = "ResponseFinal",
            config_filename = self.config_filename,
            selected_analysis_options = self.selected_analysis_options,
            additional_possible_iterables = {"pt_hard_bin": None, "jet_pt_bin": None},
            obj = ResponseMatrixBase,
        )

    def construct_pt_hard_bins_from_configuration_file(self) -> analysis_config.ConstructedObjects:
        """ Construct PtHardAnalysis objects based on iterables in a configuration file. """
        return analysis_config.construct_from_configuration_file(
            task_name = "PtHardBins",
            config_filename = self.config_filename,
            selected_analysis_options = self.selected_analysis_options,
            additional_possible_iterables = {"pt_hard_bin": None},
            obj = pt_hard_analysis.PtHardAnalysis,
        )

    def setup(self) -> None:
        """ Setup and prepare the analysis objects. """
        # Cache input hists so we can avoid repeatedly opening files
        input_hists: Dict[Any, Dict[str, Any]] = {}

        # Setup the response matrix analysis objects and run the response matrix projectors
        # By the time that this step is complete, we should have all histograms.
        with self.progress_manager.counter(total = len(self.analyses),
                                           desc = "Configuring and projecting:",
                                           unit = "responses") as setting_up:
            for pt_hard_bin in self.selected_iterables["pt_hard_bin"]:
                logger.debug(f"pt_hard_bin: {pt_hard_bin}")
                input_hists[pt_hard_bin] = {}
                for key_index, analysis in \
                        analysis_config.iterate_with_selected_objects(self.analyses, pt_hard_bin = pt_hard_bin):
                    # We should now have all RP orientations.
                    # We are caching the values here to minimize opening files.
                    if not input_hists[pt_hard_bin]:
                        input_hists[pt_hard_bin] = histogram.get_histograms_in_file(filename = analysis.input_filename)
                    logger.debug(f"key_index: {key_index}")
                    result = analysis.setup(input_hists = input_hists[pt_hard_bin])
                    if result is not True:
                        raise ValueError(f"Setup of {key_index} analysis object failed.")
                    analysis.run_projectors()

                    # Ensure that all hists have sumw2 enabled
                    analysis.set_sumw2()

                    # Update progress
                    setting_up.update()

        # Setup the pt hard bin analysis objects.
        with self.progress_manager.counter(total = len(self.pt_hard_bins),
                                           desc = "Setting up: ",
                                           unit = "pt hard bins") as setting_up:
            for key_index, pt_hard_bin in analysis_config.iterate_with_selected_objects(self.pt_hard_bins):
                pt_hard_bin.setup(input_hists = input_hists[key_index.pt_hard_bin])

                # Update progress
                setting_up.update()

    def run(self) -> bool:
        """ Run the response matrix analyses. """
        logger.debug(f"key_index: {self.key_index}, selected_option_names: {list(self.selected_iterables)}, analyses: {pprint.pformat(self.analyses)}")

        # Setup the response matrix and pt hard analysis objects.
        self.setup()

        # We have to determine the relative scale factors after the setup because they depend on the number of
        # events in all pt hard bins.
        average_number_of_events = pt_hard_analysis.calculate_average_n_events(self.pt_hard_bins)

        # Finally, remove outliers and scale the projected histograms according to their pt hard bins.
        # First, we determine the input information
        @dataclass
        class InputInfo:
            """ Helper class to store information about processing an analysis object.

            This basically just stores information in a nicely formatted and clear manner.

            Attributes:
                hist_attribute_name: Name of the attribute under which the hist is stored in the analysis object.
                outliers_removal_axis: Projection axis for the particle level used in outliers removal.
            """
            name: str
            hist_attribute_name: str
            outliers_removal_axis: projectors.TH1AxisType

        analysis_object_info = [
            # Main response hists
            InputInfo(
                name = "Response matrix",
                hist_attribute_name = "response_matrix",
                outliers_removal_axis = projectors.TH1AxisType.y_axis
            ),
            # We don't need to create the response matrix errors histogram at this point.
            # Instead, we perform the scaling and merging first, and then create it afterwards.
        ]
        # Part-, det-level spectra
        for name in ["part", "det"]:
            analysis_object_info.extend([
                InputInfo(
                    name = f"{name.capitalize()}-level matched jet spectra",
                    hist_attribute_name = f"{name}_level_hists.jet_spectra",
                    outliers_removal_axis = projectors.TH1AxisType.x_axis,
                ),
                InputInfo(
                    name = f"{name.capitalize()}-level unmatched jet spectra",
                    hist_attribute_name = f"{name}_level_hists.unmatched_jet_spectra",
                    outliers_removal_axis = projectors.TH1AxisType.x_axis,
                ),
                #InputInfo(
                #    type = name,
                #    name = f"{name.capitalize()}-level sample task spectra",
                #    hist_attribute_name = f"{name}_level_hists.sample_task_jet_spectra",
                #    outliers_removal_axis = projectors.TH1AxisType.x_axis
                #),
            ])

        # Now, perform the actual outliers removal and scaling.
        with self.progress_manager.counter(total = len(self.pt_hard_bins),
                                           desc = "Processing:",
                                           unit = "pt hard bins") as processing:
            for pt_hard_key_index, pt_hard_bin in \
                    analysis_config.iterate_with_selected_objects(self.pt_hard_bins):
                # We need to perform the outliers removal in EP groups.
                ep_analyses = {}
                for analysis_key_index, analysis in \
                        analysis_config.iterate_with_selected_objects(self.analyses, pt_hard_bin = pt_hard_key_index.pt_hard_bin):
                    ep_analyses[analysis_key_index.reaction_plane_orientation] = analysis

                for analysis_input in analysis_object_info:
                    hists = [utils.recursive_getattr(ep_analysis, analysis_input.hist_attribute_name) for ep_analysis in ep_analyses.values()]
                    logger.debug(f"hist_attribute_name: {analysis_input.hist_attribute_name}, hists: {hists}")
                    pt_hard_bin.run(
                        average_number_of_events = average_number_of_events,
                        outliers_removal_axis = analysis_input.outliers_removal_axis,
                        analyses = ep_analyses,
                        hist_attribute_name = analysis_input.hist_attribute_name,
                    )

                # Update progress
                processing.update()

        # Now merge the scale histograms into the final response matrix results.
        # +1 for the final pt hard spectra.
        with self.progress_manager.counter(total = len(self.selected_iterables["reaction_plane_orientation"]) + 1,
                                           desc = "Projecting:",
                                           unit = "EP dependent final responses") as processing:
            # First merge the pt hard bin quantities.
            pt_hard_analysis.merge_pt_hard_binned_analyses(
                analyses = analysis_config.iterate_with_selected_objects(
                    self.pt_hard_bins,
                ),
                hist_attribute_name = "pt_hard_spectra",
                output_analysis_object = self,
            )
            # Update the progress
            processing.update()

            # Then the reaction plane dependent quantities
            for reaction_plane_orientation in self.selected_iterables["reaction_plane_orientation"]:
                for analysis_input in analysis_object_info:
                    pt_hard_analysis.merge_pt_hard_binned_analyses(
                        analyses = analysis_config.iterate_with_selected_objects(
                            self.analyses,
                            reaction_plane_orientation = reaction_plane_orientation
                        ),
                        hist_attribute_name = analysis_input.hist_attribute_name,
                        output_analysis_object = self.final_responses[
                            self.final_responses_key_index(reaction_plane_orientation = reaction_plane_orientation)
                        ],
                    )

                # Update progress
                processing.update()

        # TODO: Project the final particle level spectra

        # TEMP
        example_hists = [r.response_matrix for r in self.final_responses.values()]
        logger.debug(f"pt_hard_spectra: {self.pt_hard_spectra}, final_responses: {self.final_responses}, response: {example_hists}")
        # ENDTEMP

        # Now plot the histograms
        # +1 for the final pt hard spectra.
        with self.progress_manager.counter(total = len(self.selected_iterables["reaction_plane_orientation"]) + 1,
                                           desc = "Plotting:",
                                           unit = "responses") as plotting:
            # Plot pt hard spectra
            # Pull out the dict because we need to know the length of the objects,
            # which isn't provided from a generator.
            pt_hard_analyses = dict(
                analysis_config.iterate_with_selected_objects(self.pt_hard_bins)
            )
            plot_response_matrix.plot_response_spectra(
                plot_labels = plot_base.PlotLabels(
                    title = r"$\mathit{p}_{\mathrm{T}}$ hard spectra",
                    x_label = r"$\mathit{p}_{\mathrm{T}}^{hard}$",
                    y_label = r"$\frac{dN}{d\mathit{p}_{\mathrm{T}}}$",
                ),
                output_name = "pt_hard_spectra",
                # TODO: This must be a JetHBase derived object.
                merged_analysis = self,
                pt_hard_analyses = pt_hard_analyses,
                hist_attribute_name = "pt_hard_spectra",
            )

            for reaction_plane_orientation in self.selected_iterables["reaction_plane_orientation"]:
                # Pull out the dict because we need to know the length of the objects,
                # which isn't provided from a generator.
                analyses = dict(
                    analysis_config.iterate_with_selected_objects(
                        self.analyses,
                        reaction_plane_orientation = reaction_plane_orientation
                    )
                )
                # Plot part, det level match and unmatched
                for analysis_input in analysis_object_info[1:]:
                    # This is just a proxy to get "part" or "det"
                    base_label = analysis_input.name[:analysis_input.hist_attribute_name.find("_")].lower()
                    # This will be something like "unmatched_jet_spectra"
                    output_label = analysis_input.hist_attribute_name[analysis_input.hist_attribute_name.find("."):]
                    plot_response_matrix.plot_response_spectra(
                        plot_labels = plot_base.PlotLabels(
                            title = analysis_input.name,
                            x_label = r"$\mathit{p}_{\mathrm{T,jet}}^{%(label)s}$" % {
                                "label": base_label,
                            },
                            y_label = r"$\frac{dN}{d\mathit{p}_{\mathrm{T}}}$",
                        ),
                        output_name = f"{base_label}_level_{output_label}",
                        merged_analysis = self.final_responses[
                            self.final_responses_key_index(reaction_plane_orientation)
                        ],
                        pt_hard_analyses = analyses,
                        hist_attribute_name = analysis_input.hist_attribute_name,
                    )

                # Update progress
                plotting.update()

        return True

        # TEMP
        #test_object = next(iter(self.analyses.values()))
        #logger.info(f"Attempting to dump obj of type: {type(test_object)}")
        #import ruamel.yaml
        #import sys
        #yaml = ruamel.yaml.YAML(typ = "rt")
        #yaml.register_class(ResponseMatrix)
        #yaml.dump([test_object], sys.stdout)
        # ENDTEMP

def run_from_terminal():
    # Basic setup
    # This replaces ``logging.basicConfig(...)``.
    coloredlogs.install(
        level = logging.DEBUG,
        fmt = "%(asctime)s %(name)s:%(lineno)d %(levelname)s %(message)s"
    )
    # Quiet down the matplotlib logging
    logging.getLogger("matplotlib").setLevel(logging.INFO)
    # Quiet down pachyderm generic config
    logging.getLogger("pachyderm.generic_config").setLevel(logging.INFO)

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
