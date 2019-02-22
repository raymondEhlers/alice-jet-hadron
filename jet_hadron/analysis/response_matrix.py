#!/usr/bin/env python

""" Response matrix dev.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@yale.edu>, Yale University
"""

import coloredlogs
from dataclasses import dataclass
import enlighten
import enum
import logging
import os
import pprint
from typing import Any, Callable, Dict, Iterator, List, Mapping, Tuple, Union

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
from jet_hadron.analysis import response_matrix_helpers

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
        Nothing more is needed at the moment, but we keep it to simplify customization in
        the future.
    """
    ...

# NOTE: Much of this information could be stored immediately with the histogram. This would in
#       fact be more natural. However, we also need the histogram information to be available
#       externally (particularly the attribute name and outliers_removal_axis) so that we can
#       know which hists to provide for outliers removal of a set of histograms. We didn't wrap
#       the histograms themselves because it didn't seem like the extra abstraction was helpful here.
_response_matrix_histogram_info: Dict[str, Union[analysis_objects.HistogramInformation,
                                                 pt_hard_analysis.PtHardHistogramInformation]] = {
    "response_matrix": pt_hard_analysis.PtHardHistogramInformation(
        description = "Response matrix",
        attribute_name = "response_matrix",
        outliers_removal_axis = projectors.TH1AxisType.y_axis
    ),
    "response_matrix_errors": analysis_objects.HistogramInformation(
        description = "Response matrix errors",
        attribute_name = "response_matrix_errors",
    ),
    "particle_level_spectra": analysis_objects.HistogramInformation(
        description = "Particle level spectra",
        attribute_name = "particle_level_spectra",
    ),
}

# Part-, det-level spectra
for name in ["part", "det"]:
    _response_matrix_histogram_info.update({
        f"{name}_level_hists_jet_spectra": pt_hard_analysis.PtHardHistogramInformation(
            description = f"{name.capitalize()}-level matched jet spectra",
            attribute_name = f"{name}_level_hists.jet_spectra",
            outliers_removal_axis = projectors.TH1AxisType.x_axis,
        ),
        f"{name}_level_hists_unmatched_jet_spectra": pt_hard_analysis.PtHardHistogramInformation(
            description = f"{name.capitalize()}-level unmatched jet spectra",
            attribute_name = f"{name}_level_hists.unmatched_jet_spectra",
            outliers_removal_axis = projectors.TH1AxisType.x_axis,
        ),
    })

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
        # Update the centrality range in the input.
        self.input_list_name = self.input_list_name.format(
            cent_min = self.event_activity.value_range.min,
            cent_max = self.event_activity.value_range.max
        )
        # Task settings
        # Default: No additional normalization.
        self.response_normalization = self.task_config.get(
            "response_normalization", response_matrix_helpers.ResponseNormalization.none
        )
        # Validate output filename
        if not self.output_filename.endswith(".root"):
            self.output_filename += ".root"

        # Relevant histograms
        self.response_matrix: Hist = None
        self.response_matrix_errors: Hist = None
        self.particle_level_spectra: Hist = None

        self.part_level_hists: ResponseHistograms = ResponseHistograms(None, None)
        self.det_level_hists: ResponseHistograms = ResponseHistograms(None, None)

    def __iter__(self) -> Iterator[Tuple[str, analysis_objects.HistogramInformation, Union[Hist, None]]]:
        """ Iterate over the histograms in the response matrix analysis object.

        Returns:
            Name under which the ``HistogramInformation`` object is stored, the ``HistogramInformation`` object,
            and the histogram itself.
        """
        for name, histogram_info in _response_matrix_histogram_info.items():
            yield name, histogram_info, utils.recursive_getattr(self, histogram_info.attribute_name)

    def init_hists_from_root_file(self) -> None:
        """ Initialize processed histograms from a ROOT file. """
        # We want to initialize from our saved hists - they will be at the output_prefix.
        filename = os.path.join(self.output_prefix, self.output_filename)
        with histogram.RootOpen(filename = filename, mode = "READ") as f:
            for _, hist_info, _ in self:
                h = f.Get(hist_info.hist_name)
                if not h:
                    h = None
                else:
                    # Detach it from the file so we can store it for later use.
                    h.SetDirectory(0)
                logger.debug(f"Initializing hist {h} to be stored in {hist_info.attribute_name}")
                # Finally, store the histogram
                utils.recursive_setattr(self, hist_info.attribute_name, h)

    def write_hists_to_root_file(self) -> None:
        """ Save processed histograms to a ROOT file. """
        filename = os.path.join(self.output_prefix, self.output_filename)
        # Create the output directory if necessary
        directory_name = os.path.dirname(filename)
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)
        # Then actually iterate through and save the hists.
        with histogram.RootOpen(filename = filename, mode = "RECREATE"):
            for _, hist_info, hist in self:
                # Only write the histogram if it's valid. It's possible that it's still ``None``.
                if hist:
                    logger.debug(f"Writing hist {hist} with name {hist_info.hist_name}")
                    hist.Write(hist_info.hist_name)

    def set_sumw2(self) -> None:
        """ Enable sumw2 on all hists.

        It is enabled automatically in some cases, but it's better to ensure that it is always done
        if it's not enabled.
        """
        for hist in [self.response_matrix, self.response_matrix_errors, self.particle_level_spectra]:
            if hist and not hist.GetSumw2N() > 0:
                logger.debug(f"hist: {hist.GetName()}")
                hist.Sumw2(True)

        for hists in [self.part_level_hists, self.det_level_hists]:
            for _, hist in hists:
                if hist and not hist.GetSumw2N() > 0:
                    logger.debug(f"hist: {hist.GetName()}")
                    hist.Sumw2(True)

    def project_particle_level_spectra(self) -> None:
        """ Project the selected particle level spectra from the response matrix.

        We selected a measured detector level pt range, and just project out the particle
        level jet spectra.
        """
        particle_level_spectra_bin = self.task_config["particle_level_spectra"]["particle_level_spectra_bin"]

        # Setup and run the projector
        particle_level_spectra = ResponseMatrixProjector(
            observable_to_project_from = self.response_matrix,
            output_observable = self,
            output_attribute_name = "particle_level_spectra",
            projection_name_format = "particle_level_spectra",
        )
        # Specify the detector level pt limits
        particle_level_spectra.additional_axis_cuts.append(
            projectors.HistAxisRange(
                axis_type = projectors.TH1AxisType.x_axis,
                axis_range_name = "detector_level_limits",
                min_val = projectors.HistAxisRange.apply_func_to_find_bin(
                    ROOT.TAxis.FindBin, particle_level_spectra_bin.range.min + epsilon,
                ),
                max_val = projectors.HistAxisRange.apply_func_to_find_bin(
                    ROOT.TAxis.FindBin, particle_level_spectra_bin.range.max - epsilon,
                ),
            )
        )
        # No additional cuts for the projection dependent axes
        particle_level_spectra.projection_dependent_cut_axes.append([])
        particle_level_spectra.projection_axes.append(
            projectors.HistAxisRange(
                axis_type = projectors.TH1AxisType.y_axis,
                axis_range_name = "particle_level_spectra",
                min_val = projectors.HistAxisRange.apply_func_to_find_bin(None, 1),
                max_val = projectors.HistAxisRange.apply_func_to_find_bin(ROOT.TAxis.GetNbins),
            )
        )

        # Perform the actual projection
        particle_level_spectra.project()

        # Post projection operations
        # NOTE: Sumw2 is already set.
        # Scale because we projected over 20 1 GeV bins
        projection_range = particle_level_spectra_bin.range.max - particle_level_spectra_bin.range.min
        self.particle_level_spectra.Scale(1.0 / projection_range)

        # Provide as potentially useful information
        logger.debug(f"N jets in particle_level_spectra: {self.particle_level_spectra.Integral()}")

    def particle_level_spectra_processing(self) -> None:
        """ Perform additional (final) processing on the particle level spectra. """
        # For convenience
        particle_level_spectra_config = self.task_config["particle_level_spectra"]
        hist = self.particle_level_spectra
        initial_entries = hist.GetEntries()
        initial_integral = hist.Integral()

        # Rebin to 5 GeV bin width
        rebin_width = particle_level_spectra_config["rebin_width"]
        if rebin_width > 0:
            hist.Rebin(rebin_width)
            # Scale to maintain bin width normalization (which we have had for free up to now
            # because by default the bins are 1 GeV wide).
            hist.Scale(1.0 / rebin_width)

        if particle_level_spectra_config["remove_first_bin"]:
            # Cut below 5 GeV
            # Note that this will modify the overall number of entries
            hist.SetBinContent(1, 0)
            hist.SetBinError(1, 0)

        # Scale by N_{jets}
        # The number of entries should be equal to the number of jets. However, it's not a straightforward
        # number to extract because of all of the scaling related to pt hard bins
        if particle_level_spectra_config["normalize_by_n_jets"]:
            # Integrate over the hist to determine the number of jets displayed.
            # 1e-5 is to ensure we do the integral from [0, 100) (ie not inclusive of the bin beyond 100)
            max_value = particle_level_spectra_config["particle_level_max_pt"] - utils.epsilon
            entries = hist.Integral(hist.FindBin(0), hist.FindBin(max_value))
            logger.debug(f"entries from hist: {hist.GetEntries()}, from integral: {entries}")

            # Normalize the histogram
            hist.Scale(1.0 / entries)

        logger.debug(
            f"Post particle level spectra processing information: ep_orientation: {self.reaction_plane_orientation},"
            f" initial hist entries: {initial_entries}, integral: {initial_integral}"
            f" final hist entries: {hist.GetEntries()}, integral: {hist.Integral()}"
            f" (decrease due to cutting out the 0-5 bin)"
        )

    def create_response_matrix_errors(self) -> Hist:
        """ Create response matrix errors hist from the response matrix hist.

        Args:
            None.
        Returns:
            The newly created response matrix errors histogram.
        Raises:
            ValueError: If the response matrix errors have already been created.
            ValueError: If the response matrix has not yet been created.
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

    def normalize_response_matrix(self):
        """ Normalize the response matrix. """
        response_matrix_helpers.normalize_response_matrix(
            hist = self.response_matrix,
            response_normalization = self.response_normalization
        )

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
        if self.reaction_plane_orientation == params.ReactionPlaneOrientation.inclusive:
            reaction_plane_axis_range = full_axis_range
            logger.debug("Using full EP orientation range")
        else:
            reaction_plane_axis_range = {
                "min_val": projectors.HistAxisRange.apply_func_to_find_bin(
                    None, self.reaction_plane_orientation.value.bin
                ),
                "max_val": projectors.HistAxisRange.apply_func_to_find_bin(
                    None, self.reaction_plane_orientation.value.bin
                )
            }
            logger.debug(f"Using selected EP orientation range {self.reaction_plane_orientation.name}")
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
                    ROOT.TAxis.FindBin, self.leading_hadron_bias.value + epsilon,
                ),
                max_val = projectors.HistAxisRange.apply_func_to_find_bin(
                    ROOT.TAxis.GetNbins
                ),
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
                min_val = projectors.HistAxisRange.apply_func_to_find_bin(
                    ROOT.TAxis.FindBin, self.leading_hadron_bias.value + epsilon,
                ),
                max_val = projectors.HistAxisRange.apply_func_to_find_bin(
                    ROOT.TAxis.GetNbins
                ),
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
                min_val = projectors.HistAxisRange.apply_func_to_find_bin(
                    ROOT.TAxis.FindBin, self.leading_hadron_bias.value + epsilon
                ),
                max_val = projectors.HistAxisRange.apply_func_to_find_bin(
                    ROOT.TAxis.GetNbins
                )
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
        # Base configuration
        self.config_filename = config_filename
        self.selected_analysis_options = selected_analysis_options
        self.task_name = "ResponseManager"
        # Retrieve YAML config for manager configuration
        # NOTE: We don't store the overridden selected_analysis_options because in principle they depend
        #       on the selected task. In practice, such options are unlikely to vary between the manager
        #       and the analysis tasks. However, the validation cannot handle the overridden options
        #       (because the leading hadron bias enum is converting into the object). So we just use
        #       the overridden option in formatting the output prefix (where it is required to determine
        #       the right path), and then passed the non-overridden values to the analysis objects.
        self.config, overridden_selected_analysis_options = analysis_config.read_config_using_selected_options(
            task_name = self.task_name,
            config_filename = self.config_filename,
            selected_analysis_options = self.selected_analysis_options
        )
        # Determine the formatting options needed for the output prefix
        formatting_options = analysis_config.determine_formatting_options(
            task_name = self.task_name, config = self.config,
            selected_analysis_options = overridden_selected_analysis_options
        )
        # Additional helper variables
        self.task_config = self.config[self.task_name]
        self.output_info = analysis_objects.PlottingOutputWrapper(
            # Format to ensure that the selected analysis options are filled in.
            output_prefix = self.config["outputPrefix"].format(**formatting_options),
            printing_extensions = self.config["printingExtensions"],
        )

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
        self.final_pt_hard: Mapping[Any, analysis_objects.JetHBase]
        (self.final_pt_hard_key_index, _, self.final_pt_hard) = \
            self.construct_final_pt_hard_object_from_configuration_file()

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
            additional_classes_to_register = [response_matrix_helpers.ResponseNormalization],
            obj = ResponseMatrix,
        )

    def construct_final_responses_from_configuration_file(self) -> analysis_config.ConstructedObjects:
        """ Construct final ResponseMatrixBase objects based on iterables in a configuration file. """
        return analysis_config.construct_from_configuration_file(
            task_name = "ResponseFinal",
            config_filename = self.config_filename,
            selected_analysis_options = self.selected_analysis_options,
            additional_possible_iterables = {"pt_hard_bin": None, "jet_pt_bin": None},
            additional_classes_to_register = [response_matrix_helpers.ResponseNormalization],
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

    def construct_final_pt_hard_object_from_configuration_file(self) -> analysis_config.ConstructedObjects:
        return analysis_config.construct_from_configuration_file(
            task_name = "PtHardFinal",
            config_filename = self.config_filename,
            selected_analysis_options = self.selected_analysis_options,
            additional_possible_iterables = {"pt_hard_bin": None},
            obj = pt_hard_analysis.PtHardAnalysisBase,
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

    def _run_pt_hard_bin_processing(self, histogram_info_for_processing:
                                    Mapping[str, pt_hard_analysis.PtHardHistogramInformation]) -> None:
        """ Run all pt hard bin related processing.

        Args:
            histogram_info_for_processing: Specifies which histograms to process, and how to do so.
        Returns:
            None.
        """
        # We have to determine the relative scale factors after the setup because they depend on the number of
        # events in all pt hard bins.
        average_number_of_events = pt_hard_analysis.calculate_average_n_events(self.pt_hard_bins)

        # Remove outliers and scale the projected histograms according to their pt hard bins.
        with self.progress_manager.counter(total = len(self.pt_hard_bins),
                                           desc = "Processing:",
                                           unit = "pt hard bins") as processing:
            for pt_hard_key_index, pt_hard_bin in \
                    analysis_config.iterate_with_selected_objects(self.pt_hard_bins):
                # Scale the pt hard spectra
                pt_hard_bin.run(
                    average_number_of_events = average_number_of_events,
                    outliers_removal_axis = projectors.TH1AxisType.x_axis,
                    hists = {"pt_hard_spectra": pt_hard_bin.pt_hard_spectra},
                )

                # We need to perform the outliers removal in EP groups so all EPs get a consistent
                # outlier removal index.
                ep_analyses = {}
                for analysis_key_index, analysis in \
                        analysis_config.iterate_with_selected_objects(
                            self.analyses,
                            pt_hard_bin = pt_hard_key_index.pt_hard_bin
                        ):
                    ep_analyses[analysis_key_index.reaction_plane_orientation] = analysis

                for hist_info in histogram_info_for_processing.values():
                    # We want to process each set hist_info object separately because the group of hists
                    # will share the same cut index.
                    hists = [utils.recursive_getattr(ep_analysis, hist_info.attribute_name)
                             for ep_analysis in ep_analyses.values()]
                    logger.debug(f"attribute_name: {hist_info.attribute_name}, hists: {hists}")
                    pt_hard_bin.run(
                        average_number_of_events = average_number_of_events,
                        outliers_removal_axis = hist_info.outliers_removal_axis,
                        analyses = ep_analyses,
                        hist_attribute_name = hist_info.attribute_name,
                    )

                # Update progress
                processing.update()

        # Now merge the scaled histograms into the final response matrix results.
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
                # This only contains one object, so we just take the first value.
                output_analysis_object = next(iter(self.final_pt_hard.values())),
            )
            # Update the progress
            processing.update()

            # Then the reaction plane dependent quantities
            for reaction_plane_orientation in self.selected_iterables["reaction_plane_orientation"]:
                for hist_info in histogram_info_for_processing.values():
                    pt_hard_analysis.merge_pt_hard_binned_analyses(
                        analyses = analysis_config.iterate_with_selected_objects(
                            self.analyses,
                            reaction_plane_orientation = reaction_plane_orientation
                        ),
                        hist_attribute_name = hist_info.attribute_name,
                        output_analysis_object = self.final_responses[
                            self.final_responses_key_index(reaction_plane_orientation = reaction_plane_orientation)
                        ],
                    )

                # Update progress
                processing.update()

    def _histogram_io(self, label: str, func: Callable[..., None]) -> None:
        """ Helper to handle histogram reading or writing (i/o).

        Args:
            label: Label whether we are reading or writing.
            func: Function to call to perform the reading or writing.
        """
        logger.info(f"{label} pt hard dependent response matrix histograms.")
        for _, analysis in analysis_config.iterate_with_selected_objects(self.analyses):
            func(analysis)
        logger.info(f"{label} final response matrix histograms.")
        for _, analysis in analysis_config.iterate_with_selected_objects(self.final_responses):
            func(analysis)

    def write_response_matrix_histograms(self) -> None:
        """ Write response matrix histograms.

        Note:
            We don't write the pt hard histograms because we generally don't care about them.
            We would write the merged spectra, but it isn't a high priority.
        """
        self._histogram_io(label = "Writing", func = ResponseMatrix.write_hists_to_root_file)

    def read_response_matrix_histograms(self) -> None:
        """ Read processed response matrix histograms from file.

        Note:
            We don't read the pt hard histograms because we generally don't care about them
            after scaling. We would read the merged spectra, but it isn't a high priority.
        """
        self._histogram_io(label = "Reading", func = ResponseMatrix.init_hists_from_root_file)

    def _run_final_processing(self) -> None:
        """ Run final post processing steps. """
        # Final post processing steps
        with self.progress_manager.counter(total = len(self.final_responses),
                                           desc = "Final processing:",
                                           unit = "responses") as processing:
            for _, analysis in analysis_config.iterate_with_selected_objects(self.final_responses):
                # Create the response matrix errors
                analysis.response_matrix_errors = analysis.create_response_matrix_errors()

                # Project the final particle level spectra
                analysis.project_particle_level_spectra()

            # Check and plot the agreement between the sum of the EP selected spectra and the inclusive spectra.
            # NOTE: It's important that this check is performed _before_ normalizing the particle level spectra.
            #       Otherwise, the histograms will trivially disagree because their normalizations are different.
            # NOTE: Technically, we do plotting here, but it's quite minimal, and it's all in the service of a simple
            #       crosscheck of the result, so it's not worth it to store the result in the manager.
            difference, absolute_value_of_difference = \
                self._check_particle_level_spectra_agreement_between_inclusive_and_sum_of_EP_orientations()
            plot_response_matrix.plot_particle_level_spectra_agreement(
                difference = difference,
                absolute_value_of_difference = absolute_value_of_difference,
                output_info = self.output_info,
            )

            for _, analysis in analysis_config.iterate_with_selected_objects(self.final_responses):
                # And perform some post processing.
                analysis.particle_level_spectra_processing()

                # Normalize response (if selected)
                analysis.normalize_response_matrix()

                processing.update()

    def _check_particle_level_spectra_agreement_between_inclusive_and_sum_of_EP_orientations(self) -> Tuple[Hist, Hist]:
        """ Check the agreement of the particle level spectra between the inclusive and sum of all EP orientations.

        This is basically a final crosscheck that this has been brought over from the older RM code base.

        Args:
            None.
        Returns:
            difference histogram, absolute value of difference histogram.
        """
        logger.debug("Comparing sum of EP orientations with sum of EP orientations.")
        # Take all orientations except for the inclusive.
        ep_analyses = {k: v for k, v in self.final_responses.items() if k.reaction_plane_orientation != params.ReactionPlaneOrientation.inclusive}
        inclusive = next(iter(self.final_responses.values()))

        # Sum the particle level spectra from the selected EP orientations.
        # We clone the inclusive hist for convenience. All of the particle level spectra have the same binning.
        sum_particle_level_spectra = inclusive.particle_level_spectra.Clone("sum_particle_level_spectra")
        sum_particle_level_spectra.Reset()
        for _, analysis in ep_analyses.items():
            # Useful debug information so we can keep track of how the summed hist evolves.
            logger.debug(
                f"{analysis.reaction_plane_orientation} hist:"
                f"{analysis.particle_level_spectra.GetEntries()},"
                f" integral: {analysis.particle_level_spectra.Integral()}"
            )
            ROOT.TH1.Add(sum_particle_level_spectra, analysis.particle_level_spectra)
            logger.debug(
                f"Post sum: {sum_particle_level_spectra.GetEntries()},"
                f" integral: {sum_particle_level_spectra.Integral()}"
            )

        # Some useful debug information
        logger.debug(
            f"Inclusive hist:"
            f"{inclusive.particle_level_spectra.GetEntries()},"
            f" integral: {inclusive.particle_level_spectra.Integral()}"
        )

        # Find the difference between the inclusive and selected EP orientations hists.
        difference = sum_particle_level_spectra.Clone("ep_orientations_minus_inclusive")
        difference.SetTitle("(Sum of EP selected orientations) - inclusive")
        ROOT.TH1.Add(difference, inclusive.particle_level_spectra, -1)
        # Take absolute value so we can look at the log
        absolute_value_of_difference = difference.Clone(f"{difference.GetName()}_abs")
        absolute_value_of_difference.SetTitle(f"{difference.GetTitle()} (absolute value)")
        for i in range(0, absolute_value_of_difference.GetNcells()):
            absolute_value_of_difference.SetBinContent(i, abs(absolute_value_of_difference.GetBinContent(i)))

        return difference, absolute_value_of_difference

    def _plot_results(self, histogram_info_for_processing:
                      Mapping[str, pt_hard_analysis.PtHardHistogramInformation]) -> None:
        """ Plot the results of the response matrix processing.

        Args:
            histogram_info_for_processing: Specifies which histograms to process, and how to do so.
        Returns:
            None.
        """
        # Counting of plots:
        # +1 for the final pt hard spectra.
        # +1 for particle level spectra
        # *2 for response matrix, response spectra
        with self.progress_manager.counter(total = 2 * len(self.selected_iterables["reaction_plane_orientation"]) + 1 + 1,
                                           desc = "Plotting:",
                                           unit = "responses") as plotting:

            # Plot pt hard spectra
            # Pull out the dict because we need to know the length of the objects,
            # which isn't provided from a generator.
            pt_hard_analyses = dict(
                analysis_config.iterate_with_selected_objects(self.pt_hard_bins)
            )
            # This only contains one object, so we just take the first value.
            merged_pt_hard_analysis = next(iter(self.final_pt_hard.values()))
            # If we initialize the histograms from file then we won't have the pt hard spectra.
            # In that case, we just skip trying to plot them.
            if hasattr(merged_pt_hard_analysis, "pt_hard_spectra"):

                for plot_with_ROOT in [False, True]:
                    plot_response_matrix.plot_response_spectra(
                        plot_labels = plot_base.PlotLabels(
                            title = r"$\mathit{p}_{\mathrm{T}}$ hard spectra",
                            x_label = r"$\mathit{p}_{\mathrm{T}}^{\mathrm{hard}}$",
                            y_label = r"$\frac{dN}{d\mathit{p}_{\mathrm{T}}}$",
                        ),
                        output_name = "pt_hard_spectra",
                        merged_analysis = merged_pt_hard_analysis,
                        pt_hard_analyses = pt_hard_analyses,
                        hist_attribute_name = "pt_hard_spectra",
                        plot_with_ROOT = plot_with_ROOT,
                    )
            else:
                logger.info("Skip plotting of pt hard spectra because the hists aren't initialized.")

            plotting.update()

            # Plot the particle level spectra.
            for plot_with_ROOT in [False, True]:
                plot_response_matrix.plot_particle_level_spectra(
                    ep_analyses = analysis_config.iterate_with_selected_objects(self.final_responses),
                    output_info = self.output_info,
                    plot_with_ROOT = plot_with_ROOT,
                )
            plotting.update()

            for reaction_plane_orientation in self.selected_iterables["reaction_plane_orientation"]:
                # Plot response matrix and errors
                for plot_with_ROOT in [False, True]:
                    plot_response_matrix.plot_response_matrix_and_errors(
                        obj = self.final_responses[
                            self.final_responses_key_index(reaction_plane_orientation)
                        ],
                        plot_with_ROOT = plot_with_ROOT,
                    )
                plotting.update()

                # Pull out the dict because we need to know the length of the objects,
                # which isn't provided from a generator.
                analyses = dict(
                    analysis_config.iterate_with_selected_objects(
                        self.analyses,
                        reaction_plane_orientation = reaction_plane_orientation
                    )
                )
                # Plot part, det level match and unmatched (so skip the response matrix)
                for plot_with_ROOT in [False, True]:
                    for hist_info in list(histogram_info_for_processing.values())[1:]:
                        # This is just a proxy to get "part" or "det"
                        base_label = hist_info.description[:hist_info.attribute_name.find("_")].lower()
                        # This will be something like "unmatched_jet_spectra"
                        # +1 is to skip the "." that we found.
                        output_label = hist_info.attribute_name[hist_info.attribute_name.find(".") + 1:]
                        plot_response_matrix.plot_response_spectra(
                            plot_labels = plot_base.PlotLabels(
                                title = hist_info.description,
                                x_label = r"$\mathit{p}_{\mathrm{T,jet}}^{\mathrm{%(label)s}}$" % {
                                    "label": base_label,
                                },
                                y_label = r"$\frac{dN}{d\mathit{p}_{\mathrm{T}}}$",
                            ),
                            output_name = f"{base_label}_level_{output_label}",
                            merged_analysis = self.final_responses[
                                self.final_responses_key_index(reaction_plane_orientation)
                            ],
                            pt_hard_analyses = analyses,
                            hist_attribute_name = hist_info.attribute_name,
                            plot_with_ROOT = plot_with_ROOT,
                        )

                # Update progress
                plotting.update()

    def _package_and_write_final_responses(self) -> None:
        """ Package up and write the final repsonse matrices. """
        output_filename = os.path.join(self.output_info.output_prefix, "final_responses.root")
        with histogram.RootOpen(output_filename, mode = "RECREATE"):
            for _, analysis in analysis_config.iterate_with_selected_objects(self.final_responses):
                hist = analysis.response_matrix.Clone(f"{analysis.response_matrix.GetName()}_{analysis.reaction_plane_orientation}")
                hist.Write()

    def run(self) -> bool:
        """ Run the response matrix analyses. """
        logger.debug(f"key_index: {self.key_index}, selected_option_names: {list(self.selected_iterables)}, analyses: {pprint.pformat(self.analyses)}")

        # We need to determine the input information
        histogram_info_for_processing: Dict[str, pt_hard_analysis.PtHardHistogramInformation] = {}
        for name, info in _response_matrix_histogram_info.items():
            if name not in ["response_matrix_errors", "particle_level_spectra"]:
                # Help out mypy...
                assert isinstance(info, pt_hard_analysis.PtHardHistogramInformation)
                histogram_info_for_processing[name] = info

        # Analysis steps:
        # 1. Setup response matrix and pt hard objects.
        # 2. Pt hard bin outliers removal, scaling, and merging into final response objects.
        # 3. Write histograms (or read histograms if requested).
        # 4. Final processing, including projecting particle level spectra.
        # 5. Plotting.
        # 6. Writing final hists together to a single final.
        #
        # If we are starting from reading histograms, we start from step 3.
        steps = 4 if self.task_config["read_hists_from_root_file"] else 6
        with self.progress_manager.counter(total = steps,
                                           desc = "Overall processing progress:",
                                           unit = "") as overall_progress:
            # We only need to perform the projecting and pt hard dependent part of the analysis
            # if we don't already have the histograms saved.
            if self.task_config["read_hists_from_root_file"]:
                self.read_response_matrix_histograms()
                overall_progress.update()
            else:
                # Setup the response matrix and pt hard analysis objects.
                self.setup()
                overall_progress.update()

                # Run all pt hard related processing, including outliers removal, scaling, and merging hists.
                self._run_pt_hard_bin_processing(
                    histogram_info_for_processing = histogram_info_for_processing,
                )
                overall_progress.update()

                self.write_response_matrix_histograms()
                overall_progress.update()

            # Final post processing steps.
            # This processing is performed after reading or writing the histograms because there are a
            # number of parameters here which could change. Instead of trying to deal with detecting when
            # is the proper time to reprocess, we just always reprocess.
            self._run_final_processing()
            overall_progress.update()

            # Plot the results
            self._plot_results(
                histogram_info_for_processing = histogram_info_for_processing,
            )
            overall_progress.update()

            # Package up all of the responses in one file.
            self._package_and_write_final_responses()
            overall_progress.update()

        return True

def run_from_terminal():
    # Basic setup
    # This replaces ``logging.basicConfig(...)``.
    coloredlogs.install(
        level = logging.DEBUG,
        fmt = "%(asctime)s %(name)s:%(lineno)d %(levelname)s %(message)s"
    )
    # Quiet down the matplotlib logging
    logging.getLogger("matplotlib").setLevel(logging.INFO)
    # Quiet down some pachyderm modules
    logging.getLogger("pachyderm.generic_config").setLevel(logging.INFO)
    logging.getLogger("pachyderm.histogram").setLevel(logging.INFO)
    # Turn off stats box
    ROOT.gStyle.SetOptStat(0)

    # Setup the analysis
    (config_filename, terminal_args, additional_args) = analysis_config.determine_selected_options_from_kwargs(
        task_name = "Response matrix"
    )
    selected_analysis_options = analysis_config.validate_arguments(selected_args = terminal_args)
    analysis_manager = ResponseManager(
        config_filename = config_filename,
        selected_analysis_options = selected_analysis_options,
    )
    # Finally run the analysis.
    analysis_manager.run()

    # Return it for convenience.
    return analysis_manager

if __name__ == "__main__":
    run_from_terminal()
