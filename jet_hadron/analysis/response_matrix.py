#!/usr/bin/env python

""" Jet-hardon response matrix.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@yale.edu>, Yale University
"""

import copy
import ctypes
from dataclasses import dataclass
import enum
import logging
import numpy as np
import os
import pprint
import sys
from typing import Any, Callable, Dict, Iterator, List, Mapping, Tuple, Type, Union

# NOTE: This is out of the expected order, but it must be here to prevent ROOT from stealing the command
#       line options
from jet_hadron.base.typing_helpers import Hist

from pachyderm import histogram
from pachyderm import projectors
from pachyderm import utils
from pachyderm.utils import epsilon

from jet_hadron.base import analysis_config
from jet_hadron.base import analysis_manager
from jet_hadron.base import analysis_objects
from jet_hadron.base import labels
from jet_hadron.base import params
from jet_hadron.plot import base as plot_base
from jet_hadron.plot import response_matrix as plot_response_matrix

from jet_hadron.analysis import pt_hard_analysis
from jet_hadron.analysis import response_matrix_helpers

import ROOT

logger = logging.getLogger(__name__)
this_module = sys.modules[__name__]

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

class JetHPerformanceResponseSparse(enum.Enum):
    """ Define the axes of the JetHPerformance response matrix sparse. """
    det_level_jet_pt = 0
    part_level_jet_pt = 1
    det_level_jet_area = 2
    part_level_jet_area = 3
    matching_distance = 4
    det_level_leading_particle = 5
    part_level_leading_particle = 6
    det_level_reaction_plane_orientation = 7
    part_level_reaction_plane_orientation = 8
    centrality = 9

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
# QA
_response_matrix_histogram_info["matched_jet_pt_difference"] = pt_hard_analysis.PtHardHistogramInformation(
    description = "Matched jet pt difference for hybrid and det level",
    attribute_name = "matched_jet_pt_difference",
    outliers_removal_axis = projectors.TH1AxisType.x_axis
)
_response_matrix_histogram_info["hybrid_level_spectra"] = pt_hard_analysis.PtHardHistogramInformation(
    description = "Hybrid level spectra",
    attribute_name = "hybrid_level_spectra",
    outliers_removal_axis = projectors.TH1AxisType.x_axis,
)

@dataclass
class ResponseHistograms:
    """ The main histograms for a response matrix. """
    jet_spectra: Hist
    unmatched_jet_spectra: Hist

    def __iter__(self) -> Iterator[Tuple[str, Hist]]:
        for k, v in vars(self).items():
            yield k, v

class ResponseMatrixBase(analysis_objects.JetHReactionPlane):
    """ Base response matrix class.

    Stores the response matrix histograms. Often used for final response matrix histograms
    after the intermediate steps are fully projected.
    """
    def __init__(self, *args: Any, **kwargs: Any):
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

        # QA hists
        # Difference between hybrid and detector level jet pt to characterize jet energy scale.
        self.matched_jet_pt_difference: Hist = None
        self.hybrid_level_spectra: Hist = None

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
        for hist in [self.response_matrix, self.response_matrix_errors, self.particle_level_spectra,
                     self.matched_jet_pt_difference, self.hybrid_level_spectra]:
            if hist and not hist.GetSumw2N() > 0:
                logger.debug(f"Setting Sumw2 for hist: {hist.GetName()}")
                hist.Sumw2(True)

        for hists in [self.part_level_hists, self.det_level_hists]:
            for _, hist in hists:
                if hist and not hist.GetSumw2N() > 0:
                    logger.debug(f"Setting Sumw2 for hist: {hist.GetName()}")
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

        # Sanity check
        if particle_level_spectra_config["normalize_by_n_jets"] and \
                particle_level_spectra_config["normalize_at_selected_jet_pt_bin"]:
            raise RuntimeError(
                "Cannot request normalization by both n jets and at a selected pt bin. Please check your configuration."
            )

        # Scale by N_{jets}
        # The number of entries should be equal to the number of jets. However, it's not a straightforward
        # number to extract because of all of the scaling related to pt hard bins
        if particle_level_spectra_config["normalize_by_n_jets"]:
            # Integrate over the hist to determine the number of jets displayed.
            # NOTE: This normalization also makes it into a probability distribution.
            max_value = particle_level_spectra_config["particle_level_max_pt"]
            entries = hist.Integral(hist.FindBin(0 + epsilon), hist.FindBin(max_value - epsilon))
            logger.debug(f"entries from hist: {hist.GetEntries()}, from integral: {entries}")

            # Normalize the histogram
            hist.Scale(1.0 / entries)

        if particle_level_spectra_config["normalize_at_selected_jet_pt_bin"]:
            error = ctypes.c_double(0)
            # Determine the number of entries at the selected bin value
            values = particle_level_spectra_config["normalize_at_selected_jet_pt_values"]
            entries = hist.IntegralAndError(hist.FindBin(values.min + epsilon), hist.FindBin(values.max - epsilon), error)
            #entries = hist.GetBinContent(hist.FindBin(value[0] + epsilon))
            logger.info(f"Number of entries with {values}: {entries} +/ {error.value}")

            # Normalize the histogram
            hist.Scale(1.0 / entries)

        # Check the result of the normalization (debug info to compare agreement)
        error = ctypes.c_double(0)
        entries = hist.IntegralAndError(hist.FindBin(20 + epsilon), hist.FindBin(100 - epsilon), error)
        logger.info(f"{self.reaction_plane_orientation}: Number of entries within 20-100: {entries} +/ {error.value}")

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

    def normalize_response_matrix(self) -> None:
        """ Normalize the response matrix. """
        response_matrix_helpers.normalize_response_matrix(
            hist = self.response_matrix,
            response_normalization = self.response_normalization
        )

    def project_hybrid_level_spectra(self) -> None:
        """ Project the hybrid level spectra from the response matrix. """
        # Setup and run the projector
        hybrid_level_spectra = ResponseMatrixProjector(
            observable_to_project_from = self.response_matrix,
            output_observable = self,
            output_attribute_name = "hybrid_level_spectra",
            projection_name_format = "hybrid_level_spectra",
        )
        # Specify the particle level pt limits
        hybrid_level_spectra.additional_axis_cuts.append(
            projectors.HistAxisRange(
                axis_type = projectors.TH1AxisType.y_axis,
                axis_range_name = "particle_level_limits",
                min_val = projectors.HistAxisRange.apply_func_to_find_bin(None, 1),
                max_val = projectors.HistAxisRange.apply_func_to_find_bin(ROOT.TAxis.GetNbins),
            )
        )
        # No additional cuts for the projection dependent axes
        hybrid_level_spectra.projection_dependent_cut_axes.append([])
        hybrid_level_spectra.projection_axes.append(
            projectors.HistAxisRange(
                axis_type = projectors.TH1AxisType.x_axis,
                axis_range_name = "hybrid_level_spectra",
                min_val = projectors.HistAxisRange.apply_func_to_find_bin(None, 1),
                max_val = projectors.HistAxisRange.apply_func_to_find_bin(ROOT.TAxis.GetNbins),
            )
        )

        # Perform the actual projection
        hybrid_level_spectra.project()

class ResponseMatrix(ResponseMatrixBase):
    """ Main response matrix class.

    Stores the response matrix histograms, as well as the methods to process the response matrix.
    """
    def __init__(self, *args: Any, **kwargs: Any):
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
        self.response_retrieval_info: Dict[str, Any] = self.task_config[self.task_config["responseMatrixTask"]]

    def _setup_projectors(self) -> None:
        """ Setup the sparse projectors. """
        # NOTE: The names of the projected histograms defined here must match those set in the fields
        #       defining the non-projected histograms.
        # Helper range
        full_axis_range = {
            "min_val": projectors.HistAxisRange.apply_func_to_find_bin(None, 1),
            "max_val": projectors.HistAxisRange.apply_func_to_find_bin(ROOT.TAxis.GetNbins)
        }
        # Reaction plane selection
        response_axes = getattr(this_module, self.response_retrieval_info["response"]["enum_name"])
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
            axis_type = response_axes.det_level_reaction_plane_orientation,
            axis_range_name = "detLevelReactionPlaneOrientation",
            **reaction_plane_axis_range
        )
        # Matching distance
        # Cut the matching distance at R.
        matching_distance_projector_axis = projectors.HistAxisRange(
            axis_type = response_axes.matching_distance,
            axis_range_name = "matchingDistance",
            min_val = projectors.HistAxisRange.apply_func_to_find_bin(
                None, 1
            ),
            max_val = projectors.HistAxisRange.apply_func_to_find_bin(
                ROOT.TAxis.FindBin, 0.2,
            )
        )

        #################
        # Response matrix
        #################
        if self.response_retrieval_info["response"]["sparse"]:
            # The response matrix axes are defined above so we can define a (semi-) shared reaction
            # plane orientation projector axis.
            response_matrix = ResponseMatrixProjector(
                observable_to_project_from = utils.recursive_getitem(
                    self.input_hists,
                    self.response_retrieval_info["response"]["input_name"],
                ),
                output_observable = self,
                output_attribute_name = "response_matrix",
                projection_name_format = "responseMatrix",
            )
            response_matrix.additional_axis_cuts.append(
                projectors.HistAxisRange(
                    axis_type = response_axes.det_level_leading_particle,
                    axis_range_name = "detLevelLeadingParticle",
                    min_val = projectors.HistAxisRange.apply_func_to_find_bin(
                        ROOT.TAxis.FindBin, self.leading_hadron_bias.value + epsilon,
                    ),
                    max_val = projectors.HistAxisRange.apply_func_to_find_bin(
                        ROOT.TAxis.GetNbins
                    ),
                )
            )
            # Matching distance
            response_matrix.additional_axis_cuts.append(matching_distance_projector_axis)
            # Reaction plane orientation
            response_matrix.additional_axis_cuts.append(reaction_plane_orientation_projector_axis)
            # Apply jet area cut. Remove jets with area less than 0.6 * pi * R^2
            # NOTE: This can only be applied to the response generated by the JetHPerformance task, as the
            #       cut is applied at the task level for the ResponseMaker
            if hasattr(response_axes, "det_level_jet_area"):
                logger.debug("Applying jet area cut to response.")
                response_matrix.additional_axis_cuts.append(
                    projectors.HistAxisRange(
                        axis_type = response_axes.det_level_jet_area,
                        axis_range_name = "detLevelJetArea",
                        min_val = projectors.HistAxisRange.apply_func_to_find_bin(
                            ROOT.TAxis.FindBin, 0.6 * np.pi * 0.2 * 0.2,
                        ),
                        max_val = projectors.HistAxisRange.apply_func_to_find_bin(
                            ROOT.TAxis.GetNbins,
                        ),
                    )
                )

            # No additional cuts for the projection dependent axes
            response_matrix.projection_dependent_cut_axes.append([])
            response_matrix.projection_axes.append(
                projectors.HistAxisRange(
                    axis_type = response_axes.det_level_jet_pt,
                    axis_range_name = "detLevelJetPt",
                    **full_axis_range
                )
            )
            response_matrix.projection_axes.append(
                projectors.HistAxisRange(
                    axis_type = response_axes.part_level_jet_pt,
                    axis_range_name = "partLevelJetPt",
                    **full_axis_range
                )
            )
            # Save the projector for later use
            self.projectors.append(response_matrix)

        #############################
        # Unmatched part level jet pt
        #############################
        if self.response_retrieval_info["unmatched_part_level"]["sparse"]:
            unmatched_part_level_axes = getattr(this_module,
                                                self.response_retrieval_info["unmatched_part_level"]["enum_name"])
            unmatched_part_level_jet_spectra = ResponseMatrixProjector(
                observable_to_project_from = utils.recursive_getitem(
                    self.input_hists,
                    self.response_retrieval_info["unmatched_part_level"]["input_name"],
                ),
                output_observable = self.part_level_hists,
                output_attribute_name = "unmatched_jet_spectra",
                projection_name_format = "unmatchedJetSpectraPartLevel",
            )
            # Can't apply a leading cluster cut on part level, since we don't have clusters
            unmatched_part_level_jet_spectra.projection_dependent_cut_axes.append([])
            unmatched_part_level_jet_spectra.projection_axes.append(
                projectors.HistAxisRange(
                    axis_type = unmatched_part_level_axes.jet_pt,
                    axis_range_name = "unmatchedPartLevelJetSpectra",
                    **full_axis_range
                )
            )
            # Save the projector for later use
            self.projectors.append(unmatched_part_level_jet_spectra)

        #############################
        # (Matched) Part level jet pt
        #############################
        if self.response_retrieval_info["part_level"]["sparse"]:
            part_level_axes = getattr(this_module, self.response_retrieval_info["part_level"]["enum_name"])
            part_level_jet_spectra = ResponseMatrixProjector(
                observable_to_project_from = utils.recursive_getitem(
                    self.input_hists,
                    self.response_retrieval_info["part_level"]["input_name"],
                ),
                output_observable = self.part_level_hists,
                output_attribute_name = "jet_spectra",
                projection_name_format = "jetSpectraPartLevel",
            )
            # The matching distance and reaction plane orientation axes may be different, so we copy
            # the HistAxisRange and update the axis to be the one for the part level
            # Matching distance
            part_level_matching_distance_projector_axis = copy.deepcopy(matching_distance_projector_axis)
            part_level_matching_distance_projector_axis.axis_type = part_level_axes.matching_distance
            part_level_jet_spectra.additional_axis_cuts.append(part_level_matching_distance_projector_axis)
            # Event plane orientation
            part_level_reaction_plane_orientation_projector_axis = \
                copy.deepcopy(reaction_plane_orientation_projector_axis)
            part_level_reaction_plane_orientation_projector_axis.axis_type = \
                part_level_axes.det_level_reaction_plane_orientation
            part_level_jet_spectra.additional_axis_cuts.append(part_level_reaction_plane_orientation_projector_axis)
            # No need to apply the jet area cut, as it's only needed in PbPb.
            # Can't apply a leading cluster cut on part level, since we don't have clusters
            part_level_jet_spectra.projection_dependent_cut_axes.append([])
            part_level_jet_spectra.projection_axes.append(
                projectors.HistAxisRange(
                    axis_type = part_level_axes.part_level_jet_pt,
                    axis_range_name = "partLevelJetSpectra",
                    **full_axis_range
                )
            )
            # Save the projector for later use
            self.projectors.append(part_level_jet_spectra)

        ############################
        # Unmatched det level jet pt
        ############################
        if self.response_retrieval_info["unmatched_det_level"]["sparse"]:
            unmatched_det_level_axes = getattr(this_module,
                                               self.response_retrieval_info["unmatched_det_level"]["enum_name"])
            unmatched_det_level_jet_spectra = ResponseMatrixProjector(
                observable_to_project_from = utils.recursive_getitem(
                    self.input_hists,
                    self.response_retrieval_info["unmatched_det_level"]["input_name"],
                ),
                output_observable = self.det_level_hists,
                output_attribute_name = "unmatched_jet_spectra",
                projection_name_format = "unmatchedJetSpectraDetLevel",
            )

            # The leading particle axis varies depending on whether the event plane is included in the sparse.
            leading_particle_axis = unmatched_det_level_axes.leading_particle_PP
            if self.collision_system in [params.CollisionSystem.PbPb,
                                         params.CollisionSystem.embedPythia,
                                         params.CollisionSystem.embedPP]:
                leading_particle_axis = unmatched_det_level_axes.leading_particle_PbPb
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
                    axis_type = unmatched_det_level_axes.jet_pt,
                    axis_range_name = "unmatchedDetLevelJetSpectra",
                    **full_axis_range
                )
            )
            # Save the projector for later use
            self.projectors.append(unmatched_det_level_jet_spectra)

        ############################
        # (Matched) Det level jet pt
        ############################
        if self.response_retrieval_info["det_level"]["sparse"]:
            det_level_axes = getattr(this_module, self.response_retrieval_info["det_level"]["enum_name"])
            det_level_jet_spectra = ResponseMatrixProjector(
                observable_to_project_from = utils.recursive_getitem(
                    self.input_hists,
                    self.response_retrieval_info["det_level"]["input_name"],
                ),
                output_observable = self.det_level_hists,
                output_attribute_name = "jet_spectra",
                projection_name_format = "jetSpectraDetLevel",
            )
            det_level_jet_spectra.additional_axis_cuts.append(
                projectors.HistAxisRange(
                    axis_type = det_level_axes.det_level_leading_particle,
                    axis_range_name = "detLevelLeadingParticle",
                    min_val = projectors.HistAxisRange.apply_func_to_find_bin(
                        ROOT.TAxis.FindBin, self.leading_hadron_bias.value + epsilon
                    ),
                    max_val = projectors.HistAxisRange.apply_func_to_find_bin(
                        ROOT.TAxis.GetNbins
                    )
                )
            )
            # The matching distance and reaction plane orientation axis may be different, so we copy
            # the HistAxisRange and update the axis to be the one for the det level
            # Matching distance
            det_level_matching_distance_projector_axis = copy.deepcopy(matching_distance_projector_axis)
            det_level_matching_distance_projector_axis.axis_type = det_level_axes.matching_distance
            det_level_jet_spectra.additional_axis_cuts.append(det_level_matching_distance_projector_axis)
            # Event plane orientaton
            det_level_reaction_plane_orientation_projector_axis = \
                copy.deepcopy(reaction_plane_orientation_projector_axis)
            det_level_reaction_plane_orientation_projector_axis.axis_type = \
                det_level_axes.det_level_reaction_plane_orientation
            # Apply jet area cut. Remove jets with area less than 0.6 * pi * R^2
            # NOTE: This can only be applied to the sparse generated by the JetHPerformance task, as the
            #       cut is applied at the task level for the ResponseMaker
            if hasattr(det_level_axes, "det_level_jet_area"):
                logger.debug("Applying jet area cut to matched det level jets.")
                det_level_jet_spectra.additional_axis_cuts.append(
                    projectors.HistAxisRange(
                        axis_type = det_level_axes.det_level_jet_area,
                        axis_range_name = "detLevelJetArea",
                        min_val = projectors.HistAxisRange.apply_func_to_find_bin(
                            ROOT.TAxis.FindBin, 0.6 * np.pi * 0.2 * 0.2,
                        ),
                        max_val = projectors.HistAxisRange.apply_func_to_find_bin(
                            ROOT.TAxis.GetNbins,
                        ),
                    )
                )
            det_level_jet_spectra.additional_axis_cuts.append(det_level_reaction_plane_orientation_projector_axis)
            det_level_jet_spectra.projection_dependent_cut_axes.append([])
            det_level_jet_spectra.projection_axes.append(
                projectors.HistAxisRange(
                    axis_type = det_level_axes.det_level_jet_pt,
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

    def retrieve_non_projected_hists(self) -> bool:
        """ Retrieve histograms which don't require projectors. """
        # NOTE: The names of the histograms defined here must match those set in the projectors.
        logger.debug("Retrieving non-projected hists")

        #################
        # Response matrix
        #################
        if not self.response_retrieval_info["response"]["sparse"]:
            self.response_matrix = utils.recursive_getitem(
                self.input_hists,
                self.response_retrieval_info["response"]["input_name"]
            )
            self.response_matrix.SetName("responseMatrix")

        #############################
        # Unmatched part level jet pt
        #############################
        if not self.response_retrieval_info["unmatched_part_level"]["sparse"]:
            self.part_level_hists.unmatched_jet_spectra = utils.recursive_getitem(
                self.input_hists,
                self.response_retrieval_info["unmatched_part_level"]["input_name"]
            )
            self.part_level_hists.unmatched_jet_spectra.SetName("unmatchedJetSpectraPartLevel")

        #############################
        # (Matched) Part level jet pt
        #############################
        if not self.response_retrieval_info["part_level"]["sparse"]:
            self.part_level_hists.jet_spectra = utils.recursive_getitem(
                self.input_hists,
                self.response_retrieval_info["part_level"]["input_name"]
            )
            self.part_level_hists.jet_spectra.SetName("jetSpectraPartLevel")

        ############################
        # Unmatched det level jet pt
        ############################
        if not self.response_retrieval_info["unmatched_det_level"]["sparse"]:
            self.det_level_hists.unmatched_jet_spectra = utils.recursive_getitem(
                self.input_hists,
                self.response_retrieval_info["unmatched_det_level"]["input_name"]
            )
            self.det_level_hists.unmatched_jet_spectra.SetName("unmatchedJetSpectraDetLevel")

        ############################
        # (Matched) Det level jet pt
        ############################
        if not self.response_retrieval_info["det_level"]["sparse"]:
            self.det_level_hists.jet_spectra = utils.recursive_getitem(
                self.input_hists,
                self.response_retrieval_info["det_level"]["input_name"]
            )
            self.det_level_hists.jet_spectra.SetName("jetSpectraDetLevel")

        ##########
        # QA hists
        ##########
        if not self.response_retrieval_info["jet_energy_resolution"]["sparse"]:
            event_activity_to_label = {
                params.EventActivity.central: 0,
                params.EventActivity.semi_central: 2,
            }
            hist_name = f"fh2PtJet2VsRelPt_{event_activity_to_label[self.event_activity]}"
            if self.response_retrieval_info["jet_energy_resolution"]["use_tagger"]:
                hists = histogram.get_histograms_in_file(self.input_filename)
                #logger.debug(f"hists: {hists}")
                self.matched_jet_pt_difference = hists[
                    "JetTagger_hybridLevelJets_AKTFullR020_tracks_pT3000_caloClustersCombined_E3000_pt_scheme_detLevelJets_AKTFullR020_tracks_pT3000_caloClusters_E3000_pt_scheme_TC"
                ][hist_name]
            else:
                hist_path = self.response_retrieval_info["jet_energy_resolution"]["input_name"]
                hist_path.append(hist_name)
                self.matched_jet_pt_difference = utils.recursive_getitem(
                    self.input_hists, hist_path
                )

        return True

class ResponseManager(analysis_manager.Manager):
    """ Analysis manager for creating response(s).

    Attributes:
        config_filename: Filename of the configuration
        selected_analysis_options: Options selected for this analysis.
        key_index: Key index object for the analysis.
        selected_iterables: All iterables values used to create the response matrices.
        analyses: Response matrix analysis objects.
        pt_hard_bins: Pt hard analysis objects for pt hard binned analyses (optional).
    """
    def __init__(self, config_filename: str, selected_analysis_options: params.SelectedAnalysisOptions, manager_task_name: str = "ResponseManager", **kwargs: Any):
        super().__init__(
            config_filename = config_filename, selected_analysis_options = selected_analysis_options,
            manager_task_name = manager_task_name, **kwargs
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

    def construct_responses_from_configuration_file(self) -> analysis_config.ConstructedObjects:
        """ Construct ``ResponseMatrix`` objects based on iterables in a configuration file. """
        return self._construct_responses_from_configuration_file(task_name = "Response", obj = ResponseMatrix)

    def _construct_responses_from_configuration_file(self, task_name: str = "Response",
                                                     obj: Type[ResponseMatrixBase] = ResponseMatrix
                                                     ) -> analysis_config.ConstructedObjects:
        """ Helper function for constructing the ``ResponseMatrix`` objects.

        This is a separate function to make it easy to override the arguments in inherited classes.
        """
        return analysis_config.construct_from_configuration_file(
            task_name = task_name,
            config_filename = self.config_filename,
            selected_analysis_options = self.selected_analysis_options,
            additional_possible_iterables = {"pt_hard_bin": None, "jet_pt_bin": None},
            additional_classes_to_register = [response_matrix_helpers.ResponseNormalization],
            obj = obj,
        )

    def construct_final_responses_from_configuration_file(self) -> analysis_config.ConstructedObjects:
        """ Construct final ``ResponseMatrixBase`` objects based on iterables in a configuration file. """
        return self._construct_final_responses_from_configuration_file(
            task_name = "ResponseFinal", obj = ResponseMatrixBase
        )

    def _construct_final_responses_from_configuration_file(self, task_name: str = "ResponseFinal",
                                                           obj: Type[ResponseMatrixBase] = ResponseMatrixBase
                                                           ) -> analysis_config.ConstructedObjects:
        """ Helper function for constructing the final ResponseMatrixBase objects.

        This is a separate function to make it easy to override the arguments in inherited classes.
        """
        return analysis_config.construct_from_configuration_file(
            task_name = task_name,
            config_filename = self.config_filename,
            selected_analysis_options = self.selected_analysis_options,
            additional_possible_iterables = {"pt_hard_bin": None, "jet_pt_bin": None},
            additional_classes_to_register = [response_matrix_helpers.ResponseNormalization],
            obj = obj,
        )

    def construct_pt_hard_bins_from_configuration_file(self) -> analysis_config.ConstructedObjects:
        """ Construct ``PtHardAnalysis`` objects based on iterables in a configuration file. """
        return self._construct_pt_hard_bins_from_configuration_file(
            task_name = "PtHardBins", obj = pt_hard_analysis.PtHardAnalysis
        )

    def _construct_pt_hard_bins_from_configuration_file(self, task_name: str = "PtHardBins",
                                                        obj: Type[pt_hard_analysis.PtHardAnalysis] = pt_hard_analysis.PtHardAnalysis
                                                        ) -> analysis_config.ConstructedObjects:
        """ Helper function for constructing the ``PtHardAnalysis`` objects.

        This is a separate function to make it easy to override the arguments in inherited classes.
        """
        return analysis_config.construct_from_configuration_file(
            task_name = task_name,
            config_filename = self.config_filename,
            selected_analysis_options = self.selected_analysis_options,
            additional_possible_iterables = {"pt_hard_bin": None},
            obj = obj,
        )

    def construct_final_pt_hard_object_from_configuration_file(self) -> analysis_config.ConstructedObjects:
        """ Construct final ``PtHardAnalysisBase`` objects based on iterables in a configuration file. """
        return self._construct_final_pt_hard_object_from_configuration_file(
            task_name = "PtHardFinal", obj = pt_hard_analysis.PtHardAnalysisBase
        )

    def _construct_final_pt_hard_object_from_configuration_file(self, task_name: str = "PtHardFinal",
                                                                obj: Type[pt_hard_analysis.PtHardAnalysisBase] = pt_hard_analysis.PtHardAnalysisBase
                                                                ) -> analysis_config.ConstructedObjects:
        """ Construct final ``PtHardAnalysisBase`` objects based on iterables in a configuration file. """
        return analysis_config.construct_from_configuration_file(
            task_name = task_name,
            config_filename = self.config_filename,
            selected_analysis_options = self.selected_analysis_options,
            additional_possible_iterables = {"pt_hard_bin": None},
            obj = obj,
        )

    def setup(self) -> None:
        """ Setup and prepare the analysis objects. """
        # Cache input hists so we can avoid repeatedly opening files
        input_hists: Dict[Any, Dict[str, Any]] = {}

        # Setup the response matrix analysis objects and run the response matrix projectors
        # By the time that this step is complete, we should have all histograms.
        with self._progress_manager.counter(total = len(self.analyses),
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
                pt_hard_bin.setup(input_hists = input_hists[key_index.pt_hard_bin])

                # Update progress
                setting_up.update()

    def _retrieve_scale_factors_for_logging(self) -> str:
        """ Log the scale factors that are stored in the pt hard bin analyses.

        This is really just a trivial helper.
        """
        output = []
        for pt_hard_key_index, pt_hard_bin in \
                analysis_config.iterate_with_selected_objects(self.pt_hard_bins):
            output.append(f"Pt hard bin: {pt_hard_key_index.pt_hard_bin}, scale factor: {pt_hard_bin.scale_factor}")
        return "\n".join(output)

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

        # Print out the scale factors. These are before the rescaling for the number of events. We
        # iterate over the bins separately from the loop above so # that the scale factors are all
        # listed together (ie. for convenience).
        logger.info("Pre rescaling of scale factors by average number of events")
        logger.info(self._retrieve_scale_factors_for_logging())

        # Setup
        checked_additional_arguments = False
        additional_outliers_removal_arguments = {}
        # Remove outliers and scale the projected histograms according to their pt hard bins.
        with self._progress_manager.counter(total = len(self.pt_hard_bins),
                                            desc = "Processing:",
                                            unit = "pt hard bins") as processing:
            for pt_hard_key_index, pt_hard_bin in \
                    analysis_config.iterate_with_selected_objects(self.pt_hard_bins):
                # Argument validation
                # We only want to check once because it won't vary between pt hard bin analysis, so it's
                # a waste to check multiple times.
                if checked_additional_arguments is False:
                    mean_fractional_difference_limit = \
                        pt_hard_bin.task_config.get("mean_fractional_difference_limit", None)
                    if mean_fractional_difference_limit:
                        additional_outliers_removal_arguments["mean_fractional_difference_limit"] = \
                            mean_fractional_difference_limit
                    median_fractional_difference_limit = \
                        pt_hard_bin.task_config.get("median_fractional_difference_limit", None)
                    if median_fractional_difference_limit:
                        additional_outliers_removal_arguments["median_fractional_difference_limit"] = \
                            median_fractional_difference_limit

                    # Only check once - it won't vary with pt hard bin.
                    checked_additional_arguments = True

                # Scale the pt hard spectra
                logger.debug("Scaling the pt hard spectra.")
                pt_hard_bin.run(
                    average_number_of_events = average_number_of_events,
                    outliers_removal_axis = projectors.TH1AxisType.x_axis,
                    hists = {"pt_hard_spectra": pt_hard_bin.pt_hard_spectra},
                    **additional_outliers_removal_arguments,
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
                    logger.debug(
                        f"Running pt hard analysis for attribute_name: {hist_info.attribute_name}, hists: {hists}"
                    )
                    pt_hard_bin.run(
                        average_number_of_events = average_number_of_events,
                        outliers_removal_axis = hist_info.outliers_removal_axis,
                        analyses = ep_analyses,
                        hist_attribute_name = hist_info.attribute_name,
                        **additional_outliers_removal_arguments,
                    )

                # Update progress
                processing.update()

        # Print out the scale factors. These are after the rescaling for the number of events. We
        # iterate over the bins separately from the loop above so # that the scale factors are all
        # listed together (ie. for convenience).
        logger.info("Post rescaling of scale factors by average number of events")
        logger.info(self._retrieve_scale_factors_for_logging())

        # Now merge the scaled histograms into the final response matrix results.
        # +1 for the final pt hard spectra.
        with self._progress_manager.counter(total = len(self.selected_iterables["reaction_plane_orientation"]) + 1,
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

    def _extract_hybrid_level_spectra(self) -> None:
        """ Extract the hybrid level spectra from the response matrix.

        For convenience, we first scale and remove outliers from the response, then we project
        the already scaled spectra. Last, we merge them for the final response.
        """
        with self._progress_manager.counter(total = len(self.pt_hard_bins),
                                            desc = "Projecting:",
                                            unit = "hybird level spectra") as projecting:
            for pt_hard_key_index, pt_hard_bin in \
                    analysis_config.iterate_with_selected_objects(self.pt_hard_bins):
                ep_analyses = {}
                for analysis_key_index, analysis in \
                        analysis_config.iterate_with_selected_objects(
                            self.analyses,
                            pt_hard_bin = pt_hard_key_index.pt_hard_bin
                        ):
                    ep_analyses[analysis_key_index.reaction_plane_orientation] = analysis
                    analysis.project_hybrid_level_spectra()

                # Update progress
                projecting.update()

        # Merge into the final response object. We have to do this by hand because we need to project
        # it after the pt hard bin processing is completed.
        for reaction_plane_orientation in self.selected_iterables["reaction_plane_orientation"]:
            pt_hard_analysis.merge_pt_hard_binned_analyses(
                analyses = analysis_config.iterate_with_selected_objects(
                    self.analyses,
                    reaction_plane_orientation = reaction_plane_orientation
                ),
                hist_attribute_name = "hybrid_level_spectra",
                output_analysis_object = self.final_responses[
                    self.final_responses_key_index(reaction_plane_orientation = reaction_plane_orientation)
                ],
            )

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
        for _, analysis_base in analysis_config.iterate_with_selected_objects(self.final_responses):
            func(analysis_base)

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
        with self._progress_manager.counter(total = len(self.final_responses),
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
        ep_analyses = {
            k: v for k, v in self.final_responses.items()
            if k.reaction_plane_orientation != params.ReactionPlaneOrientation.inclusive
        }
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

    def _compare_min_constituent_cut_particle_spectra(self) -> None:
        """ Plot comparison to min constituent cut particle level spectra.

        Note:
            This comparison will only be created if it's requested in the config and the min
            constituent cut data is actually available.

        Args:
            None.
        Returns:
            None. The plot is created and saved.
        """
        if self.task_config["min_constituent_comparison"]:
            inclusive_response = self.final_responses[
                self.final_responses_key_index(params.ReactionPlaneOrientation.inclusive)
            ]
            # Open the min constituent cut results.
            alice_path = os.path.join(
                "output", str(self.selected_analysis_options.collision_system),
                str(self.selected_analysis_options.collision_energy), str(self.selected_analysis_options.event_activity),
                # Need to use the inclusive response to get the proper leading hadron bias value.
                str(inclusive_response.leading_hadron_bias),
                "ResponseFinal_MinConstituent", "final_responses.root"
            )
            try:
                # First retrieve the actual histogram.
                with histogram.RootOpen(alice_path) as f:
                    min_constituent_hist = f.Get("particle_level_spectra_inclusive")
                    min_constituent_hist.SetDirectory(0)
                # Then plot the comparison
                plot_response_matrix.compare_min_constituent_cut(
                    obj = inclusive_response,
                    min_constituent_hist = min_constituent_hist,
                    output_info = self.output_info,
                )
            except OSError:
                # The file doesn't exist. Skip the plot
                logger.info("Min constituent comparison data is not available, so skipping the plot.")

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
        # +1 for particle level spectra.
        # +1 for min constituent comparison.
        # *2 for response matrix, response spectra.
        with self._progress_manager.counter(total = 2 * len(self.selected_iterables["reaction_plane_orientation"]) + 3,
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
                            title = r"$p_{\text{T}}$ hard spectra",
                            x_label = r"$p_{\text{T}}^{\text{hard}}$",
                            y_label = r"$\frac{\text{d}N}{\text{d}p_{\text{T}}}$",
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
                    ep_analyses_iter = analysis_config.iterate_with_selected_objects(self.final_responses),
                    output_info = self.output_info,
                    plot_with_ROOT = plot_with_ROOT,
                )
            # Only plot the ratios if there are actually enough RP orientations to calculate the ratio(s).
            if len(self.selected_iterables["reaction_plane_orientation"]) > 1:
                plot_response_matrix.particle_level_spectra_ratios(
                    ep_analyses_iter = analysis_config.iterate_with_selected_objects(self.final_responses),
                    output_info = self.output_info,
                )

            # Jet energy scale QA. It's not meaningful for each orientation because the tagger
            # doesn't selecte on RP orientation. So we just do it once.
            plot_response_matrix.matched_jet_energy_scale(
                plot_labels = plot_base.PlotLabels(
                    title = "Matched hybrid-detector jet energy scale",
                    x_label = fr"${labels.jet_pt_display_label('det')}\:({labels.momentum_units_label_gev()})$",
                    y_label = fr"$({labels.jet_pt_display_label('hybrid')} - {labels.jet_pt_display_label('det')})/{labels.jet_pt_display_label('det')}",
                ),
                output_name = "matched_hybrid_detctor_level_jet_energy_scale",
                output_info = self.output_info,
                obj = self.final_responses[
                    self.final_responses_key_index(params.ReactionPlaneOrientation.inclusive)
                ],
            )
            plotting.update()

            # Compare to the min constituent cut.
            self._compare_min_constituent_cut_particle_spectra()
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
                # Plot part, det level match and unmatched (so skip the response matrix (first), as well as
                # matched jet differences(last))
                for plot_with_ROOT in [False, True]:
                    for hist_info in list(histogram_info_for_processing.values())[1:-1]:
                        # This is just a proxy to get "part" or "det"
                        base_label = hist_info.description[:hist_info.attribute_name.find("_")].lower()
                        # This will be something like "unmatched_jet_spectra"
                        # +1 is to skip the "." that we found.
                        output_label = hist_info.attribute_name[hist_info.attribute_name.find(".") + 1:]
                        plot_response_matrix.plot_response_spectra(
                            plot_labels = plot_base.PlotLabels(
                                title = hist_info.description,
                                x_label = r"$p_{\text{T,jet}}^{\text{%(label)s}}$" % {
                                    "label": base_label,
                                },
                                y_label = r"$\frac{\text{d}N}{\text{d}p_{\text{T}}}$",
                            ),
                            output_name = f"{base_label}_level_{output_label}",
                            merged_analysis = self.final_responses[
                                self.final_responses_key_index(reaction_plane_orientation)
                            ],
                            pt_hard_analyses = analyses,
                            hist_attribute_name = hist_info.attribute_name,
                            plot_with_ROOT = plot_with_ROOT,
                        )
                # Plot the hybrid spectra
                plot_response_matrix.plot_response_spectra(
                    plot_labels = plot_base.PlotLabels(
                        title = "Hybrid jet spectra",
                        x_label = r"$p_{\text{T,jet}}^{\text{%(label)s}}$" % {
                            "label": "hybrid",
                        },
                        y_label = r"$\frac{\text{d}N}{\text{d}p_{\text{T}}}$",
                    ),
                    output_name = "hybrid_level_spectra",
                    merged_analysis = self.final_responses[
                        self.final_responses_key_index(reaction_plane_orientation)
                    ],
                    pt_hard_analyses = analyses,
                    hist_attribute_name = "hybrid_level_spectra",
                    plot_with_ROOT = False,
                )

                # Update progress
                plotting.update()

    def _package_and_write_final_hists(self) -> None:
        """ Package up and write the final repsonse matrices and particle level spectra. """
        # Write out the final histograms.
        output_filename = os.path.join(self.output_info.output_prefix, "final_responses.root")
        with histogram.RootOpen(output_filename, mode = "RECREATE"):
            for _, analysis in analysis_config.iterate_with_selected_objects(self.final_responses):
                # Response matrix
                response = analysis.response_matrix.Clone(
                    f"{analysis.response_matrix.GetName()}_{analysis.reaction_plane_orientation}"
                )
                response.Write()
                # Particle level spectra
                particle_level_spectra = analysis.particle_level_spectra.Clone(
                    f"{analysis.particle_level_spectra.GetName()}_{analysis.reaction_plane_orientation}"
                )
                particle_level_spectra.Write()
                # Matched jet pt
                # We didn't select on reaction plane orientation, so only write the inclusive case
                if analysis.reaction_plane_orientation == params.ReactionPlaneOrientation.inclusive:
                    matched_jet_pt_difference = analysis.matched_jet_pt_difference.Clone(
                        "matched_jet_pt_residual"
                    )
                    matched_jet_pt_difference.Write()

        # Handle HEP data.
        try:
            import hepdata_lib as hepdata
            logger.debug("Writing the responses to the HEPdata format.")
            submission = hepdata.Submission()
            for _, analysis in analysis_config.iterate_with_selected_objects(self.final_responses):
                temp_hist = analysis.response_matrix.Clone(
                    f"{analysis.response_matrix.GetName()}_{analysis.reaction_plane_orientation}_hepdata"
                )
                # We want to write it out with 5 GeV bins
                temp_hist.Rebin2D(5, 5)
                response = hepdata.root_utils.get_hist_2d_points(temp_hist)
                pt_hybrid = hepdata.Variable("Hybrid level pT", is_binned = False, units = "GeV/c")
                pt_hybrid.values = response["x"]
                pt_part = hepdata.Variable("Particle level pT", is_binned = False, is_independent = False, units = "GeV/c")
                pt_part.values = response["y"]
                resp = hepdata.Variable("Response matrix", is_binned = False, is_independent = False)
                resp.values = response["z"]

                table = hepdata.Table(f"response_matrix_{analysis.reaction_plane_orientation}_hepdata")
                table.add_variable(pt_hybrid)
                table.add_variable(pt_part)
                table.add_variable(resp)
                # NOTE: Additional labeling should usually be performed here, but I'm omitting for the sake of time
                submission.add_table(table)

            hepdata_output = os.path.join(self.output_info.output_prefix, "hepdata_responses")
            # Create the YAML files
            submission.create_files(hepdata_output)
        except ImportError:
            # It's not available here - skip it.
            ...

    def run(self) -> bool:
        """ Run the response matrix analyses. """
        logger.debug(
            f"key_index: {self.key_index}, selected_option_names: {list(self.selected_iterables)},"
            f"analyses: {pprint.pformat(self.analyses)}"
        )

        # We need to determine the input information
        histogram_info_for_processing: Dict[str, pt_hard_analysis.PtHardHistogramInformation] = {}
        for name, info in _response_matrix_histogram_info.items():
            if name not in ["response_matrix_errors", "particle_level_spectra", "hybrid_level_spectra"]:
                # Help out mypy...
                assert isinstance(info, pt_hard_analysis.PtHardHistogramInformation)
                histogram_info_for_processing[name] = info

        # Analysis steps:
        # 1. Setup response matrix and pt hard objects.
        # 2. Pt hard bin outliers removal, scaling, and merging into final response objects.
        # 3. Write histograms (or read histograms if requested).
        # 4. Final processing, including projecting particle level spectra.
        # 5. Plotting.
        # 6. Writing final hists together to a single file.
        #
        # If we are starting from reading histograms, we start from step 3.
        steps = 4 if self.task_config["read_hists_from_root_file"] else 6
        with self._progress_manager.counter(total = steps,
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

                # Extract the hybrid level spectra pt hard bin.
                self._extract_hybrid_level_spectra()

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
            self._package_and_write_final_hists()
            overall_progress.update()

        return True

def run_from_terminal() -> ResponseManager:
    """ Driver function for running the correlations analysis. """
    # Basic setup
    # Quiet down some pachyderm modules
    logging.getLogger("pachyderm").setLevel(logging.INFO)
    # Run in batch mode
    ROOT.gROOT.SetBatch(True)
    # Turn off stats box
    ROOT.gStyle.SetOptStat(0)

    # Setup and run the analysis
    manager: ResponseManager = analysis_manager.run_helper(
        manager_class = ResponseManager, task_name = "Response matrix",
    )

    # Return it for convenience.
    return manager

if __name__ == "__main__":
    run_from_terminal()
