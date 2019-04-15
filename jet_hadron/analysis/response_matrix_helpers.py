#!/usr/bin/env python

""" Helpers for creating a response matrix.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@yale.edu>, Yale University
"""

import enum
import numpy as np
from typing import Any, Callable, Tuple

from pachyderm import yaml

from jet_hadron.base.typing_helpers import Axis, Hist

import ROOT

class ResponseNormalization(enum.Enum):
    """ Selects the type of normalization to apply to the RM """
    none = 0
    normalize_each_detector_bin = 1
    normalize_each_truth_bin = 2

    def __str__(self) -> str:
        """ Returns the name of each normalization type with + " Normalization".

        For example, for none, we get "No Normalization".
        """
        return self.name

    def display_str(self) -> str:
        """ Returns the name of the normalization with "Normalization" appended in a form suitable for display.

        For example, for ``none``, it returns "No normalization".
        For the others, it returns (for example): "Normalize each detector bin"
        """
        if self == ResponseNormalization.none:
            return self.name.replace("none", "no").capitalize() + " normalization"
        return self.name.replace("_", " ").capitalize()

    # Handle YAML serialization
    to_yaml = classmethod(yaml.enum_to_yaml)
    from_yaml = classmethod(yaml.enum_from_yaml)

def normalize_response_matrix(hist: Hist, response_normalization: ResponseNormalization) -> None:
    """ Normalize response matrix.

    In the case of normalizing each detector pt bin (usually on the x axis), we take all associated truth level
    bins (usually the y axis), and normalize that array of truth bins to 1. In the case of normalizing the truth
    level bins, the case is reversed.

    Args:
        hist: The response matrix
        response_normalization: Response normalization convention, which dictates which axis to normalize.
    Returns:
        None. The response matrix is modified in place.
    """
    if response_normalization == ResponseNormalization.none:
        # Nothing to be done, so just return.
        return

    # Determine the relevant parameters for normalizing the response
    # Each entry is of the form (projection_function, max_bins)
    parameters_map = {
        ResponseNormalization.normalize_each_detector_bin: (
            ROOT.TH2.ProjectionY,
            hist.GetXaxis().GetNbins() + 1,
        ),
        ResponseNormalization.normalize_each_truth_bin: (
            ROOT.TH2.ProjectionX,
            hist.GetYaxis().GetNbins() + 1,
        ),
    }
    projection_function, max_bins = parameters_map[response_normalization]

    # We decided to ignore the overflow bins.
    for index in range(1, max_bins):
        # Access bins
        bins_content, _ = _access_set_of_values_associated_with_a_bin(
            hist = hist,
            bin_of_interest = index,
            response_normalization = response_normalization,
        )

        norm = np.sum(bins_content)
        # NOTE: The upper bound on integrals is inclusive!
        proj = projection_function(hist, f"{hist.GetName()}_projection_{index}", index, index)

        # Sanity checks
        # NOTE: The upper bound on integrals is inclusive!
        # NOTE: Integral() == Integral(1, proj.GetXaxis().GetNbins())
        if not np.isclose(norm, proj.Integral(1, proj.GetXaxis().GetNbins())):
            raise ValueError(
                f"Mismatch between sum and integral! norm: {norm},"
                f" integral: {proj.Integral(1, proj.GetXaxis().GetNbins())}"
            )
        if not np.isclose(proj.Integral(), proj.Integral(1, proj.GetXaxis().GetNbins())):
            raise ValueError(
                f"Integral mismatch! Full: {proj.Integral()} 1-nBins: {proj.Integral(1, proj.GetXaxis().GetNbins())}"
            )

        # Avoid scaling by 0
        if not norm > 0.0:
            continue

        # normalization by sum
        _scale_set_of_bins(
            hist = hist,
            bin_of_interest = index,
            response_normalization = response_normalization,
            scale_factor = norm,
        )

    # Final sanity check by checking that the normalization is correct in each bin.
    res = _check_normalization(hist = hist, response_normalization = response_normalization)

    if not res:
        raise ValueError("Normalization check failed.")

def _setup_access_bins(response_normalization: ResponseNormalization) -> Tuple[Callable[[Hist], Axis], Callable[[Hist, int, int], Any]]:
    """ Determine the proper axis and GetBin functions for accessing a set of bins.

    Args:
        response_normalization: Response normalization convention, which dictates which axis to retrieve the bins from.
    """
    axis = None
    get_bin = None
    if response_normalization == ResponseNormalization.normalize_each_detector_bin:
        axis = ROOT.TH1.GetYaxis

        # Define helper function so we can properly set the order of the parameters.
        def get_bin(hist: Hist, bin_of_interest: int, index: int) -> int:
            return hist.GetBin(bin_of_interest, index)
    elif response_normalization == ResponseNormalization.normalize_each_truth_bin:
        axis = ROOT.TH1.GetXaxis

        # Define helper function so we can properly set the order of the parameters.
        def get_bin(hist: Hist, bin_of_interest: int, index: int) -> int:
            return hist.GetBin(index, bin_of_interest)
    else:
        raise ValueError(f"RM Normalization value {response_normalization} not recognized")

    return axis, get_bin

def _access_set_of_values_associated_with_a_bin(hist: Hist,
                                                bin_of_interest: int,
                                                response_normalization: ResponseNormalization) -> Tuple[np.ndarray, np.ndarray]:
    """ Access a set of bins associated with a particular bin value in the other axis.

    For example, if the hist looks like this graphically:

    a b c
    d e f
    g h i <- values (here and above)
    1 2 3 <- bin number

    then in the case of accessing a set of y bins associated with an x bin, for example, x bin 2,
    it would return values h, e, and b.

    Args:
        hist: The histogram whose bins should be accessed. This must be a 2D hist.
        bin_of_interest: Bin which we would like to access.
        response_normalization: Response normalization convention, which dictates which axis to retrieve the bins from.
    Returns:
        Array of the bin contents, Array of bin errors
    """
    # Initial setup
    axis, get_bin = _setup_access_bins(response_normalization = response_normalization)

    #logger.debug(f"Axis: {axis(hist)}, getBin: {get_bin}")
    set_of_bins_content = np.zeros(axis(hist).GetNbins())
    set_of_bins_errors = np.zeros(axis(hist).GetNbins())
    for array_index, bin_index in enumerate(range(1, axis(hist).GetNbins() + 1)):
        set_of_bins_content[array_index] = hist.GetBinContent(get_bin(hist, bin_of_interest, bin_index))
        set_of_bins_errors[array_index] = hist.GetBinError(get_bin(hist, bin_of_interest, bin_index))

    return set_of_bins_content, set_of_bins_errors

def _scale_set_of_bins(hist: Hist, bin_of_interest: int,
                       response_normalization: ResponseNormalization,
                       scale_factor: float) -> np.ndarray:
    """ Scale a set of bins associated with a particular bin value in the other axis.

    For further information on how the bins are selected, etc, see
    ``_access_set_of_values_associated_with_a_bin(...)``.

    Args:
        hist: The histogram whose bins should be accessed. This must be a 2D hist.
        bin_of_interest: Bin which we would like to access.
        response_normalization: Response normalization convention, which dictates which axis to retrieve the bins from.
        scale_factor: Every bin will be = bin_content / scale_factor .
    Returns:
        None. The histogram is updated in place.
    """
    # Initial setup
    axis, get_bin = _setup_access_bins(response_normalization = response_normalization)

    # Get the relevant bins.
    set_of_bins_content, set_of_bins_errors = _access_set_of_values_associated_with_a_bin(
        hist = hist,
        bin_of_interest = bin_of_interest,
        response_normalization = response_normalization
    )

    # Set the scaled bin values in the histogram.
    for bin_index, (bin_content, bin_error) in enumerate(zip(set_of_bins_content, set_of_bins_errors), start = 1):
        hist.SetBinContent(get_bin(hist, bin_of_interest, bin_index), bin_content / scale_factor)
        hist.SetBinError(get_bin(hist, bin_of_interest, bin_index), bin_error / scale_factor)

def _check_normalization(hist: Hist, response_normalization: ResponseNormalization) -> bool:
    """ Check each bin to ensure that the normalization was successful.

    Args:
        hist: Response matrix to check. This must be a 2D histogram.
        response_normalization: Normalization convention for the response matrix.
    Returns:
        True if the normalization is fine.
    Raises:
        ValueError: If the normalization fails for a particular bin.
    """
    for index in range(1, hist.GetXaxis().GetNbins() + 1):
        # Access bins
        bins_content, _ = _access_set_of_values_associated_with_a_bin(
            hist = hist,
            bin_of_interest = index,
            response_normalization = response_normalization,
        )
        # Get norm
        norm = np.sum(bins_content)

        # Somewhat arbitrarily comparison limit selected. It should be sufficiently small.
        comparison_limit = 1e-9
        if not np.isclose(norm, 0, atol = comparison_limit) and not np.isclose(norm, 1, atol = comparison_limit):
            raise ValueError(f"Normalization not successful for bin {index}. Norm: {norm}")

    return True

