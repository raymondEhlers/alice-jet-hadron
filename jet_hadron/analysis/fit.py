#!/usr/bin/env python

""" Main jet-hadron correlations fit module.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import logging
import pprint
import sys
from typing import TYPE_CHECKING, Any, Sequence, Tuple, Union

import numpy as np
import pachyderm.fit
from pachyderm import histogram
from pachyderm.fit import T_FitArguments

from jet_hadron.base import params
from jet_hadron.base.typing_helpers import Hist

import ROOT

if TYPE_CHECKING:
    from jet_hadron.analysis import extracted

logger = logging.getLogger(__name__)

this_module = sys.modules[__name__]

def print_root_fit_parameters(fit: Any) -> None:
    """ Print out all of the ROOT-based fit parameters. """
    output_parameters = []
    for i in range(0, fit.GetNpar()):
        parameter = fit.GetParameter(i)
        parameter_name = fit.GetParName(i)
        lower_limit = ROOT.Double(0.0)
        upper_limit = ROOT.Double(0.0)
        fit.GetParLimits(i, lower_limit, upper_limit)

        output_parameters.append(f"{i}: {parameter_name} = {parameter} from {lower_limit} - {upper_limit}")

    pprint.pprint(output_parameters)

def fit_1d_mixed_event_normalization(hist: Hist, delta_phi_limits: Sequence[float]) -> ROOT.TF1:
    """ Alternative to determine the mixed event normalization.

    A linear function is fit to the dPhi mixed event normalization for some predefined range.

    Args:
        hist: 1D mixed event histogram to be fit.
        delta_phi_limits: Min and max fit limits in delta phi.
    Returns:
        The ROOT fit function.
    """
    fit_func = ROOT.TF1("mixedEventNormalization1D", "[0] + 0.0*x", delta_phi_limits[0], delta_phi_limits[1])

    # Fit to the given histogram
    # R uses the range defined in the fit function
    # 0 ensures that the fit isn't drawn
    # Q ensures minimum printing
    # + adds the fit to function list to ensure that it is not deleted on the creation of a new fit
    hist.Fit(fit_func, "RIB0")

    # And return the fit
    return fit_func

def fit_2d_mixed_event_normalization(hist: Hist, delta_phi_limits: Sequence[float], delta_eta_limits: Sequence[float]) -> ROOT.TF2:
    """ Alternative to determine the mixed event normalization.

    A linear function is fit to the dPhi-dEta mixed event normalization for some predefined range.

    Args:
        hist: 2D mixed event histogram to be fit.
        delta_phi_limits: Min and max fit limits in delta phi.
        delta_eta_limits: Min and max fit limits in delta eta.
    Returns:
        The ROOT fit function.
    """
    fit_func = ROOT.TF2(
        "mixedEventNormalization2D",
        "[0] + 0.0*x + 0.0*y",
        delta_phi_limits[0], delta_phi_limits[1],
        delta_eta_limits[0], delta_eta_limits[1]
    )

    # Fit to the given histogram
    # R uses the range defined in the fit function
    # 0 ensures that the fit isn't drawn
    # Q ensures minimum printing
    # + adds the fit to function list to ensure that it is not deleted on the creation of a new fit
    hist.Fit(fit_func, "RIB0")

    # And return the fit
    return fit_func

def RPF_result_to_width_fit_result(rpf_result: pachyderm.fit.FitResult,
                                   short_name: str,
                                   subtracted_hist: histogram.Histogram1D,
                                   width_obj: "extracted.ExtractedWidth") -> bool:
    """ Convert from a RPF fit result into a width fit result

    Note:
        This uses a chi squared cost function to calculate the minimum value for the width aspect of the fit.

    Args:
        rpf_result: Result from the reaction plane fit.
        short_name: Shortened version of the attribute name. Should be "ns" (for near side) or "as" (for away side).
        subtracted_hist: Subtracted histogram which this is supposed to fit.
        width_obj: Width object where the fit result will be stored.
    Returns:
        True if the fit result was successfully extracted.
    """
    # NOTE: This function is just a work in progress!!

    # Create chi squared object to calculate the minimum value.
    cost_func = pachyderm.fit.BinnedChiSquared(f = width_obj.fit_object.fit_function, data = subtracted_hist)
    # We want the func_code because it describe will then exclude the x parameter (which is what we want)
    parameters = cost_func.func_code.co_varnames
    fixed_parameters = ["mean", "pedestal"]
    free_parameters = ["width", "amplitude"]
    # Map between our new width fit parameters and those of the RPF.
    parameter_name_map = {
        "width": f"{short_name}_sigma",
        "amplitude": f"{short_name}_amplitude",
    }
    # Need to carefully grab the available values corresponding to the parameters or free_parameters, respectively.
    # NOTE: We cannot just iterate over the dict(s) themselves and check if the keys are in parameters because
    #       the parameters are de-duplicated, and thus the order can be wrong. In particular, for signal fits,
    #       B of the background fit ends up at the end of the dict because all of the other parameters are already
    #       defined for the signal fit. This approach won't have a problem with this, because we retrieve the values
    #       in the order of the parameters of the current fit component.
    # NOTE: The RPF will only have values for the free parameters in our width fit.
    values_at_minimum = {p: rpf_result.values_at_minimum[parameter_name_map[p]] for p in free_parameters}
    errors_on_parameters = {p: rpf_result.errors_on_parameters[parameter_name_map[p]] for p in free_parameters}
    covariance_matrix = {
        (a, b): rpf_result.covariance_matrix[(a, b)] for a in free_parameters for b in free_parameters
    }

    # Store the result
    width_obj.fit_object.fit_result = pachyderm.fit.FitResult(
        parameters = parameters,
        free_parameters = free_parameters,
        fixed_parameters = fixed_parameters,
        values_at_minimum = values_at_minimum,
        errors_on_parameters = errors_on_parameters,
        covariance_matrix = covariance_matrix,
        # This will be determine and set below.
        errors = np.array([]),
        x = subtracted_hist.x,
        n_fit_data_points = len(subtracted_hist.x),
        minimum_val = cost_func(*list(values_at_minimum.values())),
    )

    width_obj.fit_object.calculate_errors(x = subtracted_hist.x)

    return True

def pedestal(x: float, pedestal: float) -> float:
    """ Pedestal function.

    Note:
        If we defined this via a lambda (which would be trivial), we wouldn't be able to specify the types.
        So we define it separately.

    Args:
        x: Independent variable.
        pedestal: Pedestal value.
    Returns:
        The pedestal value.
    """
    # NOTE: We specify 0 * x so we can get proper array evaluation. If there's no explicit x
    #       dependence, it will only return a single value.
    return 0 * x + pedestal

class PedestalForDeltaEtaBackgroundDominatedRegion(pachyderm.fit.Fit):
    """ Fit a pedestal to the background dominated region of a delta eta hist.

    The initial value of the fit will be determined by the minimum y value of the histogram.

    Attributes:
        fit_range: Range used for fitting the data. Values inside of this range will be used.
        user_arguments: User arguments for the fit. Default: None.
        fit_function: Function to be fit.
        fit_result: Result of the fit. Only valid after the fit has been performed.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Finally, setup the fit function
        self.fit_function = pedestal

    def _post_init_validation(self) -> None:
        """ Validate that the fit object was setup properly.

        This can be any method that the user devises to ensure that
        all of the information needed for the fit is available.

        Args:
            None.
        Returns:
            None.
        """
        fit_range = self.fit_options.get("range", None)
        # Check that the fit range is specified
        if fit_range is None:
            raise ValueError("Fit range must be provided in the fit options.")

        # Check that the fit range is a SelectedRange (this isn't really suitable for duck typing)
        if not isinstance(fit_range, params.SelectedRange):
            raise ValueError("Must provide fit range with a selected range or a set of two values")

    def _setup(self, h: histogram.Histogram1D) -> Tuple[histogram.Histogram1D, T_FitArguments]:
        """ Setup the histogram and arguments for the fit.

        Args:
            h: Background subtracted histogram to be fit.
        Returns:
            Histogram to use for the fit, default arguments for the fit. Note that the histogram may be range
                restricted or otherwise modified here.
        """
        fit_range = self.fit_options["range"]
        restricted_range = (
            # For example, -1.2 < h.x < -0.8
            ((h.x < -1 * fit_range.min) & (h.x > -1 * fit_range.max))
            # For example, 0.8 < h.x < 1.2
            | ((h.x > fit_range.min) & (h.x < fit_range.max))
        )
        # Same conditions as above, but we need the bin edges to be inclusive.
        bin_edges_restricted_range = (
            ((h.bin_edges <= -1 * fit_range.min) & (h.bin_edges >= -1 * fit_range.max))
            | ((h.bin_edges >= fit_range.min) & (h.bin_edges <= fit_range.max))
        )
        restricted_hist = histogram.Histogram1D(
            bin_edges = h.bin_edges[bin_edges_restricted_range],
            y = h.y[restricted_range],
            errors_squared = h.errors_squared[restricted_range]
        )

        # Default arguments
        # Use the minimum of the histogram as the starting value.
        min_hist = np.min(restricted_hist.y)
        arguments: T_FitArguments = {
            "pedestal": min_hist, "error_pedestal": min_hist * 0.1,
            "limit_pedestal": (-1000, 1000),
        }

        return restricted_hist, arguments

def pedestal_with_extended_gaussian(x: Union[float, np.ndarray], mean: float, width: float,
                                    amplitude: float, pedestal: float) -> Union[float, np.ndarray]:
    """ Pedestal + extended (unnormalized) gaussian

    Args:
        x: Independent variable.
        mean: Gaussian mean.
        width: Gaussian width.
        amplitude: Amplitude of the gaussian.
        pedestal: Pedestal value.
    Returns:
        Function value.
    """
    return pedestal + pachyderm.fit.extended_gaussian(x = x, mean = mean, sigma = width, amplitude = amplitude)

class FitPedestalWithExtendedGaussian(pachyderm.fit.Fit):
    """ Fit a pedestal + extended (unnormalized) gaussian to a signal peak.

    Attributes:
        fit_range: Range used for fitting the data. Values inside of this range will be used.
        user_arguments: User arguments for the fit. Default: None.
        fit_function: Function to be fit.
        fit_result: Result of the fit. Only valid after the fit has been performed.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Finally, setup the fit function
        self.fit_function = pedestal_with_extended_gaussian

    def _post_init_validation(self) -> None:
        """ Validate that the fit object was setup properly.

        This can be any method that the user devises to ensure that
        all of the information needed for the fit is available.

        Args:
            None.
        Returns:
            None.
        """
        fit_range = self.fit_options.get("range", None)
        # Check that the fit range is specified
        if fit_range is None:
            raise ValueError("Fit range must be provided in the fit options.")

        # Check that the fit range is a SelectedRange (this isn't really suitable for duck typing)
        if not isinstance(fit_range, params.SelectedRange):
            raise ValueError("Must provide fit range with a selected range or a set of two values")

    def _setup(self, h: histogram.Histogram1D) -> Tuple[histogram.Histogram1D, T_FitArguments]:
        """ Setup the histogram and arguments for the fit.

        Args:
            h: Background subtracted histogram to be fit.
        Returns:
            Histogram to use for the fit, default arguments for the fit. Note that the histogram may be range
                restricted or otherwise modified here.
        """
        fit_range = self.fit_options["range"]
        # Restrict the range so that we only fit within the desired input.
        restricted_range = (h.x > fit_range.min) & (h.x < fit_range.max)
        restricted_hist = histogram.Histogram1D(
            # We need the bin edges to be inclusive.
            bin_edges = h.bin_edges[(h.bin_edges >= fit_range.min) & (h.bin_edges <= fit_range.max)],
            y = h.y[restricted_range],
            errors_squared = h.errors_squared[restricted_range]
        )

        # Default arguments required for the fit
        arguments: T_FitArguments = {
            "pedestal": 0, "limit_pedestal": (-1000, 1000), "error_pedestal": 0.1,
            "amplitude": 1, "limit_amplitude": (0.05, 100), "error_amplitude": 0.1 * 1,
            "mean": 0, "limit_mean": (-0.5, 0.5), "error_mean": 0.05,
            "width": 0.15, "limit_width": (0.05, 0.8), "error_width": 0.1 * 0.15,
        }

        return restricted_hist, arguments

