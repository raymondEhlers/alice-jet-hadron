#!/usr/bin/env python

from dataclasses import dataclass
import logging
import numpy as np
from pachyderm import histogram
import pprint
import scipy.optimize as optimization
from typing import Sequence, Tuple

from jet_hadron.base import params
from jet_hadron.base.typing_helpers import Hist

import ROOT

logger = logging.getLogger(__name__)

def print_root_fit_parameters(fit) -> None:
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

    A lienar function is fit to the dPhi mixed event normalization for some predefined range.

    Args:
        hist: 1D mixed event histogram to be fit.
        delta_phi_limits: Min and max fit limits in delta phi.
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

    A lienar function is fit to the dPhi-dEta mixed event normalization for some predefined range.

    Args:
        hist: 2D mixed event histogram to be fit.
        delta_phi_limits: Min and max fit limits in delta phi.
        delta_eta_limits: Min and max fit limits in delta eta.
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

def fit_pedestal_to_delta_eta_background_dominated_region(h: histogram.Histogram1D,
                                                          fit_range: params.SelectedRange) -> Tuple[float, float]:
    """ Fit a pedestal to a histogram using ``scipy.optimize.curvefit``.

    The initial value of the fit will be determined by the minimum y value of the histogram.

    Args:
        h: Histogram to be fit.
        fit_range: Min and max values within which the fit will be performed.
    Returns:
        (constant, error)
    """
    # For example, -1.2 < h.x < -0.8
    negative_restricted_range = (h.x < -1 * fit_range.min) & (h.x > -1 * fit_range.max)
    # For example, 0.8 < h.x < 1.2
    positive_restricted_range = (h.x > fit_range.min) & (h.x < fit_range.max)
    restricted_range = negative_restricted_range | positive_restricted_range
    constant, covariance_matrix = optimization.curve_fit(
        f = lambda x, c: c,
        xdata = h.x[restricted_range], ydata = h.y[restricted_range], p0 = np.min(h.y[restricted_range]),
        sigma = h.errors[restricted_range],
    )

    # Error reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
    error = float(np.sqrt(np.diag(covariance_matrix)))

    return float(constant), error

@dataclass
class GaussianFitInputs:
    """ Storage for Gaussian fit inputs.

    Attributes:
        mean: Mean value of the Gaussian.
        initial_width: Initial value of the Gaussian fit.
        fit_range: Min and max values within which the fit will be performed.
    """
    mean: float
    initial_width: float
    fit_range: params.SelectedRange

def gaussian(x: float, mu: float, sigma: float) -> float:
    """ Normalized gaussian.

    Args:
        x: Indepenednt variable.
        mu: Mean.
        sigma: Width.
    Returns:
        Normalized gaussian value.
    """
    return 1 / np.sqrt(2 * np.pi * sigma ** 2) * np.exp(-1 / 2 * ((x - mu) / sigma) ** 2)

def fit_gaussian_to_histogram(h: histogram.Histogram1D, inputs: GaussianFitInputs) -> Tuple[float, float]:
    """ Fit a guassian to a delta phi signal peak using ``scipy.optimize.curvefit``.

    Args:
        h: Background subtracted histogram to be fit.
        inputs: Fit inputs in the form of a ``GaussianFitInputs`` dataclass. Must specify the mean, the initial width,
            and the fit range.
    Returns:
        (width, error)
    """
    restricted_range = (h.x > inputs.fit_range.min) & (h.x < inputs.fit_range.max)
    width, covariance_matrix = optimization.curve_fit(
        f = lambda x, w: gaussian(x, inputs.mean, w),
        xdata = h.x[restricted_range], ydata = h.y[restricted_range], p0 = inputs.initial_width,
        sigma = h.errors[restricted_range],
    )

    # Error reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
    error = float(np.sqrt(np.diag(covariance_matrix)))

    return float(width), error

