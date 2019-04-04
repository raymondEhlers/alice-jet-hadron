#!/usr/bin/env python

from dataclasses import dataclass
import logging
import numpy as np
from pachyderm import histogram
import scipy.optimize as optimization
from typing import Sequence, Tuple

from jet_hadron.base import analysis_objects
from jet_hadron.base import params
from jet_hadron.base.typing_helpers import Hist

import ROOT

logger = logging.getLogger(__name__)

# Shared values (TODO: to be removed...)
v2_cent_00_05_values = [1, 2, 3]
v2_cent_05_10_values = [1, 2, 3]

def fit_delta_phi_background(hist: Hist, track_pt: analysis_objects.TrackPtBin, zyam: bool = True, disable_vn: bool = True, set_fixed_vn: bool = False) -> ROOT.TF1:
    """ Fit the delta phi background. """
    delta_phi = hist

    fit_func = ROOT.TF1(
        "deltaPhiBackground",
        "[0]*1 + (2*[1]*cos(2*x) + 2*[2]*cos(3*x))",
        -0.5 * ROOT.TMath.Pi(), 1.5 * ROOT.TMath.Pi()
    )

    # Backgound
    # Pedestal
    fit_func.SetParLimits(0, 0., 100)
    if zyam:
        # Seed with ZYAM value
        fit_func.SetParameter(0, delta_phi.GetBinContent(delta_phi.GetMinimumBin()))
    fit_func.SetParName(0, "Pedestal")
    # v2
    fit_func.SetParLimits(1, -1, 1)
    fit_func.SetParName(1, "v_{2}")
    # v3
    fit_func.SetParLimits(2, -1, 1)
    fit_func.SetParName(2, "v_{3}")
    if disable_vn:
        fit_func.FixParameter(1, 0)
        fit_func.FixParameter(2, 0)
    if set_fixed_vn:
        raise ValueError("v2_assoc must be properly defined!")
        v2_assoc = (v2_cent_00_05_values[track_pt.bin] + v2_cent_05_10_values[track_pt.bin]) / 2.
        #v2_assoc = math.sqrt((v2_cent_00_05_values[track_pt.bin] + v2_cent_05_10_values[track_pt.bin]) / 2.)
        # From https://arxiv.org/pdf/1509.07334v2.pdf
        v2_jet = 0.03
        v3_assoc = 0
        v3_jet = v3_assoc
        fit_func.FixParameter(7, v2_assoc * v2_jet)
        fit_func.FixParameter(8, v3_assoc * v3_jet)

    # Set styling
    fit_func.SetLineColor(ROOT.kBlue + 2)
    fit_func.SetLineStyle(1)

    # Fit to the given histogram
    # R uses the range defined in the fit function
    # 0 ensures that the fit isn't drawn
    # Q ensures minimum printing
    # + adds the fit to function list to ensure that it is not deleted on the creation of a new fit
    delta_phi.Fit(fit_func, "RIB0")

    return fit_func

def fit_delta_phi(hist: Hist, track_pt: analysis_objects.TrackPtBin, zyam: bool = True, disable_vn: bool = True, set_fixed_vn: bool = False) -> ROOT.TF1:
    """ Define 1D gaussian fit function with one gaussian each for the near and away sides, along with a gaussian offset by +/-2Pi"""
    delta_phi = hist

    #fit_func = ROOT.TF1("symmetricGaussian","[0]*exp(-0.5*((x-[1])/[2])**2)+[3]+[4]*exp(-0.5*((x-[5])/[6])**2)+[0]*exp(-0.5*((x-[1]+2.*TMath::Pi())/[2])**2)+[4]*exp(-0.5*((x-[5]-2.*TMath::Pi())/[6])**2)", -0.5*ROOT.TMath.Pi(), 1.5*ROOT.TMath.Pi())
    # NOTE: This is not symmetric! Instead, the extra fits are because of how it wraps around. Even if our data
    #       doesn't go there, it is still relevant
    fit_func = ROOT.TF1(
        "dPhiWithGaussians",
        "[6]*1 + (2*[7]*cos(2*x) + 2*[8]*cos(3*x)) + "
        "[0]*(TMath::Gaus(x, [1], [2]) + TMath::Gaus(x, [1]-2.*TMath::Pi(), [2]) + TMath::Gaus(x, [1]+2.*TMath::Pi(), [2])) + "
        "[3]*(TMath::Gaus(x, [4], [5]) + TMath::Gaus(x, [4]-2.*TMath::Pi(), [5]) + TMath::Gaus(x, [4]+2.*TMath::Pi(), [5]))",
        -0.5 * ROOT.TMath.Pi(), 1.5 * ROOT.TMath.Pi()
    )

    # Setup parameters
    amplitude_limits = [0.0, 100.0]
    sigma_limits = [0.05, 2.0]
    # Near side
    # Amplitude
    fit_func.SetParLimits(0, amplitude_limits[0], amplitude_limits[1])
    fit_func.SetParName(0, "NS Amplitude")
    # Offset
    fit_func.FixParameter(1, 0)
    fit_func.SetParName(1, "NS Offset")
    # Sigma
    fit_func.SetParLimits(2, sigma_limits[0], sigma_limits[1])
    fit_func.SetParName(2, "NS #sigma")
    # Seed for sigma
    fit_func.SetParameter(2, sigma_limits[0])

    # Away side
    # Amplitude
    fit_func.SetParLimits(3, amplitude_limits[0], amplitude_limits[1])
    fit_func.SetParName(3, "AS Amplitude")
    # Offset
    fit_func.FixParameter(4, ROOT.TMath.Pi())
    fit_func.SetParName(4, "AS Offset")
    # Sigma
    fit_func.SetParLimits(5, sigma_limits[0], sigma_limits[1])
    fit_func.SetParName(5, "AS #sigma")
    # Seed for sigma
    fit_func.SetParameter(5, sigma_limits[0])

    # Backgound
    # Pedestal
    fit_func.SetParLimits(6, 0., 100)
    if zyam:
        # Seed with ZYAM value
        fit_func.SetParameter(6, delta_phi.GetBinContent(delta_phi.GetMinimumBin()))
    fit_func.SetParName(6, "Pedestal")
    # v2
    fit_func.SetParLimits(7, -1, 1)
    fit_func.SetParName(7, "v_{2}")
    # v3
    fit_func.SetParLimits(8, -1, 1)
    fit_func.SetParName(8, "v_{3}")
    if disable_vn:
        fit_func.FixParameter(7, 0)
        fit_func.FixParameter(8, 0)
    if set_fixed_vn:
        raise ValueError("v2_assoc must be properly defined!")
        v2_assoc = (v2_cent_00_05_values[track_pt.bin] + v2_cent_05_10_values[track_pt.bin]) / 2.
        #v2_assoc = math.sqrt((v2_cent_00_05_values[track_pt.bin] + v2_cent_05_10_values[track_pt.bin]) / 2.)
        # From https://arxiv.org/pdf/1509.07334v2.pdf
        v2_jet = 0.03
        v3_assoc = 0
        v3_jet = v3_assoc
        fit_func.FixParameter(7, v2_assoc * v2_jet)
        fit_func.FixParameter(8, v3_assoc * v3_jet)

    # Set styling
    fit_func.SetLineColor(ROOT.kRed + 2)
    fit_func.SetLineStyle(1)

    # Fit to the given histogram
    # R uses the range defined in the fit function
    # 0 ensures that the fit isn't drawn
    # Q ensures minimum printing
    # + adds the fit to function list to ensure that it is not deleted on the creation of a new fit
    delta_phi.Fit(fit_func, "RIB0")

    # And return the fit
    return fit_func

def fit_delta_eta(hist: Hist, track_pt: analysis_objects.TrackPtBin, zyam: bool = True, disable_vn: bool = True, set_fixed_vn: bool = False) -> ROOT.TF1:
    """ dEta near-side fit implementation. """
    fit_func = ROOT.TF1("deltaEtaNS", "[6] + [0]*TMath::Gaus(x, [1], [2])", -1, 1)

    # Setup parameters
    amplitude_limits = [0.0, 100.0]
    sigma_limits = [0.05, 2.0]
    # Near side
    # Amplitude
    fit_func.SetParLimits(0, amplitude_limits[0], amplitude_limits[1])
    fit_func.SetParName(0, "NS Amplitude")
    # Offset
    fit_func.FixParameter(1, 0)
    fit_func.SetParName(1, "NS Offset")
    # Sigma
    fit_func.SetParLimits(2, sigma_limits[0], sigma_limits[1])
    fit_func.SetParName(2, "NS #sigma")
    # Seed for sigma
    fit_func.SetParameter(2, sigma_limits[0])

    # Backgound
    fit_func.SetParLimits(6, 0., 100)
    if zyam:
        # Seed with ZYAM value
        fit_func.SetParameter(6, hist.GetBinContent(hist.GetMinimumBin()))
    fit_func.SetParName(6, "Pedestal")

    # Set styling
    fit_func.SetLineColor(ROOT.kGreen + 2)
    fit_func.SetLineStyle(1)

    # Fit to the given histogram
    # R uses the range defined in the fit function
    # 0 ensures that the fit isn't drawn
    # Q ensures minimum printing
    # + adds the fit to function list to ensure that it is not deleted on the creation of a new fit
    hist.Fit(fit_func, "RIB0")

    # And return the fit
    return fit_func

def fit_1d_mixed_event_normalization(hist: Hist, delta_phi_limits: Sequence[float]) -> ROOT.TF1:
    """ Alternative to determine the mixed event normalization.

    A lienar function is fit to the dPhi mixed event normalization for some predefined range.
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

def fit_gaussian_to_histogram(h: histogram.Histogram1D, inputs: GaussianFitInputs) -> Tuple[float, np.array]:
    """ Fit a guassian to a delta phi signal peak using ``scipy.optimize.curvefit``.

    Args:
        h: Background subtracted histogram to be fit.
        inputs: Fit inputs in the form of a ``GaussianFitInputs`` dataclass. Must specify the mean, the initial width,
            and the fit range.
    Returns:
        (width, covariance matrix)
    """
    restricted_range = (h.x > inputs.fit_range.min) & (h.x < inputs.fit_range.max)
    width, covariance_matrix = optimization.curve_fit(
        f = lambda x, w: gaussian(x, inputs.mean, w),
        xdata = h.x[restricted_range], ydata = h.y[restricted_range], p0 = inputs.initial_width,
        sigma = h.errors[restricted_range],
    )

    return width, covariance_matrix

