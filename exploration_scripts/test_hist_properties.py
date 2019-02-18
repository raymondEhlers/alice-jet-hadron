#!/usr/bin/env python

""" Tests for histogram scaling and division.

"""

import numpy as np
from typing import Tuple

import ROOT

Hist = ROOT.TH1

def create_hists() -> Tuple[Hist, Hist]:
    bins = [100, 0, 10, 100, 0, 20]
    hist_raw = ROOT.TH2F("raw", "raw", *bins)
    hist_raw.Sumw2()
    hist_raw.Fill(5, 5)
    hist_raw.Fill(5, 5)
    hist_mixed = ROOT.TH2F("mixed", "mixed", *bins)
    hist_mixed.Sumw2()
    hist_mixed.Fill(5, 5)
    hist_mixed.Fill(5, 5)

    return hist_raw, hist_mixed

def scale_by_bin_width(hist: Hist) -> None:
    # The first bin should always exist!
    bin_width_scale_factor = hist.GetXaxis().GetBinWidth(1)
    # Because of a ROOT quirk, even a TH1* hist has a Y and Z axis, with 1 bin
    # each. This bin has bin width 1, so it doesn't change anything if we multiply
    # by that bin width. So we just do it for all histograms.
    # This has the benefit that we don't need explicit dependence on an imported
    # ROOT package.
    bin_width_scale_factor *= hist.GetYaxis().GetBinWidth(1)
    bin_width_scale_factor *= hist.GetZaxis().GetBinWidth(1)

    hist.Scale(1.0 / bin_width_scale_factor)

def test_whether_scale_factors_cancel() -> None:
    hist_raw, hist_mixed = create_hists()

    # Clone and scale a new set of hists.
    hist_raw_scaled = hist_raw.Clone(f"{hist_raw.GetName()}_scaled")
    hist_mixed_scaled = hist_mixed.Clone(f"{hist_mixed.GetName()}_scaled")
    scale_by_bin_width(hist_raw_scaled)
    scale_by_bin_width(hist_mixed_scaled)

    # Divide and obtain bin value
    hist_raw.Divide(hist_mixed)
    original_value = hist_raw.GetBinContent(hist_raw.GetXaxis().FindBin(5), hist_raw.GetYaxis().FindBin(5))
    original_value_error = hist_raw.GetBinError(hist_raw.GetXaxis().FindBin(5), hist_raw.GetYaxis().FindBin(5))

    # Divide scaled hists
    hist_raw_scaled.Divide(hist_mixed_scaled)
    scaled_value = hist_raw_scaled.GetBinContent(hist_raw_scaled.GetXaxis().FindBin(5), hist_raw_scaled.GetYaxis().FindBin(5))
    scaled_value_error = hist_raw_scaled.GetBinError(hist_raw_scaled.GetXaxis().FindBin(5), hist_raw_scaled.GetYaxis().FindBin(5))

    print("If the follow values are equal, then it scaling by the same factor and dividing cancels out the scale factor when dividing.")
    print(f"original_value: {original_value}, scaled_value: {scaled_value}")
    print(f"original_value_error: {original_value_error}, scaled_value_error: {scaled_value_error}")
    assert np.isclose(original_value, scaled_value)
    assert np.isclose(original_value_error, scaled_value_error)

def test_whether_projections_preserve_scale_factors() -> None:
    hist_raw, _ = create_hists()

    # Create, scale and project the pre projection scaling hist.
    hist_raw_pre_projection_scaling = hist_raw.Clone(f"{hist_raw.GetName}_pre_projection_scaling")
    hist_raw_pre_projection_scaling.Scale(10)
    pre_projection_scaling = hist_raw_pre_projection_scaling.ProjectionX()

    pre_projection_scaling_value = pre_projection_scaling.GetBinContent(pre_projection_scaling.GetXaxis().FindBin(5))
    pre_projection_scaling_error = pre_projection_scaling.GetBinError(pre_projection_scaling.GetXaxis().FindBin(5))

    # Project and scale the hist
    hist_raw_projection = hist_raw.ProjectionX()
    hist_raw_projection.Scale(10)

    hist_raw_projection_value = hist_raw_projection.GetBinContent(hist_raw_projection.GetXaxis().FindBin(5))
    hist_raw_projection_error = hist_raw_projection.GetBinError(hist_raw_projection.GetXaxis().FindBin(5))

    print(f"hist_raw_projection: {hist_raw_projection_value}, pre_projection_scaling: {pre_projection_scaling_value}")
    print(f"hist_raw_projection_error: {hist_raw_projection_error}, pre_projection_scaling_error: {pre_projection_scaling_error}")
    assert np.isclose(hist_raw_projection_value, pre_projection_scaling_value)
    assert np.isclose(hist_raw_projection_error, pre_projection_scaling_error)

if __name__ == "__main__":
    test_whether_scale_factors_cancel()
    test_whether_projections_preserve_scale_factors()
