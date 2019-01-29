#!/usr/bin/env python

""" Plots related to the response matrix.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import logging

import ctypes
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Any, Dict

from pachyderm import histogram
from pachyderm import utils

from jet_hadron.base import analysis_objects
from jet_hadron.base import params
from jet_hadron.base.typing_helpers import Hist
from jet_hadron.plot import base as plot_base

import ROOT

logger = logging.getLogger(__name__)

Analyses = Dict[Any, analysis_objects.JetHBase]

def plot_response_matrix_and_errors(obj: analysis_objects.JetHBase) -> None:
    """ Plot the 2D response matrix and response matrix errors hists using ROOT.

    Args:
        obj: The response matrix analysis object.
    """
    for hist, plot_errors_hist in [(obj.response_matrix, False),
                                   (obj.response_matrix_errors, True)]:
        # Plot response matrix
        _plot_response_matrix(
            hist = hist,
            plot_errors_hist = plot_errors_hist,
            output_info = obj.output_info,
        )

def _plot_response_matrix(hist: Hist, plot_errors_hist: bool, output_info: analysis_objects.PlottingOutputWrapper) -> None:
    """ Plot the given response matrix related 2D hist.

    Args:
        hist: The response matrix related 2D hist.
        errors_hist: True if the hist is the response matrix errors hist.
        output_info: Output information.
    Returns:
        None
    """
    # Setup
    canvas = ROOT.TCanvas("canvas", "canvas")
    canvas.SetLogz(1)

    # Plot the histogram
    #logger.debug(f"Response matrix n jets: {response_matrix.Integral()}".format(response_matrix.Integral()))
    name = "Response Matrix"
    if plot_errors_hist:
        name += " Errors"
    hist.SetTitle(name)
    hist.GetXaxis().SetTitle("#mathit{p}_{#mathrm{T,jet}}^{det} (GeV/#it{c})")
    hist.GetYaxis().SetTitle("#mathit{p}_{#mathrm{T,jet}}^{part} (GeV/#it{c})")
    hist.Draw("colz")

    # Set the final axis ranges.
    # Z axis
    min_val = ctypes.c_double(0)
    max_val = ctypes.c_double(0)
    hist.GetMinimumAndMaximum(min_val, max_val)
    # * 1.1 to put it slightly above the max value
    # min_val doesn't work here, because there are some entries at 0
    hist.GetZaxis().SetRangeUser(10e-7, max_val.value * 1.1)

    # Save
    output_name = "response_matrix"
    if plot_errors_hist:
        output_name += "_errors"
    plot_base.save_plot(output_info, canvas, output_name)

def plot_response_spectra(plot_labels: plot_base.PlotLabels,
                          output_name: str,
                          merged_analysis: analysis_objects.JetHBase,
                          pt_hard_analyses: Analyses,
                          hist_attribute_name: str):
    """ Plot 1D response spectra.

    Args:
        plot_labels: Labels for the plot.
        output_name: Name under which the plot should be stored.
        merged_analysis: Full merged together analysis object.
        pt_hard_analyses: Pt hard dependent analysis objects to be plotted.
        hist_attribute_name: Name of the attribute under which the histogram is stored.
    """
    # Setup
    fig, ax = plt.subplots(figsize = (8, 6))
    # NOTE: "husl" is also a good option.
    colors = iter(sns.color_palette(
        palette = "Blues_d", n_colors = len(pt_hard_analyses)
    ))
    plot_labels.apply_labels(ax)

    # First, we plot the merged analysis. This is the sum of the various pt hard bin contributions.
    merged_hist = utils.recursive_getattr(merged_analysis, hist_attribute_name)
    merged_hist = histogram.Histogram1D.from_existing_hist(merged_hist)
    ax.errorbar(
        merged_hist.x, merged_hist.y,
        yerr = merged_hist.errors,
        label = "Merged",
        color = "black",
    )

    # Now, we plot the pt hard dependent hists
    for (key_index, analysis), color in zip(pt_hard_analyses.items(), colors):
        # Determine the proper label.
        logger.debug(f"key_index: {key_index}")
        label = params.generate_pt_range_string(
            pt_bin = key_index.pt_hard_bin,
            lower_label = "",
            upper_label = "hard",
            only_show_lower_value_for_last_bin = True,
        )

        # Plot the histogram.
        hist = utils.recursive_getattr(analysis, hist_attribute_name)
        h = histogram.Histogram1D.from_existing_hist(hist)
        ax.plot(
            h.x, h.y,
            label = label,
            color = color,
        )

    # Final presentation settings
    # Ensure that the max is never beyond 300 for better presentation.
    max_limit = np.max(merged_hist.x)
    if max_limit > 300:
        max_limit = 300
    ax.set_xlim(0, max_limit)
    ax.set_yscale("log")
    ax.legend(loc = "best")
    fig.tight_layout()

    plot_base.save_plot(merged_analysis.output_info, fig, output_name)
