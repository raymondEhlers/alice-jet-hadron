#!/usr/bin/env python

""" Plots related to the response matrix.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import logging

import ctypes
import matplotlib
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

def plot_particle_level_spectra(analyses: Analyses, plot_with_ROOT: bool = False):
    """ Plot the particle level spectra associated with the response.

    """
    args: Dict[str, Any] = {}
    if plot_with_ROOT:
        _plot_particle_level_spectra_with_ROOT(**args)
    else:
        _plot_particle_level_spectra_with_matplotlib(**args)

def _plot_particle_level_spectra_with_matplotlib(temp):
    ...

def _plot_particle_level_spectra_with_ROOT(temp):
    """ Plot the particle level spectra with ROOT.

    Args:

    """
    ...

def plot_response_matrix_and_errors(obj: analysis_objects.JetHBase,
                                    plot_with_ROOT: bool = False) -> None:
    """ Plot the 2D response matrix and response matrix errors hists using ROOT.

    Args:
        obj: The response matrix analysis object.
        plot_with_ROOT: True if the plot should be done via ROOT. Default: False
    """
    for hist, plot_errors_hist in [(obj.response_matrix, False),
                                   (obj.response_matrix_errors, True)]:
        # Plot response matrix
        _plot_response_matrix(
            hist = hist,
            plot_errors_hist = plot_errors_hist,
            output_info = obj.output_info,
            plot_with_ROOT = plot_with_ROOT,
            reaction_plane_orientation = obj.reaction_plane_orientation,
        )

def _plot_response_matrix(hist: Hist,
                          plot_errors_hist: bool,
                          output_info: analysis_objects.PlottingOutputWrapper,
                          plot_with_ROOT: bool,
                          reaction_plane_orientation: params.ReactionPlaneOrientation) -> None:
    """ Plot the given response matrix related 2D hist.

    Args:
        hist: The response matrix related 2D hist.
        errors_hist: True if the hist is the response matrix errors hist.
        output_info: Output information.
        plot_with_ROOT: True if the plot should be done via ROOT.
        reaction_plane_orientation: Reaction plane orientation of the plot.
    Returns:
        None
    """
    # Determine parameters
    name = "Response Matrix"
    if plot_errors_hist:
        name += " Errors"
    name += f", event plane orientation {reaction_plane_orientation.display_str()}"
    output_name = "response_matrix"
    if plot_errors_hist:
        output_name += "_errors"
    x_label = r"$\mathit{p}_{\mathrm{T,jet}}^{\mathrm{det}} \mathrm{(GeV/\it{c})}$"
    y_label = r"$\mathit{p}_{\mathrm{T,jet}}^{\mathrm{part}} \mathrm{(GeV/\it{c})}$"

    # Determine args and call
    args = {
        "name": name, "x_label": x_label, "y_label": y_label, "output_name": output_name,
        "hist": hist,
        "plot_errors_hist": plot_errors_hist,
        "output_info": output_info,
    }

    if plot_with_ROOT:
        _plot_response_matrix_with_ROOT(**args)
    else:
        _plot_response_matrix_with_matplotlib(**args)

def _plot_response_matrix_with_matplotlib(name: str, x_label: str, y_label: str, output_name: str,
                                          hist: Hist,
                                          plot_errors_hist: bool,
                                          output_info: analysis_objects.PlottingOutputWrapper) -> None:
    """ Underlying function to actually plot a response matrix with matplotlib.

    Args:
        name: Name of the histogram.
        x_label: X axis label.
        y_label: Y axis label.
        output_name: Output name of the histogram.
        hist: The response matrix related 2D hist.
        errors_hist: True if the hist is the response matrix errors hist.
        output_info: Output information.
    Returns:
        None
    """
    # Setup
    fig, ax = plt.subplots(figsize = (8, 6))

    # Convert the histogram
    X, Y, hist_array = histogram.get_array_from_hist2D(
        hist = hist,
        set_zero_to_NaN = True,
        return_bin_edges = True,
    )

    # Determine and fill args
    kwargs = {}
    # Create a log z axis heat map.
    kwargs["norm"] = matplotlib.colors.LogNorm(vmin = np.nanmin(hist_array), vmax = np.nanmax(hist_array))
    logger.debug(f"min: {np.nanmin(hist_array)}, max: {np.nanmax(hist_array)}")
    # The colormap that we use is the default from sns.heatmap
    kwargs["cmap"] = plot_base.prepare_colormap(sns.cm.rocket)
    # Label is included so we could use a legend if we want
    kwargs["label"] = name

    logger.debug("kwargs: {}".format(kwargs))

    # Determine the edges
    extent = [
        np.amin(X), np.amax(X),
        np.amin(Y), np.amax(Y)
    ]
    # Finally, create the plot
    ax_from_imshow = ax.imshow(
        hist_array.T, extent = extent,
        interpolation = "nearest", aspect = "auto", origin = "lower",
        **kwargs
    )

    # Add colorbar
    # It needs to be defined on the figure because it is stored in a separate axis.
    fig.colorbar(ax_from_imshow, ax = ax)

    # Final styling
    ax.set_title(name)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    fig.tight_layout()

    # Save and cleanup
    output_name += "_mpl"
    plot_base.save_plot(output_info, fig, output_name)
    plt.close(fig)

def _plot_response_matrix_with_ROOT(name: str, x_label: str, y_label: str, output_name: str,
                                    hist: Hist,
                                    plot_errors_hist: bool,
                                    output_info: analysis_objects.PlottingOutputWrapper) -> None:
    """ Underlying function to actually plot a response matrix with ROOT.

    Args:
        name: Name of the histogram.
        x_label: X axis label.
        y_label: Y axis label.
        output_name: Output name of the histogram.
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
    hist.SetTitle(name)
    hist.GetXaxis().SetTitle(params.use_label_with_root(x_label))
    hist.GetYaxis().SetTitle(params.use_label_with_root(y_label))
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
    output_name += "_ROOT"
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
    # Update the plot labels as appropriate using the reaction plane orientation information
    # Help out mypy....
    assert plot_labels.title is not None
    plot_labels.title = plot_labels.title + f", reaction plane orientation {merged_analysis.reaction_plane_orientation.display_str()}"
    # Now we can apply the plot labels
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

    # Save and cleanup
    plot_base.save_plot(merged_analysis.output_info, fig, output_name)
    plt.close(fig)
