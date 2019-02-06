#!/usr/bin/env python

""" Plots related to the response matrix.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import logging

import ctypes
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from typing import Any, Dict, Sequence, Tuple

from pachyderm import histogram
from pachyderm import utils

from jet_hadron.base import analysis_objects
from jet_hadron.base import labels
from jet_hadron.base import params
from jet_hadron.base.typing_helpers import Hist
from jet_hadron.plot import base as plot_base

import ROOT

logger = logging.getLogger(__name__)

Analyses = Dict[Any, analysis_objects.JetHBase]

def plot_particle_level_spectra(ep_analyses: Analyses,
                                output_info: analysis_objects.PlottingOutputWrapper,
                                plot_with_ROOT: bool = False) -> None:
    """ Plot the particle level spectra associated with the response.

    Args:
        ep_analyses: The event plane dependent final response matrices.
        output_info: Output information.
        plot_with_ROOT: True if the plot should be done via ROOT. Default: False
    Returns:
        None. The spectra are plotted and saved.
    """
    # Pull out the dict because we need to grab individual analyses for some labeling information, which doesn't
    # play well with generators (the generator will be exhausted).
    ep_analyses = dict(ep_analyses)
    kwargs: Dict[str, Any] = {
        "ep_analyses": ep_analyses,
        "output_name": "particle_level_spectra",
        "output_info": output_info,
    }

    if plot_with_ROOT:
        _plot_particle_level_spectra_with_ROOT(**kwargs)
    else:
        _plot_particle_level_spectra_with_matplotlib(**kwargs)

def _plot_particle_level_spectra_with_matplotlib(ep_analyses: Analyses,
                                                 output_name: str,
                                                 output_info: analysis_objects.PlottingOutputWrapper) -> None:
    """ Plot the particle level spectra with matplotlib.

    Args:

    """
    ...

def _plot_particle_level_spectra_with_ROOT(ep_analyses: Analyses,
                                           output_name: str,
                                           output_info: analysis_objects.PlottingOutputWrapper) -> None:
    """ Plot the particle level spectra with ROOT.

    Args:
        ep_analyses: The final event plane dependent response matrix analysis objects.
        output_name: Name of the output plot.
        output_info: Output information.
    Returns:
        None. The created canvas is plotted and saved.
    """
    # Setup
    # Aesthetics
    # Colors and markers are from Joel's plots.
    colors = [ROOT.kBlack, ROOT.kBlue - 7, 8, ROOT.kRed - 4]
    markers = [ROOT.kFullDiamond, ROOT.kFullSquare, ROOT.kFullTriangleUp, ROOT.kFullCircle]

    # Canvas
    canvas = ROOT.TCanvas("canvas", "canvas")
    canvas.SetTopMargin(0.04)
    canvas.SetLeftMargin(0.12)
    canvas.SetRightMargin(0.04)
    canvas.SetBottomMargin(0.15)
    # These are spectra, so it makes sense to draw it in a log scale.
    canvas.SetLogy(True)

    # Legend
    legend = ROOT.TLegend(0.14, 0.17, 0.42, 0.47)
    # Remove border
    legend.SetBorderSize(0)
    # Increase text size
    legend.SetTextSize(0.06)
    # Make the legend transparent
    legend.SetFillStyle(0)

    # Retrieve the inclusive analysis to complete the labeling.
    # All of the parameters retrieved here are shared by all analyses.
    inclusive = next(iter(ep_analyses.values()))

    # Main labeling
    latex_labels = []
    # ALICE + collision energy
    latex_labels.append(ROOT.TLatex(
        0.595, 0.90,
        labels.use_label_with_root(
            rf"{inclusive.alice_label.display_str()}\:{inclusive.collision_energy.display_str()}"
        )
    ))
    # Collision system + event activity
    # We want the centrality to appear between the cross symbol and Pb--Pb
    embedded_additional_label = inclusive.event_activity.display_str()
    latex_labels.append(ROOT.TLatex(
        0.5625, 0.84,
        labels.use_label_with_root(
            rf"{inclusive.collision_system.display_str(embedded_additional_label = embedded_additional_label)}"
        ),
    ))
    # Particle level spectra range in detector pt.
    particle_level_spectra_bin = inclusive.task_config["particle_level_spectra"]["particle_level_spectra_bin"]
    latex_labels.append(ROOT.TLatex(
        0.605, 0.78,
        labels.pt_range_string(
            particle_level_spectra_bin,
            lower_label = "T,jet",
            upper_label = "det",
        ),
    ))
    # Constituent cuts
    latex_labels.append(ROOT.TLatex(
        0.5675, 0.70,
        labels.use_label_with_root(labels.constituent_cuts(additional_label = "det")),
    ))
    # Leading hadron bias
    latex_labels.append(ROOT.TLatex(
        0.6275, 0.625,
        labels.use_label_with_root(inclusive.leading_hadron_bias.display_str(additional_label = "det")),
    ))
    # Jet finding
    latex_labels.append(ROOT.TLatex(0.71, 0.56, labels.use_label_with_root(labels.jet_finding())))

    x_label = labels.use_label_with_root(
        labels.jet_pt_display_label(upper_label = "part") + r"\:" + labels.momentum_units_label_gev()
    )
    y_label = r"\mathrm{dN}/\mathrm{d}\mathit{p}_{\mathrm{T}}"

    # Plot the actual hists. The inclusive orientation will be plotted first.
    for i, (analysis, color, marker) in enumerate(zip(ep_analyses.values(), colors, markers)):
        # The hist to be plotted. We explicitly retrieve it for convenience.
        hist = analysis.particle_level_spectra

        # Set the titles
        hist.SetTitle("")
        hist.GetXaxis().SetTitle(x_label)
        full_y_label = y_label
        if analysis.task_config["particle_level_spectra"]["normalize_by_n_jets"]:
            full_y_label = r"(1/\mathrm{N}_{\mathrm{jets}})" + y_label
        hist.GetYaxis().SetTitle(full_y_label)

        # Style each individual hist. In principle, we could do this for only one # hist and then set the
        # axis labels to empty for the rest, but then we would have to empty out the labels. This is just,
        # as easy, and then we don't have to deal with changing the labels.
        # Enlarge axis title size
        hist.GetXaxis().SetTitleSize(0.055)
        hist.GetYaxis().SetTitleSize(0.055)
        # Ensure there is enough space
        hist.GetXaxis().SetTitleOffset(1.15)
        hist.GetYaxis().SetTitleOffset(1.05)
        # Enlarge axis label size
        hist.GetXaxis().SetLabelSize(0.06)
        hist.GetYaxis().SetLabelSize(0.06)
        # Center axis title
        hist.GetXaxis().CenterTitle(True)
        hist.GetYaxis().CenterTitle(True)

        # View the interesting range
        # Note that this must be set after removing any bins that we might want to remove,
        # so we set it when plotting.
        hist.GetXaxis().SetRangeUser(0, analysis.task_config["particle_level_spectra"]["particle_level_max_pt"])

        # Set histogram aesthetics
        #logger.debug(f"color: {color}")
        hist.SetLineColor(color)
        hist.SetMarkerColor(color)
        hist.SetMarkerStyle(marker)
        # Increase marker size slightly
        hist.SetMarkerSize(1.1)

        # Offset points
        # See: https://root.cern.ch/root/roottalk/roottalk03/2765.html
        if analysis.task_config["particle_level_spectra"]["plot_points_with_offset"]:
            shift = i * 0.1 * hist.GetBinWidth(1)
            xAxis = hist.GetXaxis()
            xAxis.SetLimits(xAxis.GetXmin() + shift, xAxis.GetXmax() + shift)

        # Store hist in legend
        # Remap "inclusive" -> "all" for prior consistency.
        label = analysis.reaction_plane_orientation.display_str()
        if analysis.reaction_plane_orientation == params.ReactionPlaneOrientation.inclusive:
            label = "All"
        legend.AddEntry(hist, label)

        # Last, we draw the actual hist.
        hist.Draw("same")

    # Draw all of the labels and the legend.
    for tex in latex_labels:
        tex.SetNDC(True)
        tex.Draw()
    legend.Draw()

    # Finally, save the plot
    output_name += "_ROOT"
    plot_base.save_plot(output_info, canvas, output_name)
    # Also save the plot as a c macro
    canvas.SaveAs(os.path.join(output_info.output_prefix, output_name + ".C"))

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
    name += f", {reaction_plane_orientation.display_str()} event plane orientation"
    output_name = "response_matrix"
    if plot_errors_hist:
        output_name += "_errors"
    x_label = "$%(pt_label)s %(units_label)s$" % {
        "pt_label": labels.jet_pt_display_label(upper_label = r"\mathrm{det}"),
        "units_label": labels.momentum_units_label_gev(),
    }
    y_label = "$%(pt_label)s %(units_label)s$" % {
        "pt_label": labels.jet_pt_display_label(upper_label = r"\mathrm{part}"),
        "units_label": labels.momentum_units_label_gev(),
    }

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
    hist.GetXaxis().SetTitle(labels.use_label_with_root(x_label))
    hist.GetYaxis().SetTitle(labels.use_label_with_root(y_label))
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
                          hist_attribute_name: str,
                          plot_with_ROOT: bool = False) -> None:
    """ Plot 1D response spectra.

    Args:
        plot_labels: Labels for the plot.
        output_name: Name under which the plot should be stored.
        merged_analysis: Full merged together analysis object.
        pt_hard_analyses: Pt hard dependent analysis objects to be plotted.
        hist_attribute_name: Name of the attribute under which the histogram is stored.
        plot_with_ROOT: True if the plot should be done via ROOT.
    """
    # Setup
    # NOTE: "husl" is also a good option.
    colors = sns.color_palette(
        palette = "Blues_d", n_colors = len(pt_hard_analyses)
    )
    # Update the plot labels as appropriate using the reaction plane orientation information
    # Help out mypy....
    assert plot_labels.title is not None
    # The pt hard spectra doesn't have a reaction plane orientation, so add it to the title if the
    # attribute is available
    if hasattr(merged_analysis, "reaction_plane_orientation"):
        plot_labels.title = plot_labels.title + f", {merged_analysis.reaction_plane_orientation.display_str()} reaction plane orientation"

    kwargs = {
        "plot_labels": plot_labels,
        "output_name": output_name,
        "merged_analysis": merged_analysis,
        "pt_hard_analyses": pt_hard_analyses,
        "hist_attribute_name": hist_attribute_name,
        "colors": colors,
    }

    if plot_with_ROOT:
        _plot_response_spectra_with_ROOT(**kwargs)
    else:
        _plot_response_spectra_with_matplotlib(**kwargs)

def _plot_response_spectra_with_matplotlib(plot_labels: plot_base.PlotLabels,
                                           output_name: str,
                                           merged_analysis: analysis_objects.JetHBase,
                                           pt_hard_analyses: Analyses,
                                           hist_attribute_name: str,
                                           colors: Sequence[Tuple[float, float, float]]) -> None:
    """ Plot 1D response spectra with matplotlib.

    Args:
        plot_labels: Labels for the plot.
        output_name: Name under which the plot should be stored.
        merged_analysis: Full merged together analysis object.
        pt_hard_analyses: Pt hard dependent analysis objects to be plotted.
        hist_attribute_name: Name of the attribute under which the histogram is stored.
        colors: List of colors to be used for plotting the pt hard spectra.
    """
    # Setup
    fig, ax = plt.subplots(figsize = (8, 6))
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
        label = labels.pt_range_string(
            pt_bin = key_index.pt_hard_bin,
            lower_label = "T",
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
    ax.legend(loc = "best", frameon = False)
    fig.tight_layout()

    # Save and cleanup
    output_name += "_mpl"
    plot_base.save_plot(merged_analysis.output_info, fig, output_name)
    plt.close(fig)

def _plot_response_spectra_with_ROOT(plot_labels: plot_base.PlotLabels,
                                     output_name: str,
                                     merged_analysis: analysis_objects.JetHBase,
                                     pt_hard_analyses: Analyses,
                                     hist_attribute_name: str,
                                     colors: Sequence[Tuple[float, float, float]]) -> None:
    """ Plot 1D response spectra with ROOT.

    Args:
        plot_labels: Labels for the plot.
        output_name: Name under which the plot should be stored.
        merged_analysis: Full merged together analysis object.
        pt_hard_analyses: Pt hard dependent analysis objects to be plotted.
        hist_attribute_name: Name of the attribute under which the histogram is stored.
        colors: List of colors to be used for plotting the pt hard spectra.
    """
    # Setup
    canvas = ROOT.TCanvas("canvas", "canvas")
    canvas.SetLogy(True)
    # Legend
    legend = ROOT.TLegend(0.37, 0.55, 0.9, 0.9)
    legend.SetHeader(r"\mathit{p}_{\mathrm{T}}\:\mathrm{bins}", "C")
    # Increase text size
    legend.SetTextSize(0.025)
    # Use two columns because we have a lot of entries.
    legend.SetNColumns(2)
    # Remove the legend border
    legend.SetBorderSize(0)
    # Make the legend transparent
    legend.SetFillStyle(0)

    # First, we plot the merged analysis. This is the sum of the various pt hard bin contributions.
    merged_hist = utils.recursive_getattr(merged_analysis, hist_attribute_name)
    # Apply axis labels (which must be set on the hist)
    plot_labels.apply_labels(merged_hist)
    # Style the merged hist to ensure that it is possible to see the points
    merged_hist.SetMarkerStyle(ROOT.kFullCircle)
    merged_hist.SetMarkerSize(1)
    merged_hist.SetMarkerColor(ROOT.kBlack)
    merged_hist.SetLineColor(ROOT.kBlack)
    # Ensure that the max is never beyond 300 for better presentation.
    max_limit = merged_hist.GetXaxis().GetXmax()
    if max_limit > 300:
        max_limit = 300
    merged_hist.GetXaxis().SetRangeUser(0, max_limit)

    # Label and draw
    legend.AddEntry(merged_hist, "Merged")
    merged_hist.Draw("same")

    # Now, we plot the pt hard dependent hists
    for i, ((key_index, analysis), color) in enumerate(zip(pt_hard_analyses.items(), colors)):
        # Setup
        color = ROOT.TColor.GetColor(*color)
        # Determine the proper label.
        label = labels.pt_range_string(
            pt_bin = key_index.pt_hard_bin,
            lower_label = "T",
            upper_label = "hard",
            only_show_lower_value_for_last_bin = True,
        )

        # Retrieve and style the hist
        hist = utils.recursive_getattr(analysis, hist_attribute_name)
        hist.SetMarkerStyle(ROOT.kFullCircle + i)
        hist.SetMarkerSize(1)
        hist.SetMarkerColor(color)
        hist.SetLineColor(color)

        # Label and draw
        legend.AddEntry(hist, labels.use_label_with_root(label))
        hist.Draw("same")

    # Final presentation settings
    legend.Draw()

    # Save and cleanup
    output_name += "_ROOT"
    plot_base.save_plot(merged_analysis.output_info, canvas, output_name)

