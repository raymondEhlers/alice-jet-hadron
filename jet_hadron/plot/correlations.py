#!/usr/bin/env python

""" Correlations plotting module.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import logging
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import numpy as np
import seaborn as sns
from typing import Sequence

from pachyderm import histogram

from jet_hadron.base import analysis_objects
from jet_hadron.base import params
from jet_hadron.base.typing_helpers import Canvas
from jet_hadron.plot import base as plot_base
from jet_hadron.plot import highlight_RPF
# TODO: Resolve this.... Either put this module the base package or change otherwise...
from jet_hadron.analysis import correlations_helpers

import ROOT

# Setup logger
logger = logging.getLogger(__name__)

def plot_2d_correlations(jet_hadron):
    """ Plot the 2D correlations. """
    canvas = ROOT.TCanvas("canvas2D", "canvas2D")

    # Iterate over 2D hists
    for name, initial_hist in jet_hadron.correlation_hists_2d:
        hist = initial_hist.Clone(f"{initial_hist.GetName()}_scaled")
        logger.debug(f"name: {name}, hist: {hist}")
        # We don't want to scale the mixed event hist because we already determined the normalization
        if "mixed" not in name:
            correlations_helpers.scale_by_bin_width(hist)

        # We don't need the title with all of the labeling
        hist.SetTitle("")

        # Draw plot
        hist.Draw("surf2")

        # Label axes
        hist.GetXaxis().CenterTitle(True)
        hist.GetXaxis().SetTitleSize(0.08)
        hist.GetXaxis().SetLabelSize(0.06)
        hist.GetYaxis().CenterTitle(True)
        hist.GetYaxis().SetTitleSize(0.08)
        # If I remove this, it looks worse, even though this is not supposed to do anything
        hist.GetYaxis().SetTitleOffset(1.2)
        hist.GetYaxis().SetLabelSize(0.06)
        hist.GetZaxis().CenterTitle(True)
        hist.GetZaxis().SetTitleSize(0.06)
        hist.GetZaxis().SetLabelSize(0.05)
        hist.GetZaxis().SetTitleOffset(0.8)
        canvas.SetLeftMargin(0.13)

        if "mixed" in name:
            hist.GetZaxis().SetTitle(r"$a(\Delta\varphi,\Delta\eta)$")
            hist.GetZaxis().SetTitleOffset(0.9)
        else:
            z_title = r"$1/\mathrm{N}_{\mathrm{trig}}\mathrm{d^{2}N}%(label)s/\mathrm{d}\Delta\varphi\mathrm{d}\Delta\eta$"
            if "corr" in name:
                z_title = z_title % {"label": ""}
            else:
                z_title = z_title % {"label": r"_{\mathrm{raw}}"}
                # Decrease size so it doesn't overlap with the other labels
                hist.GetZaxis().SetTitleSize(0.05)

            hist.GetZaxis().SetTitle(z_title)

        # Add labels
        # PDF DOES NOT WORK HERE: https://root-forum.cern.ch/t/latex-sqrt-problem/17442/15
        # Instead, print to EPS and then convert to PDF
        alice_label = str(jet_hadron.alice_label)
        system_label = params.system_label(
            energy = jet_hadron.collision_energy,
            system = jet_hadron.collision_system,
            activity = jet_hadron.event_activity
        )
        (jet_finding, constituent_cuts, leading_hadron, jet_pt) = params.jet_properties_label(jet_hadron.jet_pt)
        assoc_pt = params.generate_track_pt_range_string(jet_hadron.track_pt)
        logger.debug(f"label: {alice_label}, system_label: {system_label}, constituent_cuts: {constituent_cuts}, leading_hadron: {leading_hadron}, jet_pt: {jet_pt}, assoc_pt: {assoc_pt}")

        tex = ROOT.TLatex()
        tex.SetTextSize(0.04)
        # Upper left side
        tex.DrawLatexNDC(.03, .96, alice_label)
        tex.DrawLatexNDC(.005, .91, system_label)
        tex.DrawLatexNDC(.005, .86, jet_pt)
        tex.DrawLatexNDC(.005, .81, jet_finding)

        # Upper right side
        tex.DrawLatexNDC(.67, .96, assoc_pt)
        tex.DrawLatexNDC(.73, .91, constituent_cuts)
        tex.DrawLatexNDC(.75, .86, leading_hadron)

        # Save plot
        plot_base.save_canvas(jet_hadron, canvas, initial_hist.GetName())

        # Draw as colz to view more precisely
        hist.Draw("colz")
        plot_base.save_canvas(jet_hadron, canvas, initial_hist.GetName() + "colz")

        canvas.Clear()

def plot_1d_correlations(jet_hadron) -> None:
    """ Plot the 1D correlations defined here. """
    canvas = ROOT.TCanvas("canvas1D", "canvas1D")

    plot_basic_scaled_1d_correlations_root(jet_hadron = jet_hadron, canvas = canvas)
    plot_1d_signal_and_background_root(jet_hadron = jet_hadron, canvas = canvas)

def plot_basic_scaled_1d_correlations_root(jet_hadron, canvas: Canvas) -> None:
    """ Basic 1D correlation plot with ROOT.

    Note:
        We don't want to scale the histogram here by the bin width because we've already done that!
    """
    for correlations in [jet_hadron.correlation_hists_delta_phi, jet_hadron.correlation_hists_delta_eta]:
        for _, observable in correlations:
            # Draw the 1D histogram.
            observable.hist.Draw("")
            plot_base.save_canvas(jet_hadron, canvas, observable.hist.GetName())

def plot_1d_signal_and_background_root(jet_hadron, canvas: Canvas) -> None:
    """ Plot 1D signal and background ROOT hists on a single plot. """
    # Setup
    hists = jet_hadron.correlation_hists_delta_phi

    # Plot
    hists.signal_dominated.hist.Draw("")
    hists.background_dominated.hist.SetMarkerColor(ROOT.kBlue)
    hists.background_dominated.hist.Draw("same")

    # Save
    output_name = jet_hadron.hist_name_format_delta_phi.format(
        jet_pt_bin = jet_hadron.jet_pt.bin,
        track_pt_bin = jet_hadron.track_pt.bin,
        tag = "signal_background_comparion",
    )
    plot_base.save_canvas(jet_hadron, canvas, output_name)

def plot1DCorrelationsWithFits(jetH):
    canvas = ROOT.TCanvas("canvas1D", "canvas1D")

    histsWithFits = [[jetH.dPhi, jetH.dPhiFit], [jetH.dPhiSubtracted, jetH.dPhiSubtractedFit],
                     [jetH.dEtaNS, jetH.dEtaNSFit], [jetH.dEtaNSSubtracted, jetH.dEtaNSSubtractedFit]]

    for histCollection, fitCollection in histsWithFits:
        for (name, observable), fit in zip(histCollection.items(), fitCollection.values()):
            # Create scaled hist and plot it
            observable.hist.Draw("")
            fit.Draw("same")
            plot_base.save_canvas(jetH, canvas, observable.hist.GetName())

def mixed_event_normalization(jet_hadron: analysis_objects.JetHBase,
                              # For labeling purposes
                              hist_name: str, eta_limits: Sequence[float], jet_pt_title: str, track_pt_title: str,
                              # Basic data
                              lin_space: np.ndarray,       peak_finding_hist_array: np.ndarray,  # noqa: E241
                              lin_space_rebin: np.ndarray, peak_finding_hist_array_rebin: np.ndarray,
                              # CWT
                              peak_locations, peak_locations_rebin,
                              # Moving Average
                              max_moving_avg: float, max_moving_avg_rebin: float,
                              # Smoothed gaussian
                              lin_space_resample: np.ndarray, smoothed_array: np.ndarray, max_smoothed_moving_avg,
                              # Linear fits
                              max_linear_fit_1d: float, max_linear_fit_1d_rebin: float,
                              max_linear_fit_2d: float, max_linear_fit_2d_rebin: float) -> None:
    # Make the actual plot
    fig, ax = plt.subplots()
    # Add additional y margin at the bottom so the legend will fit a bit better
    # Cannot do asyemmtric padding via `ax.set_ymargin()`, so we'll do it by hand
    # See: https://stackoverflow.com/a/42804403
    data_min = min(peak_finding_hist_array.min(), peak_finding_hist_array_rebin.min())
    data_max = max(peak_finding_hist_array.max(), peak_finding_hist_array_rebin.max())
    y_min = data_min - 0.5 * (data_max - data_min)
    y_max = data_max + 0.12 * (data_max - data_min)
    ax.set_ylim(y_min, y_max)

    # Can either plot the hist or the array
    # Hist based on: https://stackoverflow.com/a/8553614
    #ax.hist(lin_space, weights=peak_finding_hist_array, bins=len(peak_finding_hist_array))
    # If plotting the hist, it's best to set the y axis limits to make it easier to view
    #ax.set_ylim(ymin=.95*min(peak_finding_hist_array), ymax=1.05*max(peak_finding_hist_array))
    # Plot array
    ax.plot(lin_space, peak_finding_hist_array, label = "ME")
    ax.plot(lin_space_rebin, peak_finding_hist_array_rebin, label = "ME rebin")
    # Peak finding
    # Set zorder of 10 to ensure that the stars are always visible
    plot_array_peak = ax.plot(
        lin_space[peak_locations],
        peak_finding_hist_array[peak_locations],
        marker = "*", markersize = 10, linestyle = "None",
        label = "CWT", zorder = 10
    )
    plot_array_rebin_peak = ax.plot(
        lin_space_rebin[peak_locations_rebin],
        peak_finding_hist_array_rebin[peak_locations_rebin],
        marker = "*", markersize = 10, linestyle = "None",
        label = "CWT rebin", zorder = 10
    )
    # Moving average
    ax.axhline(max_moving_avg, color = plot_array_peak[0].get_color(), label = r"Mov. avg. (size $\pi$)")
    ax.axhline(max_moving_avg_rebin, color = plot_array_rebin_peak[0].get_color(), linestyle = "--", label = "Mov. avg. rebin")
    # Gaussian
    # Use a mask so the range doesn't get extremely distorted when the interpolation drops around the edges
    mask = np.where(np.logical_and(lin_space_resample > -0.3 * np.pi, lin_space_resample < 1.3 * np.pi))
    plot_gaussian = ax.plot(lin_space_resample[mask], smoothed_array[mask], label = "Gauss. smooth")
    ax.axhline(max_smoothed_moving_avg, color = plot_gaussian[0].get_color(), linestyle = "--", label ="Gauss. mov. avg")
    #ax.axhline(max_smoothed, color = plot_gaussian[0].get_color(), linestyle = ":", label = "Gauss. max")

    # Linear fits
    ax.axhline(max_linear_fit_1d, color = "g", label = "1D fit")
    ax.axhline(max_linear_fit_1d_rebin, color = "g", linestyle = "--", label = "1D fit rebin")
    ax.axhline(max_linear_fit_2d, color = "b", label = "2D fit")
    ax.axhline(max_linear_fit_2d_rebin, color = "b", linestyle = "--", label = "2D fit rebin")

    eta_limits_label = AnchoredText(r"|$\Delta\eta$|<{}".format(eta_limits[1]), loc=2, frameon=False)
    ax.add_artist(eta_limits_label)

    # Legend and Labels for the plot
    #ax.set_ymargin = 0.01
    ax.legend(loc="lower left", ncol=3)
    #ax.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
    #          ncol=3, mode="expand", borderaxespad=0)
    ax.set_title(f"ME norm. for {jet_pt_title}, {track_pt_title}")
    ax.set_ylabel(r"$\Delta N/\Delta\varphi$")
    ax.set_xlabel(r"$\Delta\varphi$")

    #plt.tight_layout()
    plot_base.save_plot(jet_hadron, fig, hist_name)
    # Close the figure
    plt.close(fig)

def define_highlight_regions():
    """ Define regions to highlight.

    The user should modify or override this function if they want to define different ranges. By default,
    we highlight.

    Args:
        None
    Returns:
        list: highlightRegion objects, suitably defined for highlighting the signal and background regions.
    """
    # Select the highlighted regions.
    highlight_regions = []
    # NOTE: The edge color is still that of the colormap, so there is still a hint of the origin
    #       colormap, although the facecolors are replaced by selected highlight colors
    palette = sns.color_palette()

    # Signal
    # Blue used for the signal data color
    # NOTE: Blue really doesn't look good with ROOT_kBird, so for that case, the
    #       signal fit color, seaborn green, should be used.
    signal_color = palette[0] + (1.0,)
    signal_region = highlight_RPF.highlightRegion("Signal dom. region,\n" + r"$|\Delta\eta|<0.6$", signal_color)
    signal_region.addHighlightRegion((-np.pi / 2, 3.0 * np.pi / 2), (-0.6, 0.6))
    highlight_regions.append(signal_region)

    # Background
    # Red used for background data color
    background_color = palette[2] + (1.0,)
    background_phi_range = (-np.pi / 2, np.pi / 2)
    background_region = highlight_RPF.highlightRegion("Background dom. region,\n" + r"$0.8<|\Delta\eta|<1.2$", background_color)
    background_region.addHighlightRegion(background_phi_range, (-1.2, -0.8))
    background_region.addHighlightRegion(background_phi_range, ( 0.8,  1.2))  # noqa: E201, E241
    highlight_regions.append(background_region)

    return highlight_regions

def plot_RPF_fit_regions(jet_hadron: analysis_objects.JetHBase, filename: str) -> None:
    """ Plot showing highlighted RPF fit regions.

    Args:
        jet_hadron: Main analysis object.
        filename: Filename under which the hist should be saved.
    Returns:
        None
    """
    # Retrieve the hist to be plotted
    # Here we selected the corrected 2D correlation
    # Bins are currently selected arbitrarily
    hist = jet_hadron.correlation_hists_2d.signal

    with sns.plotting_context(context = "notebook", font_scale = 1.5):
        # Perform the plotting
        # TODO: Determine if color overlays are better here!
        (fig, ax) = highlight_RPF.plotRPFFitRegions(
            histogram.get_array_from_hist2D(hist),
            highlightRegions = define_highlight_regions(),
            useColorOverlay = False
        )

        # Add additional labeling
        # Axis
        # Needed to fix z axis rotation. See: https://stackoverflow.com/a/21921168
        ax.zaxis.set_rotate_label(False)
        ax.set_zlabel(r"$1/\mathrm{N}_{\mathrm{trig}}\mathrm{d^{2}N}/\mathrm{d}\Delta\varphi\mathrm{d}\Delta\eta$", rotation=90)
        # Set the distance from axis to label in pixels.
        # This is not ideal, but clearly tight_layout doesn't work as well for 3D plots
        ax.xaxis.labelpad = 12
        # Visually, dEta looks closer
        ax.yaxis.labelpad = 15
        ax.zaxis.labelpad = 12
        # Overall
        alice_label = str(jet_hadron.alice_label)
        system_label = params.system_label(
            energy = jet_hadron.collision_energy,
            system = jet_hadron.collision_system,
            activity = jet_hadron.event_activity
        )
        (jet_finding, constituent_cuts, leading_hadron, jet_pt) = params.jet_properties_label(jet_hadron.jet_pt)
        assoc_pt = params.generate_track_pt_range_string(jet_hadron.track_pt)

        # Upper left side
        upper_left_text = ""
        upper_left_text += alice_label
        upper_left_text += "\n" + system_label
        upper_left_text += "\n" + jet_pt
        upper_left_text += "\n" + jet_finding

        # Upper right side
        upper_right_text = ""
        upper_right_text += leading_hadron
        upper_right_text += "\n" + constituent_cuts
        upper_right_text += "\n" + assoc_pt

        # Need a different text function since we have a 3D axis
        ax.text2D(0.01, 0.99, upper_left_text,
                  horizontalalignment = "left",
                  verticalalignment = "top",
                  multialignment = "left",
                  transform = ax.transAxes)
        ax.text2D(0.00, 0.00, upper_right_text,
                  horizontalalignment = "left",
                  verticalalignment = "bottom",
                  multialignment = "left",
                  transform = ax.transAxes)

        # Finish up
        plot_base.save_plot(jet_hadron, fig, filename)
        plt.close(fig)
