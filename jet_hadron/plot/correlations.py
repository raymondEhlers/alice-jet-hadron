#!/usr/bin/env python

""" Correlations plotting module.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Sequence, TYPE_CHECKING

from pachyderm import histogram

from jet_hadron.base import analysis_objects
from jet_hadron.base import labels
from jet_hadron.base import params
from jet_hadron.plot import base as plot_base
from jet_hadron.plot import highlight_RPF

import ROOT

if TYPE_CHECKING:
    from jet_hadron.analysis import correlations

# Setup logger
logger = logging.getLogger(__name__)

def plot_2d_correlations(jet_hadron: "correlations.Correlations") -> None:
    """ Plot the 2D correlations. """
    canvas = ROOT.TCanvas("canvas2D", "canvas2D")

    # Iterate over 2D hists
    for name, observable in jet_hadron.correlation_hists_2d:
        hist = observable.hist.Clone(f"{observable.name}_scaled")
        logger.debug(f"name: {name}, hist: {hist}")
        # We don't want to scale the mixed event hist because we already determined the normalization
        if "mixed" not in observable.type:
            hist.Scale(jet_hadron.correlation_scale_factor)

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

        if "mixed" in observable.type:
            hist.GetZaxis().SetTitle(r"$a(\Delta\varphi,\Delta\eta)$")
            hist.GetZaxis().SetTitleOffset(0.9)
        else:
            z_title = r"$1/N_{\mathrm{trig}}\mathrm{d^{2}}N%(label)s/\mathrm{d}\Delta\varphi\mathrm{d}\Delta\eta$"
            if "signal" in observable.type:
                z_title = z_title % {"label": ""}
            else:
                z_title = z_title % {"label": r"_{\mathrm{raw}}"}
                # Decrease size so it doesn't overlap with the other labels
                hist.GetZaxis().SetTitleSize(0.05)

            hist.GetZaxis().SetTitle(z_title)

        # Add labels
        # PDF DOES NOT WORK HERE: https://root-forum.cern.ch/t/latex-sqrt-problem/17442/15
        # Instead, print to EPS and then convert to PDF
        alice_label = labels.make_valid_latex_string(jet_hadron.alice_label.display_str())
        system_label = labels.system_label(
            energy = jet_hadron.collision_energy,
            system = jet_hadron.collision_system,
            activity = jet_hadron.event_activity
        )
        jet_finding = labels.jet_finding()
        constituent_cuts = labels.constituent_cuts()
        leading_hadron = "$" + jet_hadron.leading_hadron_bias.display_str() + "$"
        jet_pt = labels.jet_pt_range_string(jet_hadron.jet_pt)
        assoc_pt = labels.track_pt_range_string(jet_hadron.track_pt)
        logger.debug(f"label: {alice_label}, system_label: {system_label}, constituent_cuts: {constituent_cuts}, leading_hadron: {leading_hadron}, jet_pt: {jet_pt}, assoc_pt: {assoc_pt}")

        tex = ROOT.TLatex()
        tex.SetTextSize(0.04)
        # Upper left side
        tex.DrawLatexNDC(.005, .96, labels.use_label_with_root(alice_label))
        tex.DrawLatexNDC(.005, .91, labels.use_label_with_root(system_label))
        tex.DrawLatexNDC(.005, .86, labels.use_label_with_root(jet_pt))
        tex.DrawLatexNDC(.005, .81, labels.use_label_with_root(jet_finding))

        # Upper right side
        tex.DrawLatexNDC(.67, .96, labels.use_label_with_root(assoc_pt))
        tex.DrawLatexNDC(.7275, .91, labels.use_label_with_root(leading_hadron))
        tex.DrawLatexNDC(.73, .86, labels.use_label_with_root(constituent_cuts))

        # Save plot
        plot_base.save_plot(jet_hadron.output_info, canvas, observable.name)

        # Draw as colz to view more precisely
        hist.Draw("colz")
        plot_base.save_plot(jet_hadron.output_info, canvas, observable.name + "_colz")

        canvas.Clear()

def plot_1d_correlations(jet_hadron: "correlations.Correlations", plot_with_ROOT: bool = False) -> None:
    """ Plot the 1D correlations defined here. """
    signal_background_output_name = f"jetH_delta_phi_{jet_hadron.identifier}_signal_background_comparison"
    if plot_with_ROOT:
        # With ROOT
        _plot_all_1d_correlations_with_ROOT(jet_hadron = jet_hadron)
        _plot_1d_signal_and_background_with_ROOT(
            jet_hadron = jet_hadron, output_name = signal_background_output_name
        )
    else:
        # With matplotlib
        _plot_all_1d_correlations_with_matplotlib(jet_hadron = jet_hadron)
        _plot_1d_signal_and_background_with_matplotlib(
            jet_hadron = jet_hadron, output_name = signal_background_output_name
        )

def _plot_all_1d_correlations_with_matplotlib(jet_hadron: "correlations.Correlations") -> None:
    """ Plot all 1D correlations in a very basic way with matplotlib.

    Note:
        We scaled with the correlation scale factor before plotting to ensure that the stored
        plots are always scaled properly.
    """
    # Not ideal, but it's much more convenient to import it here. Importing it at the top
    # of the file would cause an import loop.
    from jet_hadron.analysis import correlations
    fig, ax = plt.subplots()
    for correlations_groups in [jet_hadron.correlation_hists_delta_phi, jet_hadron.correlation_hists_delta_eta]:
        # Help out mypy...
        assert isinstance(correlations_groups, (correlations.CorrelationHistogramsDeltaPhi, correlations.CorrelationHistogramsDeltaEta))
        for _, observable in correlations_groups:
            # Draw the 1D histogram.
            h = histogram.Histogram1D.from_existing_hist(observable.hist)
            # Scale by the appropriate factors
            h *= jet_hadron.correlation_scale_factor
            # And then plot
            ax.errorbar(
                h.x, h.y, yerr = h.errors,
                label = observable.hist.GetName(), marker = "o", linestyle = "",
            )
            # Set labels.
            ax.set_xlabel(labels.make_valid_latex_string(observable.hist.GetXaxis().GetTitle()))
            ax.set_ylabel(labels.make_valid_latex_string(observable.hist.GetYaxis().GetTitle()))
            ax.set_title(labels.make_valid_latex_string(observable.hist.GetTitle()))

            # Final adjustments
            fig.tight_layout()

            # Save and cleanup
            output_name = observable.hist.GetName() + "_mpl"
            plot_base.save_plot(jet_hadron.output_info, fig, output_name)
            ax.clear()

    # Cleanup
    plt.close(fig)

def _plot_all_1d_correlations_with_ROOT(jet_hadron: "correlations.Correlations") -> None:
    """ Plot all 1D correlations in a very basic way with ROOT.

    Note:
        We scaled with the correlation scale factor before plotting to ensure that the stored
        plots are always scaled properly.
    """
    # Not ideal, but it's much more convenient to import it here. Importing it at the top
    # of the file would cause an import loop.
    from jet_hadron.analysis import correlations
    canvas = ROOT.TCanvas("canvas1D", "canvas1D")
    for correlations_groups in [jet_hadron.correlation_hists_delta_phi, jet_hadron.correlation_hists_delta_eta]:
        # Help out mypy...
        assert isinstance(correlations_groups, (correlations.CorrelationHistogramsDeltaPhi, correlations.CorrelationHistogramsDeltaEta))
        for name, observable in correlations_groups:
            # Draw the 1D histogram.
            hist = observable.hist.Clone(f"{name}_scaled")
            hist.Scale(jet_hadron.correlation_scale_factor)
            hist.Draw("")
            output_name = observable.hist.GetName() + "_ROOT"
            plot_base.save_plot(jet_hadron.output_info, canvas, output_name)

def plot_and_label_1d_signal_and_background_with_matplotlib_on_axis(ax: matplotlib.axes.Axes,
                                                                    jet_hadron: "correlations.Correlations",
                                                                    apply_correlation_scale_factor: bool = True) -> None:
    """ Plot and label the signal and background dominated hists on the given axis.

    This is a helper function so that we don't have to repeat code when we need to plot these hists.
    It can also be used in other modules.

    Args:
        ax: Axis on which the histograms should be plotted.
        jet_hadron: Correlations object from which the delta_phi hists should be retrieved.
        apply_correlation_scale_factor: Whether to scale the histogram by the correlation scale factor.
    Returns:
        None. The given axis is modified.
    """
    # Setup
    hists = jet_hadron.correlation_hists_delta_phi

    h_signal = histogram.Histogram1D.from_existing_hist(hists.signal_dominated.hist)
    if apply_correlation_scale_factor:
        h_signal *= jet_hadron.correlation_scale_factor
    ax.errorbar(
        h_signal.x, h_signal.y, yerr = h_signal.errors,
        label = hists.signal_dominated.type.display_str(), marker = "o", linestyle = "",
    )
    h_background = histogram.Histogram1D.from_existing_hist(hists.background_dominated.hist)
    if apply_correlation_scale_factor:
        h_background *= jet_hadron.correlation_scale_factor
    # Plot with opacity first
    background_plot = ax.errorbar(
        h_background.x, h_background.y, yerr = h_background.errors,
        marker = "o", linestyle = "", alpha = 0.5,
    )
    # Then restrict range and plot without opacity
    near_side = len(h_background.x) // 2
    ax.errorbar(
        h_background.x[:near_side], h_background.y[:near_side], yerr = h_background.errors[:near_side],
        label = hists.background_dominated.type.display_str(),
        marker = "o", linestyle = "", color = background_plot[0].get_color()
    )

    # Set labels.
    ax.set_xlabel(labels.make_valid_latex_string(hists.signal_dominated.hist.GetXaxis().GetTitle()))
    ax.set_ylabel(labels.make_valid_latex_string(hists.signal_dominated.hist.GetYaxis().GetTitle()))
    jet_pt_label = labels.jet_pt_range_string(jet_hadron.jet_pt)
    track_pt_label = labels.track_pt_range_string(jet_hadron.track_pt)
    ax.set_title(fr"Unsubtracted 1D ${hists.signal_dominated.axis.display_str()}$,"
                 f" {jet_hadron.reaction_plane_orientation.display_str()} event plane orient.,"
                 f" {jet_pt_label}, {track_pt_label}")

def _plot_1d_signal_and_background_with_matplotlib(jet_hadron: "correlations.Correlations", output_name: str) -> None:
    """ Plot 1D signal and background hists on a single plot with matplotlib. """
    # Setup
    fig, ax = plt.subplots(figsize = (8, 6))

    # Perform the actual plot
    plot_and_label_1d_signal_and_background_with_matplotlib_on_axis(
        ax = ax, jet_hadron = jet_hadron, apply_correlation_scale_factor = True
    )

    # Labeling
    ax.legend(loc = "upper right")

    # Final adjustments
    fig.tight_layout()

    # Save and cleanup
    output_name += "_mpl"
    plot_base.save_plot(jet_hadron.output_info, fig, output_name)
    plt.close(fig)

def _plot_1d_signal_and_background_with_ROOT(jet_hadron: "correlations.Correlations", output_name: str) -> None:
    """ Plot 1D signal and background hists on a single plot with ROOT. """
    # Setup
    canvas = ROOT.TCanvas("canvas1D", "canvas1D")
    hists = jet_hadron.correlation_hists_delta_phi

    # Plot
    signal_hist = hists.signal_dominated.hist.Clone(f"{hists.signal_dominated.name}_scaled")
    signal_hist.Scale(jet_hadron.correlation_scale_factor)
    signal_hist.SetLineColor(ROOT.kBlack)
    signal_hist.SetMarkerColor(ROOT.kBlack)
    signal_hist.Draw("")
    background_hist = hists.background_dominated.hist.Clone(f"{hists.background_dominated.name}_scaled")
    background_hist.Scale(jet_hadron.correlation_scale_factor)
    background_hist.SetLineColor(ROOT.kBlue)
    background_hist.SetMarkerColor(ROOT.kBlue)
    background_hist.Draw("same")

    # Save
    output_name += "_ROOT"
    plot_base.save_plot(jet_hadron.output_info, canvas, output_name)

def delta_eta_unsubtracted(hists: "correlations.CorrelationHistogramsDeltaEta",
                           correlation_scale_factor: float,
                           jet_pt: analysis_objects.JetPtBin, track_pt: analysis_objects.TrackPtBin,
                           reaction_plane_orientation: params.ReactionPlaneOrientation,
                           identifier: str, output_info: analysis_objects.PlottingOutputWrapper) -> None:
    """ Plot 1D delta eta correlations on a single plot.

    Args:
        hists: Unsubtracted delta eta histograms.
        correlation_scale_factor: Overall correlation scale factor.
        jet_pt: Jet pt bin.
        track_pt: Track pt bin.
        reaction_plane_orientation: Reaction plane orientation.
        identifier: Analysis identifier string. Usually contains jet pt, track pt, and other information.
        output_info: Standard information needed to store the output.
    Returns:
        None.
    """
    # Setup
    fig, ax = plt.subplots(figsize = (8, 6))

    # Plot NS, AS
    h_near_side = histogram.Histogram1D.from_existing_hist(hists.near_side.hist)
    h_near_side *= correlation_scale_factor
    ax.errorbar(
        h_near_side.x, h_near_side.y, yerr = h_near_side.errors,
        label = hists.near_side.type.display_str(), marker = "o", linestyle = "",
    )
    h_away_side = histogram.Histogram1D.from_existing_hist(hists.away_side.hist)
    h_away_side *= correlation_scale_factor
    ax.errorbar(
        h_away_side.x, h_away_side.y, yerr = h_away_side.errors,
        label = hists.away_side.type.display_str(), marker = "o", linestyle = "",
    )

    # Set labels.
    ax.set_xlabel(labels.make_valid_latex_string(hists.near_side.hist.GetXaxis().GetTitle()))
    ax.set_ylabel(labels.make_valid_latex_string(hists.near_side.hist.GetYaxis().GetTitle()))
    ax.set_title(fr"Unsubtracted 1D ${hists.near_side.axis.display_str()}$,"
                 f" {reaction_plane_orientation.display_str()} event plane orient.,"
                 f" {labels.jet_pt_range_string(jet_pt)}, {labels.track_pt_range_string(track_pt)}")

    # Labeling
    ax.legend(loc = "upper right")

    # Final adjustments
    fig.tight_layout()

    # Save and cleanup
    output_name = f"jetH_delta_eta_{identifier}_near_away_side_comparison"
    plot_base.save_plot(output_info, fig, output_name)
    plt.close(fig)

def comparison_1d(output_info: analysis_objects.PlottingOutputWrapper,
                  our_hist: histogram.Histogram1D,
                  their_hist: histogram.Histogram1D,
                  ratio: histogram.Histogram1D,
                  title: str, x_label: str, y_label: str,
                  output_name: str,
                  offset_our_points: bool) -> None:
    """ Compare our hist and their hist. """
    fig, ax = plt.subplots(2, 1, sharex = True, gridspec_kw = {"height_ratios": [3, 1]}, figsize = (8, 6))

    # Plot data
    offset = 0.0
    # Offset so we can see both sets of data points.
    if offset_our_points:
        offset = 0.05
    ax[0].errorbar(our_hist.x + offset, our_hist.y, yerr = our_hist.errors, label = "Our hist")
    ax[0].errorbar(their_hist.x, their_hist.y, yerr = their_hist.errors, label = "Their hist")
    # Plot ratio
    ax[1].errorbar(ratio.x, ratio.y, yerr = ratio.errors, label = "Theirs/ours")
    ax[1].set_ylim(0.5, 1.5)

    # Set plot properties
    ax[0].set_title(title)
    ax[0].set_ylabel(r"$\mathrm{d}N/\mathrm{d}\varphi$")
    ax[0].legend(loc = "best")
    ax[1].set_xlabel(r"$\Delta\varphi$")
    ax[1].set_ylabel("Theirs/ours")

    # Final adjustments
    fig.tight_layout()
    # Reduce spacing between subplots
    fig.subplots_adjust(hspace = 0, wspace = 0.05)

    # Save and cleanup
    plot_base.save_plot(output_info, fig, output_name)
    plt.close(fig)

def mixed_event_normalization(output_info: analysis_objects.PlottingOutputWrapper,
                              # For labeling purposes
                              output_name: str, eta_limits: Sequence[float], jet_pt_title: str, track_pt_title: str,
                              # Basic data
                              lin_space: np.ndarray,       peak_finding_hist_array: np.ndarray,  # noqa: E241
                              lin_space_rebin: np.ndarray, peak_finding_hist_array_rebin: np.ndarray,
                              # CWT
                              peak_locations: np.ndarray, peak_locations_rebin: np.ndarray,
                              # Moving Average
                              max_moving_avg: float, max_moving_avg_rebin: float,
                              # Smoothed gaussian
                              lin_space_resample: np.ndarray, smoothed_array: np.ndarray, max_smoothed_moving_avg: float,
                              # Linear fits
                              max_linear_fit_1d: float, max_linear_fit_1d_rebin: float,
                              max_linear_fit_2d: float, max_linear_fit_2d_rebin: float) -> None:
    # Make the actual plot
    fig, ax = plt.subplots(figsize = (8, 6))
    # Add additional y margin at the bottom so the legend will fit a bit better
    # Cannot do asymmetric padding via `ax.set_ymargin()`, so we'll do it by hand
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

    ax.text(
        0.05, 0.95, fr"$|\Delta\eta| < {eta_limits[1]}$", horizontalalignment = "left",
        verticalalignment = "top", multialignment = "left",
        transform = ax.transAxes
    )

    # Legend and Labels for the plot
    #ax.set_ymargin = 0.01
    ax.legend(loc = "lower left", ncol = 3, frameon = False)
    #ax.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
    #          ncol=3, mode="expand", borderaxespad=0)
    ax.set_title(f"ME norm. for {jet_pt_title}, {track_pt_title}")
    ax.set_ylabel(labels.delta_phi_axis_label())
    ax.set_xlabel(r"$\Delta\varphi$")

    # Final adjustments
    plt.tight_layout()
    # Cleanup and save
    plot_base.save_plot(output_info, fig, output_name)
    plt.close(fig)

def simplified_mixed_event_comparison(output_info: analysis_objects.PlottingOutputWrapper,
                                      # For labeling purposes
                                      output_name: str, eta_limits: Sequence[float],
                                      jet_pt_title: str, track_pt_title: str,
                                      # Mixed event 1D hist
                                      mixed_event_1D: histogram.Histogram1D,
                                      max_moving_avg: float,
                                      fit_1D: float, fit_2D: float) -> None:
    """ Simplified mixed event comparison.

    Args:
        output_info: Standard information needed to store the output.
        output_name: Output name.
        eta_limits: Eta limits.
        jet_pt_title: Jet pt title for labeling.
        track_pt_title: Track pt title for labeling.
        mixed_event_1D: 1D mixed event.
        max_moving_avg: Maximum of the moving average (nominal value).
        fit_1D: Max value from a 1D fit.
        fit_2D: Max value from a 2D fit.
    Returns:
        None. The comparison is plotted.
    """
    # Setup
    fig, ax = plt.subplots(figsize = (8, 6))
    hist = histogram.Histogram1D.from_existing_hist(mixed_event_1D)

    # Plot the data
    ax.errorbar(
        hist.x, hist.y, yerr = hist.errors, linestyle = "", marker = "o",
        label = "Mixed event",
    )
    # Plot the lines
    ax.axhline(
        max_moving_avg, xmin = 0.4, xmax = 1,
        label = r"Moving avg. (size $\pi$)", color = "tab:orange"
    )
    ax.axhline(
        fit_1D, xmin = 0.5, xmax = 1,
        label = r"1D fit ($\pi/2-3\pi/2$)", color = "tab:purple"
    )
    ax.axhline(
        fit_2D, xmin = 0.5, xmax = 1,
        label = r"2D fit ($\pi/2-3\pi/2$)", linestyle = "--", color = "tab:purple"
    )

    # Label delta eta range.
    ax.text(
        0.05, 0.95, fr"$|\Delta\eta| < {eta_limits[1]}$", horizontalalignment = "left",
        verticalalignment = "top", multialignment = "left",
        transform = ax.transAxes
    )

    # Legend and Labels for the plot
    ax.legend(loc = "lower right", frameon = False)
    ax.set_title(f"ME norm. for {jet_pt_title}, {track_pt_title}")
    ax.set_ylabel(labels.delta_phi_axis_label())
    ax.set_xlabel(r"$\Delta\varphi$")

    # Final adjustments
    plt.tight_layout()
    # Cleanup and save
    plot_base.save_plot(output_info, fig, output_name)
    plt.close(fig)

def define_highlight_regions(signal_dominated_eta_region: analysis_objects.AnalysisBin,
                             background_dominated_eta_region: analysis_objects.AnalysisBin,
                             near_side_phi_region: analysis_objects.AnalysisBin,
                             ) -> Sequence[highlight_RPF.HighlightRegion]:
    """ Define regions to highlight.

    The user should modify or override this function if they want to define different ranges. By default,
    we highlight the signal and background dominated regions.

    Note:
        The edge color is still that of the colormap, so there is still a hint of the original colormap, although
        the facecolors are replaced by selected highlight colors

    Args:
        signal_dominated_eta_region: Signal dominated eta range.
        background_dominated_eta_region: Background dominated eta range.
    Returns:
        list: ``highlightRegion`` objects, suitably defined for highlighting the signal and background regions.
    """
    # Select the highlighted regions.
    highlight_regions = []

    # Signal
    signal_color = matplotlib.colors.to_rgba(plot_base.AnalysisColors.signal)
    signal_region = highlight_RPF.HighlightRegion(
        "Signal,\n" + fr"$|\Delta\eta|<{signal_dominated_eta_region.max}$", signal_color
    )
    signal_region.add_highlight_region(
        phi_range = (-np.pi / 2, 3.0 * np.pi / 2),
        eta_range = (-1 * signal_dominated_eta_region.max, signal_dominated_eta_region.max),
    )
    highlight_regions.append(signal_region)

    # Background
    background_color = matplotlib.colors.to_rgba(plot_base.AnalysisColors.background)
    background_phi_range = (near_side_phi_region.min, near_side_phi_region.max)
    background_region = highlight_RPF.HighlightRegion(
        "Background,\n" + fr"${background_dominated_eta_region.min}<|\Delta\eta|<{background_dominated_eta_region.max}$",
        background_color
    )
    background_region.add_highlight_region(
        phi_range = background_phi_range,
        eta_range = (-1 * background_dominated_eta_region.max, -1 * background_dominated_eta_region.min),
    )
    background_region.add_highlight_region(
        phi_range = background_phi_range,
        eta_range = (background_dominated_eta_region.min, background_dominated_eta_region.max),
    )
    highlight_regions.append(background_region)

    return highlight_regions

def plot_RPF_fit_regions(jet_hadron: "correlations.Correlations", filename: str) -> None:
    """ Plot showing highlighted RPF fit regions.

    Args:
        jet_hadron: Main analysis object.
        filename: Filename under which the hist should be saved.
    Returns:
        None
    """
    # Retrieve the hist to be plotted
    # Here we selected the corrected 2D correlation
    hist = jet_hadron.correlation_hists_2d.signal.hist.Clone(f"{jet_hadron.correlation_hists_2d.signal.name}_RPF_scaled")
    hist.Scale(jet_hadron.correlation_scale_factor)

    with sns.plotting_context(context = "notebook", font_scale = 1.5):
        # Perform the plotting
        # The best option for clarify of viewing seems to be to just replace the colors with the overlay colors.
        # mathematical_blending and screen_colors look similar and seem to be the next most promising option.
        (fig, ax) = highlight_RPF.plot_RPF_fit_regions(
            histogram.get_array_from_hist2D(hist),
            highlight_regions = define_highlight_regions(
                signal_dominated_eta_region = jet_hadron.signal_dominated_eta_region,
                background_dominated_eta_region = jet_hadron.background_dominated_eta_region,
                near_side_phi_region = jet_hadron.near_side_phi_region,
            ),
            use_color_overlay = False,
            use_color_screen = False,
            use_mathematical_blending = False,
        )

        # Add additional labeling
        # Axis
        # Needed to fix z axis rotation. See: https://stackoverflow.com/a/21921168
        ax.zaxis.set_rotate_label(False)
        ax.set_zlabel(r"$1/N_{\mathrm{trig}}\mathrm{d^{2}}N/\mathrm{d}\Delta\varphi\mathrm{d}\Delta\eta$", rotation=90)
        # Set the distance from axis to label in pixels.
        # This is not ideal, but clearly tight_layout doesn't work as well for 3D plots
        ax.xaxis.labelpad = 12
        # Visually, dEta looks closer
        ax.yaxis.labelpad = 15
        ax.zaxis.labelpad = 12
        # Overall
        alice_label = labels.make_valid_latex_string(jet_hadron.alice_label.display_str())
        system_label = labels.system_label(
            energy = jet_hadron.collision_energy,
            system = jet_hadron.collision_system,
            activity = jet_hadron.event_activity
        )
        jet_finding = labels.jet_finding()
        constituent_cuts = labels.constituent_cuts()
        leading_hadron = "$" + jet_hadron.leading_hadron_bias.display_str() + "$"
        jet_pt = labels.jet_pt_range_string(jet_hadron.jet_pt)
        assoc_pt = labels.track_pt_range_string(jet_hadron.track_pt)

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
        plot_base.save_plot(jet_hadron.output_info, fig, filename)
        plt.close(fig)
