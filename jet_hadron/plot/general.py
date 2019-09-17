#!/usr/bin/env python

""" General plotting module.

Contains brief plotting functions which don't belong elsewhere.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import logging
import matplotlib
import matplotlib.pyplot as plt
from typing import Any, Dict, Iterator, List, Tuple, TYPE_CHECKING

from jet_hadron.base.typing_helpers import Hist

import numpy as np

from pachyderm import histogram

from jet_hadron.base import analysis_objects, labels, params
from jet_hadron.plot import base as plot_base

if TYPE_CHECKING:
    from jet_hadron.analysis import event_plane_resolution  # noqa: F401
    from jet_hadron.analysis import correlations  # noqa: F401

logger = logging.getLogger(__name__)

def event_plane_resolution_harmonic(analyses_iter: Iterator[Tuple[Any, "event_plane_resolution.EventPlaneResolution"]],
                                    harmonic: int, output_info: analysis_objects.PlottingOutputWrapper) -> None:
    """ Plot the event plane resolution for the provided detectors.

    Args:
        analyses_iter: Event plane resolution analysis objects to be plotted.
        harmonic: Resolution harmonic to be plotted. These are plotted with respect
            to the second order event plane.
        output_info: Output information.
    Returns:
        None. The figure is saved.
    """
    # Setup
    fig, ax = plt.subplots(figsize = (8, 6))
    analyses = dict(analyses_iter)

    for key_index, analysis in analyses.items():
        label = analysis.main_detector_name.replace('_', ' ')
        # Plot the resolution for a given detector.
        plot = ax.errorbar(
            analysis.resolution.x, analysis.resolution.y,
            xerr = analysis.resolution.bin_widths / 2, yerr = analysis.resolution.errors,
            marker = "o", linestyle = "", label = label,
        )

        # Plot the selected resolutions for our desired detector
        # Extract the values
        x_values = []
        x_errors = []
        y_values = []
        y_errors = []
        for selection, (value, error) in analysis.selected_resolutions.items():
            sel = list(dict(selection).values())
            x_values.append(np.mean(sel))
            x_errors.append(selection.max - np.mean(sel))
            y_values.append(value)
            y_errors.append(error)
        # And then plot them
        ax.errorbar(
            x_values, y_values, xerr = x_errors, yerr = y_errors,
            marker = "o", linestyle = "", label = label + " Sel.",
            color = plot_base.modify_brightness(plot[0].get_color(), 1.3),
        )

    # Final presentation settings
    ax.legend(frameon = False)
    ax.set_xlabel(r"Centrality (\%)")
    ax.set_ylabel(fr"$\Re_{{{harmonic}}}(\Psi_{2})$")
    fig.tight_layout()

    # Finally, save and cleanup
    output_name = f"EventPlaneResolution_R{harmonic}"
    plot_base.save_plot(output_info, fig, output_name)
    plt.close(fig)

def vn_harmonics(theta: np.ndarray, harmonics: Dict[int, float],
                 output_name: str, output_info: analysis_objects.PlottingOutputWrapper,
                 rotate_vn_relative_to_second_order: bool = False) -> None:
    """ Plot the pure v_n harmonics.

    Args:
        theta: Angular values where to evaluate the harmonics.
        harmonics: Harmonics and their coefficients to be plotted.
        output_name: Output name for the figure.
        output_info: Output information.
        rotate_vn_relative_to_second_order: Rotate the v_n relative to the n = 2 harmonic.
            This is somewhat more realistic (although still contrived). Default: False.
    Returns:
        None.
    """
    # Setup
    fig, ax = plt.subplots(figsize = (8, 8))
    rotations = {
        2: 0,
        3: 0,
        4: 0,
    }
    if rotate_vn_relative_to_second_order:
        rotations = {
            2: 0,
            # Major rotation.
            3: np.pi / 3,
            # Small rotation, since it should be correlated.
            4: np.pi / 24,
        }

    # Draw reference nuclei. This is modeled as a very central collision.
    # Draw it first so that it's underneath.
    for x_origin in [-0.05, 0.05]:
        ax.plot(x_origin + np.sin(theta), np.cos(theta), color = "black", linestyle = "dotted")

    # Draw event plane
    ax.axhline(0, xmin = 0.05, xmax = 0.95, color = "black", linestyle = "dotted")

    # Plot the harmonics parametrically.
    for n, coefficient in harmonics.items():
        # Additional rotation
        rotation = rotations[n]
        # Plot deviations from a unit circle.
        ax.plot(
            (1 + 2 * coefficient * np.cos(n * theta + rotation)) * np.cos(theta + rotation),
            (1 + 2 * coefficient * np.cos(n * theta + rotation)) * np.sin(theta + rotation),
            label = fr"$v_{{ {n} }}$: {coefficient:.2f}",
        )

    # Ensure that we maintain the proper aspect ratio.
    ax.set_aspect('equal', 'box')

    # Final presentation settings
    ax.legend(frameon = False, loc = "upper right")
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_xlabel("x (arb units)")
    ax.set_ylabel("y (arb units)")
    fig.tight_layout()

    # Finally, save and cleanup
    plot_base.save_plot(output_info, fig, output_name)
    plt.close(fig)

def rho_centrality(rho_hist: Hist, output_info: analysis_objects.PlottingOutputWrapper, includes_constituent_cut: bool = True) -> None:
    """ Plot rho as a function of centrality vs jet pt.

    Args:
        rho_hist: Rho centrality dependent hist.
        output_info: Output information.
        includes_constituent_cut: True if the plot was produced using the constituent cut.
    Returns:
        None. The figure is plotted.
    """
    # Setup
    import ROOT
    canvas = ROOT.TCanvas("c", "c")
    canvas.SetRightMargin(0.15)
    # Set labels
    rho_hist.SetTitle("")
    rho_hist.Draw("colz")
    # Keep the range more meaningful.
    rho_hist.GetYaxis().SetRangeUser(0, 100)
    # Draw a profile of the mean
    rho_hist_profile = rho_hist.ProfileX(f"{rho_hist.GetName()}_profile")
    rho_hist_profile.SetLineColor(ROOT.kRed)
    rho_hist_profile.Draw("same")

    # Finally, save and cleanup
    output_name = "rho_background"
    if includes_constituent_cut:
        output_name += "_3GeVConstituents"
    plot_base.save_plot(output_info, canvas, output_name)

def track_eta_phi(hist: Hist, event_activity: params.EventActivity, output_info: analysis_objects.PlottingOutputWrapper) -> None:
    """ Plot track eta phi.

    Also include an annotation of the EMCal eta, phi location.

    Args:
        rho_hist: Rho centrality dependent hist.
        output_info: Output information.
        includes_constituent_cut: True if the plot was produced using the constituent cut.
    Returns:
        None. The figure is plotted.
    """
    # Setup
    fig, ax = plt.subplots(figsize = (8, 6))
    x, y, hist_array = histogram.get_array_from_hist2D(
        hist,
        set_zero_to_NaN = True,
        return_bin_edges = True,
    )

    plot = ax.imshow(
        hist_array.T,
        extent = [np.amin(x), np.amax(x), np.amin(y), np.amax(y)],
        interpolation = "nearest", aspect = "auto", origin = "lower",
        norm = matplotlib.colors.Normalize(vmin = np.nanmin(hist_array), vmax = np.nanmax(hist_array)),
        cmap = "viridis",
    )
    # Draw the colorbar based on the drawn axis above.
    # NOTE: This can cause the warning:
    #       '''
    #       matplotlib/colors.py:1031: RuntimeWarning: invalid value encountered in less_equal
    #           mask |= resdat <= 0"
    #       '''
    #       The warning is due to the nan we introduced above. It can safely be ignored
    #       See: https://stackoverflow.com/a/34955622
    #       (Could suppress, but I don't feel it's necessary at the moment)
    fig.colorbar(plot)

    # Draw EMCal boundaries
    r = matplotlib.patches.Rectangle(
        xy = (-0.7, 80 * np.pi / 180),
        width = 0.7 + 0.7, height = (187 - 80) * np.pi / 180,
        facecolor = "none", edgecolor = "tab:red", linewidth = 1.5,
        label = "EMCal",
    )
    ax.add_patch(r)

    # Final presentation settings
    ax.set_xlabel(labels.make_valid_latex_string(r"\eta"))
    ax.set_ylabel(labels.make_valid_latex_string(r"\varphi"))
    ax.legend(frameon = False, loc = "upper right")
    fig.tight_layout()

    # Finally, save and cleanup
    output_name = f"track_eta_phi_{str(event_activity)}"
    plot_base.save_plot(output_info, fig, output_name)
    plt.close(fig)

    # Check the phi 1D projection
    track_phi = hist.ProjectionY()
    # Make it easier to view
    track_phi.Rebin(8)
    import ROOT
    canvas = ROOT.TCanvas("c", "c")
    # Labeling
    track_phi.SetTitle("")
    track_phi.GetXaxis().SetTitle(labels.use_label_with_root(labels.make_valid_latex_string(r"\varphi")))
    track_phi.GetYaxis().SetTitle("Counts")
    track_phi.Draw()
    # Draw lines corresponding to the EMCal
    line_min = ROOT.TLine(80 * np.pi / 180, track_phi.GetMinimum(), 80 * np.pi / 180, track_phi.GetMaximum())
    line_max = ROOT.TLine(187 * np.pi / 180, track_phi.GetMinimum(), 187 * np.pi / 180, track_phi.GetMaximum())
    for l in [line_min, line_max]:
        l.SetLineColor(ROOT.kBlue)
        l.Draw("same")

    # Save the plot
    plot_base.save_plot(output_info, canvas, f"track_phi_{str(event_activity)}")

def trigger_jets_EP(ep_analyses: List[Tuple[Any, "correlations.Correlations"]], output_info: analysis_objects.PlottingOutputWrapper) -> None:
    """ Plot jets triggers as a function of event plane orientation.

    Args:
        ep_analyses: Event plane dependent analyses.
        output_info: Output information.
    Returns:
        None. The triggers are plotted.
    """
    # Setup
    fig, ax = plt.subplots(figsize = (8, 6))

    # Plot
    for key_index, analysis in ep_analyses:
        h = histogram.Histogram1D.from_existing_hist(analysis.number_of_triggers_observable.hist)
        # Scale by the bin width
        h *= 1.0 / h.bin_widths[0]
        ax.errorbar(
            h.x, h.y,
            xerr = h.bin_widths / 2, yerr = h.errors,
            marker = "o", linestyle = "None",
            label = fr"{analysis.reaction_plane_orientation.display_str()}: $N_{{\text{{trig}}}} = {analysis.number_of_triggers:g}$",
        )
        ax.set_xlim(0, 100)

    ax.text(
        0.025, 0.025, r"$N_{\text{trig}}$ restricted to " + labels.jet_pt_range_string(analysis.jet_pt),
        transform = ax.transAxes, horizontalalignment = "left",
        verticalalignment = "bottom", multialignment = "left",
    )

    # Final presentation settings
    ax.set_xlabel(labels.make_valid_latex_string(fr"{labels.jet_pt_display_label()}\:({labels.momentum_units_label_gev()})"))
    ax.set_ylabel(labels.make_valid_latex_string(fr"d\text{{n}}/d{labels.jet_pt_display_label()}\:({labels.momentum_units_label_gev()}^{{-1}})"))
    ax.set_yscale("log")
    ax.legend(frameon = False, loc = "upper right")
    fig.tight_layout()

    # Finally, save and cleanup
    output_name = f"trigger_jet_spectra_EP"
    plot_base.save_plot(output_info, fig, output_name)
    plt.close(fig)
