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
from typing import Any, cast, Dict, Iterator, Sequence, Tuple, TYPE_CHECKING

import pachyderm.fit
from pachyderm import histogram
from pachyderm import utils

from jet_hadron.base import analysis_objects
from jet_hadron.base import labels
from jet_hadron.base import params
from jet_hadron.base.typing_helpers import Hist
from jet_hadron.plot import base as plot_base

import ROOT

if TYPE_CHECKING:
    from jet_hadron.analysis import response_matrix

logger = logging.getLogger(__name__)

Analyses = Dict[Any, "response_matrix.ResponseMatrix"]

def plot_particle_level_spectra(ep_analyses_iter: Iterator[Tuple[Any, "response_matrix.ResponseMatrix"]],
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
    ep_analyses = dict(ep_analyses_iter)

    # Determine the general and plot labels
    # First, we need some variables to define the general labels, so we retrieve the inclusive analysis.
    # All of the parameters retrieved here are shared by all analyses.
    inclusive = next(iter(ep_analyses.values()))
    # Then we define some additional helper variables
    particle_level_spectra_bin = inclusive.task_config["particle_level_spectra"]["particle_level_spectra_bin"]
    embedded_additional_label = inclusive.event_activity.display_str()

    # General labels
    general_labels = {
        "alice_and_collision_energy":
            rf"{inclusive.alice_label.display_str()}\:{inclusive.collision_energy.display_str()}",
        "collision_system_and_event_activity":
            rf"{inclusive.collision_system.display_str(embedded_additional_label = embedded_additional_label)}",
        "detector_pt_range": labels.pt_range_string(
            particle_level_spectra_bin,
            lower_label = "T,jet",
            upper_label = "det",
        ),
        "constituent_cuts": labels.constituent_cuts(additional_label = "det"),
        "leading_hadron_bias": inclusive.leading_hadron_bias.display_str(additional_label = "det"),
        "jet_finding": labels.jet_finding(),
    }
    # Ensure that each line is a valid latex line.
    # The heuristic is roughly that full statements (such as jet_finding) are already wrapped in "$",
    # while partial statements, such as the leading hadron bias, event activity, etc are not wrapped in "$".
    # This is due to the potential for such "$" to interfere with including those partial statements in other
    # statements. As an example, it would be impossible to use the ``embedded_additional_label`` above if the
    # ``event_activity`` included "$".
    for k, v in general_labels.items():
        general_labels[k] = labels.make_valid_latex_string(v)

    # Plot labels
    y_label = r"\mathrm{d}N/\mathrm{d}p_{\mathrm{T}}"
    if inclusive.task_config["particle_level_spectra"]["normalize_by_n_jets"]:
        y_label = r"(1/N_{\mathrm{jets}})" + y_label
    if inclusive.task_config["particle_level_spectra"]["normalize_at_selected_jet_pt_bin"]:
        # Assumes that we'll never set an upper bound.
        values = inclusive.task_config["particle_level_spectra"]["normalize_at_selected_jet_pt_values"]
        y_label = r"(1/N_{\text{jets}}^{p_{\text{T}} > " + fr"{values.min}\:{labels.momentum_units_label_gev()}" + r"})" + y_label
    # Add y_label units
    y_label += fr"\:({labels.momentum_units_label_gev()})^{{-1}}"
    plot_labels = plot_base.PlotLabels(
        title = "",
        x_label = fr"${labels.jet_pt_display_label(upper_label = 'part')}\:({labels.momentum_units_label_gev()})$",
        y_label = labels.make_valid_latex_string(y_label),
    )

    # Finally, we collect our arguments for the plotting functions.
    kwargs: Dict[str, Any] = {
        "ep_analyses": ep_analyses,
        "output_name": "particle_level_spectra",
        "output_info": output_info,
        "general_labels": general_labels,
        "plot_labels": plot_labels,
    }

    if plot_with_ROOT:
        _plot_particle_level_spectra_with_ROOT(**kwargs)
    else:
        _plot_particle_level_spectra_with_matplotlib(**kwargs)

def _plot_particle_level_spectra_with_matplotlib(ep_analyses: Analyses,
                                                 output_name: str,
                                                 output_info: analysis_objects.PlottingOutputWrapper,
                                                 general_labels: Dict[str, str],
                                                 plot_labels: plot_base.PlotLabels) -> None:
    """ Plot the particle level spectra with matplotlib.

    Args:
        ep_analyses: The final event plane dependent response matrix analysis objects.
        output_name: Name of the output plot.
        output_info: Output information.
        general_labels: General informational labels for the plot (ALICE, collision system, etc).
        plot_labels: plot and axis titles.
    Returns:
        None. The created canvas is plotted and saved.
    """
    # Setup
    fig, ax = plt.subplots(figsize = (8, 6))
    # Diamond, square, up triangle, circle
    markers = ["D", "s", "^", "o"]
    colors = ["black", "blue", "green", "red"]
    # Label axes
    plot_labels.apply_labels(ax)

    # Plot the actual hists. The inclusive orientation will be plotted first.
    particle_level_max_pt = 0
    for analysis, color, marker in zip(ep_analyses.values(), colors, markers):
        # Store this value for convenience. It is the same for all orientations.
        particle_level_max_pt = analysis.task_config["particle_level_spectra"]["particle_level_max_pt"]

        # For inclusive, use open markers that are plotted on top of all points.
        additional_args = {}
        if analysis.reaction_plane_orientation == params.ReactionPlaneOrientation.inclusive:
            additional_args.update({
                "fillstyle": "none",
                "zorder": 10,
            })

        # Convert and plot hist
        h = histogram.Histogram1D.from_existing_hist(analysis.particle_level_spectra)
        ax.errorbar(
            h.x, h.y,
            xerr = (h.bin_edges[1:] - h.bin_edges[:-1]) / 2,
            yerr = h.errors,
            label = analysis.reaction_plane_orientation.display_str(),
            color = color,
            marker = marker,
            linestyle = "",
            **additional_args
        )

    # Final presentation settings
    # Axis ticks
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base = 10))
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base = 2))
    tick_shared_args = {
        "axis": "both",
        "bottom": True,
        "left": True,
    }
    ax.tick_params(
        which = "major",
        # Size of the axis mark labels
        labelsize = 15,
        length = 8,
        **tick_shared_args,
    )
    ax.tick_params(
        which = "minor",
        length = 4,
        **tick_shared_args,
    )
    # Limits
    ax.set_xlim(0, particle_level_max_pt)
    # Unfortunately, MPL doesn't calculate restricted log limits very nicely, so we
    # we have to set the values by hand.
    # We grab the value from the last analysis object - the value will be the same for all of them.
    y_limits = analysis.task_config["particle_level_spectra"]["y_limits"]
    ax.set_ylim(y_limits[0], y_limits[1])
    ax.set_yscale("log")
    # Legend
    ax.legend(
        loc = "lower left",
        frameon = False,
        fontsize = 18,
    )
    # Combine the general labels and then plot them.
    label = "\n".join(general_labels.values())
    # The label is anchored in the upper right corner.
    ax.text(0.95, 0.95, s = label,
            horizontalalignment = "right",
            verticalalignment = "top",
            multialignment = "right",
            fontsize = 18,
            transform = ax.transAxes)
    fig.tight_layout()

    # Finally, save and cleanup
    output_name += "_mpl"
    plot_base.save_plot(output_info, fig, output_name)
    plt.close(fig)

def _plot_particle_level_spectra_with_ROOT(ep_analyses: Analyses,
                                           output_name: str,
                                           output_info: analysis_objects.PlottingOutputWrapper,
                                           general_labels: Dict[str, str],
                                           plot_labels: plot_base.PlotLabels) -> None:
    """ Plot the particle level spectra with ROOT.

    Args:
        ep_analyses: The final event plane dependent response matrix analysis objects.
        output_name: Name of the output plot.
        output_info: Output information.
        general_labels: General informational labels for the plot (ALICE, collision system, etc).
        plot_labels: plot and axis titles.
    Returns:
        None. The created canvas is plotted and saved.
    """
    # Setup
    # Aesthetics
    # Colors and markers are from Joel's plots.
    colors = [ROOT.kBlack, ROOT.kBlue - 7, 8, ROOT.kRed - 4]
    markers = [ROOT.kFullDiamond, ROOT.kFullSquare, ROOT.kFullTriangleUp, ROOT.kFullCircle]
    # Diamond: 1.6
    # Triangle: 1.2
    marker_sizes = [1.6, 1.1, 1.2, 1.1]

    # Canvas
    canvas = ROOT.TCanvas("canvas", "canvas")
    canvas.SetTopMargin(0.04)
    canvas.SetLeftMargin(0.14)
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

    # Main labeling
    latex_labels = []
    # ALICE + collision energy
    latex_labels.append(ROOT.TLatex(
        0.5675, 0.90,
        labels.use_label_with_root(general_labels["alice_and_collision_energy"])
    ))
    # Collision system + event activity
    # We want the centrality to appear between the cross symbol and Pb--Pb
    # NOTE: The y value is minimally adjusted down from the constant 0.06 decrease because the sqrt extends far down.
    latex_labels.append(ROOT.TLatex(
        0.5375, 0.825,
        labels.use_label_with_root(general_labels["collision_system_and_event_activity"]),
    ))
    # Particle level spectra range in detector pt.
    latex_labels.append(ROOT.TLatex(
        0.605, 0.75,
        labels.use_label_with_root(general_labels["detector_pt_range"]),
    ))
    # Constituent cuts
    latex_labels.append(ROOT.TLatex(
        0.5675, 0.675,
        labels.use_label_with_root(general_labels["constituent_cuts"]),
    ))
    # Leading hadron bias
    # The x position of this label depends on the value.
    # We need some additional parameters to determine the position, so we retrieve the inclusive analysis.
    # All of the parameters retrieved here are shared by all analyses.
    inclusive = next(iter(ep_analyses.values()))
    # We start we a semi-reasonable position with the expectation that we will usually overwrite it.
    leading_hadron_bias_label_x_position = 0.6
    # Track bias
    if inclusive.leading_hadron_bias.type == params.LeadingHadronBiasType.track:
        leading_hadron_bias_label_x_position = 0.6025
    # Cluster bias
    if inclusive.leading_hadron_bias.type == params.LeadingHadronBiasType.cluster and \
            inclusive.leading_hadron_bias.value < 10:
        leading_hadron_bias_label_x_position = 0.633
    latex_labels.append(ROOT.TLatex(
        leading_hadron_bias_label_x_position, 0.60,
        # Replace necessary because ROOT LaTeX support sux...
        # Includes "d" in finding the space because there is another space that's rendered properly
        # later in the string...
        labels.use_label_with_root(general_labels["leading_hadron_bias"]).replace(r"d\:", "d "),
    ))
    # Jet finding
    latex_labels.append(ROOT.TLatex(0.71, 0.525, labels.use_label_with_root(general_labels["jet_finding"])))

    # Plot the actual hists. The inclusive orientation will be plotted first.
    for i, (analysis, color, marker, marker_size) in enumerate(zip(ep_analyses.values(), colors, markers, marker_sizes)):
        # The hist to be plotted. We explicitly retrieve it for convenience.
        hist = analysis.particle_level_spectra

        # Set the titles
        plot_labels.apply_labels(hist)

        # Style each individual hist. In principle, we could do this for only one hist and then set the
        # axis labels to empty for the rest, but then we would have to empty out the labels. This is just,
        # as easy, and then we don't have to deal with changing the labels.
        # Enlarge axis title size
        hist.GetXaxis().SetTitleSize(0.055)
        hist.GetYaxis().SetTitleSize(0.055)
        # Ensure there is enough space
        hist.GetXaxis().SetTitleOffset(1.15)
        hist.GetYaxis().SetTitleOffset(1.22)
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
        hist.SetLineColor(color)
        hist.SetMarkerColor(color)
        hist.SetMarkerStyle(marker)
        # Increase marker size slightly
        hist.SetMarkerSize(marker_size)
        # Could increase the line width if the inclusive angle was closed, but
        # the open marker doesn't look very good...
        #hist.SetLineWidth(2)

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

    # Redraw the inclusive hist so that it's on top.
    inclusive.particle_level_spectra.Draw("same")

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

def particle_level_spectra_ratios(ep_analyses_iter: Iterator[Tuple[Any, "response_matrix.ResponseMatrix"]],
                                  output_info: analysis_objects.PlottingOutputWrapper) -> None:
    """ Create ratios relative to the particle level spectra and plot them.

    Args:
        ep_analyses: The event plane dependent final response matrices.
        output_info: Output information.
    Returns:
        None. The spectra are plotted and saved.
    """
    # Setup
    fig, ax = plt.subplots(figsize = (8, 6))
    # Diamond, square, up triangle, circle
    markers = ["D", "s", "^", "o"]
    colors = ["black", "blue", "green", "red"]
    # Pull out the dict because we need to grab individual analyses for some labeling information, which doesn't
    # play well with generators (the generator will be exhausted).
    ep_analyses = dict(ep_analyses_iter)
    # First, we need the inclusive analysis spectra to define the ratio.
    inclusive = next(iter(ep_analyses.values()))
    inclusive_hist = histogram.Histogram1D.from_existing_hist(inclusive.particle_level_spectra)

    # Setup rank 1 polynomial fit (not really the right place, but it's quick and fine
    # for these purposees).
    def degree_1_polynomial(x: float, const: float, slope: float) -> float:
        """ Degree 1 polynomial.

        Args:
            x: Independent variable.
            const: Constant offset.
            slope: Coefficient for 1st degree term.
        Returns:
            Calculated first degree polynomial.
        """
        return const + x * slope

    class Polynomial(pachyderm.fit.Fit):
        """ Fit a degree-1 to the background dominated region of a delta eta hist.

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
            self.fit_function = degree_1_polynomial

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

        def _setup(self, h: histogram.Histogram1D) -> Tuple[histogram.Histogram1D, pachyderm.fit.T_FitArguments]:
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

            # Default arguments
            # Use the minimum of the histogram as the starting value.
            arguments: pachyderm.fit.FitArguments = {
                "slope": 0, "error_slope": 0.005,
                "const": 1, "error_const": 0.005,
                "limit_slope": (-100, 100),
                "limit_const": (-10, 10),
            }

            return restricted_hist, arguments

    for analysis, color, marker in zip(ep_analyses.values(), colors, markers):
        # For inclusive, use open markers that are plotted on top of all points.
        additional_args = {}
        if analysis.reaction_plane_orientation == params.ReactionPlaneOrientation.inclusive:
            additional_args.update({
                "fillstyle": "none",
                "zorder": 10,
            })

        # Convert and plot hist
        h = histogram.Histogram1D.from_existing_hist(analysis.particle_level_spectra)
        h /= inclusive_hist
        ax.errorbar(
            h.x, h.y,
            xerr = (h.bin_edges[1:] - h.bin_edges[:-1]) / 2,
            yerr = h.errors,
            label = analysis.reaction_plane_orientation.display_str(),
            color = color,
            marker = marker,
            linestyle = "",
            **additional_args
        )

        # Fit to a degree-1 polynomial and plot
        fit_object = Polynomial(
            use_log_likelihood = False,
            fit_options = {"range": analysis.task_config["particle_level_spectra"]["normalize_at_selected_jet_pt_values"]}
        )
        fit_result = fit_object.fit(h = h)
        fit_object.fit_result = fit_result
        values = fit_object(fit_result.x, *fit_result.values_at_minimum.values())
        # Plot the values
        ax.plot(
            fit_result.x, values,
            label = rf"Fit, {fit_result.values_at_minimum['const']:.2f} $\pm$ {fit_result.errors_on_parameters['const']:.2f} + "
                    + rf"{fit_result.values_at_minimum['slope']:.1e} $\pm$ {fit_result.errors_on_parameters['slope']:.1e} "
                    + rf"* ${labels.jet_pt_display_label()}$",
            color = color,
        )
        # And the error bands
        ax.fill_between(
            fit_result.x, values - fit_result.errors,
            values + fit_result.errors,
            facecolor = color, alpha = 0.5,
        )

    # Final presentation settings
    # Legend
    ax.legend(
        # Here, we specify the location of the upper right corner of the legend box.
        loc = "upper right",
        bbox_to_anchor = (0.99, 0.99),
        borderaxespad = 0,
        fontsize = 14,
        ncol = 2,
    )
    plot_labels = plot_base.PlotLabels(
        title = "",
        x_label = fr"${labels.jet_pt_display_label(upper_label = 'part')}\:({labels.momentum_units_label_gev()})$",
        y_label = "Ratio to inclusive",
    )
    plot_labels.apply_labels(ax)
    # Use the same xlimits as for the particle level spectra
    ax.set_xlim(0, inclusive.task_config["particle_level_spectra"]["particle_level_max_pt"])
    # Should be centered around 1.
    ax.set_ylim(0.5, 1.5)
    fig.tight_layout()

    # Finally, save and cleanup
    output_name = "particle_level_ratios"
    plot_base.save_plot(output_info, fig, output_name)
    plt.close(fig)

def compare_STAR_and_ALICE(star_final_response_task: "response_matrix.ResponseMatrixBase",
                           alice_particle_level_spectra: Dict[params.CollisionEnergy, Hist],
                           output_info: analysis_objects.PlottingOutputWrapper) -> None:
    # Setup
    fig, ax = plt.subplots(figsize = (8, 6))

    # First, plot the STAR points
    star_centrality_map = {
        params.EventActivity.semi_central: r"20 \textendash 50 \%",
    }
    star_hist = histogram.Histogram1D.from_existing_hist(star_final_response_task.particle_level_spectra)
    star_label = f"STAR ${star_final_response_task.collision_energy.display_str()}$ hard-core jets"
    star_label += "\n" + f"PYTHIA with ${star_centrality_map[star_final_response_task.event_activity]}$ ${star_final_response_task.collision_system.display_str()}$ det. conditions"
    ax.errorbar(
        star_hist.x, star_hist.y,
        xerr = (star_hist.bin_edges[1:] - star_hist.bin_edges[:-1]) / 2,
        yerr = star_hist.errors,
        label = star_label,
        color = "blue",
        marker = "s",
        linestyle = "",
    )

    # Convert and plot hist
    # Markers are for 2.76, 5.02 TeV
    markers = ["s", "o"]
    for (collision_energy, part_level_hist), marker in zip(alice_particle_level_spectra.items(), markers):
        alice_label = f"ALICE ${collision_energy.display_str()}$ biased jets"
        alice_label += "\n" + f"${params.CollisionSystem.embedPythia.display_str(embedded_additional_label = star_final_response_task.event_activity.display_str())}$"
        h = histogram.Histogram1D.from_existing_hist(part_level_hist)
        ax.errorbar(
            h.x, h.y,
            xerr = (h.bin_edges[1:] - h.bin_edges[:-1]) / 2,
            yerr = h.errors,
            label = alice_label,
            color = "red",
            marker = marker,
            linestyle = "",
        )

    # Label axes
    y_label = r"\text{d}N/\text{d}p_{\text{T}}"
    if star_final_response_task.task_config["particle_level_spectra"]["normalize_by_n_jets"]:
        y_label = r"(1/N_{\text{jets}})" + y_label
    if star_final_response_task.task_config["particle_level_spectra"]["normalize_at_selected_jet_pt_bin"]:
        # Assumes that we'll never set an upper bound.
        values = star_final_response_task.task_config["particle_level_spectra"]["normalize_at_selected_jet_pt_values"]
        y_label = r"(1/N_{\text{jets}}^{p_{\text{T}} > " + fr"{values.min}\:{labels.momentum_units_label_gev()}" + r"})" + y_label
    plot_labels = plot_base.PlotLabels(
        title = "",
        x_label = fr"${labels.jet_pt_display_label(upper_label = 'part')}\:({labels.momentum_units_label_gev()})$",
        y_label = labels.make_valid_latex_string(y_label),
    )
    # Apply labels individually so we can increase the font size...
    ax.set_xlabel(plot_labels.x_label, fontsize = 16)
    ax.set_ylabel(plot_labels.y_label, fontsize = 16)
    ax.set_title("")
    # Final presentation settings
    # Axis ticks
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base = 10))
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base = 2))
    tick_shared_args = {
        "axis": "both",
        "bottom": True,
        "left": True,
    }
    ax.tick_params(
        which = "major",
        # Size of the axis mark labels
        labelsize = 15,
        length = 8,
        **tick_shared_args,
    )
    ax.tick_params(
        which = "minor",
        length = 4,
        **tick_shared_args,
    )
    # Limits
    ax.set_xlim(0, star_final_response_task.task_config["particle_level_spectra"]["particle_level_max_pt"])
    # Unfortunately, MPL doesn't calculate restricted log limits very nicely, so we
    # we have to set the values by hand.
    # We grab the value from the last analysis object - the value will be the same for all of them.
    y_limits = star_final_response_task.task_config["particle_level_spectra"]["y_limits"]
    ax.set_ylim(y_limits[0], y_limits[1])
    ax.set_yscale("log")
    # Legend
    ax.legend(
        # Here, we specify the location of the upper right corner of the legend box.
        loc = "upper right",
        bbox_to_anchor = (0.99, 0.99),
        borderaxespad = 0,
        frameon = True,
        fontsize = 15,
    )
    ax.text(0.99, 0.75, s = "Inclusive event plane orientation",
            horizontalalignment = "right",
            verticalalignment = "top",
            multialignment = "right",
            fontsize = 15,
            transform = ax.transAxes)
    fig.tight_layout()

    # Finally, save and cleanup
    output_name = "particle_level_comparison_STAR_ALICE"
    output_name += "_mpl"
    plot_base.save_plot(output_info, fig, output_name)
    plt.close(fig)

def plot_response_matrix_and_errors(obj: "response_matrix.ResponseMatrixBase",
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
    x_label = fr"{labels.jet_pt_display_label(upper_label = 'hybrid')}\:({labels.momentum_units_label_gev()})"
    y_label = fr"{labels.jet_pt_display_label(upper_label = 'part')}\:({labels.momentum_units_label_gev()})"

    # Determine args and call
    args = {
        "name": name,
        "x_label": labels.make_valid_latex_string(x_label),
        "y_label": labels.make_valid_latex_string(y_label),
        "output_name": output_name,
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
    canvas.SetLogz(True)

    # Plot the histogram
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
        # Help out mypy
        merged_analysis = cast(analysis_objects.JetHReactionPlane, merged_analysis)
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
        marker = ".",
        linestyle = "",
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
        ax.errorbar(
            h.x, h.y,
            yerr = h.errors,
            label = label,
            color = color,
            marker = ".",
            linestyle = "",
        )

    # Final presentation settings
    # Ensure that the max is never beyond 300 for better presentation.
    max_limit = np.max(merged_hist.x)
    if max_limit > 300:
        max_limit = 300
    ax.set_xlim(0, max_limit)
    ax.set_yscale("log")
    ax.legend(loc = "best", frameon = False, ncol = 2, fontsize = 11)
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
    legend.SetHeader(r"p_{\mathrm{T}}\:\mathrm{bins}", "C")
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

def plot_particle_level_spectra_agreement(difference: Hist, absolute_value_of_difference: Hist,
                                          output_info: analysis_objects.PlottingOutputWrapper) -> None:
    """ Plot the agreement of the particle level spectra between the inclusive and sum of all EP orientations.

    Args:
        difference: Hist of the sum of the EP orientations spectra minus the inclusive spectra.
        absolute_value_of_difference: Same as the difference hist, but having taken the absolute value.
            This allows us to plot the difference on a log scale (which is useful if the differences
            are small).
        output_info: Output information.
    Returns:
        None.
    """
    # Setup
    output_name = "difference_of_sum_EP_orientations_vs_inclusive"
    canvas = ROOT.TCanvas("canvas", "canvas")
    # Labeling
    x_label = labels.use_label_with_root(
        fr"{labels.jet_pt_display_label(upper_label = 'part')}\:({labels.momentum_units_label_gev()})"
    )
    y_label = r"\mathrm{d}N/\mathrm{d}p_{\mathrm{T}}"

    # Apply settings to hists
    for h in [difference, absolute_value_of_difference]:
        # Labeling
        h.GetXaxis().SetTitle(x_label)
        h.GetYaxis().SetTitle(y_label)
        # Center axis title
        h.GetXaxis().CenterTitle(True)
        h.GetYaxis().CenterTitle(True)

    # Draw and save the difference histogram.
    difference.Draw()
    plot_base.save_plot(output_info, canvas, output_name)

    # Draw and save the absolute value of the difference histogram.
    absolute_value_of_difference.Draw()
    canvas.SetLogy(True)
    output_name += "_abs"
    plot_base.save_plot(output_info, canvas, output_name)

def matched_jet_energy_scale(plot_labels: plot_base.PlotLabels, output_name: str,
                             output_info: analysis_objects.PlottingOutputWrapper,
                             obj: "response_matrix.ResponseMatrixBase") -> None:
    # Setup
    canvas = ROOT.TCanvas("canvas", "canvas")
    canvas.SetLogz(True)
    hist = obj.matched_jet_pt_difference
    logger.debug(f"hist: {hist}")

    # Plot the histogram
    plot_labels.apply_labels(hist)
    hist.Draw("colz")

    # Axis ranges
    hist.GetXaxis().SetRangeUser(0, 150)
    # Scale Z axis. Otherwise, we won't see much.
    min_val = ctypes.c_double(0)
    max_val = ctypes.c_double(0)
    hist.GetMinimumAndMaximum(min_val, max_val)
    # * 1.1 to put it slightly above the max value
    # min_val doesn't work here, because there are some entries at 0
    hist.GetZaxis().SetRangeUser(10e-7, max_val.value * 1.1)

    # Save
    plot_base.save_plot(output_info, canvas, output_name)
