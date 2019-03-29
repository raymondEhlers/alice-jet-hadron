#!/usr/bin/env python

""" Fit plotting module.

Predominately related to RPF plots

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

from dataclasses import dataclass
import logging
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from typing import Any, List, Mapping, Tuple, TYPE_CHECKING, Union

from pachyderm import histogram

import reaction_plane_fit as rpf
import reaction_plane_fit.base
import reaction_plane_fit.fit

from jet_hadron.base import analysis_objects
from jet_hadron.base import labels
from jet_hadron.base import params
from jet_hadron.plot import base as plot_base
from jet_hadron.plot import correlations as plot_correlations

if TYPE_CHECKING:
    from jet_hadron.analysis import correlations

# Setup logger
logger = logging.getLogger(__name__)

# Typing helpers
FitObjects = Mapping[Any, rpf.fit.ReactionPlaneFit]

@dataclass
class ParameterInfo:
    """ Simple helper class to store inforamtion about each fit parameter for plotting.

    Attributes:
        name: Name of the parameter in the fit. Used to extract it from the stored fit result.
        output_name: Name under which the plot will be stored.
        labels: Title and axis labels for the plot.
    """
    name: str
    output_name: str
    labels: plot_base.PlotLabels

def _plot_fit_parameter_vs_assoc_pt(fit_objects: FitObjects,
                                    parameter: ParameterInfo,
                                    output_info: analysis_objects.PlottingOutputWrapper) -> None:
    """ Implementation to plot the fit parameters vs associated track pt.  """
    fig, ax = plt.subplots(figsize = (8, 6))

    # Extract the parameter values from each fit object.
    bin_centers = np.zeros(len(fit_objects))
    parameter_values = np.zeros(len(fit_objects))
    parameter_values_errors = np.zeros(len(fit_objects))
    for i, (key_index, fit_object) in enumerate(fit_objects.items()):
        bin_centers[i] = key_index.track_pt_bin.bin_center
        parameter_values[i] = fit_object.fit_result.values_at_minimum[parameter.name]
        parameter_values_errors[i] = fit_object.fit_result.errors_on_parameters[parameter.name]

    # Plot the particular parameter.
    ax.errorbar(
        bin_centers, parameter_values, yerr = parameter_values_errors,
        marker = "o", linestyle = "",
    )

    # Labeling
    parameter.labels.apply_labels(ax)
    # Final adjustments
    fig.tight_layout()
    # Save plot and cleanup
    plot_base.save_plot(output_info, fig, parameter.output_name)
    plt.close(fig)

def fit_parameters_vs_assoc_pt(fit_objects: FitObjects,
                               selected_analysis_options: params.SelectedAnalysisOptions,
                               output_info: analysis_objects.PlottingOutputWrapper) -> None:
    """ Plot the extracted fit parameters.

    Args:
        fit_objects: Fit objects whose parameters will be plotted.
        selected_analysis_options: Selected analysis options to be used for labeling.
        output_info: Output information.
    """
    pt_assoc_label = labels.make_valid_latex_string(
        f"{labels.track_pt_display_label()} ({labels.momentum_units_label_gev()})"
    )
    prefix = "fit_parameter"
    parameters = [
        # TODO: Add the rest of the parameters.
        ParameterInfo(
            name = "v2_t",
            output_name = f"{prefix}_v2t",
            labels = plot_base.PlotLabels(title = r"Jet $v_{2}$", x_label = pt_assoc_label),
        ),
        ParameterInfo(
            name = "v2_a",
            output_name = f"{prefix}_v2a",
            labels = plot_base.PlotLabels(title = r"Associated hadron $v_{2}$", x_label = pt_assoc_label),
        ),
        ParameterInfo(
            name = "v3",
            output_name = f"{prefix}_v3",
            labels = plot_base.PlotLabels(title = r"$v_{3}$", x_label = pt_assoc_label),
        ),
        ParameterInfo(
            name = "v4_t",
            output_name = f"{prefix}_v4t",
            labels = plot_base.PlotLabels(title = r"Jet $v_{4}$", x_label = pt_assoc_label),
        ),
        ParameterInfo(
            name = "v4_a",
            output_name = f"{prefix}_v4a",
            labels = plot_base.PlotLabels(title = r"Associated hadron $v_{4}$", x_label = pt_assoc_label),
        ),
    ]

    for parameter in parameters:
        _plot_fit_parameter_vs_assoc_pt(fit_objects = fit_objects, parameter = parameter, output_info = output_info)

def rpf_covariance_matrix(fit_result: reaction_plane_fit.base.RPFitResult,
                          output_info: analysis_objects.PlottingOutputWrapper,
                          output_name: str) -> None:
    """ Plot the RP fit covariance matrix.

    Code substantially improved by using the information at
    https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html

    Args:
        fit_result: Reaction plane fit result.
        output_info: Output information.
        output_name: Name of the output plot.
    Returns:
        None. The covariance matrix is saved.
    """
    # Setup
    fig, ax = plt.subplots(figsize = (8, 6))
    # Move x labels to top to follow convention
    ax.tick_params(top = True, bottom = False,
                   labeltop = True, labelbottom = False)
    # Create a map from the labels to valid LaTeX for improved presentation
    improved_labeling_map = {
        "ns_amplitude": "A_{NS}", "as_amplitude": "A_{AS}",
        "ns_sigma": r"\sigma_{NS}", "as_sigma": r"\sigma_{AS}",
        "BG": r"\text{Signal background}", "v1": "v_{1}", "v2_t": "v_{2}^{t}", "v2_a": "v_{2}^{a}",
        "v3": "v_{3}", "v4_t": "v_{4}^{t}", "v4_a": "v_{4}^{a}",
        "B": r"\text{RPF Background}",
    }
    # Ensure that the strings are valid LaTeX
    improved_labeling_map = {k: labels.make_valid_latex_string(v) for k, v in improved_labeling_map.items()}
    # Fixed parameters aren't in the covariance matrix.
    number_of_parameters = len(fit_result.free_parameters)

    logger.debug(f"number_of_parameters: {number_of_parameters}")
    logger.debug(f"values: {list(fit_result.covariance_matrix.values())}")
    covariance_matrix_values = np.array(list(fit_result.covariance_matrix.values()))
    covariance_matrix_values = covariance_matrix_values.reshape(number_of_parameters, number_of_parameters)

    # Plot the matrix
    im = ax.imshow(covariance_matrix_values, cmap = "viridis")

    # Add the colorbar
    fig.colorbar(im, ax = ax)

    # Axis labeling
    parameter_labels = [b for a, b in list(fit_result.covariance_matrix)[:number_of_parameters]]
    parameter_labels = [improved_labeling_map[l] for l in parameter_labels]
    # Show all axis ticks and then label them
    # The first step of settings the yticks is required according to the matplotlib docs
    ax.set_xticks(range(number_of_parameters))
    ax.set_yticks(range(number_of_parameters))
    # Also rotate the x-axis labels so they are all visiable.
    ax.set_xticklabels(parameter_labels, rotation = -30, horizontalalignment = "right")
    ax.set_yticklabels(parameter_labels)
    # Use minor ticks to put a white border between each value
    ax.set_xticks(np.arange(number_of_parameters) - .5, minor = True)
    ax.set_yticks(np.arange(number_of_parameters) - .5, minor = True)
    ax.grid(which = "minor", color = "w", linestyle = '-', linewidth = 3)
    ax.tick_params(which = "minor", bottom = False, left = False)

    # Label values in each element
    threshold_for_changing_label_colors = im.norm(np.max(covariance_matrix_values)) / 2
    text_colors = ["white", "black"]
    for i in range(number_of_parameters):
        for j in range(number_of_parameters):
            color = text_colors[im.norm(covariance_matrix_values[i, j]) > threshold_for_changing_label_colors]
            im.axes.text(j, i, f"{covariance_matrix_values[i, j]:.1f}",
                         horizontalalignment="center",
                         verticalalignment="center",
                         color = color)

    # Final adjustments
    fig.tight_layout()
    # Save plot and cleanup
    plot_base.save_plot(output_info, fig, output_name)
    plt.close(fig)

def _plot_rp_fit_subtracted(ep_analyses: List[Tuple[Any, "correlations.Correlations"]], axes: matplotlib.axes.Axes) -> None:
    """ Plot the RP subtracted histograms on the given set of axes.

    Args:
        ep_analyses: Event plane dependent correlation analysis objects.
        axes: Axes on which the subtracted hists should be plotted. It must have an axis per component.
    Returns:
        None. The axes are modified in place.
    """
    #x = rp_fit.fit_result.x
    #for (fit_type, component), ax in zip(rp_fit.components.items(), axes):
    for (key_index, analysis), ax in zip(ep_analyses, axes):
        hists = analysis.correlation_hists_delta_phi_subtracted
        h = hists.signal_dominated.hist

        # Plot the subtracted hist
        ax.errorbar(
            h.x, h.y, yerr = h.errors,
            label = f"Subtracted {hists.signal_dominated.type.display_str()}", marker = "o", linestyle = "None",
        )

        # Label RP orinetation
        ax.set_title(analysis.reaction_plane_orientation.display_str())

        # Add horizontal line at 0 for comparison
        ax.axhline(y = 0, color = "black", linestyle = "dashed", zorder = 1)

    # Increase the upper range by 10% to ensure that the labels don't overlap with the data.
    lower_limit, upper_limit = ax.get_ylim()
    axes[0].set_ylim(bottom = lower_limit, top = upper_limit * 1.10)

def rp_fit_subtracted(ep_analyses: List[Tuple[Any, "correlations.Correlations"]],
                      inclusive_analysis: "correlations.Correlations",
                      output_info: analysis_objects.PlottingOutputWrapper,
                      output_name: str) -> None:
    """ Basic plot of the reaction plane fit subtracted hists.

    Args:
        ep_analyses: Event plane dependent correlation analysis objects.
        inclusive_analysis: Inclusive analysis object. Mainly used for labeling.
        output_info: Output information.
        output_name: Name of the output plot.
    Returns:
        None. The plot will be saved.
    """
    # Setup
    n_components = len(ep_analyses)
    fig, axes = plt.subplots(
        1, n_components,
        sharey = "row", sharex = True,
        #gridspec_kw = {"height_ratios": [3, 1]},
        figsize = (3 * n_components, 6)
    )
    flat_axes = axes.flatten()

    # Plot the fits on the upper panels.
    _plot_rp_fit_subtracted(ep_analyses = ep_analyses, axes = flat_axes[:n_components])

    # Define upper panel labels.
    # In-plane
    text = labels.track_pt_range_string(inclusive_analysis.track_pt)
    text += "\n" + labels.constituent_cuts()
    text += "\n" + labels.make_valid_latex_string(inclusive_analysis.leading_hadron_bias.display_str())
    _add_label_to_rpf_plot_axis(ax = flat_axes[0], label = text)
    # Mid-plane
    text = labels.make_valid_latex_string(inclusive_analysis.alice_label.display_str())
    text += "\n" + labels.system_label(
        energy = inclusive_analysis.collision_energy,
        system = inclusive_analysis.collision_system,
        activity = inclusive_analysis.event_activity
    )
    text += "\n" + labels.jet_pt_range_string(inclusive_analysis.jet_pt)
    text += "\n" + labels.jet_finding()
    _add_label_to_rpf_plot_axis(ax = flat_axes[1], label = text)
    # Out-of-plane
    #text = "Background: $0.8<|\Delta\eta|<1.2$"
    #text += "\nSignal + Background: $|\Delta\eta|<0.6$"
    #_add_label_to_rpf_plot_axis(ax = flat_axes[2], label = text)
    _add_label_to_rpf_plot_axis(ax = flat_axes[2], label = labels.make_valid_latex_string(text))

    for ax in flat_axes:
        # Increase the frequency of major ticks to once every integer.
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base = 1.0))
        # Set label
        ax.set_xlabel(labels.make_valid_latex_string(r"\Delta\varphi"))

    flat_axes[0].set_ylabel(labels.make_valid_latex_string(labels.delta_phi_axis_label()))
    #jet_pt_label = labels.jet_pt_range_string(inclusive_analysis.jet_pt)
    #track_pt_label = labels.track_pt_range_string(inclusive_analysis.track_pt)
    #ax.set_title(fr"Subtracted 1D ${inclusive_analysis.correlation_hists_delta_phi_subtracted.signal_dominated.axis.display_str()}$,"
    #             f" {inclusive_analysis.reaction_plane_orientation.display_str()} event plane orient.,"
    #             f" {jet_pt_label}, {track_pt_label}")
    ax.legend(loc = "upper right")

    # Final adjustments
    fig.tight_layout()
    # Reduce spacing between subplots
    fig.subplots_adjust(hspace = 0, wspace = 0)
    # Save plot and cleanup
    plot_base.save_plot(output_info, fig,
                        f"jetH_delta_phi_{inclusive_analysis.identifier}_rp_subtracted")

def _plot_rp_fit_components(rp_fit: reaction_plane_fit.fit.ReactionPlaneFit, ep_analyses: List[Tuple[Any, "correlations.Correlations"]], axes: matplotlib.axes.Axes) -> None:
    """ Plot the RP fit components on a given set of axes.

    Args:
        rp_fit: Reaction plane fit object.
        ep_analyses: Event plane dependent correlation analysis objects.
        axes: Axes on which the residual should be plotted. It must have an axis per component.
    Returns:
        None. The axes are modified in place.
    """
    # Validation
    if len(rp_fit.components) != len(axes):
        raise TypeError(
            f"Number of axes is not equal to the number of fit components."
            f"# of components: {len(rp_fit.components)}, # of axes: {len(axes)}"
        )
    if len(ep_analyses) != len(axes):
        raise TypeError(
            f"Number of axes is not equal to the number of EP analysis objects."
            f"# of analyis objects: {len(ep_analyses)}, # of axes: {len(axes)}"
        )

    x = rp_fit.fit_result.x
    for (key_index, analysis), ax in zip(ep_analyses, axes):
        # Setup
        # Get the relevant data
        if analysis.reaction_plane_orientation == params.ReactionPlaneOrientation.inclusive:
            h: Union["correlations.DeltaPhiSignalDominated", "correlations.DeltaPhiBackgroundDominated"] = \
                analysis.correlation_hists_delta_phi.signal_dominated
        else:
            h = analysis.correlation_hists_delta_phi.background_dominated
        hist = histogram.Histogram1D.from_existing_hist(h)
        # Determine the proper display color
        data_color = plot_base.AnalysisColors.background
        if analysis.reaction_plane_orientation == params.ReactionPlaneOrientation.inclusive:
            data_color = plot_base.AnalysisColors.signal

        # Plot the data first to ensure that the colors are consistent with previous plots
        ax.errorbar(
            x, hist.y, yerr = hist.errors, label = "Data",
            marker = "o", linestyle = "", color = data_color,
        )

        # Draw the data according to the given function
        # Determine the values of the fit function.
        fit_values = analysis.fit_object.evaluate_fit(x = x)

        # Plot the main values
        plot = ax.plot(x, fit_values, label = "Fit", color = plot_base.AnalysisColors.fit)
        # Plot the fit errors
        errors = analysis.fit_object.fit_result.errors
        ax.fill_between(x, fit_values - errors, fit_values + errors, facecolor = plot[0].get_color(), alpha = 0.8)
        ax.set_title(f"{analysis.reaction_plane_orientation.display_str()} orient.")

    # Increase the upper range by 8% to ensure that the labels don't overlap with the data.
    lower_limit, upper_limit = ax.get_ylim()
    axes[0].set_ylim(bottom = lower_limit, top = upper_limit * 1.08)

def _plot_rp_fit_residuals(rp_fit: reaction_plane_fit.fit.ReactionPlaneFit, ep_analyses: List[Tuple[Any, "correlations.Correlations"]], axes: matplotlib.axes.Axes) -> None:
    """ Plot fit residuals on a given set of axes.

    Args:
        rp_fit: Reaction plane fit object.
        ep_analyses: Event plane dependent correlation analysis objects.
        axes: Axes on which the residual should be plotted. It must have an axis per component.
    Returns:
        None. The axes are modified in place.
    """
    # Validation
    if len(rp_fit.components) != len(axes):
        raise TypeError(
            f"Number of axes is not equal to the number of fit components."
            f"# of components: {len(rp_fit.components)}, # of axes: {len(axes)}"
        )
    if len(ep_analyses) != len(axes):
        raise TypeError(
            f"Number of axes is not equal to the number of EP analysis objects."
            f"# of analyis objects: {len(ep_analyses)}, # of axes: {len(axes)}"
        )

    x = rp_fit.fit_result.x
    for (key_index, analysis), ax in zip(ep_analyses, axes):
        # Setup
        # Get the relevant data
        if analysis.reaction_plane_orientation == params.ReactionPlaneOrientation.inclusive:
            h: Union["correlations.DeltaPhiSignalDominated", "correlations.DeltaPhiBackgroundDominated"] = \
                analysis.correlation_hists_delta_phi.signal_dominated
        else:
            h = analysis.correlation_hists_delta_phi.background_dominated
        hist = histogram.Histogram1D.from_existing_hist(h)

        # We create a histogram to represent the fit so that we can take advantage
        # of the error propagation in the Histogram1D object.
        fit_hist = histogram.Histogram1D(
            # Bin edges must be the same
            bin_edges = hist.bin_edges,
            y = analysis.fit_object.evaluate_fit(x = x),
            errors_squared = analysis.fit_object.fit_result.errors ** 2,
        )
        # NOTE: Residual = data - fit / fit, not just data-fit
        residual = (hist - fit_hist) / fit_hist

        # Plot the main values
        plot = ax.plot(x, residual.y, label = "Residual", color = plot_base.AnalysisColors.fit)
        # Plot the fit errors
        ax.fill_between(
            x, residual.y - residual.errors, residual.y + residual.errors,
            facecolor = plot[0].get_color(), alpha = 0.9,
        )

        # Set the y-axis limit to be symmetric
        # Selected the value by looking at the data.
        ax.set_ylim(bottom = -0.1, top = 0.1)

def _add_label_to_rpf_plot_axis(ax: matplotlib.axes.Axes, label: str) -> None:
    """ Add a label to the middle center of a axis for the RPF plot.

    This helper is limited, but useful since we often label at the same location.

    Args:
        ax: Axis to label.
        label: Label to be added to the axis.
    Returns:
        None. The axis is modified in place.
    """
    ax.text(
        0.5, 0.97, label, horizontalalignment = "center",
        verticalalignment = "top", multialignment = "left",
        transform = ax.transAxes
    )

def plot_RP_fit(rp_fit: reaction_plane_fit.fit.ReactionPlaneFit,
                inclusive_analysis: "correlations.Correlations",
                ep_analyses: List[Tuple[Any, "correlations.Correlations"]],
                output_info: analysis_objects.PlottingOutputWrapper,
                output_name: str) -> None:
    """ Basic plot of the reaction plane fit.

    Args:
        rp_fit: Reaction plane fit object.
        inclusive_analysis: Inclusive analysis object. Mainly used for labeling.
        ep_analyses: Event plane dependent correlation analysis objects.
        output_info: Output information.
        output_name: Name of the output plot.
    Returns:
        None. The plot will be saved.
    """
    # Setup
    n_components = len(rp_fit.components)
    fig, axes = plt.subplots(
        2, n_components,
        sharey = "row", sharex = True,
        gridspec_kw = {"height_ratios": [3, 1]},
        figsize = (3 * n_components, 6)
    )
    flat_axes = axes.flatten()

    # Plot the fits on the upper panels.
    _plot_rp_fit_components(rp_fit = rp_fit, ep_analyses = ep_analyses, axes = flat_axes[:n_components])
    # Plot the residuals on the lower panels.
    _plot_rp_fit_residuals(rp_fit = rp_fit, ep_analyses = ep_analyses, axes = flat_axes[n_components:])

    # Define upper panel labels.
    # In-plane
    text = labels.track_pt_range_string(inclusive_analysis.track_pt)
    text += "\n" + labels.constituent_cuts()
    text += "\n" + labels.make_valid_latex_string(inclusive_analysis.leading_hadron_bias.display_str())
    _add_label_to_rpf_plot_axis(ax = flat_axes[0], label = text)
    # Mid-plane
    text = labels.make_valid_latex_string(inclusive_analysis.alice_label.display_str())
    text += "\n" + labels.system_label(
        energy = inclusive_analysis.collision_energy,
        system = inclusive_analysis.collision_system,
        activity = inclusive_analysis.event_activity
    )
    text += "\n" + labels.jet_pt_range_string(inclusive_analysis.jet_pt)
    text += "\n" + labels.jet_finding()
    _add_label_to_rpf_plot_axis(ax = flat_axes[1], label = text)
    # Out-of-plane
    #text = "Background: $0.8<|\Delta\eta|<1.2$"
    #text += "\nSignal + Background: $|\Delta\eta|<0.6$"
    #_add_label_to_rpf_plot_axis(ax = flat_axes[2], label = text)
    # Inclusive
    text = (
        r"\chi^{2}/\mathrm{NDF} = "
        f"{rp_fit.fit_result.minimum_val:.1f}/{rp_fit.fit_result.nDOF} = "
        f"{rp_fit.fit_result.minimum_val / rp_fit.fit_result.nDOF:.3f}"
    )
    _add_label_to_rpf_plot_axis(ax = flat_axes[2], label = labels.make_valid_latex_string(text))

    # Define lower panel labels.
    for ax in flat_axes[n_components:]:
        # Increase the frequency of major ticks to once every integer.
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base = 1.0))
        # Add axis labels
        ax.set_xlabel(labels.make_valid_latex_string(inclusive_analysis.correlation_hists_delta_phi.signal_dominated.axis.display_str()))
    # Improve the viewable range for the lower panels.
    # This value is somewhat arbitrarily selected, but seems to work well enough.
    flat_axes[n_components].set_ylim(-0.2, 0.2)

    # Specify shared y axis label
    # Delta phi correlations first
    flat_axes[0].set_ylabel(labels.delta_phi_axis_label())
    # Then label the residual
    flat_axes[n_components].set_ylabel("data - fit / fit")

    # Final adjustments
    fig.tight_layout()
    # Reduce spacing between subplots
    fig.subplots_adjust(hspace = 0, wspace = 0)
    # Save plot and cleanup
    plot_base.save_plot(output_info, fig, output_name)
    plt.close(fig)

def signal_dominated_with_background_function(analysis: "correlations.Correlations") -> None:
    """ Plot the signal dominated hist with the background function. """
    # Setup
    fig, ax = plt.subplots(figsize = (8, 6))

    # Plot signal and background dominated hists
    plot_correlations.plot_and_label_1d_signal_and_background_with_matplotlib_on_axis(ax = ax, jet_hadron = analysis)

    # Plot background function
    # First we retrieve the signal dominated histogram to get reference x values and bin edges.
    h = histogram.Histogram1D.from_existing_hist(analysis.correlation_hists_delta_phi.signal_dominated.hist)
    background = histogram.Histogram1D(
        bin_edges = h.bin_edges,
        y = analysis.fit_object.evaluate_background(h.x),
        errors_squared = analysis.fit_object.calculate_background_function_errors(h.x) ** 2,
    )
    background_plot = ax.plot(background.x, background.y, label = "Background function")
    ax.fill_between(
        background.x, background.y - background.errors, background.y + background.errors,
        facecolor = background_plot[0].get_color(), alpha = 0.9,
    )

    # Labeling
    ax.legend(loc = "upper right")

    # Final adjustments
    fig.tight_layout()
    # Save plot and cleanup
    plot_base.save_plot(analysis.output_info, fig,
                        f"jetH_delta_phi_{analysis.identifier}_signal_background_function_comparison")
    plt.close(fig)

def fit_subtracted_signal_dominated(analysis: "correlations.Correlations") -> None:
    """ Plot the subtracted signal dominated hist. """
    # Setup
    fig, ax = plt.subplots(figsize = (8, 6))
    hists = analysis.correlation_hists_delta_phi_subtracted
    h = hists.signal_dominated.hist

    # Plot the subtracted hist
    ax.errorbar(
        h.x, h.y, yerr = h.errors,
        label = f"Subtracted {hists.signal_dominated.type.display_str()}", marker = "o", linestyle = "",
    )

    # Labels.
    ax.set_xlabel(labels.make_valid_latex_string(r"\Delta\varphi"))
    ax.set_ylabel(labels.make_valid_latex_string(labels.delta_phi_axis_label()))
    jet_pt_label = labels.jet_pt_range_string(analysis.jet_pt)
    track_pt_label = labels.track_pt_range_string(analysis.track_pt)
    ax.set_title(fr"Subtracted 1D ${hists.signal_dominated.axis.display_str()}$,"
                 f" {analysis.reaction_plane_orientation.display_str()} event plane orient.,"
                 f" {jet_pt_label}, {track_pt_label}")
    ax.legend(loc = "upper right")

    # Final adjustments
    fig.tight_layout()
    # Save plot and cleanup
    plot_base.save_plot(analysis.output_info, fig,
                        f"jetH_delta_phi_{analysis.identifier}_subtracted")
    plt.close(fig)

