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
import os
import seaborn as sns
from typing import Any, List, Mapping, Tuple, TYPE_CHECKING

from pachyderm import generic_config
from pachyderm import histogram

import reaction_plane_fit as rpf
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
        # Increate the frequency of major ticks to once every integer.
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base = 1.0))
        # Set label
        ax.set_xlabel(labels.make_valid_latex_string(r"\Delta\varphi"))

    ax.set_ylabel(labels.make_valid_latex_string(labels.delta_phi_axis_label()))
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

def _plot_rp_fit_components(rp_fit: reaction_plane_fit.fit.ReactionPlaneFit, data: reaction_plane_fit.fit.Data, axes: matplotlib.axes.Axes) -> None:
    """ Plot the RP fit components on a given set of axes.

    Args:
        rp_fit: Reaction plane fit object.
        data: Reaction plane fit data.
        axes: Axes on which the residual should be plotted. It must have an axis per component.
    Returns:
        None. The axes are modified in place.
    """
    # Validation
    if len(rp_fit.components) != len(axes):
        raise TypeError(f"Number of axes is not equal to the number of fit components. # of components: {len(rp_fit.components)}, # of axes: {len(axes)}")

    # TODO: Centralize these values
    signal_color = "C0"  # Blue
    background_color = "C1"  # Orange
    fit_color = "C4"  # Purple

    x = rp_fit.fit_result.x
    for (fit_type, component), ax in zip(rp_fit.components.items(), axes):
        # Setup
        # Get the relevant data
        hist = data[fit_type]
        reaction_plane_orientation = params.ReactionPlaneOrientation[fit_type.orientation]
        data_color = background_color
        if reaction_plane_orientation == params.ReactionPlaneOrientation.inclusive:
            data_color = signal_color

        # Plot the data first to ensure that the colors are consistent with previous plots
        ax.errorbar(
            x, hist.y, yerr = hist.errors, label = "Data",
            marker = "o", linestyle = "", color = data_color,
        )

        # Draw the data according to the given function
        # Determine the values of the fit function.
        fit_values = component.evaluate_fit(x = x)

        # Plot the main values
        plot = ax.plot(x, fit_values, label = "Fit", color = fit_color)
        # Plot the fit errors
        errors = component.fit_result.errors
        ax.fill_between(x, fit_values - errors, fit_values + errors, facecolor = plot[0].get_color(), alpha = 0.8)
        # TODO: Update label.
        ax.set_title(reaction_plane_orientation.display_str())

    # Increase the upper range by 8% to ensure that the labels don't overlap with the data.
    lower_limit, upper_limit = ax.get_ylim()
    axes[0].set_ylim(bottom = lower_limit, top = upper_limit * 1.08)

def _plot_rp_fit_residuals(rp_fit: reaction_plane_fit.fit.ReactionPlaneFit, data: reaction_plane_fit.fit.Data, axes: matplotlib.axes.Axes) -> None:
    """ Plot fit residuals on a given set of axes.

    Args:
        rp_fit: Reaction plane fit object.
        data: Reaction plane fit data.
        axes: Axes on which the residual should be plotted. It must have an axis per component.
    Returns:
        None. The axes are modified in place.
    """
    # Validation
    if len(rp_fit.components) != len(axes):
        raise TypeError(f"Number of axes is not equal to the number of fit components. # of components: {len(rp_fit.components)}, # of axes: {len(axes)}")

    # TODO: Centralize these values
    fit_color = "C4"  # Purple

    x = rp_fit.fit_result.x
    for (fit_type, component), ax in zip(rp_fit.components.items(), axes):
        # Get the relevant data
        hist = data[fit_type]

        # We create a histogram to represent the fit so that we can take advantage
        # of the error propagation in the Histogram1D object.
        fit_hist = histogram.Histogram1D(
            # Bin edges must be the same
            bin_edges = hist.bin_edges,
            y = component.evaluate_fit(x = x),
            errors_squared = component.fit_result.errors ** 2,
        )
        # NOTE: Residual = data - fit / fit, not just data-fit
        residual = (hist - fit_hist) / fit_hist

        # Plot the main values
        plot = ax.plot(x, residual.y, label = "Residual", color = fit_color)
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

def plot_RP_fit(rp_fit: reaction_plane_fit.fit.ReactionPlaneFit, data: reaction_plane_fit.fit.Data,
                inclusive_analysis: "correlations.Correlations",
                output_info: analysis_objects.PlottingOutputWrapper,
                output_name: str) -> None:
    """ Basic plot of the reaction plane fit.

    Args:
        rp_fit: Reaction plane fit object.
        data: Reaction plane fit data.
        inclusive_analysis: Inclusive analysis object. Mainly used for labeling.
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
    _plot_rp_fit_components(rp_fit = rp_fit, data = data, axes = flat_axes[:n_components])
    # Plot the residuals on the lower panels.
    _plot_rp_fit_residuals(rp_fit = rp_fit, data = data, axes = flat_axes[n_components:])

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

    # Deifne lower panel labels.
    for ax in flat_axes[n_components:]:
        # Increate the frequency of major ticks to once every integer.
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

def plotMinuitQA(epFitObj, fitObj, fitsDict, minuit, jetPtBin, trackPtBin):
    """ Plot showing iminuit QA.

    This is really just a debug plot, but it is nice to have available

    NOTE: We can't really modify the plot layout much because this uses a predefined function
          in the probfit package that doesn't seem to take well to modification.

    Args:
        epFitObj (JetHEPFit): The fit object for this plot.
    """
    # NOTE: Turned off parameter printing to make it easier to see
    # NOTE: Can enable parts = true to draw each part of an added PDF separately, but it
    #       is often difficult to get a good view of it, so it's not super useful
    fitObj.draw(minuit, print_par=False)

    # Retreive fig, axes to attempt to customize the plots some
    fig = plt.gcf()
    axes = fig.get_axes()

    # Label the plots for clarity
    #logger.debug("Drawing legend with axes: {}, fits.iterkeys(): {}".format(axes, fits.keys()))
    for ax, label in zip(axes, fitsDict.keys()):
        logger.debug("label: {}".format(label))
        ax.set_title(label)

    # Reduce overlap
    fig.tight_layout()

    # Save plot
    plot_base.save_plot(epFitObj, fig, epFitObj.fitNameFormat.format(jetPtBin = jetPtBin, trackPtBin = trackPtBin, tag = epFitObj.overallFitLabel.str() + "Minuit"))
    # Cleanup
    plt.close(fig)

def PlotRPF(epFitObj):
    # Get current plotting settings to access values consistently
    plottingSettings = sns.plotting_context()
    colors = sns.color_palette()

    colorIter = iter(colors)
    colorsMap = {
        (analysis_objects.CorrelationType.signal_dominated, "Fit"): next(colorIter),
        (analysis_objects.CorrelationType.signal_dominated, "Data"): next(colorIter),
        (analysis_objects.CorrelationType.background_dominated, "Fit"): next(colorIter),
        (analysis_objects.CorrelationType.background_dominated, "Data"): next(colorIter)
    }
    zOrder = {
        (analysis_objects.CorrelationType.signal_dominated, "Fit"): 10,
        (analysis_objects.CorrelationType.signal_dominated, "FitErrorBars"): 9,
        (analysis_objects.CorrelationType.signal_dominated, "Data"): 6,
        (analysis_objects.CorrelationType.background_dominated, "Fit"): 8,
        (analysis_objects.CorrelationType.background_dominated, "FitErrorBars"): 7,
        (analysis_objects.CorrelationType.background_dominated, "Data"): 5
    }

    for (jetPtBin, trackPtBin), fitCont in epFitObj.fitContainers.items():
        # Define axes for plot
        fig, axes = plt.subplots(1, 4, sharey = True, sharex = True, figsize = (12, 6))
        # TODO: Residual = data-fit/fit, not just data-fit
        figResidual, axesResidual = plt.subplots(1, 4, sharey = True, sharex = True)

        # Store legend information
        handles = []
        labels = []

        handlesResidual = []
        labelsResidual = []

        # Store the all angles data generated from the other angles
        allAnglesSummedFromFit = {analysis_objects.CorrelationType.background_dominated: None,
                                  analysis_objects.CorrelationType.signal_dominated: None}

        # Put the all angles at the end for consistnecy
        epAngles = [angle for angle in params.ReactionPlaneOrientation]
        epAngles.append(epAngles.pop(epAngles.index(params.ReactionPlaneOrientation.all)))

        for i, (epAngle, ax, axResidual) in enumerate(zip(epAngles, axes, axesResidual)):
            # Main analysis object
            _, jetH = next(generic_config.unrollNestedDict(epFitObj.analyses[epAngle]))
            assert jetH.reaction_plane_orientation == epAngle

            # Set labels in individual panels
            # NOTE: If text is attached to the figure (fig.text()), we can just plot it whenever
            # Set title
            ax.set_title(epAngle.displayStr(), fontsize = 17)
            # Axis labels
            ax.set_xlabel(r"$\Delta\varphi$", fontsize = 17)
            # Set the number of ticks in the axis to every integer value.
            # See: https://stackoverflow.com/a/19972993
            majorTicker = matplotlib.ticker.MultipleLocator(base=1.0)
            ax.xaxis.set_major_locator(majorTicker)
            # Enable the axis minor tickers
            ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
            ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
            ax.tick_params(axis = "both",
                           which = "major",
                           labelsize = 15,
                           direction = "in",
                           length = 8,
                           bottom = True,
                           left = True)
            ax.tick_params(axis = "both",
                           which = "minor",
                           direction = "in",
                           length = 4,
                           bottom = True,
                           left = True)

            # Set y label
            (jetFinding, constituentCuts, leadingHadron, jetPt) = params.jetPropertiesLabel(jetPtBin)
            if i == 0:
                ...
            if i == 1:
                ...
            if i == 2:
                ...

            for correlationType, correlationDict in [(analysis_objects.CorrelationType.signal_dominated, jetH.dPhiArray),
                                                     (analysis_objects.CorrelationType.background_dominated, jetH.dPhiSideBandArray)]:
                # Observable name
                observableName = jetH.histNameFormatDPhiArray.format(jetPtBin = jetPtBin, trackPtBin = trackPtBin, tag = correlationType)
                observable = correlationDict[observableName]

                # Plot data
                # Plot S+B, B for all angles, but only B for EP angles
                if (correlationType == analysis_objects.CorrelationType.background_dominated and epAngle != params.ReactionPlaneOrientation.all) or (correlationType == analysis_objects.CorrelationType.signal_dominated and epAngle == params.ReactionPlaneOrientation.all):
                    x = observable.hist.x
                    y = observable.hist.array
                    errors = observable.hist.errors
                    # TODO: This should move to the enum
                    label = correlationType.displayStr()
                    if correlationType == analysis_objects.CorrelationType.background_dominated:
                        label = correlationType.displayStr() + ":\n" + r"$0.8<|\Delta\eta|<1.2$"
                    else:
                        label = correlationType.displayStr() + ":\n" + r"$|\Delta\eta|<0.6$"
                    ax.errorbar(x, y, yerr = errors, marker = "o", zorder = zOrder[(correlationType, "Data")], color = colorsMap[correlationType, "Data"], label = label)

                # Check if the fit was perofmred and therefore should be plotted
                retVal = epFitObj.CheckIfFitIsEnabled(epAngle, correlationType)
                if retVal is False:
                    # Also plot the fit in the case of background dominated in all angles
                    # Although need to clarify that we didn't actually fit - this is just showing that component
                    if not (correlationType == analysis_objects.CorrelationType.background_dominated and epAngle == params.ReactionPlaneOrientation.all):
                        continue
                    else:
                        plotLabel = "Background (Simultaneous Fit)"
                else:
                    plotLabel = correlationType.displayStr() + " Fit"

                logger.info("Plotting {}, {}".format(epAngle.str(), correlationType.str()))

                # x values for the fit to be evaluated at
                # Plot fit at same points as data
                # Defined seperately because we may want to change this in the future
                #xForFitFunc = np.linspace(-0.5*np.pi, 1.5*np.pi, 36)
                xForFitFunc = observable.hist.x

                # Evaluate fit
                fit = epFitObj.EvaluateFit(epAngle = epAngle, fitType = correlationType, xValue = xForFitFunc, fitContainer = fitCont)
                #logger.debug("fit: {}".format(fit))

                # Retrieve errors and plot
                errors = fitCont.errors[(epAngle.str(), correlationType.str())]
                #plot = ax.errorbar(xForFitFunc, fit, yerr = errors, zorder = 10, label = correlationType.displayStr() + " Fit")
                logger.debug("Label: {}".format(correlationType.displayStr() + " Fit"))
                plot = ax.plot(xForFitFunc, fit, zorder = zOrder[(correlationType, "Fit")], color = colorsMap[(correlationType, "Fit")], label = plotLabel)
                # Fill in the error band
                # See: https://stackoverflow.com/a/12958534
                ax.fill_between(xForFitFunc, fit - errors, fit + errors, facecolor = plot[0].get_color(), zorder = zOrder[(correlationType, "FitErrorBars")], alpha = 0.8)

                # Plot residual on separate axes
                residual = observable.hist.array - fit
                residualPlot = axResidual.plot(xForFitFunc, residual)
                axResidual.fill_between(xForFitFunc, residual - errors, residual + errors, facecolor = residualPlot[0].get_color(), label = correlationType.displayStr() + " fit residual")

                h, label = axResidual.get_legend_handles_labels()
                logger.debug("handlesResidual: {}, labelsResidual: {}".format(handlesResidual, labelsResidual))
                handlesResidual += h
                labelsResidual += label

                # Build up event plane fit to get all angles as a cross check
                # TODO: This should probably be refactored back to JetHFitting
                if epAngle != params.ReactionPlaneOrientation.all:
                    if allAnglesSummedFromFit[correlationType] is None:
                        allAnglesSummedFromFit[correlationType] = np.zeros(len(fit))
                    #loger.debug("fit: {}, len(fit): {}, allAnglesSummedFromFit[correlationType]: {}, len(allAnglesSummedFromFit[correlationType]): {}".format(fit, len(fit), allAnglesSummedFromFit[correlationType], len(allAnglesSummedFromFit[correlationType])))
                    # Store fit for all angles
                    allAnglesSummedFromFit[correlationType] = np.add(fit, allAnglesSummedFromFit[correlationType])

                # Store legend label
                h, label = ax.get_legend_handles_labels()
                logger.debug("handles: {}, labels: {}".format(handles, labels))
                handles += h
                labels += label

            # Need to perform after plotting all angles to ensure that we get a good
            # estimate for y max
            if i == 3:
                # Make room for the legend
                # Only needed for higher track pt bins
                if trackPtBin > 4:
                    yMin, yMax = ax.get_ylim()
                    # Scale in a data dependent manner
                    yMax = yMax + 0.3 * (yMax - yMin)
                    ax.set_ylim(yMin, yMax)

        # Plot a possible cross check
        if epFitObj.plotSummedFitCrosscheck:
            for correlationType, fit in allAnglesSummedFromFit.items():
                logger.debug("Fit: {}".format(fit))
                if fit is not None:
                    # Fit can be None if, for example, we fit the all angles signal, such that the EP signal is not fit
                    # TODO: Is the trivial factor of 3 here correct?
                    logger.info("Plotting summed all angles for correlation type {}".format(correlationType.str()))
                    ax.plot(xForFitFunc, fit / 3., zorder = 10, label = correlationType.displayStr() + " fit cross-check")
                    #ax.errorbar(xForFitFunc, fit/3., yerr = fitCont.errors["{}_{}".format(epAngle.str(), correlationType.str())], zorder = 10, label = "Signal fit")
                else:
                    logger.debug("Skipping plot of all angles summed up from each EP angle since it was empty")

        # Tight the plotting up
        fig.tight_layout()
        # Then adjust spacing between subplots
        # Must go second so it isn't reset by tight_layout()
        fig.subplots_adjust(hspace = 0, wspace = 0.05, bottom = 0.12, left = 0.07)

        # Show legend
        logger.debug("handles: {}, labels: {}".format(handles, labels))
        # Remove duplicates
        noDuplicates = {zip(labels, handles)}
        axes[3].legend(handles = noDuplicates.values(), labels = noDuplicates.keys(), loc="best", fontsize = plottingSettings["legend.fontsize"])

        # Save plot
        plot_base.save_plot(epFitObj, fig, epFitObj.fitNameFormat.format(jetPtBin = jetPtBin, trackPtBin = trackPtBin, tag = epFitObj.overallFitLabel.str()))

        # Cleanup
        plt.close(fig)

        # Tight the plotting up
        figResidual.tight_layout()
        # Then adjust spacing between subplots
        # Must go second so it isn't reset by tight_layout()
        figResidual.subplots_adjust(hspace = 0, wspace = 0.05, bottom = 0.12, left = 0.1)

        # Show legend
        logger.debug("handles: {}, labels: {}".format(handlesResidual, labelsResidual))
        # Remove duplicates
        noDuplicates = {zip(labelsResidual, handlesResidual)}
        axesResidual[3].legend(handles = noDuplicates.values(), labels = noDuplicates.keys(), loc="best", fontsize = plottingSettings["legend.fontsize"])

        # Save plot
        plot_base.save_plot(epFitObj, figResidual, epFitObj.fitNameFormat.format(jetPtBin = jetPtBin, trackPtBin = trackPtBin, tag = epFitObj.overallFitLabel.str() + "Residual"))

        # Cleanup
        plt.close(figResidual)

def PlotSubtractedEPHists(epFitObj):
    # Get current plotting settings to access values consistently
    plottingSettings = sns.plotting_context()
    colors = sns.color_palette()

    # TODO: Use the color map defined in PlotRPF (it's the same, just copied here)
    colorIter = iter(colors)
    colorsMap = {(analysis_objects.CorrelationType.signal_dominated, "Fit"): next(colorIter),
                 (analysis_objects.CorrelationType.signal_dominated, "Data"): next(colorIter),
                 (analysis_objects.CorrelationType.background_dominated, "Fit"): next(colorIter),
                 (analysis_objects.CorrelationType.background_dominated, "Data"): next(colorIter)}

    # Iterate over the data and subtract the hists
    for (jetPtBin, trackPtBin), fitCont in epFitObj.fitContainers.items():

        # Define axes for plot
        fig, axes = plt.subplots(1, 4, sharey = True, sharex = True)

        # Just for the all angles subtracted
        figAll, axisAll = plt.subplots(figsize=(5, 7.5))

        # Store legend information
        handles = []
        labels = []

        # Put the all angles at the end for consistnecy
        epAngles = [angle for angle in params.ReactionPlaneOrientation]
        epAngles.append(epAngles.pop(epAngles.index(params.ReactionPlaneOrientation.all)))

        for i, (epAngle, ax) in enumerate(zip(epAngles, axes)):
            # Set labels in individual panels
            # Set title
            ax.set_title(epAngle.displayStr(), fontsize = 17)
            # Axis labels
            ax.set_xlabel(r"$\Delta\varphi$", fontsize = 17)
            # Set y label
            if i == 0:
                ax.set_ylabel(r"1/$\mathrm{N}_{\mathrm{trig}}$d(N-B)/d$\Delta\varphi$", fontsize = 17)

            # Main analysis object
            _, jetH = next(generic_config.unrollNestedDict(epFitObj.analyses[epAngle]))

            observableName = jetH.histNameFormatDPhiSubtractedArray.format(jetPtBin = jetPtBin, trackPtBin = trackPtBin, tag = analysis_objects.CorrelationType.signal_dominated)
            logger.debug("Processing observable {}".format(observableName))
            logger.debug("Subtracted hist arrays: {}".format(jetH.dPhiSubtractedArray))
            observable = jetH.dPhiSubtractedArray[observableName]

            # x values for the fit to be evaluated at
            # Plot fit at same points as data
            # Defined seperately because we may want to change this in the future
            xForFitFunc = observable.hist.binCenters

            # Retrieve fit data
            #fit = epFitObj.EvaluateFit(epAngle = epAngle, fitType = analysis_objects.CorrelationType.background_dominated, xValue = xForFitFunc, fitContainer = fitCont)
            # We want to subtract the background function for each EP angle, so we need the errors from the background dominated fit params
            #logger.debug("fitCont.errors: {}".format(fitCont.errors))
            # TODO: Is it right to be retrieving the background errors here?? I'm not so certain for the all angles case that this is right...
            fitErrors = fitCont.errors[(epAngle.str(), analysis_objects.CorrelationType.background_dominated.str())]

            ax.errorbar(xForFitFunc, observable.hist.array, yerr = observable.hist.errors, zorder = 5, color = colorsMap[(observable.correlationType, "Data")], label = observable.correlationType.displayStr() + " Subtracted")
            # Following Joel's example, plot the fit error on the same points as the correlation error
            # Fill in the error band
            # See: https://stackoverflow.com/a/12958534
            ax.fill_between(xForFitFunc, observable.hist.array - fitErrors, observable.hist.array + fitErrors, label = "Fit error", facecolor = colorsMap[(observable.correlationType, "Fit")], zorder = 10, alpha = 0.8)

            # Store legend label
            h, label = ax.get_legend_handles_labels()
            logger.debug("handles: {}, labels: {}".format(handles, labels))
            handles += h
            labels += label

            if epAngle == params.ReactionPlaneOrientation.all:
                # TODO: Include all angles in label
                axisAll.set_title(epAngle.displayStr(), fontsize = 17)
                # Axis labels
                axisAll.set_xlabel(r"$\Delta\varphi$", fontsize = 17)
                # Set y label
                axisAll.set_ylabel(r"1/$\mathrm{N}_{\mathrm{trig}}$d(N-B)/d$\Delta\varphi$", fontsize = 17)
                # TODO: Consoldiate into a function
                # Set the number of ticks in the axis to every integer value.
                # See: https://stackoverflow.com/a/19972993
                majorTicker = matplotlib.ticker.MultipleLocator(base=1.0)
                axisAll.xaxis.set_major_locator(majorTicker)
                # Enable the axis minor tickers
                axisAll.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
                axisAll.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
                axisAll.tick_params(axis = "both",
                                    which = "major",
                                    labelsize = 15,
                                    direction = "in",
                                    length = 8,
                                    bottom = True,
                                    left = True)
                axisAll.tick_params(axis = "both",
                                    which = "minor",
                                    direction = "in",
                                    length = 4,
                                    bottom = True,
                                    left = True)

                # Add labels
                # NOTE: Cannot end in "\n". It will cause an crash.
                # TODO: Add to other subtracted plots
                (jetFinding, constituentCuts, leadingHadron, jetPt) = params.jetPropertiesLabel(jetPtBin)
                text = ""
                if not epFitObj.minimalLabelsForAllAnglesSubtracted:
                    text += jetH.aliceLabel.str()
                    text += "\n" + params.systemLabel(energy = jetH.collisionEnergy, system = jetH.collisionSystem, activity = jetH.eventActivity)
                    text += "\n" + jetFinding + constituentCuts
                    text += "\n" + jetPt + ", " + leadingHadron
                    # Extra "\n" at the end because we can't lead with a bare "\n".
                    text += "\n" + params.generateTrackPtRangeString(trackPtBin) + "\n"
                    textArgs = {"x": 0.5, "y": 0.82,
                                "horizontalalignment": "center",
                                "verticalalignment": "center",
                                "multialignment": "center"}
                else:
                    textArgs = {"x": 0.92, "y": 0.84,
                                "horizontalalignment": "right",
                                "verticalalignment": "top"}
                text += "Scale uncertainty: 6%"
                #logger.debug("text: {}".format(text))
                textArgs["fontsize"] = 16
                textArgs["transform"] = axisAll.transAxes
                textArgs["s"] = text
                axisAll.text(**textArgs)

                axisAll.errorbar(xForFitFunc, observable.hist.array, yerr = observable.hist.errors, zorder = 5, color = colorsMap[(observable.correlationType, "Data")], label = observable.correlationType.displayStr() + "\nSubtracted: " + r"$|\Delta\eta|<0.6$")
                axisAll.fill_between(xForFitFunc, observable.hist.array - fitErrors, observable.hist.array + fitErrors, label = "Fit error", facecolor = colorsMap[(observable.correlationType, "Fit")], zorder = 10, alpha = 0.8)

                # Adjust after we know the range of the data
                yMin, yMax = axisAll.get_ylim()
                # Scale in a data dependent manner
                yMax = yMax + 0.35 * (yMax - yMin)
                axisAll.set_ylim(yMin, yMax)

        # Tight the plotting up
        fig.tight_layout()
        # Then adjust spacing between subplots
        # Must go second so it isn't reset by tight_layout()
        fig.subplots_adjust(hspace = 0, wspace = 0.05, bottom = 0.12, left = 0.1)

        # Show legend
        logger.debug("handles: {}, labels: {}".format(handles, labels))
        # Remove duplicates
        noDuplicates = {zip(labels, handles)}
        axes[3].legend(handles = noDuplicates.values(), labels = noDuplicates.keys(), loc="best", fontsize = plottingSettings["legend.fontsize"])

        # Save plot
        plot_base.save_plot(epFitObj, fig, epFitObj.fitNameFormat.format(jetPtBin = jetPtBin, trackPtBin = trackPtBin, tag = epFitObj.overallFitLabel.str() + "_subtracted"))

        # Cleanup
        plt.close(fig)

        # Tight the plotting up
        figAll.tight_layout()
        # Then adjust spacing between subplots
        # Must go second so it isn't reset by tight_layout()
        #figAll.subplots_adjust(hspace = 0, wspace = 0.05, bottom = 0.12, left = 0.1)

        #axisAll.legend(loc="best", fontsize = plottingSettings["legend.fontsize"])
        axisAll.legend(loc="best", fontsize = 16)

        # Save plot
        plot_base.save_plot(epFitObj, figAll, epFitObj.fitNameFormat.format(jetPtBin = jetPtBin, trackPtBin = trackPtBin, tag = epFitObj.overallFitLabel.str() + "AllAngles_subtracted"))

        # Cleanup
        plt.close(figAll)

def CompareToJoel(epFitObj):
    # Open ROOT file
    filename = "RPF_sysScaleCorrelations{trackPtBin}rebinX2bg.root"

    joelAllAnglesName = "allReconstructedSignalwithErrorsNOM"
    joelAllAnglesErrorMinName = "allReconstructedSignalwithErrorsMIN"
    joelAllAnglesErrorMaxName = "allReconstructedSignalwithErrorsMAX"

    # Iterate over the data and subtract the hists
    for (jetPtBin, trackPtBin), fitCont in epFitObj.fitContainers.items():
        import ROOT
        logger.info("Comparing with Joel's code for trackPtBin {}".format(trackPtBin))

        # TODO: Remove hard code
        # TODO: Move this out of here if possible (but perhaps it's fine)
        fIn = ROOT.TFile.Open(os.path.join("output", "plotting", "PbPb", "joelCentral", filename.format(trackPtBin = trackPtBin)), "READ")
        joelAllAngles = fIn.Get(joelAllAnglesName)
        joelAllAnglesErrorMin = fIn.Get(joelAllAnglesErrorMinName)
        joelAllAnglesErrorMax = fIn.Get(joelAllAnglesErrorMaxName)

        # Define axes for plot
        fig, ax = plt.subplots()

        jetPtString = params.generateJetPtRangeString(jetPtBin)
        trackPtString = params.generateTrackPtRangeString(trackPtBin)
        formatStr = """{jetPtString}\n{trackPtString}""".format(jetPtString = jetPtString, trackPtString = trackPtString)
        ax.set_title(r"$\Delta\varphi$ subtracted correlations " + " for {}".format(formatStr))
        # Axis labels
        ax.set_xlabel(r"$\Delta\varphi$")
        ax.set_ylabel(r"$\mathrm{dN}/\mathrm{d}\Delta\varphi$")

        epAngle = params.ReactionPlaneOrientation.all
        #data = epFitObj.subtractedHistData[(jetPtBin, trackPtBin)][epAngle][analysis_objects.CorrelationType.signal_dominated]
        _, jetH = next(generic_config.unrollNestedDict(epFitObj.analyses[epAngle]))
        observableName = jetH.histNameFormatDPhiSubtractedArray.format(jetPtBin = jetPtBin, trackPtBin = trackPtBin, tag = analysis_objects.CorrelationType.signal_dominated)
        observable = jetH.dPhiSubtractedArray[observableName]

        # Plot my dada
        myDataPlot = ax.errorbar(observable.hist.binCenters, observable.hist.array, yerr = observable.hist.errors, label = "This analysis")
        myDataPlotColor = myDataPlot[0].get_color()
        fitErrors = fitCont.errors[(epAngle.str(), analysis_objects.CorrelationType.signal_dominated.str())]
        ax.fill_between(observable.hist.binCenters, observable.hist.array - fitErrors, observable.hist.array + fitErrors, facecolor = myDataPlotColor, zorder = 10, alpha = 0.8)

        # Plot joel data
        joelData = histogram.getArrayFromHist(joelAllAngles)
        joelDataPlot = ax.errorbar(joelData["binCenters"], joelData["y"], yerr = joelData["errors"], label = "Joel")
        joelDataPlotColor = joelDataPlot[0].get_color()
        joelErrorMin = histogram.getArrayFromHist(joelAllAnglesErrorMin)
        ax.fill_between(joelData["binCenters"], joelData["y"], joelErrorMin["y"], facecolor = joelDataPlotColor)
        joelErrorMax = histogram.getArrayFromHist(joelAllAnglesErrorMax)
        ax.fill_between(joelData["binCenters"], joelData["y"], joelErrorMax["y"], facecolor = joelDataPlotColor)

        # Tight the plotting up
        # TODO: Shorten up title (convert the information to a text box), and re-enable this option
        #       For explanation of the error, see: https://github.com/mwaskom/seaborn/issues/954
        #fig.tight_layout()
        # Then adjust spacing between subplots
        # Must go second so it isn't reset by tight_layout()
        # Tuned for "paper" context
        fig.subplots_adjust(hspace = 0, wspace = 0.05, bottom = 0.12, left = 0.1)

        # Show legend
        plt.legend(loc="best")

        # Save plot
        # TODO: Define this name in the class!
        plot_base.save_plot(epFitObj, fig, "joelComparison_jetPt{jetPtBin}_trackPt{trackPtBin}_{tag}_subtracted".format(jetPtBin = jetPtBin, trackPtBin = trackPtBin, tag = epFitObj.overallFitLabel.str()))

        # Cleanup
        plt.close(fig)
