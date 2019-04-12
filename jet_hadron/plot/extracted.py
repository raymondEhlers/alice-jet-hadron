#!/usr/bin/env python

""" Extracted values plotting module.

Includes quantities such as widths and yields.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

from cycler import cycler
from fractions import Fraction
import logging
from pachyderm import utils
import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, TYPE_CHECKING, Union

from jet_hadron.base import analysis_config
from jet_hadron.base import analysis_objects
from jet_hadron.base import labels
from jet_hadron.base import params
from jet_hadron.plot import base as plot_base
# Careful to watch out for import loops...
from jet_hadron.analysis import fit as fitting

if TYPE_CHECKING:
    from jet_hadron.analysis import correlations

logger = logging.getLogger(__name__)

# Typing helpers
Correlations = Union["correlations.DeltaPhiSignalDominated",
                     "correlations.DeltaEtaNearSide", "correlations.DeltaEtaAwaySide"]

def delta_eta_with_gaussian(analysis: "correlations.Correlations") -> None:
    """ Plot the subtracted delta eta near-side. """
    # Setup
    fig, ax = plt.subplots(figsize = (8, 6))

    # Plot only the near side for now because the away-side doesn't have a gaussian shape
    # Of the form (attribute_name, mean)
    attribute_names = [
        ("near_side", 0.0),
    ]
    for attribute_name, mean in attribute_names:
        # Setup
        # Correlation
        correlation: Union["correlations.DeltaEtaNearSide"] = \
            getattr(analysis.correlation_hists_delta_eta_subtracted, attribute_name)
        # Extracted width
        extracted_width: analysis_objects.ExtractedObservable = getattr(analysis.widths_delta_eta, attribute_name)

        # Plot the data.
        h = correlation.hist
        ax.errorbar(
            h.x, h.y, yerr = h.errors,
            marker = "o", linestyle = "",
            label = f"{correlation.type.display_str()}",
        )

        # Plot the fit
        logger.debug(f"mean: {type(mean)}, width: {type(extracted_width.value)}")
        gauss = fitting.gaussian(h.x, mu = mean, sigma = extracted_width.value)
        fit_plot = ax.plot(
            h.x, gauss,
            label = fr"Gaussian fit: $\mu = $ {mean:.2f}, $\sigma = $ {extracted_width.value:.2f}",
        )
        # Fill in the error band.
        error = extracted_width.error * np.ones(len(h.x))
        ax.fill_between(
            h.x, gauss - error, gauss + error,
            facecolor = fit_plot[0].get_color(), alpha = 0.5,
        )

        # Labels.
        ax.set_xlabel(labels.make_valid_latex_string(correlation.axis.display_str()))
        ax.set_ylabel(labels.make_valid_latex_string(labels.delta_eta_axis_label()))
        jet_pt_label = labels.jet_pt_range_string(analysis.jet_pt)
        track_pt_label = labels.track_pt_range_string(analysis.track_pt)
        ax.set_title(fr"Subtracted 1D ${correlation.axis.display_str()}$,"
                     f" {analysis.reaction_plane_orientation.display_str()} event plane orient.,"
                     f" {jet_pt_label}, {track_pt_label}")
        ax.legend(loc = "upper right")

        # Final adjustments
        fig.tight_layout()
        # Save plot and cleanup
        plot_base.save_plot(analysis.output_info, fig,
                            f"jetH_delta_eta_{analysis.identifier}_width_{attribute_name}_fit")
        # Reset for the next iteration of the loop
        ax.clear()

    # Final cleanup.
    plt.close(fig)

def delta_phi_with_gaussians(analysis: "correlations.Correlations") -> None:
    """ Plot the subtracted delta phi correlation with gaussian fits to the near and away side. """
    # Setup
    fig, ax = plt.subplots(figsize = (8, 6))
    correlation = analysis.correlation_hists_delta_phi_subtracted.signal_dominated
    h = correlation.hist

    # First we plot the data
    ax.errorbar(
        h.x, h.y, yerr = h.errors,
        marker = "o", linestyle = "",
        label = f"{correlation.type.display_str()}",
    )

    # Of the form (attribute_name, mean)
    delta_phi_regions = [
        ("near_side", 0.0),
        ("away_side", np.pi),
    ]
    for attribute_name, mean in delta_phi_regions:
        # Setup
        # Extracted width
        extracted_width: analysis_objects.ExtractedObservable = getattr(analysis.widths_delta_phi, attribute_name)
        # Convert the attribute name to display better. Ex: "near_side" -> "Near side"
        attribute_display_name = attribute_name.replace("_", " ").capitalize()

        # Plot the fit
        gauss = fitting.gaussian(h.x, mu = mean, sigma = extracted_width.value)
        fit_plot = ax.plot(
            h.x, gauss,
            label = fr"{attribute_display_name} gaussian fit: $\mu = $ {mean:.2f}"
                    fr", $\sigma = $ {extracted_width.value:.2f}",
        )
        # Fill in the error band.
        error = extracted_width.error * np.ones(len(h.x))
        ax.fill_between(
            h.x, gauss - error, gauss + error,
            facecolor = fit_plot[0].get_color(), alpha = 0.5,
        )

    # Labels.
    ax.set_xlabel(labels.make_valid_latex_string(correlation.axis.display_str()))
    ax.set_ylabel(labels.make_valid_latex_string(labels.delta_phi_axis_label()))
    jet_pt_label = labels.jet_pt_range_string(analysis.jet_pt)
    track_pt_label = labels.track_pt_range_string(analysis.track_pt)
    ax.set_title(fr"Subtracted 1D ${correlation.axis.display_str()}$,"
                 f" {analysis.reaction_plane_orientation.display_str()} event plane orient.,"
                 f" {jet_pt_label}, {track_pt_label}")
    ax.legend(loc = "upper right")

    # Final adjustments
    fig.tight_layout()
    # Save plot and cleanup
    plot_base.save_plot(analysis.output_info, fig,
                        f"jetH_delta_phi_{analysis.identifier}_width_signal_dominated_fit")
    plt.close(fig)

def _extracted_values(analyses: Mapping[Any, "correlations.Correlations"],
                      selected_iterables: Dict[str, Sequence[Any]],
                      attribute_name: str,
                      plot_labels: plot_base.PlotLabels,
                      output_info: analysis_objects.PlottingOutputWrapper,
                      projection_range_func: Optional[Callable[["correlations.Correlations"], str]] = None,
                      extraction_range_func: Optional[Callable[["correlations.Correlations"], str]] = None) -> None:
    """ Plot extracted values.

    Args:

    """
    # Setup
    fig, ax = plt.subplots(figsize = (8, 6))
    # Specify plotting properties
    # color, marker, fill marker or not
    # NOTE: Fill marker is specified when plotting becuase of a matplotlib bug
    # NOTE: This depends on iterating over the EP orientation in the exact manner specified below.
    ep_plot_properties = {
        # black, diamond, no fill
        params.ReactionPlaneOrientation.inclusive: ("black", "D", "none"),
        # blue = "C0", square, fill
        params.ReactionPlaneOrientation.in_plane: ("tab:blue", "s", "full"),
        # green = "C2", triangle up, fill
        params.ReactionPlaneOrientation.mid_plane: ("tab:green", "^", "full"),
        # red = "C3", circle, fill
        params.ReactionPlaneOrientation.out_of_plane: ("tab:red", "o", "full"),
    }
    cyclers = []
    plot_property_values = list(ep_plot_properties.values())
    for i, prop in enumerate(["color", "marker", "fillstyle"]):
        cyclers.append(cycler(prop, [p[i] for p in plot_property_values]))
    # We skip the fillstyle because apprently it doesn't work with the cycler at the moment due to a bug...
    combined_cyclers = sum(cyclers[1:-1], cyclers[0])
    ax.set_prop_cycle(combined_cyclers)

    # Used for labeling purposes. The values that are used are identical for all analyses.
    inclusive_analysis: Optional["correlations.Correlations"] = None
    for displace_index, ep_orientation in enumerate(selected_iterables["reaction_plane_orientation"]):
        # Store the values to be plotted
        values: Dict[analysis_objects.PtBin, analysis_objects.ExtractedObservable] = {}
        for key_index, analysis in \
                analysis_config.iterate_with_selected_objects(
                    analyses, reaction_plane_orientation = ep_orientation
                ):
            # Store each extracted value.
            values[analysis.track_pt] = utils.recursive_getattr(analysis, attribute_name)
            # These are both used for labeling purposes and are identical for all analyses that are iterated over.
            if ep_orientation == params.ReactionPlaneOrientation.inclusive and inclusive_analysis is None:
                inclusive_analysis = analysis

        # Plot the values
        bin_centers = np.array([k.bin_center for k in values])
        bin_centers = bin_centers + displace_index * 0.05
        ax.errorbar(
            bin_centers, [v.value for v in values.values()], yerr = [v.error for v in values.values()],
            label = ep_orientation.display_str(), linestyle = "",
            fillstyle = "none" if ep_orientation == params.ReactionPlaneOrientation.inclusive else "full",
        )

    # Help out mypy...
    assert inclusive_analysis is not None

    # Labels.
    # TODO: Extraction limits
    # General
    text = labels.make_valid_latex_string(inclusive_analysis.alice_label.display_str())
    text += "\n" + labels.system_label(
        energy = inclusive_analysis.collision_energy,
        system = inclusive_analysis.collision_system,
        activity = inclusive_analysis.event_activity
    )
    text += "\n" + labels.jet_pt_range_string(inclusive_analysis.jet_pt)
    text += "\n" + labels.jet_finding()
    text += "\n" + labels.constituent_cuts()
    text += "\n" + labels.make_valid_latex_string(inclusive_analysis.leading_hadron_bias.display_str())
    # Deal with projection range, extraction range string.
    # Attempt to put them on the same line if they are both defined.
    # Otherwise, it will just be one of them.
    projection_range_string = ""
    if projection_range_func:
        projection_range_string = projection_range_func(inclusive_analysis)
    extraction_range_string = ""
    if extraction_range_func:
        extraction_range_string = extraction_range_func(inclusive_analysis)
    if projection_range_string or extraction_range_string:
        additional_text = ""
        if projection_range_string:
            if extraction_range_string:
                additional_text += f"{projection_range_string}, {extraction_range_string}"
            else:
                additional_text += f"{projection_range_string}"
        else:
            additional_text += f"{extraction_range_string}"
        text += "\n" + additional_text
    # Finally, add the text to the axis.
    ax.text(
        0.97, 0.97, text, horizontalalignment = "right",
        verticalalignment = "top", multialignment = "right",
        transform = ax.transAxes
    )
    # Axes and titles
    ax.set_xlabel(labels.make_valid_latex_string(labels.track_pt_display_label()))
    # Apply any specified labels
    if plot_labels.title is not None:
        plot_labels.title = plot_labels.title + f" for {labels.jet_pt_range_string(inclusive_analysis.jet_pt)}"
    plot_labels.apply_labels(ax)
    ax.legend(loc = "center right", frameon = False)

    # Final adjustments
    fig.tight_layout()
    # Save plot and cleanup
    plot_base.save_plot(output_info, fig,
                        f"jetH_delta_phi_{inclusive_analysis.jet_pt_identifier}_{attribute_name.replace('.', '_')}")
    plt.close(fig)

def delta_phi_plot_projection_range_string(inclusive_analysis: "correlations.Correlations") -> str:
    """ Provides a string that describes the delta eta projection range for delta phi plots. """
    return labels.make_valid_latex_string(
        fr"$|\Delta\eta|<{inclusive_analysis.signal_dominated_eta_region.max}$"
    )

def delta_phi_near_side_widths(analyses: Mapping[Any, "correlations.Correlations"],
                               selected_iterables: Dict[str, Sequence[Any]],
                               output_info: analysis_objects.PlottingOutputWrapper) -> None:
    """ Plot the delta phi near-side widths. """
    _extracted_values(
        analyses = analyses, selected_iterables = selected_iterables,
        attribute_name = "widths_delta_phi.near_side",
        plot_labels = plot_base.PlotLabels(
            y_label = "Near-side width",
            title = "Near-side width",
        ),
        output_info = output_info,
        projection_range_func = delta_phi_plot_projection_range_string,
    )

def delta_phi_away_side_widths(analyses: Mapping[Any, "correlations.Correlations"],
                               selected_iterables: Dict[str, Sequence[Any]],
                               output_info: analysis_objects.PlottingOutputWrapper) -> None:
    """ Plot the delta phi away-side widths. """
    _extracted_values(
        analyses = analyses, selected_iterables = selected_iterables,
        attribute_name = "widths_delta_phi.away_side",
        plot_labels = plot_base.PlotLabels(
            y_label = "Away-side width",
            title = "Away-side width",
        ),
        output_info = output_info,
        projection_range_func = delta_phi_plot_projection_range_string,
    )

def delta_phi_near_side_yields(analyses: Mapping[Any, "correlations.Correlations"],
                               selected_iterables: Dict[str, Sequence[Any]],
                               output_info: analysis_objects.PlottingOutputWrapper) -> None:
    """ Plot the delta phi near-side yields. """
    _extracted_values(
        analyses = analyses, selected_iterables = selected_iterables,
        attribute_name = "yields_delta_phi.near_side",
        plot_labels = plot_base.PlotLabels(
            y_label = labels.make_valid_latex_string(
                fr"\mathrm{{d}}N/\mathrm{{d}}{labels.pt_display_label()} ({labels.momentum_units_label_gev()})^{{-1}}",
            ),
            title = "Near-side yield",
        ),
        output_info = output_info,
        projection_range_func = delta_phi_plot_projection_range_string,
    )

def delta_phi_away_side_yields(analyses: Mapping[Any, "correlations.Correlations"],
                               selected_iterables: Dict[str, Sequence[Any]],
                               output_info: analysis_objects.PlottingOutputWrapper) -> None:
    """ Plot the delta phi away-side yields. """
    _extracted_values(
        analyses = analyses, selected_iterables = selected_iterables,
        attribute_name = "yields_delta_phi.away_side",
        plot_labels = plot_base.PlotLabels(
            y_label = labels.make_valid_latex_string(
                fr"\mathrm{{d}}N/\mathrm{{d}}{labels.pt_display_label()} ({labels.momentum_units_label_gev()})^{{-1}}",
            ),
            title = "Away-side yield",
        ),
        output_info = output_info,
        projection_range_func = delta_phi_plot_projection_range_string,
    )

def delta_eta_plot_projection_range_string(inclusive_analysis: "correlations.Correlations") -> str:
    """ Provides a string that describes the delta phi projection range for delta eta plots. """
    # The limit is almost certainly a multiple of pi, so we try to express it more naturally
    # as a value like pi/2 or 3*pi/2
    # This relies on this dividing cleanly. It usually seems  to work.
    coefficient = Fraction(inclusive_analysis.near_side_phi_region.max / np.pi)
    leading_coefficient = ""
    if coefficient.numerator != 1:
        leading_coefficient = f"{coefficient.numerator}"
    value = fr"{leading_coefficient}\pi/{coefficient.denominator}"
    return labels.make_valid_latex_string(
        fr"$|\Delta\varphi|<{value}$"
    )

def delta_eta_near_side_widths(analyses: Mapping[Any, "correlations.Correlations"],
                               selected_iterables: Dict[str, Sequence[Any]],
                               output_info: analysis_objects.PlottingOutputWrapper) -> None:
    """ Plot the delta eta near-side widths. """
    _extracted_values(
        analyses = analyses, selected_iterables = selected_iterables,
        attribute_name = "widths_delta_eta.near_side",
        plot_labels = plot_base.PlotLabels(
            y_label = "Near-side width",
            title = "Near-side width",
        ),
        output_info = output_info,
        projection_range_func = delta_eta_plot_projection_range_string,
    )

def delta_eta_near_side_yields(analyses: Mapping[Any, "correlations.Correlations"],
                               selected_iterables: Dict[str, Sequence[Any]],
                               output_info: analysis_objects.PlottingOutputWrapper) -> None:
    """ Plot the delta eta near-side yields. """
    _extracted_values(
        analyses = analyses, selected_iterables = selected_iterables,
        attribute_name = "yields_delta_eta.near_side",
        plot_labels = plot_base.PlotLabels(
            y_label = labels.make_valid_latex_string(
                fr"\mathrm{{d}}N/\mathrm{{d}}{labels.pt_display_label()} ({labels.momentum_units_label_gev()})^{{-1}}",
            ),
            title = "Near-side yield",
        ),
        output_info = output_info,
        projection_range_func = delta_eta_plot_projection_range_string,
    )

