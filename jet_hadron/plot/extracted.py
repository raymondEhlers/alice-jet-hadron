#!/usr/bin/env python

""" Extracted values plotting module.

Includes quantities such as widths and yields.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

from cycler import cycler
from fractions import Fraction
import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pachyderm import histogram
from pachyderm.utils import epsilon
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING, Union, cast

from jet_hadron.base import analysis_config
from jet_hadron.base import analysis_objects
from jet_hadron.base import labels
from jet_hadron.base import params
from jet_hadron.plot import base as plot_base

# Careful here! We're taking the unusual step of import a module in the analysis package.
# However, it's extremely convenient, so we break convention here.
from jet_hadron.analysis import extracted

if TYPE_CHECKING:
    from jet_hadron.analysis import correlations

logger = logging.getLogger(__name__)

# Typing helpers
Correlations = Union["correlations.DeltaPhiSignalDominated",
                     "correlations.DeltaEtaNearSide", "correlations.DeltaEtaAwaySide"]

def _find_pi_coefficient(value: float, target_denominator: int = 12) -> str:
    """ Convert a given value to a string with coefficients of pi.

    For example, for a given value of 4.71..., it will return "3 * \\pi / 2."

    Args:
        value: Value for which we want the pi coefficient.
        target_denominator: Maximum denominator. We will try to round to a fraction that is close
            to a fraction where this is the maximum denominator. Default: 12, which will work for
            both fractions that are factors of 3 and 4.
    Returns:
        LaTeX string containing the coefficient with pi.
    """
    # Since any value we pass here is almost certainly a multiple of pi, we try to express it
    # more naturally as a value like pi/2 or 3*pi/2
    # We use a rounding trick to try to account for floating point rounding issues.
    # See: https://stackoverflow.com/a/11269128
    coefficient = Fraction(int(round(value / np.pi * target_denominator)), target_denominator)
    leading_coefficient = ""
    if coefficient.numerator != 1:
        leading_coefficient = f"{coefficient.numerator}"
    s = fr"{leading_coefficient}\pi/{coefficient.denominator}"
    return s

def _find_extraction_range_min_and_max_in_hist(h: histogram.Histogram1D,
                                               extraction_range: params.SelectedRange) -> Tuple[float, float]:
    """ Find the min and max given the extraction range subject to the binning of the hist.

    The idea is that if the hist has bins every 0.1, and our limits are [0.02, 0.58], then the actual limits
    are [0.0, 0.6].

    Args:
        h: Histogram whose binning will be used.
        extraction_range: Extraction range that is used with the histogram.
    Returns:
        The extraction range based on the histogram binning.
    """
    min_value = h.bin_edges[h.find_bin(extraction_range.min + epsilon)]
    # We need a +1 on the upper limit because the integral is inclusive of the upper bin.
    max_value = h.bin_edges[h.find_bin(extraction_range.max - epsilon) + 1]
    logger.debug(f"extraction_range: {extraction_range}, min_value: {min_value}, max_value: {max_value}")
    return min_value, max_value

def delta_eta_with_gaussian(analysis: "correlations.Correlations") -> None:
    """ Plot the subtracted delta eta near-side. """
    # Setup
    fig, ax = plt.subplots(figsize = (8, 6))

    for (attribute_name, width_obj), (correlation_attribute_name, correlation) in \
            zip(analysis.widths_delta_eta, analysis.correlation_hists_delta_eta_subtracted):
        # Setup
        # Sanity check
        if attribute_name != correlation_attribute_name:
            raise ValueError(
                "Issue extracting width and hist together."
                f"Width obj name: {attribute_name}, hist obj name: {correlation_attribute_name}"
            )
        # Plot only the near side for now because the away-side doesn't have a gaussian shape
        if attribute_name == "away_side":
            continue

        # Plot the data.
        h = correlation.hist
        ax.errorbar(
            h.x, h.y, yerr = h.errors,
            marker = "o", linestyle = "",
            label = f"{correlation.type.display_str()}",
        )

        # Plot the fit
        gauss = width_obj.fit_object(h.x, **width_obj.fit_result.values_at_minimum)
        fit_plot = ax.plot(
            h.x, gauss,
            label = fr"Gaussian fit: $\mu = $ {width_obj.mean:.2f}, $\sigma = $ {width_obj.width:.2f}",
        )
        # Fill in the error band.
        error = width_obj.fit_object.calculate_errors(x = h.x)
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

    # Plot the fit.
    for attribute_name, width_obj in analysis.widths_delta_phi:
        # Setup
        # Convert the attribute name to display better. Ex: "near_side" -> "Near side"
        attribute_display_name = attribute_name.replace("_", " ").capitalize()
        # We only want to plot the fit over the range that it was fit.
        restricted_range = (h.x > width_obj.fit_object.fit_options["range"].min) & \
            (h.x < width_obj.fit_object.fit_options["range"].max)
        x = h.x[restricted_range]

        # Plot the fit
        gauss = width_obj.fit_object(x, **width_obj.fit_result.values_at_minimum)
        fit_plot = ax.plot(
            x, gauss,
            label = fr"{attribute_display_name} gaussian fit: $\mu = $ {width_obj.mean:.2f}"
                    fr", $\sigma = $ {width_obj.width:.2f}",
        )
        # Fill in the error band.
        error = width_obj.fit_object.calculate_errors(x = x)
        ax.fill_between(
            x, gauss - error, gauss + error,
            facecolor = fit_plot[0].get_color(), alpha = 0.5,
        )

        # This means that we extracted values from the RP fit. Let's also plot them for comparison
        if width_obj.fit_args != {}:
            args = dict(width_obj.fit_result.values_at_minimum)
            args.update({
                # Help out mypy...
                k: cast(float, v) for k, v in width_obj.fit_args.items() if "error_" not in k
            })
            rpf_gauss = width_obj.fit_object(x, **args)
            rp_fit_plot = ax.plot(
                x, rpf_gauss,
                label = fr"RPF {attribute_display_name} gaussian fit: $\mu = $ {width_obj.mean:.2f}"
                        fr", $\sigma = $ {width_obj.fit_args['width']:.2f}",
            )
            # Fill in the error band.
            # NOTE: Strictly speaking, this error band isn't quite right (since it is dependent on the fit result
            # of the actual width fit), but I think it's fine for these purposes.
            error = width_obj.fit_object.calculate_errors(x = x)
            ax.fill_between(
                x, rpf_gauss - error, rpf_gauss + error,
                facecolor = rp_fit_plot[0].get_color(), alpha = 0.5,
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

def _proj_and_extract_range_label(inclusive_analysis: "correlations.Correlations",
                                  projection_range_func: Optional[Callable[["correlations.Correlations"], str]] = None,
                                  extraction_range_func: Optional[Callable[["correlations.Correlations"], str]] = None,
                                  ) -> str:
    """ Determine the projection range and extraction range label.

    We attempt to put them on the same line if they are both defined. Otherwise, it will just be one of them.

    Args:
        inclusive_analysis: Inclusive correlations analysis object for labeling purposes.
        projection_range_func: Function which will provide the projection range of the extracted value given
            the inclusive object.
        extraction_range_func: Function which will provide the extraction range of the extracted value given
            the inclusive object.
    Returns:
        The properly formatted label potentially containing the projection and extraction range(s).
    """
    labels = []
    for f in [projection_range_func, extraction_range_func]:
        if f:
            res = f(inclusive_analysis)
            if res:
                labels.append(res)

    return ", ".join(labels)

def _extracted_values(analyses: Mapping[Any, "correlations.Correlations"],
                      selected_iterables: Mapping[str, Sequence[Any]],
                      extract_value_func: Callable[["correlations.Correlations"], analysis_objects.ExtractedObservable],
                      plot_labels: plot_base.PlotLabels,
                      logy: bool,
                      output_name: str,
                      fit_type: str,
                      output_info: analysis_objects.PlottingOutputWrapper,
                      projection_range_func: Optional[Callable[["correlations.Correlations"], str]] = None,
                      extraction_range_func: Optional[Callable[["correlations.Correlations"], str]] = None) -> None:
    """ Plot extracted values.

    Note:
        It's best to fully define the ``extract_value_func`` function even though it can often be easily accomplished
        with a lambda because only a full function definition can use explicit type checking. Since this function uses
        a variety of different sources for the data, this type checking is particularly helpful. So writing a full
        function with full typing is strongly preferred to ensure that we get it right.

    Args:
        analyses: Correlation analyses.
        selected_iterables: Iterables that were used in constructing the analysis objects. We use them to iterate
            over some iterators in a particular order (particularly the reaction plane orientation).
        extract_value_func: Function to retrieve the extracted value and error.
        plot_labels: Titles and axis labels for the plot.
        logy: True if we should plot with log y.
        output_name: Base of name under which the plot will be stored.
        fit_type: Name of the RP fit type used to get to this extracted value.
        output_info: Information needed to determine where to store the plot.
        projection_range_func: Function which will provide the projection range of the extracted value given
            the inclusive object. Default: None.
        extraction_range_func: Function which will provide the extraction range of the extracted value given
            the inclusive object. Default: None.
    """
    # Setup
    fig, ax = plt.subplots(figsize = (8, 6))
    # Specify plotting properties
    # color, marker, fill marker or not
    # NOTE: Fill marker is specified when plotting because of a matplotlib bug
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
    # We skip the fillstyle because apparently it doesn't work with the cycler at the moment due to a bug...
    # They didn't implement their add operation to handle 0, so we have to give it the explicit start value.
    combined_cyclers = sum(cyclers[1:-1], cyclers[0])
    ax.set_prop_cycle(combined_cyclers)

    # Store handles for adding to the legend later. Stored as dict so we don't have to
    # de-duplicate later.
    error_boxes = {}
    # Used for labeling purposes. The values that are used are identical for all analyses.
    inclusive_analysis: "correlations.Correlations"
    for displace_index, ep_orientation in enumerate(selected_iterables["reaction_plane_orientation"]):
        # Store the values to be plotted
        values: Dict[analysis_objects.PtBin, analysis_objects.ExtractedObservable] = {}
        for key_index, analysis in \
                analysis_config.iterate_with_selected_objects(
                    analyses, reaction_plane_orientation = ep_orientation
                ):
            # Store each extracted value.
            values[analysis.track_pt] = extract_value_func(analysis)
            # These are both used for labeling purposes and are identical for all analyses that are iterated over.
            if ep_orientation == params.ReactionPlaneOrientation.inclusive:
                inclusive_analysis = analysis

        # Plot the values
        bin_centers = np.array([k.bin_center for k in values])
        bin_centers = bin_centers + displace_index * 0.1
        ax.errorbar(
            bin_centers, [v.value for v in values.values()], yerr = [v.error for v in values.values()],
            label = ep_orientation.display_str(), linestyle = "",
            fillstyle = ep_plot_properties[ep_orientation][2],
        )
        # Plot the RP fit error if it's available.
        if "fit_error" in values[analysis.track_pt].metadata:
            logger.debug(f"Plotting fit errors for {output_name}, {ep_orientation}")
            boxes = plot_base.error_boxes(
                ax = ax, x_data = bin_centers, y_data = np.array([v.value for v in values.values()]),
                x_errors = np.array([0.1 / 2.0] * len(bin_centers)),
                y_errors = np.array([v.metadata["fit_error"] for v in values.values()]),
                label = "Background", color = plot_base.AnalysisColors.fit,
            )
            error_boxes["fit_error"] = boxes
        # Plot the scale uncertainty systematic if it's available.
        if "mixed_event_scale_systematic" in values[analysis.track_pt].metadata:
            logger.debug(f"Plotting the mixed event scale systematic for {output_name}, {ep_orientation}")
            boxes = plot_base.error_boxes(
                ax = ax, x_data = bin_centers, y_data = np.array([v.value for v in values.values()]),
                x_errors = np.array([0.1 / 2.0] * len(bin_centers)),
                # Transposed so that it's in the right format for plotting
                y_errors = np.array([v.metadata["mixed_event_scale_systematic"] for v in values.values()]).T,
                label = "Correlated uncertainty", color = plot_base.AnalysisColors.systematic,
            )
            error_boxes["systematic"] = boxes

    # Labels.
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
    # Finally, add the text to the axis.
    ax.text(
        0.97, 0.97, text, horizontalalignment = "right",
        verticalalignment = "top", multialignment = "right",
        transform = ax.transAxes
    )
    # Deal with projection range, extraction range string.
    # We always want to include the global scale uncertainty.
    lower_left_label = r"Scale uncertainty: 5\%"
    additional_label = _proj_and_extract_range_label(
        inclusive_analysis = inclusive_analysis,
        projection_range_func = projection_range_func,
        extraction_range_func = extraction_range_func,
    )
    if additional_label:
        # Left here in case we decide to make it optional.
        #if lower_left_label == "":
        #    lower_left_label = additional_label
        #else:
        #    lower_left_label += "\n" + additional_label
        lower_left_label += "\n" + additional_label
    if lower_left_label:
        ax.text(
            0.03, 0.03, lower_left_label, horizontalalignment = "left",
            verticalalignment = "bottom", multialignment = "left",
            transform = ax.transAxes, fontsize = 14,
        )

    # Axes and titles
    ax.set_xlabel(labels.make_valid_latex_string(fr"{labels.track_pt_display_label()}\:({labels.momentum_units_label_gev()})"))
    # Apply any specified labels
    if plot_labels.title is not None:
        plot_labels.title = plot_labels.title + f" for {labels.jet_pt_range_string(inclusive_analysis.jet_pt)}"
    plot_labels.apply_labels(ax)
    # Add the error boxes to the handles
    handles, _ = ax.get_legend_handles_labels()
    # To add the PatchCollection to the legend, we have to jump through a number of hoops.
    # We have to create a new set of patches which have the same label and then take the first facecolor.
    # Despite the singular name, facecolor returns a collection of colors, so we have to take the first entry.
    handles.extend([matplotlib.patches.Patch(label = v.get_label(), color = v.get_facecolor()[0]) for v in error_boxes.values()])
    ax.legend(loc = (0.03, 0.08), frameon = False, fontsize = 14, handles = handles)
    # Apply log if requested.
    if logy:
        ax.set_yscale("log")
        # Adjust the formatting if using log on the y axis.
        ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.yaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(plot_base.log_minor_tick_formatter))
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min, y_max if y_max > 5 else 5)

    # Final adjustments
    fig.tight_layout()
    # Save plot and cleanup
    plot_base.save_plot(output_info, fig,
                        f"{fit_type}_{output_name}_{inclusive_analysis.jet_pt_identifier}")
    plt.close(fig)

def delta_phi_plot_projection_range_string(inclusive_analysis: "correlations.Correlations") -> str:
    """ Provides a string that describes the delta eta projection range for delta phi plots. """
    return labels.make_valid_latex_string(
        fr"$|\Delta\eta|<{inclusive_analysis.signal_dominated_eta_region.max}$"
    )

def delta_phi_near_side_widths(analyses: Mapping[Any, "correlations.Correlations"],
                               selected_iterables: Mapping[str, Sequence[Any]],
                               fit_type: str,
                               output_info: analysis_objects.PlottingOutputWrapper) -> None:
    """ Plot the delta phi near-side widths. """
    def near_side_widths(analysis: "correlations.Correlations") -> analysis_objects.ExtractedObservable:
        """ Simple helper function to extract the delta phi near-side widths """
        return analysis_objects.ExtractedObservable(
            value = analysis.widths_delta_phi.near_side.width,
            error = analysis.widths_delta_phi.near_side.fit_result.errors_on_parameters["width"],
            metadata = analysis.widths_delta_phi.near_side.metadata,
        )

    _extracted_values(
        analyses = analyses, selected_iterables = selected_iterables,
        extract_value_func = near_side_widths,
        plot_labels = plot_base.PlotLabels(
            y_label = "Near-side width",
            title = "Near-side width",
        ),
        logy = False,
        output_name = "widths_delta_phi_near_side",
        fit_type = fit_type,
        output_info = output_info,
        projection_range_func = delta_phi_plot_projection_range_string,
    )

def delta_phi_away_side_widths(analyses: Mapping[Any, "correlations.Correlations"],
                               selected_iterables: Mapping[str, Sequence[Any]],
                               fit_type: str,
                               output_info: analysis_objects.PlottingOutputWrapper) -> None:
    """ Plot the delta phi away-side widths. """
    def away_side_widths(analysis: "correlations.Correlations") -> analysis_objects.ExtractedObservable:
        """ Simple helper function to extract the delta phi away-side widths """
        return analysis_objects.ExtractedObservable(
            value = analysis.widths_delta_phi.away_side.width,
            error = analysis.widths_delta_phi.away_side.fit_result.errors_on_parameters["width"],
            metadata = analysis.widths_delta_phi.away_side.metadata,
        )

    _extracted_values(
        analyses = analyses, selected_iterables = selected_iterables,
        extract_value_func = away_side_widths,
        plot_labels = plot_base.PlotLabels(
            y_label = "Away-side width",
            title = "Away-side width",
        ),
        logy = False,
        output_name = "widths_delta_phi_away_side",
        fit_type = fit_type,
        output_info = output_info,
        projection_range_func = delta_phi_plot_projection_range_string,
    )

def delta_phi_near_side_yields(analyses: Mapping[Any, "correlations.Correlations"],
                               selected_iterables: Mapping[str, Sequence[Any]],
                               fit_type: str,
                               output_info: analysis_objects.PlottingOutputWrapper) -> None:
    """ Plot the delta phi near-side yields. """
    def near_side_yields(analysis: "correlations.Correlations") -> analysis_objects.ExtractedObservable:
        """ Helper function to provide the ExtractedObservable. """
        return analysis.yields_delta_phi.near_side.value

    def near_side_extraction_range(analysis: "correlations.Correlations") -> str:
        """ Helper function to provide the yield extraction range.

        We want to extract the value from the hist itself in case the binning is off (it shouldn't be).
        """
        # Setup
        min_value, max_value = _find_extraction_range_min_and_max_in_hist(
            h = analysis.correlation_hists_delta_phi_subtracted.signal_dominated.hist,
            extraction_range = analysis.yields_delta_phi.near_side.extraction_range,
        )
        return labels.make_valid_latex_string(
            r"\text{Yield extraction range:}\:|\Delta\varphi| < " + f"{_find_pi_coefficient(max_value)}"
        )

    _extracted_values(
        analyses = analyses, selected_iterables = selected_iterables,
        extract_value_func = near_side_yields,
        plot_labels = plot_base.PlotLabels(
            y_label = labels.make_valid_latex_string(
                fr"\mathrm{{d}}N/\mathrm{{d}}{labels.pt_display_label()}\:({labels.momentum_units_label_gev()})^{{-1}}",
            ),
            title = "Near-side yield",
        ),
        logy = True,
        output_name = "yields_delta_phi_near_side",
        fit_type = fit_type,
        output_info = output_info,
        projection_range_func = delta_phi_plot_projection_range_string,
        extraction_range_func = near_side_extraction_range,
    )

def delta_phi_away_side_yields(analyses: Mapping[Any, "correlations.Correlations"],
                               selected_iterables: Mapping[str, Sequence[Any]],
                               fit_type: str,
                               output_info: analysis_objects.PlottingOutputWrapper) -> None:
    """ Plot the delta phi away-side yields. """
    def away_side_yields(analysis: "correlations.Correlations") -> analysis_objects.ExtractedObservable:
        """ Helper function to provide the ExtractedObservable. """
        return analysis.yields_delta_phi.away_side.value

    def away_side_extraction_range(analysis: "correlations.Correlations") -> str:
        """ Helper function to provide the yield extraction range.

        We want to extract the value from the hist itself in case the binning is off (it shouldn't be).
        """
        # Setup
        min_value, max_value = _find_extraction_range_min_and_max_in_hist(
            h = analysis.correlation_hists_delta_phi_subtracted.signal_dominated.hist,
            extraction_range = analysis.yields_delta_phi.away_side.extraction_range,
        )
        return labels.make_valid_latex_string(
            r"\text{Yield extraction range:}\:"
            fr" {_find_pi_coefficient(min_value)}"
            r" < |\Delta\varphi| <"
            fr" {_find_pi_coefficient(max_value)}"
        )

    _extracted_values(
        analyses = analyses, selected_iterables = selected_iterables,
        extract_value_func = away_side_yields,
        plot_labels = plot_base.PlotLabels(
            y_label = labels.make_valid_latex_string(
                fr"\mathrm{{d}}N/\mathrm{{d}}{labels.pt_display_label()}\:({labels.momentum_units_label_gev()})^{{-1}}",
            ),
            title = "Away-side yield",
        ),
        logy = True,
        output_name = "yields_delta_phi_away_side",
        fit_type = fit_type,
        output_info = output_info,
        projection_range_func = delta_phi_plot_projection_range_string,
        extraction_range_func = away_side_extraction_range,
    )

def delta_eta_plot_projection_range_string(inclusive_analysis: "correlations.Correlations") -> str:
    """ Provides a string that describes the delta phi projection range for delta eta plots. """
    # The limit is almost certainly a multiple of pi, so we try to express it more naturally
    # as a value like pi/2 or 3*pi/2
    value = _find_pi_coefficient(value = inclusive_analysis.near_side_phi_region.max)
    return labels.make_valid_latex_string(
        fr"$|\Delta\varphi|<{value}$"
    )

def delta_eta_near_side_widths(analyses: Mapping[Any, "correlations.Correlations"],
                               selected_iterables: Mapping[str, Sequence[Any]],
                               fit_type: str,
                               output_info: analysis_objects.PlottingOutputWrapper) -> None:
    """ Plot the delta eta near-side widths. """
    def near_side_widths(analysis: "correlations.Correlations") -> analysis_objects.ExtractedObservable:
        """ Helper function to provide the ExtractedObservable. """
        return analysis_objects.ExtractedObservable(
            value = analysis.widths_delta_phi.near_side.width,
            error = analysis.widths_delta_phi.near_side.fit_result.errors_on_parameters["width"]
        )

    _extracted_values(
        analyses = analyses, selected_iterables = selected_iterables,
        extract_value_func = near_side_widths,
        plot_labels = plot_base.PlotLabels(
            y_label = "Near-side width",
            title = "Near-side width",
        ),
        logy = False,
        output_name = "widths_delta_eta_near_side",
        fit_type = fit_type,
        output_info = output_info,
        projection_range_func = delta_eta_plot_projection_range_string,
    )

def delta_eta_near_side_yields(analyses: Mapping[Any, "correlations.Correlations"],
                               selected_iterables: Mapping[str, Sequence[Any]],
                               fit_type: str,
                               output_info: analysis_objects.PlottingOutputWrapper) -> None:
    """ Plot the delta eta near-side yields. """
    def near_side_widths(analysis: "correlations.Correlations") -> analysis_objects.ExtractedObservable:
        """ Helper function to provide the ExtractedObservable. """
        return analysis.yields_delta_eta.near_side.value

    def near_side_extraction_range(analysis: "correlations.Correlations") -> str:
        """ Helper function to provide the yield extraction range.

        We want to extract the value from the hist itself in case the binning is off (it shouldn't be).
        """
        # Setup
        min_value, max_value = _find_extraction_range_min_and_max_in_hist(
            h = analysis.correlation_hists_delta_eta_subtracted.near_side.hist,
            extraction_range = analysis.yields_delta_eta.near_side.extraction_range,
        )
        # Due to floating point rounding issues, we need to apply our rounding trick here.
        return labels.make_valid_latex_string(
            r"\text{Yield extraction range:}\:"
            fr" |\Delta\eta| < {int(round(max_value * 10)) / 10}"
        )

    _extracted_values(
        analyses = analyses, selected_iterables = selected_iterables,
        extract_value_func = near_side_widths,
        plot_labels = plot_base.PlotLabels(
            y_label = labels.make_valid_latex_string(
                fr"\mathrm{{d}}N/\mathrm{{d}}{labels.pt_display_label()}\:({labels.momentum_units_label_gev()})^{{-1}}",
            ),
            title = "Near-side yield",
        ),
        output_name = "yields_delta_eta_near_side",
        logy = True,
        fit_type = fit_type,
        output_info = output_info,
        projection_range_func = delta_eta_plot_projection_range_string,
        extraction_range_func = near_side_extraction_range,
    )

def _determine_contributors_label_names(yield_value: Union[extracted.ExtractedYieldRatio,
                                                           extracted.ExtractedYieldDifference]) -> List[str]:
    """ Determine contributor label names from a stored yield object.

    For example, it converts "out_of_plane" -> "out".

    Args:
        yield_value: A single yield value.
    Returns:
        Tuple of the names of the contributors, ordered as they were stored.
    """
    names = []
    for c in yield_value.contributors:
        # Converts "out_of_plane" -> "out"
        name = str(c)
        name = name[:name.find("_")].capitalize()
        names.append(name)

    return names

def _determine_JEWEL_prediction_observable(yield_value: Union[extracted.ExtractedYieldRatio,
                                                              extracted.ExtractedYieldDifference]) -> str:
    """ Determine JEWEL prediction filename from a stored yield object.

    For example, it converts "out_of_plane" -> "out".

    Args:
        yield_value: A single yield value.
    Returns:
        Tuple of the names of the contributors, ordered as they were stored.
    """
    names = _determine_contributors_label_names(yield_value)
    if isinstance(yield_value, extracted.ExtractedYieldRatio):
        middle_term = "Over"
    else:
        middle_term = "Minus"
    names = [n.capitalize() for n in names]
    return f"{names[0]}{middle_term}{names[1]}"

def _yield_ratio(yield_ratios: Dict[Any, extracted.ExtractedYieldRatio],
                 extract_value_func: Callable[["correlations.CorrelationsYields"], analysis_objects.ExtractedObservable],
                 an_analysis: "correlations.Correlations",
                 label: str,
                 plot_labels: plot_base.PlotLabels,
                 output_name: str,
                 fit_type: str,
                 output_info: analysis_objects.PlottingOutputWrapper,
                 JEWEL_predictions: Optional[extracted.JEWELPredictions] = None,
                 projection_range_func: Optional[Callable[["correlations.Correlations"], str]] = None,
                 extraction_range_func: Optional[Callable[["correlations.Correlations"], str]] = None) -> None:
    """ Plot yield ratios for out / in and mid / in.

    Args:
        yield_ratios: The yield ratios to be plotted.
        extract_value_func: Function to retrieve the extracted value and error.
        an_analysis: Any correlations analyiss. Just needed for generic labels.
        labels: Output labels.
        plot_labels: Titles and axis labels for the plot.
        output_name: Base of name under which the plot will be stored.
        fit_type: Name of the RP fit type used to get to this extracted value.
        output_info: Information needed to determine where to store the plot.
        JEWEL_predictions: JEWEL predictions to plot. Default: None.
        projection_range_func: Function which will provide the projection range of the extracted value given
            the inclusive object. Default: None.
        extraction_range_func: Function which will provide the extraction range of the extracted value given
            the inclusive object. Default: None.
    Returns:
        None. The plot is stored.
    """
    # Setup
    fig, ax = plt.subplots(figsize = (8, 6))

    # Store handles for adding to the legend later. Stored as dict so we don't have to
    # de-duplicate later.
    error_boxes = {}
    values: Dict[analysis_objects.PtBin, analysis_objects.ExtractedObservable] = {}
    for key_index, ratios in yield_ratios.items():
        logger.debug(f"key_index: {key_index}")
        values[key_index.track_pt_bin] = extract_value_func(ratios)

    # Plot the values
    bin_centers = np.array([k.bin_center for k in values])
    ax.errorbar(
        bin_centers, [v.value for v in values.values()], yerr = [v.error for v in values.values()],
        marker = "o", linestyle = "",
    )
    # Plot the fit uncertainty if it's available.
    if "fit_error" in values[key_index.track_pt_bin].metadata:
        logger.debug(f"Plotting the fit errors.")
        boxes = plot_base.error_boxes(
            ax = ax, x_data = bin_centers, y_data = np.array([v.value for v in values.values()]),
            x_errors = np.array([0.1 / 2.0] * len(bin_centers)),
            y_errors = np.array([v.metadata["fit_error"] for v in values.values()]),
            label = "Background", color = plot_base.AnalysisColors.fit,
        )
        error_boxes["fit_error"] = boxes

    # Add JEWEL predictions if available
    if JEWEL_predictions:
        for prediction_type, prediction in JEWEL_predictions:
            # Plot the JEWEL predictions as a line with bands.
            ax.plot(
                prediction.x, prediction.y,
                label = f"JEWEL {JEWEL_predictions.display_name(prediction_type)}",
                color = plot_base.AnalysisColors.prediction,
                # Use a solid line for keeping recoils, and a dashed line for no recoils.
                linestyle = "-" if prediction_type == "keep_recoils" else "--"
            )
            ax.fill_between(
                prediction.x, prediction.y - prediction.errors, prediction.y + prediction.errors,
                color = plot_base.AnalysisColors.prediction, alpha = 0.3,
            )

    # Labels.
    # General
    text = labels.make_valid_latex_string(an_analysis.alice_label.display_str())
    text += "\n" + labels.system_label(
        energy = an_analysis.collision_energy,
        system = an_analysis.collision_system,
        activity = an_analysis.event_activity
    )
    text += "\n" + labels.jet_pt_range_string(an_analysis.jet_pt)
    text += "\n" + labels.jet_finding()
    text += "\n" + labels.constituent_cuts()
    text += "\n" + labels.make_valid_latex_string(an_analysis.leading_hadron_bias.display_str())
    # Finally, add the text to the axis.
    ax.text(
        0.97, 0.97, text, horizontalalignment = "right",
        verticalalignment = "top", multialignment = "right",
        transform = ax.transAxes
    )
    # Yield ratio label
    ax.text(
        0.03, 0.97, label, horizontalalignment = "left",
        verticalalignment = "top", multialignment = "left",
        transform = ax.transAxes
    )
    # Deal with projection range, extraction range string.
    additional_label = _proj_and_extract_range_label(
        inclusive_analysis = an_analysis,
        projection_range_func = projection_range_func,
        extraction_range_func = extraction_range_func,
    )
    if additional_label:
        ax.text(
            0.03, 0.03, additional_label, horizontalalignment = "left",
            verticalalignment = "bottom", multialignment = "left",
            transform = ax.transAxes, fontsize = 14,
        )

    # Set a uniform range
    ax.set_ylim(0.1, 1.9)
    # Add a line at 1 for reference
    ax.axhline(y = 1, color = "black", linestyle = "dashed", zorder = 1)

    # Axes and titles
    ax.set_xlabel(
        labels.make_valid_latex_string(fr"{labels.track_pt_display_label()}\:({labels.momentum_units_label_gev()})")
    )
    # Apply any specified labels
    if plot_labels.title is not None:
        plot_labels.title = plot_labels.title + f" for {labels.jet_pt_range_string(an_analysis.jet_pt)}"
    plot_labels.apply_labels(ax)
    # Add the error boxes to the handles
    handles, _ = ax.get_legend_handles_labels()
    # To add the PatchCollection to the legend, we have to jump through a number of hoops.
    # We have to create a new set of patches which have the same label and then take the first facecolor.
    # Despite the singular name, facecolor returns a collection of colors, so we have to take the first entry.
    handles.extend(
        [matplotlib.patches.Patch(label = v.get_label(), color = v.get_facecolor()[0]) for v in error_boxes.values()]
    )
    # The handles are reversed so the background will show up first with the JEWEL labels below.
    ax.legend(loc = (0.03, 0.08), frameon = False, fontsize = 14, handles = list(reversed(handles)))

    # Final adjustments
    fig.tight_layout()
    # Convert "Out / in" -> "out_vs_in"
    file_label = label.lower().replace(" / ", "_vs_")
    # Save plot and cleanup
    plot_base.save_plot(output_info, fig,
                        f"{fit_type}_{output_name}_{file_label}_{an_analysis.jet_pt_identifier}")
    plt.close(fig)

def delta_phi_near_side_yield_ratio(yield_ratios: Dict[Any, "correlations.CorrelationsYields"],
                                    an_analysis: "correlations.Correlations",
                                    fit_type: str,
                                    label: str,
                                    output_info: analysis_objects.PlottingOutputWrapper) -> None:
    def near_side_yields(yields: "correlations.CorrelationsYields") -> analysis_objects.ExtractedObservable:
        """ Helper function to provide the ExtractedObservable. """
        return cast(analysis_objects.ExtractedObservable, yields.near_side.value)

    def near_side_extraction_range(analysis: "correlations.Correlations") -> str:
        """ Helper function to provide the yield extraction range.

        We want to extract the value from the hist itself in case the binning is off (it shouldn't be).
        """
        # Setup
        min_value, max_value = _find_extraction_range_min_and_max_in_hist(
            h = analysis.correlation_hists_delta_phi_subtracted.signal_dominated.hist,
            extraction_range = analysis.yields_delta_phi.near_side.extraction_range,
        )
        return labels.make_valid_latex_string(
            r"\text{Yield extraction range:}\:|\Delta\varphi| < " + f"{_find_pi_coefficient(max_value)}"
        )

    # Determine the y axis label based on the contributors.
    single_yield = next(iter(yield_ratios.values())).near_side
    numerator_name, denominator_name = _determine_contributors_label_names(single_yield)

    # Determine the JEWEL predictions.
    JEWEL_predictions = extracted.load_JEWEL_predictions(
        an_analysis.collision_system, an_analysis.collision_energy,
        an_analysis.event_activity, _determine_JEWEL_prediction_observable(single_yield), side = "NS",
    )

    # Plot both with and without JEWEL
    for plot_JEWEL_predictions in [False, True]:
        _yield_ratio(
            yield_ratios = yield_ratios, an_analysis = an_analysis,
            extract_value_func = near_side_yields,
            label = label,
            plot_labels = plot_base.PlotLabels(
                y_label = labels.make_valid_latex_string(
                    fr"R = \text{{Y}}_{{\text{{{numerator_name}}}}} / \text{{Y}}_{{\text{{{denominator_name}}}}}",
                ),
                title = "Near-side yield ratio",
            ),
            output_name = "yield_ratio_delta_phi_near_side" + ("_JEWEL" if plot_JEWEL_predictions else ""),
            fit_type = fit_type,
            output_info = output_info,
            JEWEL_predictions = JEWEL_predictions if plot_JEWEL_predictions else None,
            projection_range_func = delta_phi_plot_projection_range_string,
            extraction_range_func = near_side_extraction_range,
        )

def delta_phi_away_side_yield_ratio(yield_ratios: Dict[Any, "correlations.CorrelationsYields"],
                                    an_analysis: "correlations.Correlations",
                                    label: str,
                                    fit_type: str,
                                    output_info: analysis_objects.PlottingOutputWrapper) -> None:
    """ Plot the delta phi away-side yields. """
    def away_side_yields(yields: "correlations.CorrelationsYields") -> analysis_objects.ExtractedObservable:
        """ Helper function to provide the ExtractedObservable. """
        return cast(analysis_objects.ExtractedObservable, yields.away_side.value)

    def away_side_extraction_range(analysis: "correlations.Correlations") -> str:
        """ Helper function to provide the yield extraction range.

        We want to extract the value from the hist itself in case the binning is off (it shouldn't be).
        """
        # Setup
        min_value, max_value = _find_extraction_range_min_and_max_in_hist(
            h = analysis.correlation_hists_delta_phi_subtracted.signal_dominated.hist,
            extraction_range = analysis.yields_delta_phi.away_side.extraction_range,
        )
        return labels.make_valid_latex_string(
            r"\text{Yield extraction range:}\:"
            fr" {_find_pi_coefficient(min_value)}"
            r" < |\Delta\varphi| <"
            fr" {_find_pi_coefficient(max_value)}"
        )

    # Determine the y axis label based on the contributors.
    single_yield = next(iter(yield_ratios.values())).away_side
    numerator_name, denominator_name = _determine_contributors_label_names(single_yield)

    # Determine the JEWEL predictions.
    JEWEL_predictions = extracted.load_JEWEL_predictions(
        an_analysis.collision_system, an_analysis.collision_energy,
        an_analysis.event_activity, _determine_JEWEL_prediction_observable(single_yield), side = "AS",
    )

    # Plot both with and without JEWEL
    for plot_JEWEL_predictions in [False, True]:
        _yield_ratio(
            yield_ratios = yield_ratios, an_analysis = an_analysis,
            extract_value_func = away_side_yields,
            label = label,
            plot_labels = plot_base.PlotLabels(
                y_label = labels.make_valid_latex_string(
                    fr"R = \text{{Y}}_{{\text{{{numerator_name}}}}} / \text{{Y}}_{{\text{{{denominator_name}}}}}",
                ),
                title = "Away-side yield ratio",
            ),
            output_name = "yield_ratio_delta_phi_away_side" + ("_JEWEL" if plot_JEWEL_predictions else ""),
            fit_type = fit_type,
            output_info = output_info,
            JEWEL_predictions = JEWEL_predictions if plot_JEWEL_predictions else None,
            projection_range_func = delta_phi_plot_projection_range_string,
            extraction_range_func = away_side_extraction_range,
        )

###################
# Yield differences
###################

def _yield_difference(yield_differences: Dict[Any, extracted.ExtractedYieldDifference],
                      extract_value_func: Callable[["correlations.CorrelationsYields"],
                                                   analysis_objects.ExtractedObservable],
                      an_analysis: "correlations.Correlations",
                      label: str,
                      plot_labels: plot_base.PlotLabels,
                      output_name: str,
                      fit_type: str,
                      output_info: analysis_objects.PlottingOutputWrapper,
                      JEWEL_predictions: Optional[extracted.JEWELPredictions] = None,
                      projection_range_func: Optional[Callable[["correlations.Correlations"], str]] = None,
                      extraction_range_func: Optional[Callable[["correlations.Correlations"], str]] = None) -> None:
    """ Plot yield differences for out / in and mid / in.

    Args:
        yield_differences: The yield differences to be plotted.
        extract_value_func: Function to retrieve the extracted value and error.
        an_analysis: Any correlations analyiss. Just needed for generic labels.
        labels: Output labels.
        plot_labels: Titles and axis labels for the plot.
        output_name: Base of name under which the plot will be stored.
        fit_type: Name of the RP fit type used to get to this extracted value.
        output_info: Information needed to determine where to store the plot.
        JEWEL_predictions: JEWEL predictions to plot. Default: None.
        projection_range_func: Function which will provide the projection range of the extracted value given
            the inclusive object. Default: None.
        extraction_range_func: Function which will provide the extraction range of the extracted value given
            the inclusive object. Default: None.
    Returns:
        None. The plot is stored.
    """
    # Setup
    fig, ax = plt.subplots(figsize = (8, 6))

    # Store handles for adding to the legend later. Stored as dict so we don't have to
    # de-duplicate later.
    error_boxes = {}
    values: Dict[analysis_objects.PtBin, analysis_objects.ExtractedObservable] = {}
    for key_index, differences in yield_differences.items():
        logger.debug(f"key_index: {key_index}")
        values[key_index.track_pt_bin] = extract_value_func(differences)

    # Plot the values
    bin_centers = np.array([k.bin_center for k in values])
    ax.errorbar(
        bin_centers, [v.value for v in values.values()], yerr = [v.error for v in values.values()],
        marker = "o", linestyle = "",
    )
    # Plot the fit uncertainty if it's available.
    if "fit_error" in values[key_index.track_pt_bin].metadata:
        logger.debug(f"Plotting the fit errors")
        boxes = plot_base.error_boxes(
            ax = ax, x_data = bin_centers, y_data = np.array([v.value for v in values.values()]),
            x_errors = np.array([0.1 / 2.0] * len(bin_centers)),
            y_errors = np.array([v.metadata["fit_error"] for v in values.values()]),
            label = "Background", color = plot_base.AnalysisColors.fit,
        )
        error_boxes["fit_error"] = boxes
    # Plot the scale uncertainty systematic if it's available.
    if "mixed_event_scale_systematic" in values[key_index.track_pt_bin].metadata:
        logger.debug(f"Plotting the mixed event scale systematic.")
        boxes = plot_base.error_boxes(
            ax = ax, x_data = bin_centers, y_data = np.array([v.value for v in values.values()]),
            x_errors = np.array([0.1 / 2.0] * len(bin_centers)),
            # Transposed so that it's in the right format for plotting
            y_errors = np.array([v.metadata["mixed_event_scale_systematic"] for v in values.values()]).T,
            label = "Correlated uncertainty", color = plot_base.AnalysisColors.systematic,
        )
        error_boxes["systematic"] = boxes

    # Add JEWEL predictions if available
    if JEWEL_predictions:
        for prediction_type, prediction in JEWEL_predictions:
            # Plot the JEWEL predictions as a line with bands.
            ax.plot(
                prediction.x, prediction.y,
                label = f"JEWEL {JEWEL_predictions.display_name(prediction_type)}",
                color = plot_base.AnalysisColors.prediction,
                # Use a solid line for keeping recoils, and a dashed line for no recoils.
                linestyle = "-" if prediction_type == "keep_recoils" else "--"
            )
            ax.fill_between(
                prediction.x, prediction.y - prediction.errors, prediction.y + prediction.errors,
                color = plot_base.AnalysisColors.prediction, alpha = 0.3,
            )

    # Labels.
    # General
    text = labels.make_valid_latex_string(an_analysis.alice_label.display_str())
    text += "\n" + labels.system_label(
        energy = an_analysis.collision_energy,
        system = an_analysis.collision_system,
        activity = an_analysis.event_activity
    )
    text += "\n" + labels.jet_pt_range_string(an_analysis.jet_pt)
    text += "\n" + labels.jet_finding()
    text += "\n" + labels.constituent_cuts()
    text += "\n" + labels.make_valid_latex_string(an_analysis.leading_hadron_bias.display_str())
    # Finally, add the text to the axis.
    ax.text(
        0.97, 0.97, text, horizontalalignment = "right",
        verticalalignment = "top", multialignment = "right",
        transform = ax.transAxes
    )
    # Yield difference label
    ax.text(
        0.03, 0.97, label, horizontalalignment = "left",
        verticalalignment = "top", multialignment = "left",
        transform = ax.transAxes
    )
    # Deal with projection range, extraction range string.
    lower_left_label = r"Scale uncertainty: 5\%"
    additional_label = _proj_and_extract_range_label(
        inclusive_analysis = an_analysis,
        projection_range_func = projection_range_func,
        extraction_range_func = extraction_range_func,
    )
    if additional_label:
        lower_left_label += "\n" + additional_label
    if lower_left_label:
        ax.text(
            0.03, 0.03, additional_label, horizontalalignment = "left",
            verticalalignment = "bottom", multialignment = "left",
            transform = ax.transAxes, fontsize = 14,
        )

    # Set a uniform range
    ax.set_ylim(-1.5, 1.5)
    # Add a line at 0 for reference
    ax.axhline(y = 0, color = "black", linestyle = "dashed", zorder = 1)

    # Axes and titles
    ax.set_xlabel(
        labels.make_valid_latex_string(fr"{labels.track_pt_display_label()}\:({labels.momentum_units_label_gev()})")
    )
    # Apply any specified labels
    if plot_labels.title is not None:
        plot_labels.title = plot_labels.title + f" for {labels.jet_pt_range_string(an_analysis.jet_pt)}"
    plot_labels.apply_labels(ax)
    # Add the error boxes to the handles
    handles, _ = ax.get_legend_handles_labels()
    # To add the PatchCollection to the legend, we have to jump through a number of hoops.
    # We have to create a new set of patches which have the same label and then take the first facecolor.
    # Despite the singular name, facecolor returns a collection of colors, so we have to take the first entry.
    handles.extend(
        [matplotlib.patches.Patch(label = v.get_label(), color = v.get_facecolor()[0]) for v in error_boxes.values()]
    )
    # TODO: Update the loc with the scale uncertainty label.
    # The handles are reversed so the background will show up first with the JEWEL labels below.
    ax.legend(loc = (0.03, 0.12), frameon = False, fontsize = 14, handles = list(reversed(handles)))

    # Final adjustments
    fig.tight_layout()
    # Convert "Out-/ in" -> "out_vs_in"
    file_label = label.lower().replace(" - ", "_vs_")
    # Save plot and cleanup
    plot_base.save_plot(output_info, fig,
                        f"{fit_type}_{output_name}_{file_label}_{an_analysis.jet_pt_identifier}")
    plt.close(fig)

def delta_phi_near_side_yield_difference(yield_differences: Dict[Any, "correlations.CorrelationsYields"],
                                         an_analysis: "correlations.Correlations",
                                         fit_type: str,
                                         label: str,
                                         output_info: analysis_objects.PlottingOutputWrapper) -> None:
    def near_side_yields(yields: "correlations.CorrelationsYields") -> analysis_objects.ExtractedObservable:
        """ Helper function to provide the ExtractedObservable. """
        return cast(analysis_objects.ExtractedObservable, yields.near_side.value)

    def near_side_extraction_range(analysis: "correlations.Correlations") -> str:
        """ Helper function to provide the yield extraction range.

        We want to extract the value from the hist itself in case the binning is off (it shouldn't be).
        """
        # Setup
        min_value, max_value = _find_extraction_range_min_and_max_in_hist(
            h = analysis.correlation_hists_delta_phi_subtracted.signal_dominated.hist,
            extraction_range = analysis.yields_delta_phi.near_side.extraction_range,
        )
        return labels.make_valid_latex_string(
            r"\text{Yield extraction range:}\:|\Delta\varphi| < " + f"{_find_pi_coefficient(max_value)}"
        )

    # Determine the y axis label based on the contributors.
    single_yield = next(iter(yield_differences.values())).near_side
    first_term_name, second_term_name = _determine_contributors_label_names(single_yield)

    # Determine the JEWEL predictions.
    JEWEL_predictions = extracted.load_JEWEL_predictions(
        an_analysis.collision_system, an_analysis.collision_energy,
        an_analysis.event_activity, _determine_JEWEL_prediction_observable(single_yield), side = "NS",
    )

    # Plot both with and without JEWEL
    for plot_JEWEL_predictions in [False, True]:
        _yield_difference(
            yield_differences = yield_differences, an_analysis = an_analysis,
            extract_value_func = near_side_yields,
            label = label,
            plot_labels = plot_base.PlotLabels(
                y_label = labels.make_valid_latex_string(
                    fr"\text{{D}}_{{\text{{RP}}}} = \text{{Y}}_{{\text{{{first_term_name}}}}} - \text{{Y}}_{{\text{{{second_term_name}}}}}"
                    + fr"\:({labels.momentum_units_label_gev()}^{{-1}})",
                ),
                title = "Near-side yield difference",
            ),
            output_name = "yield_difference_delta_phi_near_side" + ("_JEWEL" if plot_JEWEL_predictions else ""),
            fit_type = fit_type,
            output_info = output_info,
            JEWEL_predictions = JEWEL_predictions if plot_JEWEL_predictions else None,
            projection_range_func = delta_phi_plot_projection_range_string,
            extraction_range_func = near_side_extraction_range,
        )

def delta_phi_away_side_yield_difference(yield_differences: Dict[Any, "correlations.CorrelationsYields"],
                                         an_analysis: "correlations.Correlations",
                                         label: str,
                                         fit_type: str,
                                         output_info: analysis_objects.PlottingOutputWrapper) -> None:
    """ Plot the delta phi away-side yields differences. """
    def away_side_yields(yields: "correlations.CorrelationsYields") -> analysis_objects.ExtractedObservable:
        """ Helper function to provide the ExtractedObservable. """
        return cast(analysis_objects.ExtractedObservable, yields.away_side.value)

    def away_side_extraction_range(analysis: "correlations.Correlations") -> str:
        """ Helper function to provide the yield extraction range.

        We want to extract the value from the hist itself in case the binning is off (it shouldn't be).
        """
        # Setup
        min_value, max_value = _find_extraction_range_min_and_max_in_hist(
            h = analysis.correlation_hists_delta_phi_subtracted.signal_dominated.hist,
            extraction_range = analysis.yields_delta_phi.away_side.extraction_range,
        )
        return labels.make_valid_latex_string(
            r"\text{Yield extraction range:}\:"
            fr" {_find_pi_coefficient(min_value)}"
            r" < |\Delta\varphi| <"
            fr" {_find_pi_coefficient(max_value)}"
        )

    # Determine the y axis label based on the contributors.
    single_yield = next(iter(yield_differences.values())).away_side
    first_term_name, second_term_name = _determine_contributors_label_names(single_yield)

    # Determine the JEWEL predictions.
    JEWEL_predictions = extracted.load_JEWEL_predictions(
        an_analysis.collision_system, an_analysis.collision_energy,
        an_analysis.event_activity, _determine_JEWEL_prediction_observable(single_yield), side = "AS",
    )

    # Plot both with and without JEWEL
    for plot_JEWEL_predictions in [False, True]:
        _yield_difference(
            yield_differences = yield_differences, an_analysis = an_analysis,
            extract_value_func = away_side_yields,
            label = label,
            plot_labels = plot_base.PlotLabels(
                y_label = labels.make_valid_latex_string(
                    fr"\text{{D}}_{{\text{{RP}}}} = \text{{Y}}_{{\text{{{first_term_name}}}}} - \text{{Y}}_{{\text{{{second_term_name}}}}}"
                    + fr"\:({labels.momentum_units_label_gev()}^{{-1}})",
                ),
                title = "Away-side yield difference",
            ),
            output_name = "yield_difference_delta_phi_away_side" + ("_JEWEL" if plot_JEWEL_predictions else ""),
            fit_type = fit_type,
            output_info = output_info,
            JEWEL_predictions = JEWEL_predictions if plot_JEWEL_predictions else None,
            projection_range_func = delta_phi_plot_projection_range_string,
            extraction_range_func = away_side_extraction_range,
        )
