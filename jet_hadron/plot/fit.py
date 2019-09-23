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
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING, Union

from pachyderm import histogram
from pachyderm import yaml

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
ReferenceData = Mapping[str, Any]

@dataclass
class ParameterInfo:
    """ Simple helper class to store information about each fit parameter for plotting.

    Attributes:
        name: Name of the parameter in the fit. Used to extract it from the stored fit result.
        output_name: Name under which the plot will be stored.
        labels: Title and axis labels for the plot.
        plot_reference_data_func: Function to handle plotting provided reference data. Optional.
        transform_fit_data: Function to transform the fit data for plotting.
    """
    name: str
    output_name: str
    labels: plot_base.PlotLabels
    plot_reference_data_func: Optional[Callable[..., Optional[List[Any]]]] = None
    transform_fit_data: Optional[Callable[[histogram.Histogram1D], histogram.Histogram1D]] = None
    additional_plot_options: Optional[Dict[str, Union[str, bool]]] = None

def _plot_fit_parameter_vs_assoc_pt(fit_objects: FitObjects,
                                    parameter: ParameterInfo,
                                    reference_data: ReferenceData,
                                    selected_analysis_options: params.SelectedAnalysisOptions,
                                    output_info: analysis_objects.PlottingOutputWrapper) -> None:
    """ Implementation of plotting the fit parameters vs associated track pt.

    Args:
        fit_objects: RP Fit objects.
        parameter: Information about the parameter to be plotted.
        reference_data: Reference data to compare the parameter against.
        selected_analysis_options: Selected analysis options for determining which data to plot.
        output_info: Output information.
    Returns:
        None. The figure is plotted and saved.
    """
    fig, ax = plt.subplots(figsize = (8, 6))

    # Extract the parameter values from each fit object.
    bin_edges = []
    parameter_values = []
    parameter_errors = []
    for key_index, fit_object in fit_objects.items():
        # First take all of the lower edges.
        # This assumes that the bins are continuous.
        bin_edges.append(key_index.track_pt_bin.min)
        parameter_values.append(fit_object.fit_result.values_at_minimum[parameter.name])
        parameter_errors.append(fit_object.fit_result.errors_on_parameters[parameter.name])
    # Now grab the last upper edge. The last key_index is still valid.
    bin_edges.append(key_index.track_pt_bin.max)

    # Store the data into a convenient form.
    data = histogram.Histogram1D(
        bin_edges = bin_edges, y = parameter_values, errors_squared = np.array(parameter_errors) ** 2
    )

    # Plug-in to transform the plotted data.
    if parameter.transform_fit_data:
        data = parameter.transform_fit_data(data)

    # Plot the particular parameter.
    ax.errorbar(
        data.x, data.y, yerr = data.errors,
        marker = "o", linestyle = "",
        label = parameter.labels.title,
    )
    # Handle parameter specific options
    additional_plot_options = parameter.additional_plot_options
    if additional_plot_options is None:
        additional_plot_options = {}
    logy = additional_plot_options.get("logy", False)
    if logy:
        ax.set_yscale("log")

    handles: Optional[List[Any]] = []
    if parameter.plot_reference_data_func:
        handles = parameter.plot_reference_data_func(
            reference_data, ax,
            selected_analysis_options,
        )

    # Labeling
    parameter.labels.apply_labels(ax)
    legend_kwargs: Dict[str, Any] = dict(
        loc = additional_plot_options.get("legend_location", "upper left"), frameon = False,
    )
    # Add custom legend handles from the reference data.
    if handles:
        legend_kwargs["handles"] = handles
    ax.legend(**legend_kwargs)
    # Final adjustments
    fig.tight_layout()
    # Save plot and cleanup
    plot_base.save_plot(output_info, fig, parameter.output_name)
    plt.close(fig)

def _plot_reference_vn_data(variable_name: str, harmonic: int, reference_data: ReferenceData,
                            ax: matplotlib.axes.Axes,
                            selected_analysis_options: params.SelectedAnalysisOptions) -> None:
    """ Plot the reference vn data histogram on the given axis.

    Args:
        variable_name: Name of the variable to plot.
        harmonic: Harmonic of the data to be plotted. Usually related to the variable name.
        reference_data: Reference data to be used for plotting.
        ax: Axis where the data should be plotted.
        selected_analysis_options: Selected analysis options for determining which data to plot.
    Returns:
        None. The data is plotted on the given axis.
    """
    # Determine the centrality key (because they don't map cleanly onto our centrality ranges)
    reference_data_centrality_map = {
        params.EventActivity.central: "0-5",
        params.EventActivity.semi_central: "30-40",
    }
    centrality_label = reference_data_centrality_map[selected_analysis_options.event_activity]
    # Retrieve the data
    hist = reference_data["ptDependent"][variable_name][centrality_label]

    # Plot and label it on the existing axis.
    ax.errorbar(
        hist.x, hist.y, yerr = hist.errors,
        marker = "o", linestyle = "",
        label = fr"ALICE {centrality_label}\% "
                + labels.make_valid_latex_string(f"v_{{ {harmonic} }}") + r"\{2, $| \Delta\eta |>1$\}",
    )

def _reference_v2_a_data(reference_data: ReferenceData,
                         ax: matplotlib.axes.Axes,
                         selected_analysis_options: params.SelectedAnalysisOptions) -> None:
    """ Plot reference v2_a data.

    Args:
        reference_data: Reference data to be used for plotting.
        ax: Axis where the data should be plotted.
        selected_analysis_options: Selected analysis options for determining which data to plot.
    Returns:
        None. The data is plotted on the given axis.
    """
    _plot_reference_vn_data(variable_name = "v2_a", harmonic = 2, reference_data = reference_data,
                            ax = ax, selected_analysis_options = selected_analysis_options)

def _reference_v2_t_data(reference_data: ReferenceData,
                         ax: matplotlib.axes.Axes,
                         selected_analysis_options: params.SelectedAnalysisOptions) -> List[Any]:
    """ Plot reference v2_t data point.

    Args:
        reference_data: Reference data to be used for plotting.
        ax: Axis where the data should be plotted.
        selected_analysis_options: Selected analysis options for determining which data to plot.
    Returns:
        None. The data is plotted on the given axis.
    """
    # Determine the centrality key (because they don't map cleanly onto our centrality ranges)
    reference_data_centrality_map = {
        params.EventActivity.central: "0-5",
        params.EventActivity.semi_central: "30-50",
    }
    centrality_label = reference_data_centrality_map[selected_analysis_options.event_activity]
    # Retrieve the data. It's a single point in contrast to the other reference data
    data = reference_data["ptDependent"]["v2_t"][centrality_label]
    pt_range = params.SelectedRange(*data["jet_pt_range"])
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(
        0 if y_min > 0 else y_min,
        (data["value"] + data["error"]) * 1.1 if y_max < data["value"] + data["error"] else y_max
    )

    # Draw the data as an arrow
    # The arrow points from xytext to xy
    x_min, x_max = ax.get_xlim()
    ax.annotate(
        "", xytext = (x_max, data["value"]),
        # The arrow should go in 10% of the plot.
        xy = (x_max - (x_max - x_min) * 0.10, data["value"]), xycoords = "data",
        arrowprops = dict(arrowstyle = "simple")
    )
    # And then label. We can't add the label with the arrow together because we
    # can't control the text position well enough.
    label = "Jet $v_{2}$"
    ax.annotate(
        label, xy = (x_max - (x_max - x_min) * 0.05, data["value"] * 1.35),
        horizontalalignment="center",
        verticalalignment="center",
    )
    # And then the error band
    r = matplotlib.patches.Rectangle(
        (x_max - (x_max - x_min) * 0.08, data["value"] - data["error"]),
        x_max, 2 * data["error"],
        alpha = 0.5
    )
    ax.add_patch(r)

    # Add the full information to the legend
    legend_label = (fr"ALICE {centrality_label}\%, Jet $v_{2}$"
                    fr" {pt_range.min}-{pt_range.max} ${labels.momentum_units_label_gev()}$")
    # Help out mypy...
    handles: List[Any]
    handles, legend_labels = ax.get_legend_handles_labels()
    handles.append(matplotlib.lines.Line2D([0], [0], label = legend_label))

    return handles

def _reference_v3_data(reference_data: ReferenceData,
                       ax: matplotlib.axes.Axes,
                       selected_analysis_options: params.SelectedAnalysisOptions) -> None:
    """ Plot reference v3 data.

    Args:
        reference_data: Reference data to be used for plotting.
        ax: Axis where the data should be plotted.
        selected_analysis_options: Selected analysis options for determining which data to plot.
    Returns:
        None. The data is plotted on the given axis.
    """
    _plot_reference_vn_data(variable_name = "v3", harmonic = 3, reference_data = reference_data,
                            ax = ax, selected_analysis_options = selected_analysis_options)

def _square_root_v3_2(data: histogram.Histogram1D) -> histogram.Histogram1D:
    """ Transform the v3^2 data into v3 data.

    Args:
        data: Input data.
    Returns:
        Transformed data.
    """
    # Take the square root of the v3^2 values to compare to the reference data.
    parameter_values = np.sqrt(data.y)
    errors = 1 / 2 * data.errors / parameter_values
    return histogram.Histogram1D(bin_edges = data.bin_edges, y = parameter_values, errors_squared = errors ** 2)

def _reference_v4_a_data(reference_data: ReferenceData,
                         ax: matplotlib.axes.Axes,
                         selected_analysis_options: params.SelectedAnalysisOptions) -> None:
    """ Plot reference v4_a data.

    Args:
        reference_data: Reference data to be used for plotting.
        ax: Axis where the data should be plotted.
        selected_analysis_options: Selected analysis options for determining which data to plot.
    Returns:
        None. The data is plotted on the given axis.
    """
    _plot_reference_vn_data(variable_name = "v4_a", harmonic = 4, reference_data = reference_data,
                            ax = ax, selected_analysis_options = selected_analysis_options)

def fit_parameters_vs_assoc_pt(fit_objects: FitObjects,
                               fit_type: str,
                               selected_analysis_options: params.SelectedAnalysisOptions,
                               reference_data_path: str,
                               output_info: analysis_objects.PlottingOutputWrapper) -> None:
    """ Plot the extracted fit parameters as a function of associated pt.

    Args:
        fit_objects: Fit objects whose parameters will be plotted.
        fit_type: Name of the type of fit.
        selected_analysis_options: Selected analysis options to be used for labeling.
        reference_data_path: Path to the reference data.
        output_info: Output information.
    """
    # Extract data from the reference dataset. We extract it here to save file IO.
    reference_data_path = reference_data_path.format(
        **dict(selected_analysis_options),
        #collision_energy = selected_analysis_options.collision_energy
    )
    try:
        reference_data: Mapping[str, Mapping[str, histogram.Histogram1D]] = {}
        with open(reference_data_path, "r") as f:
            y = yaml.yaml(modules_to_register = [histogram])
            reference_data = y.load(f)
    except IOError:
        # We will just work with the empty reference data.
        pass

    pt_assoc_label = labels.make_valid_latex_string(
        fr"{labels.track_pt_display_label()}\:({labels.momentum_units_label_gev()})"
    )
    prefix = f"{fit_type}_fit_parameter"
    parameters = [
        ParameterInfo(
            name = "v2_t",
            output_name = f"{prefix}_v2t",
            labels = plot_base.PlotLabels(title = r"Jet $v_{2}$", x_label = pt_assoc_label),
            plot_reference_data_func = _reference_v2_t_data,
        ),
        ParameterInfo(
            name = "v2_a",
            output_name = f"{prefix}_v2a",
            labels = plot_base.PlotLabels(title = r"Associated hadron $v_{2}$", x_label = pt_assoc_label),
            plot_reference_data_func = _reference_v2_a_data,
        ),
        ParameterInfo(
            name = "v3",
            output_name = f"{prefix}_v3",
            labels = plot_base.PlotLabels(
                title = r"$\tilde{v}_{3}^{a}\tilde{v}_{3}^{\text{jet}}$",
                x_label = pt_assoc_label
            ),
            # Don't plot the reference v3 data - it's not a meaningful comparison. See why the transform function
            # is also disabled below.
            #plot_reference_data_func = _reference_v3_data,
            # Don't transform the v3 data! It's difficult to interpret the v_3 data as v_3^2, as it's really the
            # product of v_3^{t} and v_3^{a}. The actual value is expected to be 0, and certainly could be negative.
            #transform_fit_data = _square_root_v3_2,
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
            plot_reference_data_func = _reference_v4_a_data,
        ),
    ]

    # Plot the signal parameters (but only if they exist).
    fit_obj = next(iter(fit_objects.values()))
    if "ns_amplitude" in fit_obj.fit_result.parameters:
        parameters.append(
            ParameterInfo(
                name = "ns_amplitude",
                output_name = f"{prefix}_ns_amplitude",
                labels = plot_base.PlotLabels(title = r"Near side gaussian amplitude", x_label = pt_assoc_label),
            )
        )
    if "ns_sigma" in fit_obj.fit_result.parameters:
        parameters.append(
            ParameterInfo(
                name = "ns_sigma",
                output_name = f"{prefix}_ns_sigma",
                labels = plot_base.PlotLabels(
                    title = r"Near side $\sigma$", x_label = pt_assoc_label, y_label = r"$\sigma_{\text{ns}}$",
                ),
            )
        )
    if "as_amplitude" in fit_obj.fit_result.parameters:
        parameters.append(
            ParameterInfo(
                name = "as_amplitude",
                output_name = f"{prefix}_as_amplitude",
                labels = plot_base.PlotLabels(title = r"Away side gaussian amplitude", x_label = pt_assoc_label),
            )
        )
    if "as_sigma" in fit_obj.fit_result.parameters:
        parameters.append(
            ParameterInfo(
                name = "as_sigma",
                output_name = f"{prefix}_as_sigma",
                labels = plot_base.PlotLabels(
                    title = r"Away side $\sigma$", x_label = pt_assoc_label, y_label = r"$\sigma_{\text{as}}$"
                ),
            )
        )
    if "BG" in fit_obj.fit_result.parameters:
        parameters.append(
            ParameterInfo(
                name = "BG",
                output_name = f"{prefix}_signal_background",
                labels = plot_base.PlotLabels(title = r"Effective RPF background", x_label = pt_assoc_label),
                additional_plot_options = {"logy": True, "legend_location": "upper right"},
            )
        )
    if "B" in fit_obj.fit_result.parameters:
        parameters.append(
            ParameterInfo(
                name = "B",
                output_name = f"{prefix}_background",
                labels = plot_base.PlotLabels(title = r"RPF background", x_label = pt_assoc_label),
                additional_plot_options = {"logy": True, "legend_location": "upper right"},
            )
        )

    for parameter in parameters:
        _plot_fit_parameter_vs_assoc_pt(
            fit_objects = fit_objects, parameter = parameter,
            reference_data = reference_data, selected_analysis_options = selected_analysis_options,
            output_info = output_info
        )

def _matrix_values(free_parameters: Sequence[str],
                   matrix: Dict[Tuple[str, str], float],
                   output_info: analysis_objects.PlottingOutputWrapper,
                   output_name: str) -> None:
    """ Plot the RP fit covariance matrix.

    Code substantially improved by using the information at
    https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html

    Args:
        free_parameters: Names of the free parameters used in the fit (which will be included
            in the plot).
        matrix: Matrix values to plot. Should be either the correlation matrix or the covariance
            matrix.
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
        "v3": "v_{3}^{2}", "v4_t": "v_{4}^{t}", "v4_a": "v_{4}^{a}",
        "B": r"\text{RPF Background}",
    }
    # Add Signal fit parameters.
    for ep in ["in_plane", "mid_plane", "out_of_plane"]:
        short_name = ep[:ep.find("_")]
        improved_labeling_map.update({
            f"{ep}_ns_amplitude": fr"A_{{NS}}^{{\text{{ {short_name} }}}}", f"{ep}_as_amplitude": fr"A_{{AS}}^{{\text{{ {short_name} }}}}",
            f"{ep}_ns_sigma": fr"\sigma_{{NS}}^{{\text{{ {short_name} }}}}", f"{ep}_as_sigma": fr"\sigma_{{AS}}^{{\text{{ {short_name} }}}}",
        })
    # Ensure that the strings are valid LaTeX
    improved_labeling_map = {k: labels.make_valid_latex_string(v) for k, v in improved_labeling_map.items()}
    # Fixed parameters aren't in the covariance matrix.
    number_of_parameters = len(free_parameters)

    # Put the values into a form that can actually be plotted. Note that this assumes a square matrix.
    matrix_values = np.array(list(matrix.values()))
    matrix_values = matrix_values.reshape(number_of_parameters, number_of_parameters)

    # Plot the matrix
    im = ax.imshow(matrix_values, cmap = "viridis")

    # Add the colorbar
    fig.colorbar(im, ax = ax)

    # Axis labeling
    parameter_labels = [b for a, b in list(matrix)[:number_of_parameters]]
    parameter_labels = [improved_labeling_map[l] for l in parameter_labels]
    # Show all axis ticks and then label them
    # The first step of settings the yticks is required according to the matplotlib docs
    ax.set_xticks(range(number_of_parameters))
    ax.set_yticks(range(number_of_parameters))
    # Also rotate the x-axis labels so they are all readable.
    ax.set_xticklabels(parameter_labels,
                       rotation = -30, rotation_mode = "anchor",
                       horizontalalignment = "right")
    ax.set_yticklabels(parameter_labels)

    # Turn spines off and create white grid
    for edge, spine in ax.spines.items():
        spine.set_visible(False)
    # Use minor ticks to put a white border between each value
    ax.set_xticks(np.arange(number_of_parameters + 1) - .5, minor = True)
    ax.set_yticks(np.arange(number_of_parameters + 1) - .5, minor = True)
    ax.grid(which = "minor", color = "w", linestyle = '-', linewidth = 3)
    ax.tick_params(which = "minor", bottom = False, left = False)
    # Have outward ticks just for this plot. They seem to look better here
    ax.tick_params(direction = "out")

    # Label values in each element
    threshold_for_changing_label_colors = im.norm(np.max(matrix_values)) / 2
    text_colors = ["white", "black"]
    for i in range(number_of_parameters):
        for j in range(number_of_parameters):
            color = text_colors[im.norm(matrix_values[i, j]) > threshold_for_changing_label_colors]
            im.axes.text(j, i, f"{matrix_values[i, j]:.1f}",
                         horizontalalignment="center",
                         verticalalignment="center",
                         color = color)

    # Final adjustments
    fig.tight_layout()
    # Save plot and cleanup
    plot_base.save_plot(output_info, fig, output_name)
    plt.close(fig)

def rpf_correlation_matrix(fit_result: reaction_plane_fit.base.FitResult,
                           output_info: analysis_objects.PlottingOutputWrapper,
                           identifier: str) -> None:
    """ Plot the RP fit correlation matrix.

    Args:
        fit_result: Reaction plane fit result.
        output_info: Output information.
        identifier: Identify the fit. Will be used to build the output name.
    Returns:
        None. The covariance matrix is saved.
    """
    _matrix_values(
        free_parameters = fit_result.free_parameters,
        matrix = fit_result.correlation_matrix,
        output_info = output_info,
        output_name = f"{identifier}_correlation_matrix",
    )

def rpf_covariance_matrix(fit_result: reaction_plane_fit.base.FitResult,
                          output_info: analysis_objects.PlottingOutputWrapper,
                          identifier: str) -> None:
    """ Plot the RP fit covariance matrix.

    Args:
        fit_result: Reaction plane fit result.
        output_info: Output information.
        identifier: Identify the fit. Will be used to build the output name.
    Returns:
        None. The covariance matrix is saved.
    """
    _matrix_values(
        free_parameters = fit_result.free_parameters,
        matrix = fit_result.covariance_matrix,
        output_info = output_info,
        output_name = f"{identifier}_covariance_matrix",
    )

def _plot_rp_fit_subtracted(ep_analyses: List[Tuple[Any, "correlations.Correlations"]], axes: matplotlib.axes.Axes) -> None:
    """ Plot the RP subtracted histograms on the given set of axes.

    Args:
        ep_analyses: Event plane dependent correlation analysis objects.
        axes: Axes on which the subtracted hists should be plotted. It must have an axis per component.
    Returns:
        None. The axes are modified in place.
    """
    for (key_index, analysis), ax in zip(ep_analyses, axes):
        hists = analysis.correlation_hists_delta_phi_subtracted
        h = hists.signal_dominated.hist

        # Plot the subtracted hist
        ax.errorbar(
            h.x, h.y, yerr = h.errors,
            label = f"Sub. {hists.signal_dominated.type.display_str()}", marker = "o", linestyle = "None",
            # Make the marker smaller to ensure that we can see the other uncertainties.
            markersize = 6,
        )

        # Plot the background uncertainty separately.
        background_error = h.metadata["RPF_background_errors"]
        ax.fill_between(
            h.x, h.y - background_error, h.y + background_error,
            label = "RP fit uncertainty",
            color = plot_base.AnalysisColors.fit,
        )

        # Label RP orientation
        ax.set_title(analysis.reaction_plane_orientation.display_str())

        # Add horizontal line at 0 for comparison
        ax.axhline(y = 0, color = "black", linestyle = "dashed", zorder = 1)

        # Plot the scale uncertainty if available
        if "mixed_event_scale_systematic" in h.metadata:
            ax.fill_between(
                h.x,
                h.metadata["mixed_event_scale_systematic"][0],
                h.metadata["mixed_event_scale_systematic"][1],
                label = "Correlated uncertainty", color = plot_base.AnalysisColors.systematic,
            )

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
        figsize = (3 * n_components, 6)
    )
    flat_axes = axes.flatten()

    # Plot the fits on the upper panels.
    _plot_rp_fit_subtracted(ep_analyses = ep_analyses, axes = flat_axes[:n_components])

    # Define upper panel labels.
    # Inclusive
    text = labels.make_valid_latex_string(inclusive_analysis.alice_label.display_str())
    _add_label_to_rpf_plot_axis(ax = flat_axes[0], label = text)
    # In-plane
    text = labels.track_pt_range_string(inclusive_analysis.track_pt)
    text += "\n" + labels.constituent_cuts()
    text += "\n" + labels.make_valid_latex_string(inclusive_analysis.leading_hadron_bias.display_str())
    _add_label_to_rpf_plot_axis(ax = flat_axes[1], label = text)
    # Mid-plane
    text = labels.system_label(
        energy = inclusive_analysis.collision_energy,
        system = inclusive_analysis.collision_system,
        activity = inclusive_analysis.event_activity
    )
    text += "\n" + labels.jet_pt_range_string(inclusive_analysis.jet_pt)
    text += "\n" + labels.jet_finding()
    # NOTE: Assumes that the signal and background dominated ranges are symmetric
    text += "\n" + (fr"Background: ${inclusive_analysis.background_dominated_eta_region.min:.1f}"
                    r"<|\Delta\eta|<"
                    fr"{inclusive_analysis.background_dominated_eta_region.max:.1f}$")
    text += "\n" + (r"Signal + Background: $|\Delta\eta|<"
                    fr"{inclusive_analysis.signal_dominated_eta_region.max:.1f}$")
    text += "\n" + r"Scale uncertainty: 5\%"
    _add_label_to_rpf_plot_axis(ax = flat_axes[2], label = text, size = 12.5)
    # Out-of-plane orientation
    flat_axes[3].legend(
        frameon = False, loc = "upper center", fontsize = 15
    )

    # Improve the viewable range for the upper panels.
    # Namely, we want to move it down such that the data doesn't overlap with the
    # labels, but oscillations in the data are still viewable.
    # These values are determine empirically.
    y_min, y_max = flat_axes[0].get_ylim()
    scale_factor = 1.12
    if inclusive_analysis.track_pt.min >= 4.:
        scale_factor = 1.35
    flat_axes[0].set_ylim(y_min, y_max * scale_factor)

    # Improve x axis presentation
    for ax in flat_axes:
        # Increase the frequency of major ticks to once every integer.
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base = 1.0))
        # Add axis labels
        ax.set_xlabel(labels.make_valid_latex_string(inclusive_analysis.correlation_hists_delta_phi.signal_dominated.axis.display_str()))
    flat_axes[0].set_ylabel(labels.make_valid_latex_string(labels.delta_phi_axis_label()))

    # Final adjustments
    fig.tight_layout()
    # We need to do some additional axis adjustment  after the tight layout, so we
    # perform that here.
    fig.subplots_adjust(
        # Reduce spacing between subplots
        hspace = 0, wspace = 0,
        # Reduce external spacing
        left = 0.10, right = 0.99,
        top = 0.96, bottom = 0.11,
    )
    # Save plot and cleanup
    plot_base.save_plot(output_info, fig,
                        f"{output_name}_rp_subtracted")
    plt.close(fig)

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
    if len(ep_analyses) != len(axes):
        raise TypeError(
            f"Number of axes is not equal to the number of EP analysis objects."
            f" # of analysis objects: {len(ep_analyses)}, # of axes: {len(axes)}"
        )

    for (key_index, analysis), ax in zip(ep_analyses, axes):
        # Setup
        # Get the relevant data. We define the background first so that it is plotted underneath.
        data: Dict[str, histogram.Histogram1D] = {
            "Background dominated":
            histogram.Histogram1D.from_existing_hist(analysis.correlation_hists_delta_phi.background_dominated),
            "Signal dominated":
            histogram.Histogram1D.from_existing_hist(analysis.correlation_hists_delta_phi.signal_dominated),
        }

        # Scale the data properly.
        for h in data.values():
            h *= analysis.correlation_scale_factor

        # Plot the data first to ensure that the colors are consistent with previous plots
        for label, hist in data.items():
            plotting_signal = "Signal" in label
            ax.errorbar(
                hist.x, hist.y, yerr = hist.errors, label = label,
                marker = "o", linestyle = "",
                color = plot_base.AnalysisColors.signal if plotting_signal else plot_base.AnalysisColors.background,
                # Open markers for background, closed for signal.
                fillstyle = "full" if plotting_signal else "none",
            )

        # Draw the fit
        fit_hist = analysis.fit_hist
        plot = ax.plot(fit_hist.x, fit_hist.y, label = "RP fit", color = plot_base.AnalysisColors.fit)
        # Plot the fit errors.
        # We need to ensure that the errors are copied so we don't accidentally modify the fit result.
        ax.fill_between(
            fit_hist.x,
            fit_hist.y - fit_hist.errors, fit_hist.y + fit_hist.errors,
            facecolor = plot[0].get_color(), alpha = 0.8
        )
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
    if len(ep_analyses) != len(axes):
        raise TypeError(
            f"Number of axes is not equal to the number of EP analysis objects."
            f" # of analysis objects: {len(ep_analyses)}, # of axes: {len(axes)}"
        )

    x = rp_fit.fit_result.x
    for (key_index, analysis), ax in zip(ep_analyses, axes):
        # Setup
        # Get the relevant data
        h: Union["correlations.DeltaPhiSignalDominated", "correlations.DeltaPhiBackgroundDominated"]
        if isinstance(analysis.fit_object, reaction_plane_fit.fit.BackgroundFitComponent):
            h = analysis.correlation_hists_delta_phi.background_dominated
        else:
            h = analysis.correlation_hists_delta_phi.signal_dominated
        hist = histogram.Histogram1D.from_existing_hist(h)

        # We create a histogram to represent the fit so that we can take advantage
        # of the error propagation in the Histogram1D object.
        fit_hist = histogram.Histogram1D(
            # Bin edges must be the same
            bin_edges = hist.bin_edges,
            y = analysis.fit_object.evaluate_fit(x = x),
            errors_squared = analysis.fit_object.fit_result.errors ** 2,
        )

        # Properly scale the hists
        hist *= analysis.correlation_scale_factor
        fit_hist *= analysis.correlation_scale_factor
        # NOTE: Residual = data - fit / fit, not just data-fit
        residual = (hist - fit_hist) / fit_hist

        # Plot the main values
        plot = ax.plot(x, residual.y, label = "Residual", color = plot_base.AnalysisColors.fit)
        # Plot the fit errors
        ax.fill_between(
            x, residual.y - residual.errors, residual.y + residual.errors,
            facecolor = plot[0].get_color(), alpha = 0.9,
        )

def _add_label_to_rpf_plot_axis(ax: matplotlib.axes.Axes, label: str,
                                **kwargs: Any) -> None:
    """ Add a label to the middle center of a axis for the RPF plot.

    This helper is limited, but useful since we often label at the same location.

    Args:
        ax: Axis to label.
        label: Label to be added to the axis.
        kwargs: Additional arguments to pass to text. They will override any defaults.
    Returns:
        None. The axis is modified in place.
    """
    # Default arguments
    text_kwargs = dict(
        x = 0.5, y = 0.97, s =label,
        transform = ax.transAxes, horizontalalignment = "center",
        verticalalignment = "top", multialignment = "left",
    )
    # Override with any additional passed kwargs
    text_kwargs.update(kwargs)

    # Draw the text
    ax.text(**text_kwargs)

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
    #n_components = len(rp_fit.components)
    n_components = 4
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
    # Inclusive
    text = labels.make_valid_latex_string(inclusive_analysis.alice_label.display_str())
    _add_label_to_rpf_plot_axis(ax = flat_axes[0], label = text)
    # In-plane
    text = labels.track_pt_range_string(inclusive_analysis.track_pt)
    text += "\n" + labels.constituent_cuts()
    text += "\n" + labels.make_valid_latex_string(inclusive_analysis.leading_hadron_bias.display_str())
    _add_label_to_rpf_plot_axis(ax = flat_axes[1], label = text)
    # Mid-plane
    text = labels.system_label(
        energy = inclusive_analysis.collision_energy,
        system = inclusive_analysis.collision_system,
        activity = inclusive_analysis.event_activity
    )
    text += "\n" + labels.jet_pt_range_string(inclusive_analysis.jet_pt)
    text += "\n" + labels.jet_finding()
    # NOTE: Assumes that the signal and background dominated ranges are symmetric
    text += "\n" + (fr"Background: ${inclusive_analysis.background_dominated_eta_region.min:.1f}"
                    r"<|\Delta\eta|<"
                    fr"{inclusive_analysis.background_dominated_eta_region.max:.1f}$")
    text += "\n" + (r"Signal + Background: $|\Delta\eta|<"
                    fr"{inclusive_analysis.signal_dominated_eta_region.max:.1f}$")
    text += "\n" + r"Scale uncertainty: 5\%"
    _add_label_to_rpf_plot_axis(ax = flat_axes[2], label = text, size = 12.5)
    # Out-of-plane orientation
    flat_axes[3].legend(
        frameon = False, loc = "upper center", fontsize = 15
    )
    # Use effective chi squared rather than the function minimum to ensure that we have
    # a valid way to characterize the fit even if we're using a log likelihood cost function.
    effective_chi_squared = rp_fit.fit_result.effective_chi_squared(rp_fit.cost_func)
    # This sanity check is only meaningful if not using a log likelihood fit.
    if rp_fit.use_log_likelihood is False:
        assert np.isclose(effective_chi_squared, rp_fit.fit_result.minimum_val)
    text = (
        r"\chi^{2}/\mathrm{NDF} = "
        f"{effective_chi_squared:.1f}/{rp_fit.fit_result.nDOF} = "
        f"{effective_chi_squared / rp_fit.fit_result.nDOF:.3f}"
    )
    _add_label_to_rpf_plot_axis(
        ax = flat_axes[3], label = labels.make_valid_latex_string(text),
        x = 0.5, y = 0.71, size = 15
    )
    # Improve the viewable range for the upper panels.
    # Namely, we want to move it down such that the data doesn't overlap with the
    # labels, but oscillations in the data are still viewable.
    # These values are determine empirically.
    y_min, y_max = flat_axes[0].get_ylim()
    scale_factor = 1.12
    if inclusive_analysis.track_pt.min >= 4.:
        scale_factor = 1.35
    flat_axes[0].set_ylim(y_min, y_max * scale_factor)

    # Define lower panel labels.
    for ax in flat_axes[n_components:]:
        # Increase the frequency of major ticks to once every integer.
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base = 1.0))
        # Add axis labels
        ax.set_xlabel(labels.make_valid_latex_string(inclusive_analysis.correlation_hists_delta_phi.signal_dominated.axis.display_str()))
    # Improve the viewable range for the lower panels.
    # This values are selected empirically. Note that we often select values slightly
    # less than the round value. This way, we are less likely to have labels overlap
    # with the upper label y-axis.
    y_min, y_max = -0.49, 0.49
    if inclusive_analysis.track_pt.min >= 4.:
        y_min, y_max = -0.95, 0.95
    if inclusive_analysis.track_pt.min >= 5.:
        y_min, y_max = -9, 9
    if inclusive_analysis.track_pt.min >= 6.:
        y_min, y_max = -19, 19
    flat_axes[n_components].set_ylim(y_min, y_max)

    # Specify shared y axis label
    # Delta phi correlations first
    # Extra 0.75 in because the aligned baseline isn't exactly the same (probably due to latex)
    flat_axes[0].set_ylabel(labels.delta_phi_axis_label(), labelpad = -2.75)
    # Then label the residual
    flat_axes[n_components].set_ylabel("data - fit / fit", labelpad = -2)
    # And then align them. Note that the padding set above is still meaningful,
    # but this will put them on the same baseline (which we want to move in a bit)
    fig.align_ylabels()

    # Final adjustments
    fig.tight_layout()
    # We need to do some additional axis adjustment  after the tight layout, so we
    # perform that here.
    fig.subplots_adjust(
        # Reduce spacing between subplots
        hspace = 0, wspace = 0,
        # Reduce external spacing
        left = 0.08, right = 0.99,
        top = 0.96, bottom = 0.11,
    )
    # Save plot and cleanup
    plot_base.save_plot(output_info, fig, output_name)
    plt.close(fig)

def delta_eta_fit(analysis: "correlations.Correlations") -> None:
    """ Plot the delta eta correlations with the fit. """
    # Setup
    fig, ax = plt.subplots(figsize = (8, 6))

    # Plot both the near side and the away side.
    for (attribute_name, correlation), (fit_attribute_name, fit_object) in \
            zip(analysis.correlation_hists_delta_eta, analysis.fit_objects_delta_eta):
        if attribute_name != fit_attribute_name:
            raise ValueError(
                "Issue extracting hist and pedestal fit object together."
                f"Correlation obj name: {attribute_name}, pedestal fit obj name: {fit_attribute_name}"
            )
        # Setup an individual hist
        h = correlation.hist
        label = correlation.type.display_str()

        # Determine the fit range so we can show it in the plot.
        # For example, -1.2 < h.x < -0.8
        negative_restricted_range = ((h.x < -1 * analysis.background_dominated_eta_region.min)
                                     & (h.x > -1 * analysis.background_dominated_eta_region.max))
        # For example, 0.8 < h.x < 1.2
        positive_restricted_range = ((h.x > analysis.background_dominated_eta_region.min)
                                     & (h.x < analysis.background_dominated_eta_region.max))
        restricted_range = negative_restricted_range | positive_restricted_range

        # First plot all of the data with opacity
        data_plot = ax.errorbar(
            h.x, h.y, yerr = h.errors, marker = "o", linestyle = "", alpha = 0.5,
        )
        # Then plot again without opacity highlighting the fit range.
        ax.errorbar(
            h.x[restricted_range], h.y[restricted_range], yerr = h.errors[restricted_range],
            label = label,
            marker = "o", linestyle = "", color = data_plot[0].get_color()
        )

        # Next, plot the pedestal following the same format
        # First plot the restricted values
        # We have to plot the fit data in two separate halves to prevent the lines
        # from being connected across the region where were don't fit.
        # Plot the left half
        pedestal_plot = ax.plot(
            h.x[negative_restricted_range],
            fit_object(h.x[negative_restricted_range], **fit_object.fit_result.values_at_minimum),
            label = "Pedestal",
        )
        # And then the right half
        ax.plot(
            h.x[positive_restricted_range],
            fit_object(h.x[positive_restricted_range], **fit_object.fit_result.values_at_minimum),
            color = pedestal_plot[0].get_color(),
        )
        # Then plot the errors over the entire range.
        fit_values = fit_object(h.x, **fit_object.fit_result.values_at_minimum)
        fit_errors = fit_object.calculate_errors(h.x)
        ax.fill_between(
            h.x,
            fit_values - fit_errors,
            fit_values + fit_errors,
            facecolor = pedestal_plot[0].get_color(), alpha = 0.7,
        )
        # Then plot over the entire range using a dashed line.
        ax.plot(
            h.x,
            fit_values,
            linestyle = "--",
            color = pedestal_plot[0].get_color(),
        )

        # Labels.
        ax.set_xlabel(labels.make_valid_latex_string(correlation.axis.display_str()))
        ax.set_ylabel(labels.make_valid_latex_string(labels.delta_eta_axis_label()))
        jet_pt_label = labels.jet_pt_range_string(analysis.jet_pt)
        track_pt_label = labels.track_pt_range_string(analysis.track_pt)
        ax.set_title(fr"Unsubtracted 1D ${correlation.axis.display_str()}$,"
                     f" {analysis.reaction_plane_orientation.display_str()} event plane orient.,"
                     f" {jet_pt_label}, {track_pt_label}")
        ax.legend(loc = "upper right")

        # Final adjustments
        fig.tight_layout()
        # Save plot and cleanup
        plot_base.save_plot(analysis.output_info, fig,
                            f"jetH_delta_eta_{analysis.identifier}_{attribute_name}_fit")
        # Reset for the next iteration of the loop
        ax.clear()

    # Final cleanup
    plt.close(fig)

def delta_eta_fit_subtracted(analysis: "correlations.Correlations") -> None:
    """ Plot the subtracted delta eta near-side and away-side. """
    # Setup
    fig, ax = plt.subplots(figsize = (8, 6))

    # Plot both the near side and the away side.
    attribute_names = ["near_side", "away_side"]
    for attribute_name in attribute_names:
        # Setup an individual hist
        correlation = getattr(analysis.correlation_hists_delta_eta_subtracted, attribute_name)
        h = correlation.hist

        # Plot the data
        ax.errorbar(
            h.x, h.y, yerr = h.errors,
            marker = "o", linestyle = "",
            label = f"Subtracted {correlation.type.display_str()}",
        )

        # Add horizontal line at 0 for comparison.
        ax.axhline(y = 0, color = "black", linestyle = "dashed", zorder = 1)

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
                            f"jetH_delta_eta_{analysis.identifier}_{attribute_name}_subtracted")
        # Reset for the next iteration of the loop
        ax.clear()

    # Final cleanup
    plt.close(fig)

def signal_dominated_with_background_function(analysis: "correlations.Correlations") -> None:
    """ Plot the signal dominated hist with the background function. """
    # Setup
    fig, ax = plt.subplots(figsize = (8, 6))

    # Plot signal and background dominated hists
    plot_correlations.plot_and_label_1d_signal_and_background_with_matplotlib_on_axis(
        ax = ax, jet_hadron = analysis, apply_correlation_scale_factor = True,
    )

    # Plot background function
    # First we retrieve the signal dominated histogram to get reference x values and bin edges.
    fit_hist = analysis.fit_hist
    background_plot = ax.plot(
        fit_hist.x, fit_hist.y, label = "Background function",
        color = plot_base.AnalysisColors.fit
    )
    ax.fill_between(
        fit_hist.x, fit_hist.y - fit_hist.errors, fit_hist.y + fit_hist.errors,
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
    # Plot the background uncertainty separately.
    fit_hist = analysis.fit_hist
    ax.fill_between(
        h.x,
        h.y - fit_hist.errors,
        h.y + fit_hist.errors,
        label = "RP background uncertainty",
        color = plot_base.AnalysisColors.fit,
    )
    if "mixed_event_scale_systematic" in h.metadata:
        ax.fill_between(
            h.x,
            h.metadata["mixed_event_scale_systematic"][0],
            h.metadata["mixed_event_scale_systematic"][1],
            label = "Correlated uncertainty", color = plot_base.AnalysisColors.systematic,
        )
    # Line for comparison
    ax.axhline(y = 0, color = "black", linestyle = "dashed", zorder = 1)

    # Labels.
    ax.set_xlabel(labels.make_valid_latex_string(hists.signal_dominated.axis.display_str()))
    ax.set_ylabel(labels.make_valid_latex_string(labels.delta_phi_axis_label()))
    jet_pt_label = labels.jet_pt_range_string(analysis.jet_pt)
    track_pt_label = labels.track_pt_range_string(analysis.track_pt)
    ax.set_title(fr"Subtracted 1D ${hists.signal_dominated.axis.display_str()}$,"
                 f" {analysis.reaction_plane_orientation.display_str()} event plane orient.,"
                 f" {jet_pt_label}, {track_pt_label}")
    ax.legend(loc = "upper right", frameon = False)

    # Final adjustments
    fig.tight_layout()
    # Save plot and cleanup
    plot_base.save_plot(analysis.output_info, fig,
                        f"jetH_delta_phi_{analysis.identifier}_subtracted")
    plt.close(fig)

