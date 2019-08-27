#!/usr/bin/env python

""" General plotting module.

Contains brief plotting functions which don't belong elsewhere.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import logging
import matplotlib.pyplot as plt
from typing import Any, Dict, Iterator, Tuple, TYPE_CHECKING

import numpy as np

from jet_hadron.base import analysis_objects
from jet_hadron.plot import base as plot_base

if TYPE_CHECKING:
    from jet_hadron.analysis import event_plane_resolution  # noqa: F401

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
            (1 + 2 * coefficient * np.cos(n * theta + rotation)) * np.sin(theta + rotation),
            (1 + 2 * coefficient * np.cos(n * theta + rotation)) * np.cos(theta + rotation),
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
