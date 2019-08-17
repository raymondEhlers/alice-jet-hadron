#!/usr/bin/env python

""" General plotting module.

Contains brief plotting functions which don't belong elsewhere.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import logging
import matplotlib.pyplot as plt
from typing import Any, Iterator, Tuple, TYPE_CHECKING

import numpy as np

from jet_hadron.base import analysis_objects
from jet_hadron.plot import base as plot_base

if TYPE_CHECKING:
    from jet_hadron.analysis import event_plane_resolution  # noqa: F401

logger = logging.getLogger(__name__)

def event_plane_resolution_harmonic(analyses_iter: Iterator[Tuple[Any, "event_plane_resolution.EventPlaneResolution"]],
                                    harmonic: int, output_info: analysis_objects.PlottingOutputWrapper) -> None:
    """ Plot the event plane resolution for the provided detectors. """
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
