#!/usr/bin/env python

""" Plots related to the response matrix.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import logging

import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Dict

from pachyderm import histogram
from pachyderm import utils

from jet_hadron.base import analysis_objects
from jet_hadron.base import params
from jet_hadron.base.typing_helpers import Hist
from jet_hadron.plot import base as plot_base

logger = logging.getLogger(__name__)

Analyses = Dict[Any, analysis_objects.JetHBase]

def plot_response_matrix(hist: Hist):
    ...

def plot_response_spectra(plot_labels: plot_base.PlotLabels,
                          merged_analysis: analysis_objects.JetHBase,
                          pt_hard_analyses: Analyses,
                          hist_attribute_name: str):
    """ Plot 1D response spectra.

    Args:
        plot_labels:
        merged_analysis: Full merged together analysis object.
        pt_hard_analyses: Pt hard dependent analysis objects to be plotted.
        hist_attribute_name: Name of the attribute under which the histogram is stored.
    """
    # Setup
    fig, ax = plt.subplots(figsize = (8, 6))
    # NOTE: "husl" is also a good option.
    colors = iter(sns.color_palette(
        palette = "Blues_d", n_colors = len(pt_hard_analyses)
    ))
    plot_labels.apply_labels(ax)

    # First, we plot the merged analysis. This is the sum of the various pt hard bin contributions.
    merged_hist = utils.recursive_getattr(merged_analysis, hist_attribute_name)
    merged_hist = histogram.Histogram1D.from_existing_hist(merged_hist)
    ax.plot(
        merged_hist.x, merged_hist.y,
        label = "Merged",
    )

    # Now, we plot the pt hard dependent hists
    for (key_index, analysis), color in zip(pt_hard_analyses.items(), colors):
        # Determine the proper label.
        logger.debug(f"key_index: {key_index}")
        label = params.generate_pt_range_string(
            pt_bin = key_index.pt_hard_bin,
            lower_label = "",
            upper_label = "hard",
            only_show_lower_value_for_last_bin = True,
        )

        # Plot the histogram.
        hist = utils.recursive_getattr(analysis, hist_attribute_name)
        h = histogram.Histogram1D.from_existing_hist(hist)
        ax.plot(
            h.x, h.y,
            label = label,
            color = color,
        )

    ax.legend(loc = "best")

    plot_base.save_plot(merged_analysis, fig, "TEMP")
