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
from jet_hadron.base.typing_helpers import Hist

logger = logging.getLogger(__name__)

Analyses = Dict[Any, analysis_objects.JetHBase]

def plot_response_matrix(hist: Hist):
    ...

def plot_response_spectra(merged_analysis: analysis_objects.JetHBase,
                          pt_hard_analyses: Analyses,
                          hist_attribute_name: str):
    """ Plot 1D response spectra.

    Args:
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

    for (_, analysis), color in zip(pt_hard_analyses.items(), colors):

        # TODO: Determine the label...
        label = ""

        # Plot the histogram.
        hist = utils.recursive_getattr(analysis, hist_attribute_name)
        h = histogram.Histogram1D(hist)
        ax.plot(
            h.bin_centers, h.y,
            label = label,
            color = color,
        )

    ax.legend(loc = "best")
