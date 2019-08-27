#!/usr/bin/env python3

""" Collection of miscellaneous figures for my thesis.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import logging

import IPython
import numpy as np

from jet_hadron.base import analysis_manager, params
from jet_hadron.plot import general as plot_general

logger = logging.getLogger(__name__)

class FiguresManager(analysis_manager.Manager):
    """ Manages creating standalone or miscellaneous figures for my thesis or papers.

    The manager is a bit of overkill for this purpose, but it's a convenient way to access
    the YAML configuration.

    Args:
        config_filename: Path to the configuration filename.
        selected_analysis_options: Selected analysis options.
        manager_task_name: Name of the analysis manager task name in the config.
    """
    def __init__(self, config_filename: str, selected_analysis_options: params.SelectedAnalysisOptions, **kwargs: str):
        super().__init__(
            config_filename = config_filename, selected_analysis_options = selected_analysis_options,
            manager_task_name = "FiguresManager", **kwargs,
        )

    def run(self) -> bool:
        """ Create the figures. """
        steps = 1
        with self._progress_manager.counter(total = steps,
                                            desc = "Overall processing progress:",
                                            unit = "") as overall_progress:
            # Pure v_n harmonics example plot
            if self.task_config["tasks"]["vn_harmonics"]:
                theta = np.linspace(0, 2 * np.pi, 100)
                # Define the harmonics settings. We intentionally select the same coefficients
                # despite this not being physically relevant for clarity in seeing the shapes.
                # Format of harmonic: coefficient (ie. magnitude)
                harmonics = self.task_config["vn_harmonics"]
                plot_general.vn_harmonics(
                    theta = theta, harmonics = harmonics,
                    output_name = "vn_harmonics", output_info = self.output_info,
                    rotate_vn_relative_to_second_order = True,
                )

            # Also update regardless of whether we make the plot to ensure that the count is correct.
            overall_progress.update()

        return True

def run_from_terminal() -> FiguresManager:
    # Basic setup
    # Quiet down pachyderm
    logging.getLogger("pachyderm").setLevel(logging.INFO)

    # Setup and run the analysis
    manager: FiguresManager = analysis_manager.run_helper(
        manager_class = FiguresManager, task_name = "Figures",
    )

    # Quiet down IPython.
    logging.getLogger("parso").setLevel(logging.INFO)
    # Embed IPython to allow for some additional exploration
    IPython.embed()

    # Return the manager for convenience.
    return manager

if __name__ == "__main__":
    run_from_terminal()

