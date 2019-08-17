#!/usr/bin/env python3

""" Calculate the event plane resolution.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@yale.edu>, Yale University
"""

import logging
from dataclasses import dataclass
from functools import reduce
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from pachyderm import histogram
from pachyderm.utils import epsilon

from jet_hadron.base import analysis_config, analysis_manager, analysis_objects, params

logger = logging.getLogger(__name__)

@dataclass
class Detector:
    name: str
    data: histogram.Histogram1D

class EventPlaneResolution(analysis_objects.JetHBase):
    """ Calculate the event plane resolution.

    Args:
        n: Harmonic for calculating the resolution
        detector: Name of the main detector for which we are calculating the resolution.

    Attributes:
        harmonic: Harmonic for calculating the resolution
        main_detector_name: Name of the main detector for which we are calculating the resolution.
        other_detector_names: Names of the other detectors (not including the main detector).
        output_ranges: Centrality ranges over which the output resolutions should be calculated.
        main_detector: The main detector for which we are calculating the resolution.
        other_detectors: The two other detectors used for calculating the resolution. The
            order doesn't matter because cosine is an even function.
        resolutions: Resolutions which are calculated over the output_ranges. Keys are the output_ranges.
    """
    def __init__(self, harmonic: int, detector: str, *args: Any, **kwargs: Any):
        # Base class
        super().__init__(*args, **kwargs)
        # Configuration
        self.harmonic = harmonic
        self.main_detector_name = detector

        # Properties for determining the event plane resolution
        # Here we take all other detectors that aren't the main detector
        self.other_detector_names = [name for name in self.task_config.get("detectors") if name != self.main_detector_name]
        self.output_ranges = self.task_config.get("output_ranges", None)
        # Validation
        if self.output_ranges is None:
            # Default should be to process each centrality range.
            self.output_ranges = [params.SelectedRange(min, max)
                                  for min, max in zip(np.linspace(0, 100, 11)[:-1], np.linsapce(0, 100, 11)[1:])]

        # Objects that will be created during the calculation
        self.main_detector: Detector
        self.other_detectors: List[Detector]
        self.resolutions: Dict[params.SelectedRange, Tuple[float, float]] = {}

    def setup(self, input_hists: Optional[Dict[str, Any]]) -> None:
        """ Setup the histograms needed for the calculation.

        Args:
            input_hists: Input histograms. We attempt to pass them in so we don't have to open the same file repeatedly.
        Returns:
            None.
        """
        if input_hists is None:
            # Retrieve the histograms
            input_hists = histogram.get_histograms_in_list(self.input_filename, self.input_list_name)

        self.main_detector = Detector(
            self.main_detector_name,
            histogram.Histogram1D.from_existing_hist(
                input_hists["QA"]["eventPlaneRes"][self.main_detector_name][f"R{self.harmonic}"]
            )
        )
        self.other_detectors = [
            Detector(name, histogram.Histogram1D.from_existing_hist(
                input_hists["QA"]["eventPlaneRes"][name][f"R{self.harmonic}"]
            ))
            for name in self.other_detector_names if name != self.main_detector_name
        ]

    def _calculate_resolution(self) -> histogram.Histogram1D:
        """ Calculate the event plane resolution.

        Args:
            None.
        Returns:
            Resolutions and errors calculated for each centrality bin.
        """
        # R = sqrt(multiply all other detectors/main detector)
        # NOTE: main_detector actually contains the cosine of the difference of the other detectors. The terminology
        #       is a bit confusing. As an example, for VZERO resolution, sqrt((TPC_Pos * TPC_Neg)/VZERO)
        # NOTE: The output of reduce is an element-wise multiplication of all other_detector arrays. For
        #       the example of two other detectors, this is the same as just other[0].data * other[1].data
        resolution_squared = reduce(
            lambda x, y: x * y, [other.data for other in self.other_detectors]) / self.main_detector.data
        # We sometimes end up with a few very small negative values. This is a problem for sqrt, so we
        # explicitly set them to 0.
        resolution_squared.y[resolution_squared.y < 0] = 0
        resolutions = np.sqrt(resolution_squared.y)
        # Sometimes the resolution is 0, so we carefully divide to avoid NaNs
        errors = np.divide(
            1. / 2 * resolution_squared.errors, resolutions,
            out = np.zeros_like(resolution_squared.errors), where = resolutions != 0,
        )

        # Return the values in a histogram for convenience
        return histogram.Histogram1D(
            bin_edges = self.main_detector.data.bin_edges, y = resolutions, errors_squared = errors ** 2
        )

    def run(self, event_counts: histogram.Histogram1D) -> bool:
        """ Run the event plane resolution calculation.

        Args:
            event_counts: Number of events as a function of centrality.
        Returns:
            Calculated resolution and error for each output range provided.
        """
        # Validation
        if len(event_counts.y) != len(self.main_detector.data.y):
            raise ValueError("Event counts binning must match that of the resolution data.")

        # Calculate the resolutions for each bin.
        resolutions = self._calculate_resolution()

        # Calculate the resolutions of interest
        for r in self.output_ranges:
            lower_bin = resolutions.find_bin(r.min + epsilon)
            upper_bin = resolutions.find_bin(r.max - epsilon)
            # +1 because the upper bin edge is exclusive.
            selected_range = slice(lower_bin, upper_bin + 1)
            # Use data from the selected range, and average it, weighting by number
            # of events in each bin.
            res = np.sum(event_counts.y[selected_range] * resolutions.y[selected_range]) / np.sum(event_counts.y[selected_range])
            error = np.sum(event_counts.y[selected_range] * resolutions.errors[selected_range]) / np.sum(event_counts.y[selected_range])
            self.resolutions[r] = (res, error)

        return True

class EventPlaneResolutionManager(analysis_manager.Manager):
    """ Steer the calculation of the event plane resolution. """
    def __init__(self, config_filename: str, selected_analysis_options: params.SelectedAnalysisOptions, **kwargs: str):
        super().__init__(
            config_filename = config_filename, selected_analysis_options = selected_analysis_options,
            manager_task_name = "EventPlaneResolutionManager", **kwargs,
        )

        # Analysis task
        # Create the actual analysis objects.
        self.analyses: Mapping[Any, EventPlaneResolution]
        self.selected_iterables: Mapping[str, Sequence[Any]]
        (self.key_index, self.selected_iterables, self.analyses) = self.construct_from_configuration_file()

        # Event counts are shared between objects, so we store them here so we don't repeated do the same thing.
        self.event_counts: histogram.Histogram1D

    def construct_from_configuration_file(self) -> analysis_config.ConstructedObjects:
        """ Construct EventPlaneResolution objects based on iterables in a configuration file. """
        return analysis_config.construct_from_configuration_file(
            task_name = "EventPlaneResolution",
            config_filename = self.config_filename,
            selected_analysis_options = self.selected_analysis_options,
            additional_possible_iterables = {"detector": None, "harmonic": None},
            obj = EventPlaneResolution,
        )

    def _setup(self) -> bool:
        """ Setup the analysis tasks. """
        # Setup the analysis objects.
        input_hists: Dict[str, Any] = {}
        with self._progress_manager.counter(total = len(self.analyses),
                                            desc = "Setting up:",
                                            unit = "analysis objects") as setting_up:
            for key_index, analysis in analysis_config.iterate_with_selected_objects(self.analyses):
                # We are effectively caching the values here.
                if not input_hists:
                    input_hists = histogram.get_histograms_in_list(
                        filename = analysis.input_filename, list_name = analysis.input_list_name
                    )

                # Setup input histograms and projectors.
                analysis.setup(input_hists = input_hists)

                # Cache the event counts for convenience.
                if not hasattr(self, "event_counts"):
                    # Determine the number of events in each centrality bin.
                    event_counts = input_hists["Centrality_selected"]
                    # Rebin from bin width of 1 to bin width of 10. We don't want to rescale because
                    # we're interested in the raw counts, not normalized by the bin width.
                    event_counts.Rebin(10)
                    self.event_counts = histogram.Histogram1D.from_existing_hist(
                        event_counts
                    )

                # Keep track of progress
                setting_up.update()

        # Successfully setup the tasks
        return True

    def run(self) -> bool:
        """ Setup and run the actual analysis. """
        # Setup
        result = self._setup()
        if not result:
            raise RuntimeError("Setup failed")

        # Run the calculations
        with self._progress_manager.counter(total = len(self.analyses),
                                            desc = "Calculating:",
                                            unit = "EP resolutions") as running:
            for key_index, analysis in analysis_config.iterate_with_selected_objects(self.analyses):
                analysis.run(event_counts = self.event_counts)
                logger.debug(f"resolutions: {key_index} {analysis.resolutions}")

                running.update()

        # TODO: Output and plot the results...

        return True

def run_from_terminal() -> EventPlaneResolutionManager:
    """ Driver function for calculating the event plane resolution. """
    # Basic setup
    # Quiet down some pachyderm modules
    logging.getLogger("pachyderm.generic_config").setLevel(logging.INFO)
    logging.getLogger("pachyderm.yaml").setLevel(logging.INFO)
    logging.getLogger("pachyderm.histogram").setLevel(logging.INFO)

    # Setup and run the analysis
    manager: EventPlaneResolutionManager = analysis_manager.run_helper(
        manager_class = EventPlaneResolutionManager, task_name = "Event plane resolution",
    )

    # Return it for convenience.
    return manager

if __name__ == "__main__":
    run_from_terminal()

