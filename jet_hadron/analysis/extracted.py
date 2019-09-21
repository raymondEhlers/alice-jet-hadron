#!/usr/bin/env python

""" Handle extracted widths and yields.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple, Union

import numpy as np
import pachyderm.fit
from pachyderm import histogram

from jet_hadron.base import analysis_objects
from jet_hadron.base import params

@dataclass
class ExtractedYield:
    value: analysis_objects.ExtractedObservable
    central_value: float
    extraction_limit: float

    @property
    def extraction_range(self) -> params.SelectedRange:
        """ Helper to retrieve the extraction range. """
        return params.SelectedRange(
            min = self.central_value - self.extraction_limit,
            max = self.central_value + self.extraction_limit
        )

@dataclass
class ExtractedYieldRatio(ExtractedYield):
    contributors: List[params.ReactionPlaneOrientation]

@dataclass
class ExtractedYieldDifference(ExtractedYield):
    contributors: List[params.ReactionPlaneOrientation]

@dataclass
class ExtractedWidth:
    fit_object: pachyderm.fit.Fit
    fit_args: pachyderm.fit.T_FitArguments
    metadata: Dict[str, Any] = field(default_factory = dict)

    @property
    def fit_result(self) -> pachyderm.fit.FitResult:
        """ Helper to retrieve the overall fit result. """
        return self.fit_object.fit_result

    @fit_result.setter
    def fit_result(self, result: pachyderm.fit.FitResult) -> None:
        """ Helper to simplify setting the fit result. """
        self.fit_object.fit_result = result

    @property
    def width(self) -> float:
        """ Helper to retrieve the width and the error on the width. """
        return self.fit_result.values_at_minimum["width"]

    @property
    def mean(self) -> float:
        """ Helper to retrieve the mean (should be fixed). """
        return self.fit_result.values_at_minimum["mean"]

@dataclass
class JEWELPredictions:
    """ Contains JEWEL predictions with w/ and w/out recoil. """
    keep_recoils: histogram.Histogram1D
    no_recoils: histogram.Histogram1D

    def __iter__(self) -> Iterator[Tuple[str, histogram.Histogram1D]]:
        # NOTE: dataclasses.asdict(...) is recursive, so it's far
        #       too aggressive for our purposes!
        for k, v in vars(self).items():
            yield k, v

    @classmethod
    def display_name(cls, name: str) -> str:
        name_map = {
            "keep_recoils": "Inc. recoils",
            "no_recoils": "No recoils",
        }
        return name_map[name]

def _load_JEWEL_predictions_from_file(path: Path) -> histogram.Histogram1D:
    """ Load JEWEL predictions from file.

    The file format is:

    ```
    pt_bin_center pt_bin_center_uncertainty value value_uncertainty
    ```

    Args:
        path: Path to the file containing the JEWEL predictions.
    Returns:
        The data loaded into a histogram.
    """
    bin_edges = np.array([0.5, 1, 1.5, 2, 3, 4, 5, 6, 10])
    bin_centers_of_interest = bin_edges[:-1] + (bin_edges[1:] - bin_edges[:-1]) / 2
    values = []
    errors = []
    with open(path, "r") as f:
        for line in f:
            # Remove remaining newline.
            line = line.rstrip("\n")
            # Convert to floats
            converted_values = [float(v) for v in line.split("\t")]
            bin_center, _, value, value_uncertainty = converted_values
            # Only store if the bin center is of interest.
            for c in bin_centers_of_interest:
                if np.isclose(bin_center, c):
                    break
            else:
                continue
            values.append(value)
            errors.append(value_uncertainty)

    return histogram.Histogram1D(
        bin_edges = bin_edges, y = np.array(values), errors_squared = np.array(errors) ** 2
    )

def load_JEWEL_predictions(collision_system: params.CollisionSystem, collision_energy: params.CollisionEnergy,
                           event_activity: params.EventActivity, observable: str, side: str,
                           input_data_path: Union[str, Path] = "inputData") -> JEWELPredictions:
    """ Load the requested JEWEL predictions.

    Returns the predictions both with and without recoils.

    Args:
        collision_system: Collision system.
        collision_energy: Collision energy.
        event_activity: Event activity.
        observable: Type of observable to retrieve.
        side: Near side or away side. Accepts "NS" or "AS".
        input_data_path: Path to the input data. Default: "inputData".
    Returns:
        Retrieved JEWEL data for both with and without recoils.
    """
    # Validation
    if side != "NS" and side != "AS":
        raise ValueError("Invalid side of {side} passed. Must be 'NS' or 'AS'.")

    base_path = Path(input_data_path) / str(collision_system) / str(collision_energy) / "JEWEL" / str(event_activity)
    predictions = {}
    for recoils in ["KeepRecoils", "NoRecoils"]:
        full_path = base_path / recoils / f"{observable}_{side}_jetPt_20_40.dat"
        predictions[recoils] = _load_JEWEL_predictions_from_file(full_path)

    return JEWELPredictions(
        keep_recoils = predictions["KeepRecoils"],
        no_recoils = predictions["NoRecoils"],
    )

