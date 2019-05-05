#!/usr/bin/env python

"""

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import enlighten
import logging
import numpy as np
import pyjet
from scipy.spatial import KDTree
from typing import Any, List, Sequence, Tuple

from jet_hadron.event_gen import generator
from jet_hadron.event_gen import pythia6 as gen_pythia6

# Type helpers
PseudoJet: pyjet._libpyjet.PseudoJet = pyjet._libpyjet.PseudoJet

logger = logging.getLogger(__name__)

def fields_view(arr: np.ndarray, fields: Sequence[str]) -> np.ndarray:
    """ Provide a view of a selection of fields of a structured array.

    Code from: https://stackoverflow.com/a/21819324

    Args:
        arr: Array to select some fields from.
        fields: Fields which should be selected from the array.
    Returns:
        Array view which selects only those fields.
    """
    new_dtype = np.dtype({name: arr.dtype.fields[name] for name in fields})
    return np.ndarray(arr.shape, new_dtype, arr, 0, arr.strides)

def find_jets(input_array: np.ndarray, algorithm: str = "antikt", R: float = 0.4, min_jet_pt: float = 2) -> np.array:
    """ Perform jet finding use FastJet.

    Assumes that the input is of the form (E, p_x, p_y, p_z, ...)

    Args:
        input_array: Array containing the input particles.
        algorithm: Name of the jet-algorithm to use. Default: "antikt".
        R: Jet finding resolution parameter. Default: R = 0.2.
        min_jet_pt: Minimum jet pt. Default: 2.
    Returns:
        Array of PsuedoJets found by FastJet.
    """
    jet_def = pyjet.JetDefinition(algo = algorithm, R = R)
    cluster_sequence = pyjet.ClusterSequenceArea(
        inputs = input_array, jetdef = jet_def,
        areatype = "active_explicit_ghosts",
    )
    jets: np.array = np.array(cluster_sequence.inclusive_jets(ptmin = min_jet_pt))

    return jets

def max_constituent_pt(jet: PseudoJet) -> float:
    """ Find max constituent pt.

    Args:
        jet: The jet of interest.
    Returns:
        Maximum constituent pt.
    """
    max_pt = 0
    for c in jet.constituents():
        if c.pt > max_pt:
            max_pt = c.pt
    return max_pt

def match_jets(particle_level_jets: np.ndarray, detector_level_jets: np.ndarray, R: float) -> List[Tuple[int, int]]:
    """

    Args:

    Returns:
    """
    part_level_positions = np.array([(j.eta, j.phi) for j in particle_level_jets])
    det_level_positions = np.array([(j.eta, j.phi) for j in detector_level_jets])
    logger.debug(f"part_level_positions: {part_level_positions}, det_level_positions: {det_level_positions}")
    part_level_tree = KDTree(part_level_positions)
    det_level_tree = KDTree(det_level_positions)

    part_level_matches = part_level_tree.query_ball_tree(det_level_tree, r = 0.6 * R)
    det_level_matches = det_level_tree.query_ball_tree(part_level_tree, r = 0.6 * R)

    indices = []
    for i, part_match in enumerate(part_level_matches):
        for det_match in det_level_matches:
            for m in det_match:
                if m in part_match:
                    indices.append((i, m))

    logger.debug(f"part_level_matches: {part_level_matches}, det_level_matches: {det_level_matches}")
    logger.debug(f"indices: {indices}")

    return indices

class JetAnalysis:
    def __init__(self, generator: generator.Generator, jet_radius: float = 0.4):
        self.generator = generator
        self.jet_radius = jet_radius

        # Monitor the progress of the analysis.
        self._progress_manager = enlighten.get_manager()

    def setup(self) -> bool:
        """ Setup the generator and the outputs. """
        # Setup the tree / hists

        return True

    def _process_event(self, event: generator.Event) -> bool:
        """ """
        # TODO: Fill this in a bit
        # Jet finding
        find_jets(event)

        # Save the result
        ...

        return True

    def event_loop(self, n_events: int) -> bool:
        """

        """
        n_accepted = 0
        with self._progress_manager.counter(total = n_events,
                                            desc = "Generating",
                                            unit = "events") as progress:
            for event in self.generator(n_events = n_events):
                # Process event
                result = self._process_event(event)

                # Keep track of the number of events which actually passed all of the conditions.
                if result:
                    n_accepted += 1

                # Update the progress bar
                progress.update()

        print(f"n_accepted: {n_accepted}")

        # Disable enlighten so that it won't mess with any later steps (such as exploration with IPython).
        # Otherwise, IPython will act very strangely and is basically impossible to use.
        self._progress_manager.stop()

        return True

    def finalize(self) -> None:
        """ Finalize the analysis. """
        ...

class STARJetAnalysis(JetAnalysis):
    """

    """
    def __init__(self, *args: Any, **kwargs: Any):
        # Setup base class
        super().__init__(*args, **kwargs)

        # Efficiency hists
        self.efficiency_hists: List[np.ndarray]

        # Output
        self.response: np.ndarray = []

    def _apply_STAR_detector_effects(self, particles: np.ndarray) -> np.ndarray:
        """

        """
        detector_level = np.copy(particles)
        # STAR momentum resolution
        # Modeled by a gaussian width of parameters provided by Hanseul (from Nick originally).
        # Functionally, it goes as sigma_pt / pt = (0.005 + 0.0025 * pt)
        detector_level["pT"] = np.random.normal(
            detector_level["pT"],
            detector_level["pT"] * (0.005 + 0.0025 * detector_level["pT"]),
        )

        # TODO: STAR tracking efficiency

        return particles

    def _process_event(self, event: generator.Event) -> bool:
        """

        """
        # Acceptance cuts
        particle_level_particles = event[np.abs(event["eta"]) < 1]

        # Apply efficiency.
        detector_level_particles = self._apply_STAR_detector_effects(particle_level_particles)

        # Apply particle selections
        # Only keep detector level particles above 2 GeV
        detector_level_particles = detector_level_particles[detector_level_particles["pT"] > 2]

        # Jet finding
        particle_level_jets = find_jets(particle_level_particles)
        detector_level_jets = find_jets(detector_level_particles)

        # Apply jet cuts
        # Only keep detector level jets with an additional 4 GeV particle bias.
        max_const_pt = np.array([max_constituent_pt(j) for j in detector_level_jets])
        detector_level_jets = detector_level_jets[max_const_pt > 4.]

        # Match jets
        if len(particle_level_jets) == 0 or len(detector_level_jets) == 0:
            return False

        matches = match_jets(particle_level_jets, detector_level_jets, R = self.jet_radius)

        for (part_index, det_index) in matches:
            logger.debug(f"part_level: {particle_level_jets[part_index]}, det_level: {detector_level_jets[det_index]}")
            self.response.append((detector_level_jets[det_index].pt, particle_level_jets[part_index].pt))

        # Store data
        return True

    def finalize(self) -> None:
        """ Finalize the analysis. """
        self.response = np.array(self.response)
        #print(f"self.response: {self.response}")

        # Create histogram
        h, x_edges, y_edges = np.histogram2d(
            self.response[:, 0], self.response[:, 1],
            bins = (60, 60), range = ((0, 60), (0, 60))
        )

        # Plot
        import matplotlib
        import matplotlib.pyplot as plt
        # Fix normalization
        h[h == 0] = np.nan
        fig, ax = plt.subplots(figsize = (8, 6))
        resp = ax.imshow(
            h, extent = (x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
            interpolation = "nearest",
            aspect = "auto",
            origin = "lower",
            norm = matplotlib.colors.Normalize(vmin = np.nanmin(h), vmax = np.nanmax(h)),
        )
        fig.colorbar(resp)
        fig.tight_layout()
        fig.subplots_adjust(hspace = 0, wspace = 0)
        fig.savefig("response.pdf")

def run_jet_analysis() -> None:
    """ """
    # TODO: Dask!
    analysis = STARJetAnalysis(
        generator = gen_pythia6.Pythia6(
            sqrt_s = 200,
            random_seed = 10,
            pt_hard = (20, 30),
        ),
        jet_radius = 0.4,
    )

    # Setup and run the analysis
    res = analysis.setup()
    if not res:
        raise RuntimeError("Setup failed!")
    analysis.event_loop(n_events = 10000)
    analysis.finalize()

if __name__ == "__main__":
    run_jet_analysis()
