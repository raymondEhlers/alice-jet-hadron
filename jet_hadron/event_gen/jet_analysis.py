#!/usr/bin/env python

"""

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import enlighten
import numpy as np
import pyjet
from typing import Any, List, Tuple

from jet_hadron.event_gen import generator
from jet_hadron.event_gen import pythia6 as gen_pythia6

PseudoJet: pyjet._libpyjet.PseudoJet = pyjet._libpyjet.PseudoJet

def find_jets(input_array: np.ndarray, algorithm: str = "antikt", R: float = 0.4) -> np.array:
    """ Perform jet finding use FastJet.

    Assumes that the input is of the form (E, p_x, p_y, p_z, ...)

    Args:
        input_array: Array containing the input particles.
        algorithm: Name of the jet-algorithm to use. Default: "antikt"
        R: Jet finding resolution parameter. Default: R = 0.2.
    Returns:
        Array of PsuedoJets found by FastJet.
    """
    jet_def = pyjet.JetDefinition(algo = algorithm, R = R)
    cluster_sequence = pyjet.ClusterSequenceArea(
        inputs = input_array, jetdef = jet_def,
        areatype = "active_explicit_ghosts",
        ep = True,
    )
    jets: np.array = np.array(cluster_sequence.inclusive_jets())

    return jets

def max_jet_constituent(jet: PseudoJet) -> float:
    """

    """
    max_pt = 0
    print(f"constituents: {jet.constituents()}")
    for c in jet.constituents():
        if c.pt > max_pt:
            max_pt = c.pt
    return max_pt

def match_jets(particle_level_jets: np.ndarray, detector_level_jets: np.ndarray) -> List[Tuple[int, int]]:
    """

    """
    ...

class JetAnalysis:
    def __init__(self, generator: generator.Generator):
        self.generator = generator

        # Monitor the progress of the analysis.
        self._progress_manager = enlighten.get_manager()

    def setup(self) -> bool:
        """ Setup the generator and the outputs. """
        # Setup the tree / hists

        return True

    def _process_event(self, event: generator.Event) -> None:
        """ """
        # Jet finding
        #jets = find_jets(event)
        find_jets(event)

        # Save the result
        ...

    def event_loop(self, n_events: int) -> bool:
        """

        """
        with self._progress_manager.counter(total = n_events,
                                            desc = "Generating",
                                            unit = "events") as progress:
            for event in self.generator(n_events = n_events):
                # Process event
                self._process_event(event)

                # Apply the efficiency
                ...

                # Update the progress bar
                progress.update()

        # Disable enlighten so that it won't mess with any later steps (such as exploration with IPython).
        # Otherwise, IPython will act very strangely and is basically impossible to use.
        self._progress_manager.stop()

        return True

class STARJetAnalysis(JetAnalysis):
    """

    """
    def __init__(self, *args: Any, **kwargs: Any):
        # Setup base class
        super().__init__(*args, **kwargs)

        # Efficiency hists
        self.efficiency_hists: List[np.ndarray]

    def _apply_STAR_detector_effects(self, particles: np.ndarray) -> np.ndarray:
        """

        """
        print(f"particles dtype: {particles.dtype}")
        # STAR momentum resolution
        # Modeled by a gaussian width of parameters provided by Hanseul (from Nick originally).
        # Functionally, it goes as sigma_pt / pt = (0.005 + 0.0025 * pt)
        #pt = np.sqrt(detector_level_particles["px"] ** 2 + detector_level_particles["py"] ** 2)
        #particles["pt"] = np.random.normal(particles["pt"], particles["pt"] * (0.005 + 0.0025 * particles["pt"]))

        # STAR tracking efficiency

        return particles

    def _process_event(self, event: generator.Event) -> None:
        """

        """
        # Acceptance cuts
        particle_level_particles = event[np.abs(event["eta"]) < 1]

        # Apply efficiency.
        detector_level_particles = self._apply_STAR_detector_effects(particle_level_particles)

        # Apply particle selections
        # Only keep detector level particles above 2 GeV
        pt = np.sqrt(detector_level_particles["px"] ** 2 + detector_level_particles["py"] ** 2)
        detector_level_particles = detector_level_particles[pt > 2]

        # Jet finding
        particle_level_jets = find_jets(particle_level_particles)
        detector_level_jets = find_jets(detector_level_particles)

        # Apply jet cuts
        # Only keep detector level jets with an additional 4 GeV particle bias.
        max_jet_consts = np.array([max_jet_constituent(j) for j in detector_level_jets])
        detector_level_jets = detector_level_jets[max_jet_consts > 4.]

        # Match jets
        match_jets(particle_level_jets, detector_level_jets)

        # Store out

def run_jet_analysis() -> None:
    analysis = STARJetAnalysis(
        generator = gen_pythia6.Pythia6(
            sqrt_s = 200,
            random_seed = 10,
            pt_hard = (10, 20),
        ),
    )
    res = analysis.setup()
    if not res:
        raise RuntimeError("Setup failed!")

    analysis.event_loop(n_events = 5)

if __name__ == "__main__":
    run_jet_analysis()
