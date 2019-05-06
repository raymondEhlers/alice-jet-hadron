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

from pachyderm import histogram

from jet_hadron.base import analysis_objects
from jet_hadron.base import labels
from jet_hadron.base import params
from jet_hadron.plot import base as plot_base
from jet_hadron.event_gen import generator
from jet_hadron.event_gen import pythia6 as gen_pythia6

# Type helpers
PseudoJet: pyjet._libpyjet.PseudoJet = pyjet._libpyjet.PseudoJet

logger = logging.getLogger(__name__)

def fields_view(arr: np.ndarray, fields: Sequence[str]) -> np.ndarray:
    """ Provides a view of a selection of fields of a structured array.

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

def match_jets(particle_level_jets: np.ndarray, detector_level_jets: np.ndarray, matching_distance: float) -> List[Tuple[int, int]]:
    """ Match particle and detector level jets geometrically.

    Matching is performed via KDTrees. The particle level jet is required to match the
    detector level jet and vice-versa.

    Args:
        particle_level_jets: Particle level jets.
        detector_level_jets: Detector level jets.
        matching_distance: Maximum matching distance between jets. Default guidance is
            to use 0.6 * R.
    Returns:
        List of pairs of (particle level index, detector level index).
    """
    # Extract the jet locations from the PSeudoJets.
    part_level_positions = np.array([(j.eta, j.phi) for j in particle_level_jets])
    det_level_positions = np.array([(j.eta, j.phi) for j in detector_level_jets])
    logger.debug(f"part_level_positions: {part_level_positions}, det_level_positions: {det_level_positions}")

    # Construct the KDTress. They default to using the L^2 norm (ie our expected distance measure).
    part_level_tree = KDTree(part_level_positions)
    det_level_tree = KDTree(det_level_positions)
    # Perform the actual matching.
    part_level_matches = part_level_tree.query_ball_tree(det_level_tree, r = matching_distance)
    det_level_matches = det_level_tree.query_ball_tree(part_level_tree, r = matching_distance)

    # Only keep the closest match where the particle level jet points to the detector level
    # jet and vise-versa.
    indices = []
    for i, part_match in enumerate(part_level_matches):
        min_distance = 1000
        min_distance_index = -1
        for det_match in det_level_matches:
            for m in det_match:
                if m in part_match:
                    # Calculate the distance
                    dist = np.sqrt(
                        (part_level_positions[i][0] - det_level_positions[m][0]) ** 2
                        + (part_level_positions[i][1] - det_level_positions[m][1]) ** 2
                    )
                    logger.debug(f"part_level_index: {i}, Potential match: {m}, distance: {dist}")
                    if dist < min_distance:
                        logger.debug(f"Found match! Previous min_distance: {min_distance}")
                        min_distance = dist
                        min_distance_index = m

        if min_distance_index != -1:
            logger.debug(f"Final match: {i}, {min_distance_index}")
            indices.append((i, min_distance_index))

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
        """ Process each event.

        Args:
            event: Event level information and the input particles.
        Returns:
            True if the event was successfully processed.
        """
        # TODO: Fill this in a bit
        # Jet finding
        find_jets(event)

        # Save the result
        ...

        return True

    def event_loop(self, n_events: int) -> bool:
        """ Loop over the generator to generate events.

        Args:
            n_events: Number of events to generate.
        Returns:
            True if the generation was run successfully.
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

    Args:
        event_activity: Centrality selection for determining the momentum resolution.
    """
    def __init__(self, event_activity: params.EventActivity, *args: Any, **kwargs: Any):
        # Setup base class
        super().__init__(*args, **kwargs)

        # Efficiency hists
        self.event_activity = event_activity
        self.efficiency_hists: List[np.ndarray] = []
        self.efficiency_sampling: List[int] = []

        # Output
        self.response: np.ndarray = []

    def setup(self) -> bool:
        """

        """
        # Setup the efficiency histograms
        # Distribution to sample for determining which efficiency hist to use.
        # Based on year 14 ZDC rates (plot from Dan).
        # 0 = 0-33 kHz -> 1/9
        # 1 = 33-66 kHz -> 5/9
        # 2 = 66-100 kHz -> 3/9
        self.efficiency_sampling: List[int] = [
            0,
            1, 1, 1, 1, 1,
            2, 2, 2
        ]

        # Retrieve the efficiency histograms
        hists = histogram.get_histograms_in_file(filename = "inputData/AuAu/200/y14_efficiency_dca1.root")

        centrality_index_map = {
            # 0-10% in most analyses.
            # 0 = 0-5%, 1 = 5-10%
            params.EventActivity.central: list(range(0, 2)),
            # 20-50% in Joel's STAR analysis.
            # 4 = 20-25%, 5 = 25-30%, 6 = 30-35%, 7 = 35-40%, 8 = 40-45%, 9 = 45-50%
            params.EventActivity.semi_central: list(range(4, 10)),
        }

        for interation_rate_index in range(3):
            centrality_hists = []
            for centrality_index in centrality_index_map[self.event_activity]:
                h_root = hists[f"efficiency_lumi_{interation_rate_index}_cent_{centrality_index}"]
                _, _, h_temp = histogram.get_array_from_hist2D(h_root, set_zero_to_NaN = False)
                centrality_hists.append(h_temp)

            # Average the efficiency over the centrality bins.
            final_efficiency_hist = sum(centrality_hists) / len(centrality_hists)
            self.efficiency_hists.append(final_efficiency_hist)

            if interation_rate_index == 0:
                # h_root was set from the last iteration, so we take advantage of it.
                # Take advantage of the last iteration
                self.efficiency_pt_bin_edges = histogram.get_bin_edges_from_axis(h_root.GetXaxis())
                self.efficiency_eta_bin_edges = histogram.get_bin_edges_from_axis(h_root.GetYaxis())

        return True

    def _apply_STAR_detector_effects(self, particles: np.ndarray) -> np.ndarray:
        """ Apply the STAR detector effects.

        In particular, applies the momentum resolution and efficiency.

        Args:
            particles: Input particles.
        Returns:
            "Detector level" particles.
        """
        # Make a copy so that we don't propagate the changes to the particle level particles.
        detector_level = np.copy(particles)

        # STAR momentum resolution
        # Modeled by a gaussian width of parameters provided by Hanseul (from Nick originally).
        # Functionally, it goes as sigma_pt / pt = (0.005 + 0.0025 * pt)
        detector_level["pT"] = np.random.normal(
            detector_level["pT"],
            detector_level["pT"] * (0.005 + 0.0025 * detector_level["pT"]),
        )

        # STAR tracking efficiency.
        # Determine the expected efficiency based on the particle pt and eta, and then drop the particle
        # if the tracking efficiency is less than a flat random distribution.
        random_values = np.random.rand(len(particles))
        # Need to decide which efficiency histogram to use. See the definition of the efficiency_sampling
        # for further information on how and why these values are used.
        efficiency_hist = self.efficiency_hists[np.random.choice(self.efficiency_sampling)]

        # Determine the efficiency histogram indices.
        # This means that if we have the bin edges [0, 1, 2], and we pass value 1.5, it will return
        # index 2, but we want to return bin 1, so we subtract one from the result. For more, see
        # ``histogram.Histogram1D.find_bin(...)``.
        efficiency_pt_index = np.searchsorted(self.efficiency_pt_bin_edges, detector_level["pT"], side = "right") - 1
        efficiency_eta_index = np.searchsorted(self.efficiency_eta_bin_edges, detector_level["eta"], side = "right") - 1
        # Deal with particles outside of the efficiency histograms
        # We could have particles over 5 GeV, so we assume that the efficiency is flat above 5 GeV and
        # assign any of the particles above 5 GeV to the last efficiency bin.
        # - 1 because because the efficiency hist values are 0 indexed.
        efficiency_pt_index[efficiency_pt_index >= efficiency_hist.shape[0]] = efficiency_hist.shape[0] - 1
        # Since we have an eta cut at 1, we don't need to check for particles outside of this range.

        # Keep any particles where the efficiency is higher than the random value.
        keep_particles_mask = efficiency_hist[efficiency_pt_index, efficiency_eta_index] > random_values
        detector_level = detector_level[keep_particles_mask]

        return detector_level

    def _process_event(self, event: generator.Event) -> bool:
        """ Process the generated event.

        Args:
            event: Event level information and the input particles.
        Returns:
            True if the event was successfully processed.
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

        matches = match_jets(particle_level_jets, detector_level_jets, matching_distance = 0.6 * self.jet_radius)

        for (part_index, det_index) in matches:
            logger.debug(f"part_level: {particle_level_jets[part_index]}, det_level: {detector_level_jets[det_index]}")
            self.response.append((detector_level_jets[det_index].pt, particle_level_jets[part_index].pt))

        # Store data
        return True

    def finalize(self) -> None:
        """ Finalize the analysis. """
        self.response = np.array(self.response)
        print(f"number of jets: {len(self.response)}")
        #logger.debug(f"self.response: {self.response}")

        # Create histogram
        h, x_edges, y_edges = np.histogram2d(
            self.response[:, 0], self.response[:, 1],
            bins = (60, 60), range = ((0, 60), (0, 60))
        )

        # Plot
        output_info = analysis_objects.PlottingOutputWrapper(
            output_prefix = "output/AuAu/200",
            printing_extensions = ["pdf"],
        )
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

        ax.set_xlabel(labels.make_valid_latex_string(labels.jet_pt_display_label("det")))
        ax.set_ylabel(labels.make_valid_latex_string(labels.jet_pt_display_label("part")))
        fig.tight_layout()
        fig.subplots_adjust(hspace = 0, wspace = 0)
        plot_base.save_plot(output_info, fig, "response")

def run_jet_analysis() -> None:
    """ """
    # TODO: Dask!
    # Pt hard bins: [5, 10, 15, 20, 25, 35, 45]
    analysis = STARJetAnalysis(
        event_activity = params.EventActivity.semi_central,
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
