#!/usr/bin/env python

"""

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import enlighten
import logging
import numpy as np
import os
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
DTYPE_PT = np.dtype([("pT", np.float64), ("eta", np.float64), ("phi", np.float64), ("m", np.float64)])
# Stores particle and detector level jets.
DTYPE_JETS = np.dtype(
    [(f"{label}_{name}", dtype) for label in ["part", "det"] for name, dtype in DTYPE_PT.descr]
)
DTYPE_EVENT_PROPERTIES = np.dtype([("cross_section", np.float64), ("pt_hard", np.float64)])

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

def save_tree(output_info: analysis_objects.PlottingOutputWrapper,
              arr: np.ndarray, output_name: str) -> str:
    """ Write the tree stored in a numpy array to a file.

    Args:
        output_index: Output information.
        arr: Tree stored in an array to write out.
        output_name: Filename under which the tree should be saved, but without the file extension.
    Returns:
        The filename under which the tree was written.
    """
    # Determine filename
    if not output_name.endswith(".npy"):
        output_name += ".npy"
    full_path = os.path.join(output_info.output_prefix, output_name)

    # Write
    with open(full_path, "wb") as f:
        np.save(f, arr)

    return full_path

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
    """ Direct the generation and processing of events from a generator through jet finding to final processing.

    Args:
        generator: Generator object.
        identifier: String by which the analysis should be identified.
        jet_radius: Jet finding parameter.

    Attributes:
        generator: Generator object.
        identifier: String by which the analysis should be identified.
        jet_radius: Jet finding parameter.
        jets: Array of jets found from particles from the generator.
        _progress_manager: Used to display processing progress.
    """
    def __init__(self, generator: generator.Generator, identifier: str, jet_radius: float = 0.4):
        self.generator = generator
        self.identifier = identifier
        self.jet_radius = jet_radius

        # Output variables
        # They start out as lists, but will be converted to numpy arrays when finalized.
        self.jets: np.ndarray = []
        self.events: np.ndarray = []

        # Output info
        self.output_info = analysis_objects.PlottingOutputWrapper(
            output_prefix = ".",
            printing_extensions = ["pdf"],
        )

        # Monitor the progress of the analysis.
        self._progress_manager = enlighten.get_manager()

    def setup(self) -> bool:
        """ Setup the generator and the outputs.

        If use of TTress is desired, a separate tree for the event level properties and a flat tree
        for all of the jets is probably recommended. Precisely what will be done depends on the user requirements.
        """
        ...

        return True

    def _process_event(self, event: generator.Event) -> bool:
        """ Process each event.

        Args:
            event: Event level information and the input particles.
        Returns:
            True if the event was successfully processed.
        """
        # Setup
        event_properties, event_particles = event

        # Jet finding
        particle_level_jets = find_jets(event_particles)

        # Store the jet properties
        for jet in particle_level_jets:
            self.jets.append(
                np.array((jet.pt, jet.eta, jet.phi, jet.mass), dtype = DTYPE_PT)
            )

        # Store the event properties
        self.events.append(
            np.array((event_properties.cross_section, event_properties.pt_hard), dtype = DTYPE_EVENT_PROPERTIES)
        )

        return True

    def event_loop(self, n_events: int) -> int:
        """ Loop over the generator to generate events.

        Args:
            n_events: Number of events to generate.
        Returns:
            Number of accepted events.
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

        return n_accepted

    def finalize(self, n_events_accepted: int) -> None:
        """ Finalize the analysis. """
        # Finally, convert to a proper numpy array. It's only converted here because it's not efficient to expand
        # existing numpy arrays.
        self.jets = np.array(self.jets, dtype = DTYPE_JETS)
        self.events = np.array(self.events, dtype = DTYPE_EVENT_PROPERTIES)

        # And save out the tree so we don't have to calculate it again later.
        self.save_tree(arr = self.jets, output_name = self.identifier + "_jets")

    def save_tree(self, *args: Any, **kwargs: Any) -> str:
        """ Helper for saving a tree to file.

        Args:
            output_index: Output information.
            arr: Tree stored in an array to write out.
            output_name: Filename under which the tree should be saved, but without the file extension.
        Returns:
            The filename under which the tree was written.
        """
        return save_tree(self.output_info, *args, **kwargs)

class STARJetAnalysis(JetAnalysis):
    """ Find and analyze jets using STAR Au--Au data taking conditions.

    This allows us to simulate.

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
        self.output_info = analysis_objects.PlottingOutputWrapper(
            output_prefix = "output/AuAu/200",
            printing_extensions = ["pdf"],
        )

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
        # Setup
        event_properties, event_particles = event

        # Acceptance cuts
        particle_level_particles = event_particles[np.abs(event_particles["eta"]) < 1]

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

        # Store data
        for (part_index, det_index) in matches:
            logger.debug(f"part_level: {particle_level_jets[part_index]}, det_level: {detector_level_jets[det_index]}")
            self.jets.append(
                np.array(
                    (
                        particle_level_jets[part_index].pt,
                        particle_level_jets[part_index].eta,
                        particle_level_jets[part_index].phi,
                        particle_level_jets[part_index].mass,
                        detector_level_jets[det_index].pt,
                        detector_level_jets[det_index].eta,
                        detector_level_jets[det_index].phi,
                        detector_level_jets[det_index].mass,
                    ),
                    dtype = DTYPE_JETS,
                )
            )
        # Store event properties
        self.events.append(
            np.array((event_properties.cross_section, event_properties.pt_hard), dtype = DTYPE_EVENT_PROPERTIES)
        )

        return True

    def finalize(self, n_events_accepted: int) -> None:
        """ Finalize the analysis. """
        # Sanity check
        assert len(self.events) == n_events_accepted

        print(f"number of accepted events: {len(self.events)}, jets in tree: {len(self.jets)}")
        # Finally, convert to a proper numpy array. It's only converted here because it's not efficient to expand
        # existing numpy arrays.
        self.jets = np.array(self.jets, dtype = DTYPE_JETS)
        self.events = np.array(self.events, dtype = DTYPE_EVENT_PROPERTIES)

        # And save out the tree so we don't have to calculate it again later.
        self.save_tree(arr = self.jets, output_name = self.identifier + "_jets")
        self.save_tree(arr = self.events, output_name = self.identifier + "_event_properties")

        # Create histogram
        h, x_edges, y_edges = np.histogram2d(
            self.jets["det_pT"], self.jets["part_pT"],
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

        ax.set_xlabel(labels.make_valid_latex_string(labels.jet_pt_display_label("det")))
        ax.set_ylabel(labels.make_valid_latex_string(labels.jet_pt_display_label("part")))
        fig.tight_layout()
        fig.subplots_adjust(hspace = 0, wspace = 0)
        plot_base.save_plot(
            self.output_info, fig,
            f"response_{self.identifier}"
        )

def run_jet_analysis() -> None:
    """ Run the jet analysis. """
    # Pt hard bins: [5, 10, 15, 20, 25, 35, 45]
    analyses = []
    pt_hard_bins = [5, 10, 15, 20, 25, 35, 45]
    for low_bin, high_bin in zip(pt_hard_bins[:-1], pt_hard_bins[1:]):
        analysis = STARJetAnalysis(
            event_activity = params.EventActivity.semi_central,
            generator = gen_pythia6.Pythia6(
                sqrt_s = 200,
                random_seed = 10,
                pt_hard = (low_bin, high_bin),
            ),
            identifier = f"STAR_jets_ptHard_{low_bin}_{high_bin}",
            jet_radius = 0.4,
        )
        analyses.append(analysis)

    # Setup and run the analysis
    for analysis in analyses:
        res = analysis.setup()
        if not res:
            raise RuntimeError("Setup failed!")

    for analysis in analyses:
        n_events_accepted = analysis.event_loop(n_events = 200)
        analysis.finalize(n_events_accepted)

if __name__ == "__main__":
    run_jet_analysis()
