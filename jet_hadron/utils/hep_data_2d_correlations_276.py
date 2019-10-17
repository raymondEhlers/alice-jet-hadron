#!/usr/bin/env python3

""" Create HEPdata file from a set of specified histograms.

It would be straightforward to generalize this into a general conversion function,
but it's not worth the effort at the moment.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import logging
from pathlib import Path
from typing import List, Tuple

import coloredlogs

from pachyderm import histogram

logger = logging.getLogger(__name__)

# Map the short names to full names for convenience.
_ep_orientation_full_name_map = {
    "all": "all angles",
    "in": "in-plane",
    "mid": "mid-plane",
    "out": "out-of-plane",
}
# This location depends on the layout of the public note.
_figure_location_map = {
    "all": 2,
    "in": 3,
    "mid": 4,
    "out": 5,
}

def convert_histograms_to_hepdata(input_path: Path, output_path: Path) -> bool:
    """ Convert 2D correlations into HEPdata YAML files.

    Args:
        input_path: Path to the files containing the correlations to be converted.
        output_path: Path where the HEPdata files should be written.
    Returns:
        True if successful.
    """
    # Move the import inside the function so that we don't have to explicitly depend on it.
    import hepdata_lib as hepdata
    # NOTE: hist_names are the same as file names!
    ep_orientations, assoc_pt_bin_edges, hist_names = define_2d_correlations_hists()

    logger.debug("Writing the correlations to the HEPdata format.")
    submission = hepdata.Submission()
    for ep, pt_1, pt_2, hist_name in hist_names:
        logger.debug(f"Converting ep: {ep}, pt: {pt_1}-{pt_2}, hist_name: {hist_name}")
        filename = input_path / f"{hist_name}.root"
        # There's only one histogram, so we just take it.
        h = histogram.get_histograms_in_file(str(filename))[hist_name]
        correlation = hepdata.root_utils.get_hist_2d_points(h)
        phi = hepdata.Variable(r"$\varphi$", is_binned = False)
        phi.values = correlation["x"]
        eta = hepdata.Variable(r"$\eta$", is_binned = False)
        eta.values = correlation["y"]
        corr = hepdata.Variable(r"(1/N_{\textrm{trig}})\textrm{d}^{2}N/\textrm{d}\Delta\varphi\textrm{d}\Delta\eta", is_binned = False, is_independent = False)
        corr.values = correlation["z"]

        table = hepdata.Table(f"raw_correlation_ep_{ep}_pt_assoc_{pt_1}_{pt_2}_hepdata")
        table.add_variable(phi)
        table.add_variable(eta)
        table.add_variable(corr)
        # Add basic labeling.
        table.description = fr"Correlation function for 20-40 $\textrm{{GeV}}/c$ {_ep_orientation_full_name_map[ep]} jets with ${pt_1} < p_{{\textrm{{T}}}} {pt_2}$ $\textrm{{GeV}}/c$ hadrons."
        # This location depends on the layout of the public note.
        table.location = fr"Figure {_figure_location_map[ep]}"
        submission.add_table(table)

    hepdata_output = output_path / "hepdata"
    # Create the YAML files
    submission.create_files(str(hepdata_output))

    return True

def define_2d_correlations_hists() -> Tuple[List[str], List[float], List[Tuple[str, float, float, str]]]:
    """ Define the 2D correlation hists.

    Args:
        None.
    Returns:
        (ep_orientation values, assoc_pt values, (ep_orientation, lower_pt_bin_edge, upper_pt_bin_edge, name))
    """
    ep_orientations = ["all", "in", "mid", "out"]
    assoc_pt = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0]
    # Example name: "raw2DratioallAss1015J2040C3050bg0812sig06rbX2rbY2.root",
    names = [
        (ep, pt_1, pt_2, f"raw2Dratio{ep}Ass{int(pt_1 * 10)}{int(pt_2 * 10)}J2040C3050bg0812sig06rbX2rbY2")
        for ep in ep_orientations
        for pt_1, pt_2 in zip(assoc_pt[:-1], assoc_pt[1:])
    ]

    return ep_orientations, assoc_pt, names

if __name__ == "__main__":
    # Basic setup
    coloredlogs.install(
        level = logging.DEBUG,
        fmt = "%(asctime)s %(name)s:%(lineno)d %(levelname)s %(message)s"
    )

    convert_histograms_to_hepdata(
        input_path = Path("inputData/PbPb/2.76/semi_central/Histos2D"),
        output_path = Path("output/PbPb/2.76/semi_central/joelsCorrelations"),
    )
