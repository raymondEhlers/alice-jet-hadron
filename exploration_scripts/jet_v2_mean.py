#!/usr/bin/env python3

"""

"""

import numpy as np
from pathlib import Path
from typing import Dict

from pachyderm import histogram
from pachyderm import yaml

def retrieve_jet_v2_hists(filename: Path) -> Dict[str, histogram.Histogram1D]:
    yml = yaml.yaml(classes_to_register = [histogram.Histogram1D])
    with open(filename, "r") as f:
        hists = yml.load(f)

    return {
        "central": hists["0-5"],
        "semi-central": hists["30-50"],
    }

def jet_v2_properties() -> None:
    hists = retrieve_jet_v2_hists(filename = Path("inputData/PbPb/2.76/aliceJetV2.yaml"))

    errors = {}
    for name, hist in hists.items():
        errors[name] = np.sqrt(np.sum(hist.errors_squared))

    print("Mean +/- error for jet v2")
    for (name, hist), error in zip(hists.items(), errors.values()):
        print(f"{name}: {np.mean(hist.y):.4f} +/- {error:.4f}")
    print("NOTE: Uncertanties are summed in quadrature, so the uncertanties are estimates at best")

if __name__ == "__main__":
    jet_v2_properties()
