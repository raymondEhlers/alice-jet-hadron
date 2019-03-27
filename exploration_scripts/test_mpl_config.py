#!/usr/bin/env python

""" Tests to compare color schemes.

Those schemes are defined in matplotlib, seaborn, pachyderm, and formerly jet_hadron.plot.base
"""

import matplotlib
import pachyderm.plot
import pprint

original = matplotlib.rcParams.copy()

def compare_config():
    import jet_hadron.plot.base as plot_base  # noqa: F401
    # There are the jet_hadron params where are automatically loaded in plot_base.
    jet_hadron_params = matplotlib.rcParams.copy()

    # Now reset before continuing
    pachyderm.plot.restore_defaults()

    pachyderm.plot.configure()
    new = matplotlib.rcParams.copy()

    compare = {k: f"pachyderm: {new[k]}, jet_hadron: {v}" for k, v in jet_hadron_params.items() if v != new[k]}

    pprint.pprint(compare)

if __name__ == "__main__":
    compare_config()
