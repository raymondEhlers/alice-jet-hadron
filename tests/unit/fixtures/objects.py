#!/usr/bin/env python

""" Objects related fixtures to aid testing.

.. code-author: Raymond Ehlers <raymond.ehlers@yale.edu>, Yale University
"""

from dataclasses import dataclass
import numpy as np
import pytest

@pytest.fixture
@pytest.mark.ROOT
def testRootHists():
    """ Create minimal TH*F hists in 1D, 2D, and 3D. Each has been filled once.

    Args:
        None
    Returns:
        tuple: (TH1F, TH2F, TH3F) for testing
    """
    import ROOT

    @dataclass
    class RootHists:
        """ ROOT histograms for testing.

        Just for convenience.
        """
        hist1D: ROOT.TH1
        hist2D: ROOT.TH2
        hist3D: ROOT.TH3

    # Define the hist to use for testing
    hist = ROOT.TH1F("test", "test", 10, 0, 1)
    hist.Fill(.1)
    hist2D = ROOT.TH2F("test2", "test2", 10, 0, 1, 10, 0, 20)
    hist2D.Fill(.1, 1)
    hist3D = ROOT.TH3F("test3", "test3", 10, 0, 1, 10, 0, 20, 10, 0, 100)
    hist3D.Fill(.1, 1, 10)

    return RootHists(hist1D = hist, hist2D = hist2D, hist3D = hist3D)

@pytest.fixture
@pytest.mark.ROOT
def testSparse():
    """ Create a THnSparseF for testing.

    Fills in a set of values for testing.

    Args:
        None.
    Returns:
        tuple: (THnSparseF, fill_value) for testing.
    """
    import ROOT

    @dataclass
    class SparseAxis:
        """ THnSparse axis information.

        Just for convenience.
        """
        n_bins: int
        min: float
        max: float

    ignoredAxis   = SparseAxis(n_bins =  1, min =   0.0, max =  1.0)  # noqa: E221, E222
    selectedAxis1 = SparseAxis(n_bins = 10, min =   0.0, max = 20.0)  # noqa: E222
    selectedAxis2 = SparseAxis(n_bins = 20, min = -10.0, max = 10.0)
    selectedAxis3 = SparseAxis(n_bins = 30, min =   0.0, max = 30.0)  # noqa: E222
    # We want to select axes 2, 4, 5
    axes = [ignoredAxis, ignoredAxis, selectedAxis1, ignoredAxis, selectedAxis2, selectedAxis3, ignoredAxis]

    # Create the actual sparse
    # NOTE: dtype is required here for the sparse to be created successfully.
    bins = np.array([el.n_bins for el in axes], dtype=np.int32)
    mins = np.array([el.min for el in axes])
    maxes = np.array([el.max for el in axes])
    #logger.debug("bins: {}, mins: {}, maxs: {}".format(bins, mins, maxes))
    sparse = ROOT.THnSparseF("testSparse", "testSparse", len(axes), bins, mins, maxes)

    # Fill in some strategic values.
    # Wrapper function is for convenience.
    def fill_sparse(one, two, three):
        sparse.Fill(np.array([0., 0., one, 0., two, three, 0.], dtype = np.float64))
    fill_values = [
        (4., -2., 10.),
        (4., 2., 10.)
    ]
    for values in fill_values:
        fill_sparse(*values)

    return (sparse, fill_values)

