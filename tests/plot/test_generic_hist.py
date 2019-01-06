#!/usr/bin/env python

""" Tests for plotting generic hists.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import pytest  # noqa: F401
import numpy as np
import root_numpy

from pachyderm import histogram

def test_root_numpy_removal_1D(logging_mixin, test_root_hists):
    """ Test removing root_numpy for 1D hists. """
    test_hist = test_root_hists.hist1D

    (hist_array, bin_edges) = root_numpy.hist2array(test_hist, return_edges=True)

    # Expected
    expected_hist = histogram.Histogram1D.from_existing_hist(test_hist)

    assert np.allclose(hist_array, expected_hist.y)
    # ``bin_edges[0]`` corresponds to x axis bin edges (which are the only relevant ones here).
    assert np.allclose(bin_edges[0], expected_hist.bin_edges)

    # ROOT bin edges
    root_edges = np.empty(test_hist.GetXaxis().GetNbins() + 1)
    root_edges[:-1] = [test_hist.GetXaxis().GetBinLowEdge(i) for i in range(1, test_hist.GetXaxis().GetNbins() + 1)]
    root_edges[-1] = test_hist.GetXaxis().GetBinUpEdge(test_hist.GetXaxis().GetNbins())

    assert np.allclose(expected_hist.bin_edges, root_edges)

def test_root_numpy_removal_bin_edges_2D(logging_mixin, test_root_hists):
    """ Test removing root_numpy for 2D hists with x and y as bin edges. """
    test_hist = test_root_hists.hist2D

    (hist_array, bin_edges) = root_numpy.hist2array(test_hist, return_edges=True)
    hist_array[hist_array == 0] = np.nan

    epsilon = 1e-9
    x_range = np.arange(
        np.amin(bin_edges[0]),
        np.amax(bin_edges[0]) + epsilon,
        test_hist.GetXaxis().GetBinWidth(1)
    )
    y_range = np.arange(
        np.amin(bin_edges[1]),
        np.amax(bin_edges[1]) + epsilon,
        test_hist.GetYaxis().GetBinWidth(1)
    )
    X, Y = np.meshgrid(x_range, y_range)

    # Expected
    expected_X, expected_Y, expected_hist_array = histogram.get_array_from_hist2D(hist = test_hist, set_zero_to_NaN = True, return_bin_edges = True)

    assert np.allclose(X, expected_X)
    assert np.allclose(Y, expected_Y)
    assert np.allclose(hist_array, expected_hist_array, equal_nan = True)

def test_root_numpy_removal_bin_centers_2D(logging_mixin, test_root_hists):
    """ Test removing root_numpy for 2D hists with x and y as bin centers. """
    test_hist = test_root_hists.hist2D

    (hist_array, _) = root_numpy.hist2array(test_hist, return_edges=True)
    hist_array[hist_array == 0] = np.nan

    # We want an array of bin centers
    x_range = np.array([test_hist.GetXaxis().GetBinCenter(i) for i in range(1, test_hist.GetXaxis().GetNbins() + 1)])
    y_range = np.array([test_hist.GetYaxis().GetBinCenter(i) for i in range(1, test_hist.GetYaxis().GetNbins() + 1)])
    X, Y = np.meshgrid(x_range, y_range)

    # Expected
    expected_X, expected_Y, expected_hist_array = histogram.get_array_from_hist2D(hist = test_hist, set_zero_to_NaN = True)

    assert np.allclose(X, expected_X)
    assert np.allclose(Y, expected_Y)
    assert np.allclose(hist_array, expected_hist_array, equal_nan = True)

