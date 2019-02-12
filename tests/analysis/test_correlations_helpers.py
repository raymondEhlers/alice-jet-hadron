#!/usr/bin/env python

""" Tests for the correlations helper module.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import dataclasses
import logging
import pytest

from jet_hadron.analysis import correlations_helpers

logger = logging.getLogger(__name__)

@pytest.mark.parametrize("hist_index, expected", [
    (0, {"scale_factor": 10.0}),
    (1, {"scale_factor": 5.0}),
    (2, {"scale_factor": 0.5}),
], ids = ["hist1D", "hist2D", "hist3D"])
def test_hist_container_scale_factor(logging_mixin, hist_index, expected, test_root_hists):
    """ Test hist container scale factor calculation. """
    hist = dataclasses.astuple(test_root_hists)[hist_index]
    assert correlations_helpers._calculate_bin_width_scale_factor(hist) == expected["scale_factor"]
    additional_scale_factor = 0.5
    assert correlations_helpers._calculate_bin_width_scale_factor(
        hist = hist,
        additional_scale_factor = additional_scale_factor) == \
        expected["scale_factor"] * additional_scale_factor

@pytest.mark.parametrize("hist_index, expected", [
    (0, {"scale_factor": 10.0}),
    (1, {"scale_factor": 5.0}),
    (2, {"scale_factor": 0.5}),
], ids = ["hist1D", "hist2D", "hist3D"])
def test_scale_histogram(logging_mixin, hist_index, expected, test_root_hists):
    """ Test scaling ROOT hists. """

    hist = dataclasses.astuple(test_root_hists)[hist_index]
    expected_hist = hist.Clone("{hist.GetName()}_expected")
    expected_hist.Scale(expected["scale_factor"])

    correlations_helpers.scale_by_bin_width(hist)

    assert expected_hist.GetEntries() == hist.GetEntries()
    # Compare all bins
    for bin in range(1, hist.GetNcells() + 1):
        assert expected_hist.GetBinContent(bin) == hist.GetBinContent(bin)

