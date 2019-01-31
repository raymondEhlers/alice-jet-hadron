#!/usr/bin/env python

""" Tests for the response matrix module.

Note that these tests aren't comprehensive, but they provide a few useful tests.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import pytest

from jet_hadron.analysis import response_matrix_helpers

@pytest.mark.parametrize("normalization, expected", [
    ("none",
        {"str": "none",
            "display_str": "No normalization"}),
    ("normalize_each_detector_bin",
        {"str": "normalize_each_detector_bin",
            "display_str": "Normalize each detector bin"})
], ids = ["None", "Each detector bin"])
def test_response_normalization(logging_mixin, normalization, expected):
    """ Test response normalization strings. """
    norm = response_matrix_helpers.ResponseNormalization[normalization]
    assert str(norm) == expected["str"]
    assert norm.display_str() == expected["display_str"]

