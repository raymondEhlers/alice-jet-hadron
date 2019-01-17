#!/usr/bin/env python

""" Tests for the correlations analysis module.

By necessity, these won't be extensive, but we can cover some functionality.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import pytest  # noqa: F401

from jet_hadron.analysis import correlations

def test_jet_hadron_correlation_axis_display(logging_mixin):
    """ Test for displaying a jet_hadron correlation axis. """

    axis = correlations.JetHCorrelationAxis.delta_phi

    assert axis.display_str() == r"$\Delta\varphi$"
