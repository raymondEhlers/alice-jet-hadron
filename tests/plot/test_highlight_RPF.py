#!/usr/bin/env python

""" Tests for plotting highlights of RPF regions

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import pytest  # noqa: F401
import numpy
import logging

from jet_hadron.plot import highlight_RPF

# Setup logger
logger = logging.getLogger(__name__)

def scale_color_to_max_1(colors):
    """ Scale all colors to max 1 (from 255). """
    return tuple(map(highlight_RPF.convert_color_to_max_1, colors))

def test_convert_colors_to_max_1(logging_mixin):
    """ Test the scaling of a color from [0, 255] -> [0, 1]. """
    assert highlight_RPF.convert_color_to_max_1(0) == 0
    assert highlight_RPF.convert_color_to_max_1(51) == 0.2
    assert highlight_RPF.convert_color_to_max_1(64) == 64 / 255.
    assert highlight_RPF.convert_color_to_max_1(255) == 1

def test_convert_colros_to_max_255(logging_mixin):
    """ Test the scaling of a color from [0,1] -> [0, 255]. """
    assert highlight_RPF.convert_color_to_max_255(0) == 0
    assert highlight_RPF.convert_color_to_max_255(0.2) == 51
    assert highlight_RPF.convert_color_to_max_255(0.25) == 64
    assert highlight_RPF.convert_color_to_max_255(1) == 255

def test_overlay_colors(logging_mixin):
    """ Test determining colors using the "overlay" method. """
    # Example 1
    background = scale_color_to_max_1((169, 169, 169))
    foreground = scale_color_to_max_1((102, 205, 96))
    result = scale_color_to_max_1((152, 221, 148))
    # NOTE: We use allclose because the rounding errors seem to accumulate here
    assert numpy.allclose(highlight_RPF.overlay_colors(foreground = foreground, background = background), result, 0.005)

    # Example 2
    background = scale_color_to_max_1((171, 56, 56))
    foreground = scale_color_to_max_1((102, 205, 96))
    result = scale_color_to_max_1((154, 90, 42))
    assert numpy.allclose(highlight_RPF.overlay_colors(foreground = foreground, background = background), result, 0.005)

def test_screen_colors(logging_mixin):
    """ Test determining colors using the "screen" method. """
    # Start with a simple example
    foreground = scale_color_to_max_1((0, 0, 0))
    background = scale_color_to_max_1((51, 51, 51))
    result = scale_color_to_max_1((52, 52, 52))
    # NOTE: allclose doesn't seem to be necessary here, likely because the formula uses a bitshift
    #       and then just subtracts. So we should have exact values.
    assert highlight_RPF.screen_colors(foreground = foreground, background = background) == result
    # More complex
    foreground = scale_color_to_max_1((171, 56, 56))
    background = scale_color_to_max_1((102, 205, 96))
    result = scale_color_to_max_1((205, 217, 132))
    assert highlight_RPF.screen_colors(foreground = foreground, background = background) == result

