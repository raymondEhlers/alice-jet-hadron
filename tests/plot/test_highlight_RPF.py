#!/usr/bin/env python

# Tests for plotting highlights of RPF regions

import pytest  # noqa: F401
import numpy
import logging

from jet_hadron.plot import highlight_RPF

# Setup logger
logger = logging.getLogger(__name__)

def scaleColorToMax1(colors):
    """ Scale all colors to max 1 (from 255). """
    return tuple(map(highlight_RPF.convertColorToMax1, colors))

def testConvertColorsToMax1(loggingMixin):
    """ Test the scaling of a color from [0, 255] -> [0, 1]. """
    assert highlight_RPF.convertColorToMax1(0) == 0
    assert highlight_RPF.convertColorToMax1(51) == 0.2
    assert highlight_RPF.convertColorToMax1(64) == 64 / 255.
    assert highlight_RPF.convertColorToMax1(255) == 1

def testConvertColrosToMax255(loggingMixin):
    """ Test the scaling of a color from [0,1] -> [0, 255]. """
    assert highlight_RPF.convertColorToMax255(0) == 0
    assert highlight_RPF.convertColorToMax255(0.2) == 51
    assert highlight_RPF.convertColorToMax255(0.25) == 64
    assert highlight_RPF.convertColorToMax255(1) == 255

def testOverlayColors(loggingMixin):
    """ Test determining colors using the "overlay" method. """
    # Example 1
    background = scaleColorToMax1((169, 169, 169))
    foreground = scaleColorToMax1((102, 205, 96))
    result = scaleColorToMax1((152, 221, 148))
    # NOTE: We use allclose because the rounding errors seem to accumulate here
    assert numpy.allclose(highlight_RPF.overlayColors(foreground = foreground, background = background), result, 0.005)

    # Example 2
    background = scaleColorToMax1((171, 56, 56))
    foreground = scaleColorToMax1((102, 205, 96))
    result = scaleColorToMax1((154, 90, 42))
    assert numpy.allclose(highlight_RPF.overlayColors(foreground = foreground, background = background), result, 0.005)

def testScreenColors(loggingMixin):
    """ Test determining colors using the "screen" method. """
    # Start with a simple example
    foreground = scaleColorToMax1((0, 0, 0))
    background = scaleColorToMax1((51, 51, 51))
    result = scaleColorToMax1((52, 52, 52))
    # NOTE: allclose doesn't seem to be necessary here, likely because the formula uses a bitshift
    #       and then just subtracts. So we should have exact values.
    assert highlight_RPF.screenColors(foreground = foreground, background = background) == result
    # More complex
    foreground = scaleColorToMax1((171, 56, 56))
    background = scaleColorToMax1((102, 205, 96))
    result = scaleColorToMax1((205, 217, 132))
    assert highlight_RPF.screenColors(foreground = foreground, background = background) == result

