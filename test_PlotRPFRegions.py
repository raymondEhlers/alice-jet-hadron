#!/usr/bin/env python

# Tests for plotting highlights of RPF regions

import pytest
import numpy
import logging
# Setup logger
logger = logging.getLogger(__name__)

import PlotRPFRegions

# Set logging level as a global variable to simplify configuration.
# This is not ideal, but fine for simple tests.
loggingLevel = logging.DEBUG

def scaleColorToMax1(colors):
    """ Scale all colors to max 1 (from 255). """
    return tuple(map(PlotRPFRegions.convertColorToMax1, colors))

def testConvertColorsToMax1(caplog):
    """ Test the scaling of a color from [0, 255] -> [0, 1]. """
    caplog.set_level(loggingLevel)

    assert PlotRPFRegions.convertColorToMax1(0) == 0
    assert PlotRPFRegions.convertColorToMax1(51) == 0.2
    assert PlotRPFRegions.convertColorToMax1(64) == 64/255.
    assert PlotRPFRegions.convertColorToMax1(255) == 1

def testConvertColrosToMax255(caplog):
    """ Test the scaling of a color from [0,1] -> [0, 255]. """
    caplog.set_level(loggingLevel)

    assert PlotRPFRegions.convertColorToMax255(0) == 0
    assert PlotRPFRegions.convertColorToMax255(0.2) == 51
    assert PlotRPFRegions.convertColorToMax255(0.25) == 64
    assert PlotRPFRegions.convertColorToMax255(1) == 255

def testOverlayColors(caplog):
    """ Test determining colors using the "overlay" method. """
    caplog.set_level(loggingLevel)

    # Example 1
    background = scaleColorToMax1((169, 169, 169))
    foreground = scaleColorToMax1((102, 205, 96))
    result = scaleColorToMax1((152, 221, 148))
    # NOTE: We use allclose because the rounding errors seem to accumulate here
    assert numpy.allclose(PlotRPFRegions.overlayColors(foreground = foreground, background = background), result, 0.005)

    # Example 2
    background = scaleColorToMax1((171, 56, 56))
    foreground = scaleColorToMax1((102, 205, 96))
    result = scaleColorToMax1((154, 90, 42))
    assert numpy.allclose(PlotRPFRegions.overlayColors(foreground = foreground, background = background), result, 0.005)

def testScreenColors(caplog):
    """ Test determining colors using the "screen" method. """
    caplog.set_level(loggingLevel)

    # Start with a simple example
    foreground = scaleColorToMax1((0, 0, 0))
    background = scaleColorToMax1((51, 51, 51))
    result = scaleColorToMax1((52, 52, 52))
    # NOTE: allclose doesn't seem to be necessary here, likely because the formula uses a bitshift
    #       and then just subtracts. So we should have exact values.
    assert PlotRPFRegions.screenColors(foreground = foreground, background = background) == result
    # More complex
    foreground = scaleColorToMax1((171, 56, 56))
    background = scaleColorToMax1((102, 205, 96))
    result = scaleColorToMax1((205, 217, 132))
    assert PlotRPFRegions.screenColors(foreground = foreground, background = background) == result

