#!/usr/bin/env python

# Tests for base plotting module.
#
# author: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
# date: 6 June 2018

import pytest
import os
import logging
# Setup logger
logger = logging.getLogger(__name__)

import jetH.plot.base as plotBase

@pytest.mark.parametrize("outputPrefix,printingExtensions", [
    ("a/b", ["png"]),
    ("a/b", ["png", "pdf"]),
    ("a/b/c", ["pdf"])
], ids = ["standard", "multipleExtension", "variationOfOutputPrefix"])
def testPlottingOutputWrapper(loggingMixin, outputPrefix, printingExtensions):
    """ Test the plottingOutputWrapper object. """
    obj = plotBase.plottingOutputWrapper(outputPrefix = outputPrefix, printingExtensions = printingExtensions)
    assert obj.outputPrefix == outputPrefix
    assert obj.printingExtensions == printingExtensions

@pytest.fixture(params = [
    ("a/b", ["png"]),
    ("a/b", ["png", "pdf"]),
    ("a/b/c", ["pdf"])
], ids = ["standard", "multipleExtension", "variationOfOutputPrefix"])
def setupSaveTests(request, mocker):
    """ Provides mock objects for testing the save oriented functionality. """
    (outputPrefix, printingExtensions) = request.param
    obj = mocker.MagicMock(spec = ["outputPrefix", "printingExtensions"],
                           outputPrefix = outputPrefix,
                           printingExtensions = printingExtensions)
    # Only the functions in the spec are allowed, which is why the objects are separate.
    figure = mocker.MagicMock(spec = ["savefig"])
    canvas = mocker.MagicMock(spec = ["SaveAs"])

    expectedFilenames = []
    for ext in printingExtensions:
        expectedFilenames.append(os.path.join(outputPrefix, "{filename}." + ext))
    return (obj, figure, canvas, expectedFilenames)

def testSavePlot(loggingMixin, setupSaveTests):
    """ Test the wrapper for saving a matplotlib plot. """
    (obj, figure, canvas, expectedFilenames) = setupSaveTests
    filename = "filename"
    filenames = plotBase.savePlot(obj, figure, filename)
    figure.savefig.assert_called()

    assert filenames == [name.format(filename = filename) for name in expectedFilenames]

def testSaveCanvas(loggingMixin, setupSaveTests):
    """ Test the wrapper for saving a ROOT canvas. """
    (obj, figure, canvas, expectedFilenames) = setupSaveTests
    filename = "filename"
    filenames = plotBase.saveCanvas(obj, canvas, filename)
    canvas.SaveAs.assert_called()

    assert filenames == [name.format(filename = filename) for name in expectedFilenames]

def testSavePlotImpl(loggingMixin, setupSaveTests):
    """ Test the implementation for saving a matplotlib plot. """
    (obj, figure, canvas, expectedFilenames) = setupSaveTests
    filename = "filename"
    filenames = plotBase.savePlotImpl(figure, obj.outputPrefix, filename, obj.printingExtensions)
    figure.savefig.assert_called()

    assert filenames == [name.format(filename = filename) for name in expectedFilenames]

def testSaveCanvasImpl(loggingMixin, mocker, setupSaveTests):
    """ Test the implementation for saving a ROOT canvas. """
    (obj, figure, canvas, expectedFilenames) = setupSaveTests
    filename = "filename"
    filenames = plotBase.saveCanvasImpl(canvas, obj.outputPrefix, filename, obj.printingExtensions)
    canvas.SaveAs.assert_called()

    assert filenames == [name.format(filename = filename) for name in expectedFilenames]

def testRegistrationOfKBirdColormap(loggingMixin):
    """ Test to ensure that the ROOT kBird colormap is registered in matplotlib successfully. """
    import matplotlib.pyplot as plt
    kBirdName = "ROOT_kBird"
    # If it doesn't exist, it will throw an exception, which will fail the test
    kBird = plt.get_cmap(kBirdName)

    assert kBird.name == kBirdName

def testFixBadColormapValue(loggingMixin):
    """ Test that the bad (ie. nan or similar) value of a given color scheme is successfully reassigned. """
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("viridis")
    # NOTE: There isn't any direct access to this property.
    initialBadValue = cmap._rgba_bad
    cmap = plotBase.prepareColormap(cmap)
    finalBadValue = cmap._rgba_bad

    assert initialBadValue != finalBadValue
    assert finalBadValue == (1.0, 1.0, 1.0, 1.0)
