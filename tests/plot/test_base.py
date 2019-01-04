#!/usr/bin/env python

""" Tests for base plotting module.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import pytest
import os
import logging

from jet_hadron.plot import base as plot_base

# Setup logger
logger = logging.getLogger(__name__)

@pytest.mark.parametrize("output_prefix,printing_extensions", [
    ("a/b", ["png"]),
    ("a/b", ["png", "pdf"]),
    ("a/b/c", ["pdf"])
], ids = ["standard", "multipleExtension", "variationOfOutputPrefix"])
def testPlottingOutputWrapper(logging_mixin, output_prefix, printing_extensions):
    """ Test the plottingOutputWrapper object. """
    obj = plot_base.plottingOutputWrapper(output_prefix = output_prefix, printing_extensions = printing_extensions)
    assert obj.output_prefix == output_prefix
    assert obj.printing_extensions == printing_extensions

@pytest.fixture(params = [
    ("a/b", ["png"]),
    ("a/b", ["png", "pdf"]),
    ("a/b/c", ["pdf"])
], ids = ["standard", "multipleExtension", "variationOfOutputPrefix"])
def setupSaveTests(request, mocker):
    """ Provides mock objects for testing the save oriented functionality. """
    (output_prefix, printing_extensions) = request.param
    obj = mocker.MagicMock(spec = ["output_prefix", "printing_extensions"],
                           output_prefix = output_prefix,
                           printing_extensions = printing_extensions)
    # Only the functions in the spec are allowed, which is why the objects are separate.
    figure = mocker.MagicMock(spec = ["savefig"])
    canvas = mocker.MagicMock(spec = ["SaveAs"])

    expectedFilenames = []
    for ext in printing_extensions:
        expectedFilenames.append(os.path.join(output_prefix, "{filename}." + ext))
    return (obj, figure, canvas, expectedFilenames)

def testSavePlot(logging_mixin, setupSaveTests):
    """ Test the wrapper for saving a matplotlib plot. """
    (obj, figure, canvas, expectedFilenames) = setupSaveTests
    filename = "filename"
    filenames = plot_base.savePlot(obj, figure, filename)
    figure.savefig.assert_called()

    assert filenames == [name.format(filename = filename) for name in expectedFilenames]

def testSaveCanvas(logging_mixin, setupSaveTests):
    """ Test the wrapper for saving a ROOT canvas. """
    (obj, figure, canvas, expectedFilenames) = setupSaveTests
    filename = "filename"
    filenames = plot_base.saveCanvas(obj, canvas, filename)
    canvas.SaveAs.assert_called()

    assert filenames == [name.format(filename = filename) for name in expectedFilenames]

def testSavePlotImpl(logging_mixin, setupSaveTests):
    """ Test the implementation for saving a matplotlib plot. """
    (obj, figure, canvas, expectedFilenames) = setupSaveTests
    filename = "filename"
    filenames = plot_base.savePlotImpl(figure, obj.output_prefix, filename, obj.printing_extensions)
    figure.savefig.assert_called()

    assert filenames == [name.format(filename = filename) for name in expectedFilenames]

def testSaveCanvasImpl(logging_mixin, mocker, setupSaveTests):
    """ Test the implementation for saving a ROOT canvas. """
    (obj, figure, canvas, expectedFilenames) = setupSaveTests
    filename = "filename"
    filenames = plot_base.saveCanvasImpl(canvas, obj.output_prefix, filename, obj.printing_extensions)
    canvas.SaveAs.assert_called()

    assert filenames == [name.format(filename = filename) for name in expectedFilenames]

def testRegistrationOfKBirdColormap(logging_mixin):
    """ Test to ensure that the ROOT kBird colormap is registered in matplotlib successfully. """
    import matplotlib.pyplot as plt
    kBirdName = "ROOT_kBird"
    # If it doesn't exist, it will throw an exception, which will fail the test
    kBird = plt.get_cmap(kBirdName)

    assert kBird.name == kBirdName

def testFixBadColormapValue(logging_mixin):
    """ Test that the bad (ie. nan or similar) value of a given color scheme is successfully reassigned. """
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("viridis")
    # NOTE: There isn't any direct access to this property.
    initialBadValue = cmap._rgba_bad
    cmap = plot_base.prepareColormap(cmap)
    finalBadValue = cmap._rgba_bad

    assert initialBadValue != finalBadValue
    assert finalBadValue == (1.0, 1.0, 1.0, 1.0)
