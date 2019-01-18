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
], ids = ["standard", "Multiple extensions", "Variation of output prefix"])
def test_plotting_output_wrapper(logging_mixin, output_prefix, printing_extensions):
    """ Test the plottingOutputWrapper object. """
    obj = plot_base.PlottingOutputWrapper(output_prefix = output_prefix, printing_extensions = printing_extensions)
    assert obj.output_prefix == output_prefix
    assert obj.printing_extensions == printing_extensions

@pytest.fixture(params = [
    ("a/b", ["png"]),
    ("a/b", ["png", "pdf"]),
    ("a/b/c", ["pdf"])
], ids = ["standard", "Multiple extensions", "Variation of output prefix"])
def setup_save_tests(request, mocker):
    """ Provides mock objects for testing the save oriented functionality. """
    (output_prefix, printing_extensions) = request.param
    obj = mocker.MagicMock(spec = ["output_prefix", "printing_extensions"],
                           output_prefix = output_prefix,
                           printing_extensions = printing_extensions)
    # Only the functions in the spec are allowed, which is why the objects are separate.
    figure = mocker.MagicMock(spec = ["savefig"])
    canvas = mocker.MagicMock(spec = ["SaveAs"])

    expected_filenames = []
    for ext in printing_extensions:
        expected_filenames.append(os.path.join(output_prefix, "{filename}." + ext))
    return (obj, figure, canvas, expected_filenames)

def test_save_plot(logging_mixin, setup_save_tests):
    """ Test the wrapper for saving a matplotlib plot. """
    (obj, figure, canvas, expected_filenames) = setup_save_tests
    filename = "filename"
    filenames = plot_base.save_plot(obj, figure, filename)
    figure.savefig.assert_called()

    assert filenames == [name.format(filename = filename) for name in expected_filenames]

def test_save_canvas(logging_mixin, setup_save_tests):
    """ Test the wrapper for saving a ROOT canvas. """
    (obj, figure, canvas, expected_filenames) = setup_save_tests
    filename = "filename"
    filenames = plot_base.save_canvas(obj, canvas, filename)
    canvas.SaveAs.assert_called()

    assert filenames == [name.format(filename = filename) for name in expected_filenames]

def test_save_plot_impl(logging_mixin, setup_save_tests):
    """ Test the implementation for saving a matplotlib plot. """
    (obj, figure, canvas, expected_filenames) = setup_save_tests
    filename = "filename"
    filenames = plot_base.save_plot_impl(figure, obj.output_prefix, filename, obj.printing_extensions)
    figure.savefig.assert_called()

    assert filenames == [name.format(filename = filename) for name in expected_filenames]

def test_save_canvas_impl(logging_mixin, mocker, setup_save_tests):
    """ Test the implementation for saving a ROOT canvas. """
    (obj, figure, canvas, expected_filenames) = setup_save_tests
    filename = "filename"
    filenames = plot_base.save_canvas_impl(canvas, obj.output_prefix, filename, obj.printing_extensions)
    canvas.SaveAs.assert_called()

    assert filenames == [name.format(filename = filename) for name in expected_filenames]

def test_registration_of_kBird_colormap(logging_mixin):
    """ Test to ensure that the ROOT kBird colormap is registered in matplotlib successfully. """
    import matplotlib.pyplot as plt
    kBirdName = "ROOT_kBird"
    # If it doesn't exist, it will throw an exception, which will fail the test
    kBird = plt.get_cmap(kBirdName)

    assert kBird.name == kBirdName

def test_fix_bad_colormap_value(logging_mixin):
    """ Test that the bad (ie. nan or similar) value of a given color scheme is successfully reassigned. """
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("viridis")
    # NOTE: There isn't any direct access to this property.
    initial_bad_value = cmap._rgba_bad
    cmap = plot_base.prepare_colormap(cmap)
    final_bad_value = cmap._rgba_bad

    assert initial_bad_value != final_bad_value
    assert final_bad_value == (1.0, 1.0, 1.0, 1.0)
