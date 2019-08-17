#!/usr/bin/env python

""" Base plotting module.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

from dataclasses import dataclass
import logging
import numpy as np
import os
from typing import List, Optional, Sequence, Tuple, TypeVar, Union

# Import and configure plotting packages
import matplotlib
import matplotlib.pyplot as plt

import pachyderm.plot

from jet_hadron.base import analysis_objects
from jet_hadron.base import labels
from jet_hadron.base.typing_helpers import Canvas, Hist

# Setup logger
logger = logging.getLogger(__name__)

# Configure plot styling.
pachyderm.plot.configure()

class AnalysisColors:
    """ Exceedingly simple class to store analysis colors. """
    signal = "tab:blue"  # "C0" in the default MPL color cycle
    background = "tab:orange"  # "C1" in the default MPL color cycle
    fit = "tab:purple"  # "C4" in the default MPL color cycle

@dataclass
class PlotLabels:
    """ Simple wrapper for keeping plot labels together.

    Note:
        The attributes are initialized to and compared against ``None`` rather than the empty
        string because empty string is a valid value.

    Attributes:
        title: Title of the plot.
        x_label: x axis label of the plot.
        y_label: y axis label of the plot.
    """
    title: Optional[str] = None
    x_label: Optional[str] = None
    y_label: Optional[str] = None

    def apply_labels(self, obj: Union[matplotlib.axes.Axes, Hist]) -> None:
        if hasattr(obj, "GetXaxis"):
            self._apply_labels_ROOT(hist = obj)
        else:
            self._apply_labels_mpl(ax = obj)

    def _apply_labels_mpl(self, ax: matplotlib.axes.Axes) -> None:
        if self.title is not None:
            ax.set_title(self.title)
        if self.x_label is not None:
            ax.set_xlabel(self.x_label)
        if self.y_label is not None:
            ax.set_ylabel(self.y_label)

    def _apply_labels_ROOT(self, hist: Hist) -> None:
        if self.title is not None:
            hist.SetTitle(labels.use_label_with_root(self.title))
        if self.x_label is not None:
            hist.GetXaxis().SetTitle(labels.use_label_with_root(self.x_label))
        if self.y_label is not None:
            hist.GetYaxis().SetTitle(labels.use_label_with_root(self.y_label))

def save_plot(obj: analysis_objects.PlottingOutputWrapper,
              figure: Union[matplotlib.figure.Figure, Canvas],
              output_name: str,
              pdf_with_ROOT: bool = False) -> List[str]:
    """ Loop over all requested file extensions and save the current plot in matplotlib.

    Uses duck typing to properly save both matplotlib figures and ROOT canvases.

    Args:
        obj: Contains the output_prefix and printing_extensions
        figure: Figure or ROOT canvas on which the plot was drawn.
        output_name: Filename under which the plot should be saved, but without the file extension.
        pdf_with_ROOT: True if the ROOT canvas should be saved as a PDF (assuming that it was requested in
            the list of printing extensions). ROOT + pdf + latex labels often fails dramatically, so
            this should only be enabled in rare cases.
    Returns:
        Filenames under which the plot was saved.
    """
    # Setup output area
    if not os.path.exists(obj.output_prefix):
        os.makedirs(obj.output_prefix)

    # Check for the proper attribute for a ROOT canvas
    if hasattr(figure, "SaveAs"):
        return save_canvas_impl(
            canvas = figure, output_prefix = obj.output_prefix, output_name = output_name,
            printing_extensions = obj.printing_extensions, pdf_with_ROOT = pdf_with_ROOT
        )
    # If not, we plot it with MPL.
    return save_plot_impl(
        fig = figure, output_prefix = obj.output_prefix, output_name = output_name,
        printing_extensions = obj.printing_extensions
    )

# Base functions
def save_canvas_impl(canvas: Canvas,
                     output_prefix: str, output_name: str,
                     printing_extensions: Sequence[str],
                     pdf_with_ROOT: bool = False) -> List[str]:
    """ Implementation of generic save canvas function.

    It loops over all requested file extensions and save the ROOT canvas.

    Args:
        canvas: Canvas on which the plot was drawn.
        output_prefix: File path to where files should be saved.
        output_name: Filename under which the plot should be saved, but without the file extension.
        printing_extensions: List of file extensions under which plots should be saved. They should
            not contain the dot!
        pdf_with_ROOT: True if the ROOT canvas should be saved as a PDF (assuming that it was requested in
            the list of printing extensions). ROOT + pdf + latex labels often fails dramatically, so
            this should only be enabled in rare cases.
    Returns:
        list: Filenames under which the plot was saved.
    """
    filenames = []
    for extension in printing_extensions:
        # Skip drawing PDFs with ROOT because it handles LaTeX so poorly.
        # We will only produce it if explicitly requested.
        if extension == "pdf" and not pdf_with_ROOT:
            continue

        filename = os.path.join(output_prefix, output_name + "." + extension)
        # Probably don't want this log message since ROOT will also generate a message
        #logger.debug("Saving ROOT canvas to \"{}\"".format(filename))
        canvas.SaveAs(filename)
        filenames.append(filename)
    return filenames

def save_plot_impl(fig: matplotlib.figure.Figure,
                   output_prefix: str, output_name: str,
                   printing_extensions: Sequence[str]) -> List[str]:
    """ Implementation of generic save plot function.

    It loops over all requested file extensions and save the matplotlib fig.

    Args:
        fig: Figure on which the plot was drawn.
        output_prefix: File path to where files should be saved.
        output_name: Filename under which the plot should be saved, but without the file extension.
        printing_extensions: List of file extensions under which plots should be saved. They should
            not contain the dot!
    Returns:
        Filenames under which the plot was saved.
    """
    filenames = []
    for extension in printing_extensions:
        filename = os.path.join(output_prefix, output_name + "." + extension)
        logger.debug(f"Saving matplotlib figure to \"{filename}\"")
        fig.savefig(filename)
        filenames.append(filename)
    return filenames

def error_boxes(ax: matplotlib.axes.Axes,
                x_data: np.ndarray, y_data: np.ndarray,
                y_errors: np.ndarray, x_errors: np.ndarray = None,
                **kwargs: Union[str, float]) -> matplotlib.collections.PatchCollection:
    """ Plot error boxes for the given data.

    Inpsired by: https://matplotlib.org/gallery/statistics/errorbars_and_boxes.html and
    https://github.com/HDembinski/pyik/blob/217ae25bbc316c7a209a1a4a1ce084f6ca34276b/pyik/mplext.py#L138

    Args:
        ax: Axis onto which the rectangles will be drawn.
        x_data: x location of the data.
        y_data: y location of the data.
        y_errors: y errors of the data. The array can either be of length n, or of length (n, 2)
            for asymmetric errors.
        x_errors: x errors of the data. The array can either be of length n, or of length (n, 2)
            for asymmetric errors. Default: None. This corresponds to boxes that are 10% of the
            distance between the two given point and the previous one.
    """
    # Validation
    if x_errors is None:
        # Default to 10% of the distance between the two points.
        x_errors = (x_data[1:] - x_data[:-1]) * 0.1
        # Use the last width for the final point. (This is a bit of a hack).
        x_errors = np.append(x_errors, x_errors[-1])
        logger.debug(f"x_errors: {x_errors}")

    # Validate input data.
    if len(x_data) != len(y_data):
        raise ValueError("Length of x_data and y_data doesn't match! x_data: {len(x_data)}, y_data: {len(y_data)}")
    if len(x_errors.T) != len(x_data):
        raise ValueError("Length of x_data and x_errors doesn't match! x_data: {len(x_data)}, x_errors: {len(x_errors)}")
    if len(y_errors.T) != len(y_data):
        raise ValueError("Length of y_data and y_errors doesn't match! y_data: {len(y_data)}, y_errors: {len(y_errors)}")

    # Default arguments
    if "alpha" not in kwargs:
        kwargs["alpha"] = 0.5

    # Create the rectangles
    error_boxes = []
    # We need to transpose the errors, because they are expected to be of the shape (n, 2).
    # NOTE: It will still work as expected if they are only of length n.
    for x, y, xerr, yerr in zip(x_data, y_data, x_errors.T, y_errors.T):
        # For the errors, we want to support symmetric and asymmetric errors.
        # Thus, for asymmetric errors, we sum up the distance, but for symmetric
        # errors, we want to take * 2 of the error.
        xerr = np.atleast_1d(xerr)
        yerr = np.atleast_1d(yerr)
        logger.debug(f"yerr: {yerr}")
        r = matplotlib.patches.Rectangle(
            (x - xerr[0], y - yerr[0]),
            xerr.sum() if len(xerr) == 2 else xerr * 2,
            yerr.sum() if len(yerr) == 2 else yerr * 2,
        )
        error_boxes.append(r)

    # Create the patch collection and add it to the given axis.
    patch_collection = matplotlib.collections.PatchCollection(
        error_boxes, **kwargs,
    )
    ax.add_collection(patch_collection)

    return patch_collection

_T_Color = TypeVar("_T_Color")

def modify_brightness(color: Union[str, Tuple[float, float, float]], amount: float = 0.5) -> Tuple[float, float, float]:
    """ Modifies a color's brightness.

    Lightens or darkens the given color by multiplying (1-luminosity) by the given amount.
    Function from: https://stackoverflow.com/a/49601444

    Examples:

        >>> lighten_color('g', 0.3)
        >>> lighten_color('#F034A3', 0.6)
        >>> lighten_color((.3,.55,.1), 0.5)

    Args:
        color: Color to be lightened or darkened. Can be matplotlib color string, hex string, or RGB tuple.
        amount: Amount to modify the brightness by. Less than 1 makes it lighter,
            greater than 1 makes it darker.
    Returns:
        Modified color in the given format.
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except KeyError:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

# ROOT 6 default color scheme
# Colors extracted from [TColor::SetPalette()](https://root.cern.ch/doc/master/TColor_8cxx_source.html#l02209):
# Added to matplotlib as "ROOT_kBird"
# Color entries are of the form (r, g, b)
# Creation of the color scheme inspired by: https://github.com/matplotlib/matplotlib/blob/master/examples/images_contours_and_fields/custom_cmap.py
bird_root = matplotlib.colors.LinearSegmentedColormap.from_list(
    name = "ROOT_kBird", N = 256,
    colors = [
        (0.2082, 0.1664, 0.5293),
        (0.0592, 0.3599, 0.8684),
        (0.0780, 0.5041, 0.8385),
        (0.0232, 0.6419, 0.7914),
        (0.1802, 0.7178, 0.6425),
        (0.5301, 0.7492, 0.4662),
        (0.8186, 0.7328, 0.3499),
        (0.9956, 0.7862, 0.1968),
        (0.9764, 0.9832, 0.0539)
    ]
)
# Register the colormap with matplotlib
plt.register_cmap(name = bird_root.name, cmap = bird_root)

def prepare_colormap(colormap: matplotlib.colors.Colormap) -> matplotlib.colors.Colormap:
    """ Apply fix to colormaps to remove the need for transparency.

    Since transparency is not support EPS, we change "bad" values (such as NaN in a plot)
    from (0,0,0,0) (this value can be accessed via `cm._rgba_bad`) to white with
    alpha = 1 (no transparency).

    Args:
        colormap: Colormap used to map data to colors.
    Returns:
        The updated colormap.
    """
    # Set bad values to white instead of transparent because EPS doesn't support transparency
    colormap.set_bad("w", 1)

    return colormap

