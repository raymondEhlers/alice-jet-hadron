#!/usr/bin/env python

""" Base plotting module.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

from dataclasses import dataclass
import logging
import os
from typing import List, Optional, Sequence, Union

# Import and configure plotting packages
import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns

from jet_hadron.base import analysis_objects
from jet_hadron.base import labels
from jet_hadron.base.typing_helpers import Canvas, Hist

# Setup logger
logger = logging.getLogger(__name__)

# Setup seaborn
# NOTE: This needs to go before customizing matplotlib because this will override some of those settings.
sns.set(context = "notebook", style = "white")

# Customize matplotlib
# Enable latex
matplotlib.rcParams["text.usetex"] = True
# Enable axis ticks (after they are disabled by seaborn)
matplotlib.rcParams["xtick.bottom"] = True
matplotlib.rcParams["ytick.left"] = True
# Make minor axis ticks visible (but only on left and bottom)
matplotlib.rcParams["xtick.minor.visible"] = True
matplotlib.rcParams["ytick.minor.visible"] = True
matplotlib.rcParams["xtick.minor.top"] = False
matplotlib.rcParams["ytick.minor.right"] = False
# Ensure that axis ticks go inward instead of outward
matplotlib.rcParams["xtick.direction"] = "in"
matplotlib.rcParams["ytick.direction"] = "in"
# Enable AMS math package (for among other things, "\text")
matplotlib.rcParams["text.latex.preamble"].append(r"\usepackage{amsmath}")
# Below, we set the LaTeX fornts to be the same as those used in matplotlib.
# For sans serif fonts in LaTeX (required for setting the fonts below), see: https://stackoverflow.com/a/11612347
# To set the tex fonts to be the same as the normal matplotlib fonts, see: https://stackoverflow.com/a/27697390
matplotlib.rcParams["text.latex.preamble"].append(r"\usepackage{sfmath}")
matplotlib.rcParams["mathtext.fontset"] = "custom"
matplotlib.rcParams["mathtext.rm"] = "Bitstream Vera Sans"
matplotlib.rcParams["mathtext.it"] = "Bitstream Vera Sans:italic"
matplotlib.rcParams["mathtext.bf"] = "Bitstream Vera Sans:bold"

class AnalysisColors:
    """ Exceedingly simple class to store analysis colors. """
    signal = "C0"  # Blue from MPL
    background = "C1"  # Orange from MPL
    fit = "C4"  # Purple from MPL

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

# ROOT 6 default color scheme
# Colors extracted from [TColor::SetPalette()](https://root.cern.ch/doc/master/TColor_8cxx_source.html#l02209):
# Added to matplotlib as "ROOT_kBird"
# Color entries are of the form (r, g, b)
# Creation of the colorscheme inspired by: https://github.com/matplotlib/matplotlib/blob/master/examples/images_contours_and_fields/custom_cmap.py
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

