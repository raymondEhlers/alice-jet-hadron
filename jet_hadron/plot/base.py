#!/usr/bin/env python

""" Base plotting module.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import logging
import os
from typing import Any, Union, List, Sequence

# Import and configure plotting packages
import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns

from jet_hadron.base.typing_helpers import Canvas

# Setup logger
logger = logging.getLogger(__name__)
# Enable latex
plt.rc('text', usetex=True)
# Potentially improve the layout
# See: https://stackoverflow.com/a/17390833
#from matplotlib import rcParams
#rcParams.update({'figure.autolayout': True})
# Setup seaborn
sns.set(context = "notebook", style = "white")

# For sans serif fonts in LaTeX (required for setting the fonts below)
# See: https://stackoverflow.com/a/11612347
# Set the tex fonts to be the same as the normal matplotlib fonts
# See: https://stackoverflow.com/a/27697390
plt.rc("text.latex", preamble = r"\usepackage{sfmath}")
matplotlib.rcParams["mathtext.fontset"] = "custom"
matplotlib.rcParams["mathtext.rm"] = "Bitstream Vera Sans"
matplotlib.rcParams["mathtext.it"] = "Bitstream Vera Sans:italic"
matplotlib.rcParams["mathtext.bf"] = "Bitstream Vera Sans:bold"

# Wrappers
class PlottingOutputWrapper:
    """ Simple wrapper to allow use of the save_canvas and save_plot wrappers.

    Args:
        output_prefix: File path to where files should be saved.
        printing_extensions: List of file extensions under which plots should be saved.
    """
    def __init__(self, output_prefix: str, printing_extensions: Sequence[str]):
        self.output_prefix = output_prefix
        self.printing_extensions = printing_extensions

def save_canvas(obj: Union[PlottingOutputWrapper, Any], canvas: Canvas, output_path: str) -> List[str]:
    """ Loop over all requested file extensions and save the canvas.

    Args:
        obj (PlottingOutputWrapper or similar): Contains the output_prefix and printing_extensions
        canvas: Canvas on which the plot was drawn.
        output_path: Filename under which the plot should be saved, but without the file extension.
    Returns:
        Filenames under which the plot was saved.
    """
    return save_canvas_impl(canvas, obj.output_prefix, output_path, obj.printing_extensions)

def save_plot(obj: Union[PlottingOutputWrapper, Any], figure: matplotlib.figure.Figure, output_path: str) -> List[str]:
    """ Loop over all requested file extensions and save the current plot in matplotlib.

    Args:
        obj (PlottingOutputWrapper or similar): Contains the output_prefix and printing_extensions
        figure: Figure on which the plot was drawn.
        output_path: Filename under which the plot should be saved, but without the file extension.
    Returns:
        Filenames under which the plot was saved.
    """
    return save_plot_impl(figure, obj.output_prefix, output_path, obj.printing_extensions)

# Base functions
def save_canvas_impl(canvas: Canvas, output_prefix: str, output_path: str, printing_extensions: Sequence[str]) -> List[str]:
    """ Implementation of generic save canvas function.

    It loops over all requested file extensions and save the ROOT canvas.

    Args:
        canvas: Canvas on which the plot was drawn.
        output_prefix: File path to where files should be saved.
        output_path: Filename under which the plot should be saved, but without the file extension.
        printing_extensions (list): List of file extensions under which plots should be saved. They should
            not contain the dot!
    Returns:
        list: Filenames under which the plot was saved.
    """
    filenames = []
    for extension in printing_extensions:
        filename = os.path.join(output_prefix, output_path + "." + extension)
        # Probably don't want this log message since ROOT will also generate a message
        #logger.debug("Saving ROOT canvas to \"{}\"".format(filename))
        canvas.SaveAs(filename)
        filenames.append(filename)
    return filenames

def save_plot_impl(fig: matplotlib.figure.Figure, output_prefix: str, output_path: str, printing_extensions: Sequence[str]) -> List[str]:
    """ Implementation of generic save plot function.

    It loops over all requested file extensions and save the matplotlib fig.

    Args:
        fig: Figure on which the plot was drawn.
        output_prefix: File path to where files should be saved.
        output_path: Filename under which the plot should be saved, but without the file extension.
        printing_extensions: List of file extensions under which plots should be saved. They should
            not contain the dot!
    Returns:
        Filenames under which the plot was saved.
    """
    filenames = []
    for extension in printing_extensions:
        filename = os.path.join(output_prefix, output_path + "." + extension)
        logger.debug("Saving matplotlib figure to \"{}\"".format(filename))
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

