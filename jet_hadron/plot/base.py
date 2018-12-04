#!/usr/bin/env python

######################
# Base plotting module
######################

import os
# Setup logger
import logging
logger = logging.getLogger(__name__)

# Import and configure plotting packages
import matplotlib
import matplotlib.pyplot as plt
# Enable latex
plt.rc('text', usetex=True)
# Potentially improve the layout
# See: https://stackoverflow.com/a/17390833
#from matplotlib import rcParams
#rcParams.update({'figure.autolayout': True})
import seaborn as sns
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
class plottingOutputWrapper(object):
    """ Simple wrapper to allow use of the saveCanvas and savePlot wrappers.

    Args:
        outputPrefix (str): File path to where files should be saved.
        printingExtensions (list): List of file extensions under which plots should be saved.
    """
    def __init__(self, outputPrefix, printingExtensions):
        self.outputPrefix = outputPrefix
        self.printingExtensions = printingExtensions

def saveCanvas(obj, canvas, outputPath):
    """ Loop over all requested file extensions and save the canvas.

    Args:
        obj (plottingOutputWrapper or similar): Contains the outputPrefix and printingExtensions
        canvas (ROOT.TCanvas): Canvas on which the plot was drawn.
        outputPath (str): Filename under which the plot should be saved, but without the file extension.
    Returns:
        list: Filenames under which the plot was saved.
    """
    return saveCanvasImpl(canvas, obj.outputPrefix, outputPath, obj.printingExtensions)

def savePlot(obj, figure, outputPath):
    """ Save the current plot in matplotlib.

    Args:
        obj (plottingOutputWrapper or similar): Contains the outputPrefix and printingExtensions
        figure (matplotlib.Figure): Figure on which the plot was drawn.
        outputPath (str): Filename under which the plot should be saved, but without the file extension.
    Returns:
        list: Filenames under which the plot was saved.
    """
    return savePlotImpl(figure, obj.outputPrefix, outputPath, obj.printingExtensions)

# Base functions
def saveCanvasImpl(canvas, outputPrefix, outputPath, printingExtensions):
    """ Implementation of generic save canvas function. It loops over all requested file
    extensions and save the ROOT canvas.

    Cannot be named the same because python won't differeniate by number of arguments.

    Args:
        canvas (ROOT.TCanvas): Canvas on which the plot was drawn.
        outputPrefix (str): File path to where files should be saved.
        outputPath (str): Filename under which the plot should be saved, but without the file extension.
        printingExtensions (list): List of file extensions under which plots should be saved. They should
            not contain the dot!
    Returns:
        list: Filenames under which the plot was saved.
    """
    filenames = []
    for extension in printingExtensions:
        filename = os.path.join(outputPrefix, outputPath + "." + extension)
        # Probably don't want this log message since ROOT will also generate a message
        #logger.debug("Saving ROOT canvas to \"{}\"".format(filename))
        canvas.SaveAs(filename)
        filenames.append(filename)
    return filenames

def savePlotImpl(fig, outputPrefix, outputPath, printingExtensions):
    """ Implementation of generic save plot function. It loops over all requested file
    extensions and save the matplotlib fig.

    Cannot be named the same because python won't differeniate by number of arguments.

    Args:
        fig (matplotlib.figure): Figure on which the plot was drawn.
        outputPrefix (str): File path to where files should be saved.
        outputPath (str): Filename under which the plot should be saved, but without the file extension.
        printingExtensions (list): List of file extensions under which plots should be saved. They should
            not contain the dot!
    Returns:
        list: Filenames under which the plot was saved.
    """
    filenames = []
    for extension in printingExtensions:
        filename = os.path.join(outputPrefix, outputPath + "." + extension)
        logger.debug("Saving matplotlib figure to \"{}\"".format(filename))
        fig.savefig(filename)
        filenames.append(filename)
    return filenames

# ROOT 6 default color scheme
# Colors extracted from [TColor::SetPalette()](https://root.cern.ch/doc/master/TColor_8cxx_source.html#l02209):
# Added to matplotlib as "ROOT_kBird"
# Color entries are of the form (r, g, b)
# Creation of the colorscheme inspired by: https://github.com/matplotlib/matplotlib/blob/master/examples/images_contours_and_fields/custom_cmap.py
birdRoot = matplotlib.colors.LinearSegmentedColormap.from_list(name = "ROOT_kBird", N = 256,
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
                                                               ])
# Register the colormap with matplotlib
plt.register_cmap(name = birdRoot.name, cmap = birdRoot)

def prepareColormap(colormap):
    """ Apply fix to colormaps to remove the need for transparency.

    Since transparency is not support EPS, we change "bad" values (such as NaN in a plot)
    from (0,0,0,0) (this value can be accessed via `cm._rgba_bad`) to white with
    alpha = 1 (no transparency).

    Args:
        colormap (matplotlib.colors colormap): Colormap used to map data to colors.
    Returns:
        colormap: The updated colormap.
    """
    # Set bad values to white instead of transparent because EPS doesn't support transparency
    colormap.set_bad("w", 1)

    return colormap

