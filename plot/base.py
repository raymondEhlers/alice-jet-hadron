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
sns.set(context = "notebook", style = "whitegrid")

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
    """ Loop over all requested file extensions and save the canvas. """
    saveCanvasImpl(canvas, obj.outputPrefix, outputPath, obj.printingExtensions)

def savePlot(obj, figure, outputPath):
    """ Save the current plot in matplotlib. """
    savePlotImpl(figure, obj.outputPrefix, outputPath, obj.printingExtensions)

# Base functions
def saveCanvasImpl(canvas, outputPrefix, outputPath, printingExtensions):
    """ Implementation of generic save canvas function.
    Cannot be named the same because python won't differeniate by number of arguments.

    Loop over all requested file extensions and save the canvas. """
    for extension in printingExtensions:
        filename = os.path.join(outputPrefix, outputPath + "." + extension)
        # Probably don't want this log message since ROOT will also generate a message
        #logger.debug("Saving ROOT canvas to \"{}\"".format(filename))
        canvas.SaveAs(filename)

def savePlotImpl(fig, outputPrefix, outputPath, printingExtensions):
    """ Implementation of generic save plot function.
    Cannot be named the same because python won't differeniate by number of arguments.

    Loop over all requested file extensions and save the matplotlib fig. """
    for extension in printingExtensions:
        filename = os.path.join(outputPrefix, outputPath + "." + extension)
        logger.debug("Saving matplotlib figure to \"{}\"".format(filename))
        fig.savefig(filename)

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

    Since transparency is not support EPS, we change "bad" values (such as nan) from (0,0,0,0)
    (this value can be accessed via `cm._rgba_bad`) to white with alpha = 1 (no transparency)

    Args:
        colormap (matplotlib.colors colormap): Colormap used to map data to colors.
    """
    # Set bad values to white instead of transparent because EPS doesn't support transparency
    colormap.set_bad("w",1)

    return colormap

