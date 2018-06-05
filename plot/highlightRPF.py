#!/usr/bin/env python

# Highlight Reaction Plane Fit fit regions corresponding to the signal
# and background regions.
#
# NOTE: This doesn't quite follow the traditional split between functions and plots
#       because it also works as a standalone split. However, most data functions are
#       split off into the RPF utils module.
#
# Dependencies:
#   - rootpy
#   - numpy
#   - root_numpy
#   - matplotlib
#   - seaborn
#
# author: Raymond Ehlers <raymond.ehlers@yale.edu>, Yale University
# date: 03 May 2018

import argparse
import logging
# Setup logger
logger = logging.getLogger(__name__)
import sys
import warnings
# Handle rootpy warning
warnings.filterwarnings(action='ignore', category=RuntimeWarning, message=r'creating converter for unknown type "_Atomic\(bool\)"')
thisModule = sys.modules[__name__]

import numpy as np

import PlotBase

# Import plotting packages
import matplotlib
import matplotlib.pyplot as plt
# Needed for 3D plots
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

import rootpy.ROOT as ROOT
# Tell ROOT to ignore command line options so args are passed to python
# NOTE: Must be immediately after import ROOT!
ROOT.PyConfig.IgnoreCommandLineOptions = True
import rootpy
import rootpy.io

import JetHUtils

##########
# Plotting
##########
def convertColorToMax255(color):
    """ Convert color to be in the range [0,255]. """
    return int(round(color * 255))

def convertColorToMax1(color):
    """ Convert color to be in the range [0,1]. """
    return color / 255.0

def overlayColors(foreground, background):
    """ Combine two colors together using an "overlay" method.

    Implemented using the formula from [colorblendy](http://colorblendy.com/). Specifically, see
    static/js/colorlib.js:blend_filters in the colorblendy github.

    NOTE: These formulas are based on colors from [0, 255], so we need to scale our [0, 1] colors
        up and down.

    Args:
        foreground (tuple): (R,G,B), where each color is between [0, 1]
        background (tuple): (R,G,B), where each color is between [0, 1]
    Returns:
        tuple: (R,G,B) overlay of colors
    """

    output = []
    for fg, bg in zip(foreground, background):
        # Need to scale up to the 255 scale to use the formula
        fg = convertColorToMax255(fg)
        bg = convertColorToMax255(bg)
        overlayColor = (2 * fg * bg / 255.) if bg < 128. else (255 - 2 * (255 - bg) * (255 - fg) / 255.)
        # We then need to scale the color to be between 0 and 1.
        overlayColor = convertColorToMax1(overlayColor)
        output.append(overlayColor)

    return tuple(output)

def screenColors(foreground, background):
    """ Combine two colors together using a "screen" method.

    Implemented using the formula from [colorblendy](http://colorblendy.com/). Specifically, see
    static/js/colorlib.js:blend_filters in the colorblendy github.

    NOTE: These formulas are based on colors from [0, 255], so we need to scale our [0, 1] colors
        up and down.

    Args:
        foreground (tuple): (R,G,B), where each color is between [0, 1]
        background (tuple): (R,G,B), where each color is between [0, 1]
    Returns:
        tuple: (R,G,B) overlay of colors
    """
    output = []
    for fg, bg in zip(foreground, background):
        # Need to scale up to the 255 scale to use the formula
        fg = convertColorToMax255(fg)
        bg = convertColorToMax255(bg)
        screenColor = 255 - (((255-fg)*(255-bg)) >> 8)
        # We then need to scale the color to be between 0 and 1.
        screenColor = convertColorToMax1(screenColor)
        output.append(screenColor)

    return tuple(output)

def highlightRegionOfSurface(surf, highlightColors, X, Y, phiRange, etaRange, useEdgeColors = False, useColorOverlay = False, useColorScreen = False):
    """ Highlight a region of a surface plot.

    The colors of the selected region are changed from their default value on the surface plot to the selected
    highliht colors.  Adapted from: https://stackoverflow.com/a/5276486

    Args:
        surf (Poly3DCollection): Surface on which regions will be highlighted.
        highlightColors (tuple): The highlight color in the form (r,g,b, alpha).
        X (numpy.ndarray): X bin centers.
        Y (numpy.ndarray): Y bin centers.
        phiRange (float, float): Min, max phi of highlight region.
        etaRange (float, float): Min, max eta of highlight region.
        useEdgeColors (bool): Use edge colors instead of face colors. Default: False
        useColorOverlay (bool): Combine the highlight color and existing color using overlayColors(). Default: False
        useColorScreen (bool): Combine the highlight color and existing color using screenColors(). Default: False
    Return:
        numpy.ndarray: Indices selected for highlighting
    """
    # Get tuple values
    phiMin, phiMax = phiRange
    etaMin, etaMax = etaRange

    # define a region to highlight
    highlight = (X > phiMin) & (X < phiMax) & (Y > etaMin) & (Y < etaMax)
    # Get the original colors so they can be updated.
    # NOTE: Using get_facecolors() requires a workaround which copies 3d colors to 2d colors.
    #       Otherwise, the colors be retrieved directly via: `colors = surf._facecolors3d`.
    #       The equivalent also applies to edge colors
    if useEdgeColors:
        colors = surf.get_edgecolors()
    else:
        colors = surf.get_facecolors()
    #logger.debug("colors.shape: {}, highlight: {}".format(colors.shape, highlight.shape))
    #logger.debug("colors: {}".format(colors))

    # Update the colors with the highlight color.
    # The colors are stored as a list for some reason so get the flat indicies
    logger.debug("max idx: {}".format(np.max(np.where(highlight[:-1,:-1].flat)[0])))
    for idx in np.where(highlight[:-1,:-1].flat)[0]:
        # Modify each color one-by-one
        if useColorOverlay:
            colors[idx] = list(overlayColors(foreground = highlightColors, background = tuple(colors[idx])))
        elif useColorScreen:
            colors[idx] = list(screenColors(foreground = highlightColors, background = tuple(colors[idx])))
        else:
            colors[idx] = list(highlightColors)

    # Reset the colors
    if useEdgeColors:
        surf.set_edgecolors(colors)
    else:
        surf.set_facecolors(colors)

    return highlight

def surfacePlotForHighlighting(ax, X, Y, histArray, colormap, colorbar = False):
    """ Plot a surface with the passed arguments for use with highlighting a region.

    This function could be generally useful for plotting surfaces. However, it should
    be noted that the arguments are specifically optimized for plotting a highlighted
    region.

    Args:
        X (numpy.ndarray): X bin centers.
        Y (numpy.ndarray): Y bin centers.
        histArray (numpy.ndarray): Histogram data as 2D array.
        colormap (matplotlib.colors colormap): Colormap used to map the data to colors.
        colorbar (bool): True if a colorbar should be added.

    Returns:
        Poly3DCollection: Value returned by the surface plot.
    """
    # NOTE: {r,c}count tells the surf plot how many times to sample the data. By using len(),
    #       we ensure that every point is plotted.
    # NOTE: Cannot just use norm and cmap as args to plot_surface, as this seems to return colors only based
    #       on value, instead of based on (x, y). To work around this, we calculate the norm separately,
    #       and then use that to set the facecolors manually, which appears to set the colors based on
    #       (x, y) position. Inspired by https://stackoverflow.com/a/42927880
    norm = matplotlib.colors.Normalize(vmin = np.min(histArray), vmax = np.max(histArray))
    surf = ax.plot_surface(X, Y, histArray.T,
            facecolors = colormap(norm(histArray.T)),
            #norm = matplotlib.colors.Normalize(vmin = np.min(histArray), vmax = np.max(histArray)),
            #cmap = sns.cm.rocket,
            rcount = len(histArray.T[:,0]),
            ccount = len(histArray.T[0]))

    if colorbar:
        # NOTE: Cannot use surf directly, because it doesn't contain the mapping from data to color
        #       Instead, we basically handle it by hand.
        m = matplotlib.cm.ScalarMappable(cmap = colormap, norm = surf.norm)
        m.set_array(histArray.T)
        ax.colorbar(m)

    # Need to manually update the 2d colors based on the 3d colors to be able to use `get_facecolors()`.
    # This workaround is from: https://github.com/matplotlib/matplotlib/issues/4067
    surf._facecolors2d = surf._facecolors3d
    surf._edgecolors2d = surf._edgecolors3d

    return surf

def contourPlotForHighlighting(ax, X, Y, histArray, colormap, levelStep = 0.002):
    """ Plot a contour for the region highlighting plot.

    Included as an option because ROOT more or less includes one, but it doesn't really
    have an impact on the output, so it can probably be ignored.

    Args:
        X (numpy.ndarray): X bin centers.
        Y (numpy.ndarray): Y bin centers.
        histArray (numpy.ndarray): Histogram data as 2D array.
        colormap (matplotlib.colors colormap): Colormap used to map the data to colors.
        levelStep (float): Step in z between each contour

    Returns:
        matplotlib.axes.Axes.contour: Value returned by the contour plot.
    """
    contour = ax.contour(X, Y, histArray.T,
                         zdir='z',
                         alpha = 1,
                         cmap = colormap,
                         levels = np.arange(np.min(histArray), np.max(histArray), levelStep),
                         norm = norm)

    return contour

def plotRPFFitRegions(hist, highlightRegions, colormap = sns.cm.rocket, useTransparencyInHighlights = False, plotContour = False, viewAngle = (35, 225), **highlightArgs):
    """ Highlight the RPF fit range corresponding to the selected highlightRegions

    Recommended colormap options include:
        - seaborn heatmap: `sns.cm.rocket`
        - ROOT kBird (apparently basically the same as "parula" in Matlab...), "ROOT_kBird"
        - Better matplotlib default colormaps: "viridis" looks promising. Other options
          are described here: https://bids.github.io/colormap/

    Args:
        hist (numpy.ndarray, numpy.ndarray, numpy.ndarray): Tuple of (x bin centers, y bin centers, array
            of hist data).  Useful to use in conjunction with getHistogramDataForSurfacePlot(hist).
        highlightRegions (list): highlightRegion objects corresponding to the signal and background regions to
            be highlighted.
        colormap (colormap or str): Either a colormap or the name of a colormap to retrieve via `plt.get_cmap(colormap)`.
            Default: seaborn.cm.rocket.
        useTransparencyInHighlights (bool): If true, allow for transparency in the highlight colors by plotting the
            surface twice.  Only the colors of the upper surface will be modified when highlighting. Default: False.
        plotContour (bool): If true, also plot a contour. Included as an option because ROOT seems to use it, but
            it doesn't seem to have an impact on the image here. Default: False.
        viewAngle (float, float): Set the view angle of the output, in the form
            of (elevation, azimuth). Default: (35, 225).
        highlightArgs (dict): Additional arguments for how colors highlighting the surface are blended. Passed to
            highlightRegionOfSurface(), so see the arguments there.
    Returns:
        tuple: (fig, ax)
    """
    X, Y, histArray = hist
    # Set view angle after the function definition because python3 doesn't support
    # tuple parameter unpacking in functions
    #if viewAngle is None:
    #    viewAngle = None

    # Setup plot
    fig = plt.figure(figsize=(10,7.5))
    # We need to create the axis in a special manner to be able to use 3d projections such
    # as surface or contour.
    ax = plt.axes(projection='3d')

    # Configure the colormap
    # If passed a name instead of a colormap directly, retrieve the colormap via mpl
    if isinstance(colormap, str):
        colormap = plt.get_cmap(colormap)
    # Set bad values to white with not transparency
    colormap = PlotBase.prepareColormap(colormap)

    # Plot surface(s)
    surf = surfacePlotForHighlighting(ax, X, Y, histArray, colormap = colormap)
    # Surface to be used for highlighting the regions
    highlightSurf = surf
    if useTransparencyInHighlights:
        surf2 = surfacePlotForHighlighting(ax, X, Y, histArray, colormap = colormap)
        highlightSurf = surf2

    if plotContour:
        contour = contourPlotForHighlighting(ax, X, Y, histArray, colormap = colormap)
    
    # Draw highlights
    for region in highlightRegions:
        region.drawHighlights(highlightSurf, X, Y, **highlightArgs)

    # Add labels
    ax.set_xlabel(r"$\Delta\varphi$")
    ax.set_ylabel(r"$\Delta\eta$")
    # Create entries for the highlight regions
    handles = []
    for region in highlightRegions:
        handles.append(matplotlib.patches.Patch(color = region.color, label = region.label))
    ax.legend(handles = handles)

    # View options
    ax.view_init(*viewAngle)
    plt.tight_layout()

    return (fig, ax)

###################
# Highlight regions
###################
class highlightRegion(object):
    """ Manages regions for highlighting.

    Args:
        label (str): Label for the highlighted region(s)
        color (tuple): The highlight color in the form (r,g,b, alpha).
    """
    def __init__(self, label, color):
        self.label = label
        self.color = color
        self.regions = []

    def addHighlightRegion(self, phiRange, etaRange):
        """ Add a region to this highlight object.
        
        Args:
            phiRange (float, float): Tuple of min, max phi of highlight region.
            etaRange (float, float): Tuple of min, max eta of highlight region.
        """
        self.regions.append([phiRange, etaRange])

    def drawHighlights(self, highlightSurf, X, Y, **kwargs):
        """ Highlight the seleteced region(s) on the passed surface.

        It will loop over all selected regions for this highlight object.

        Args:
            highlightSurf (Poly3DCollection): Surface on which regions will be highlighted.
            X (numpy.ndarray): X bin centers.
            Y (numpy.ndarray): Y bin centers.
            kwargs (dict): Additional options for highlightRegionOfSurface().
        """
        for region in self.regions:
            # 0 is the phi tuple, while 1 is the eta tuple
            highlightRegionOfSurface(highlightSurf, self.color, X, Y, region[0], region[1], **kwargs)

def defineHighlightRegions():
    """ Define regions to highlight.

    The user should modify or override this function if they want to define different ranges. By default,
    we highlight.

    Args:
        None
    Returns:
        list: highlightRegion objects, suitably defined for highlighting the signal and background regions.
    """
    # Select the highlighted regions.
    highlightRegions = []
    # NOTE: The edge color is still that of the colormap, so there is still a hint of the origin
    #       colormap, although the facecolors are replaced by selected highlight colors

    # Signal
    signalColor = (1, 0, 0, 1.0,)
    signalRegion = highlightRegion("Signal region", signalColor)
    signalRegion.addHighlightRegion((-np.pi/2, 3.0*np.pi/2), (-0.6, 0.6))
    highlightRegions.append(signalRegion)

    # Background
    backgroundColor = (0, 1, 0, 1.0,)
    backgroundPhiRange = (-np.pi/2, np.pi/2)
    backgroundRegion = highlightRegion("Background region", backgroundColor)
    backgroundRegion.addHighlightRegion(backgroundPhiRange, (-1.2, -0.8))
    backgroundRegion.addHighlightRegion(backgroundPhiRange, ( 0.8,  1.2))
    highlightRegions.append(backgroundRegion)

    return highlightRegions

##########################
# Standalone functionality
##########################
def plotRPFRegions(inputFile, histName, outputPrefix = ".", printingExtensions = ["pdf"]):
    # Basic setup
    # Create logger
    logging.basicConfig(level=logging.DEBUG)
    # Quiet down the matplotlib logging
    logging.getLogger("matplotlib").setLevel(logging.INFO)

    # Retrieve hist
    with rootpy.io.root_open(inputFile, "READ") as f:
        hist = f.Get(histName)
        hist.SetDirectory(0)

    # Increase the size of the fonts, etc
    with sns.plotting_context(context = "notebook", font_scale = 1.5):
        # See the possible arguments to highlightRegionOfSurface()
        # For example, to turn on the color overlay, it would be:
        #highlightArgs = {"useColorScreen" : True}
        highlightArgs = {}
        # Call plotting functions
        (fig, ax) = plotRPFFitRegions(JetHUtils.getArrayFromHist2D(hist),
                highlightRegions = defineHighlightRegions(),
                colormap = "ROOT_kBird",
                **highlightArgs)

        # Modify axis labels
        # Set the distance from axis to label in pixels.
        # This is not ideal, but clearly tight_layout doesn't work as well for 3D plots
        ax.xaxis.labelpad = 12
        # Visually, dEta looks closer
        ax.yaxis.labelpad = 15

        # If desired, add additional labeling here.
        # Probably want to use, for examxple, `ax.text2D(0.1, 0.9, "test text", transform = ax.transAxes)`
        # This would place the text in the upper left corner (it's like NDC)

        # Save and finish up
        # The figure will be saved at outputPrefix/outputPath.printingExtension
        outputWrapeer = PlotBase.plottingOutputWrapper(outputPrefix = outputPrefix, printingExtensions = printingExtensions)
        PlotBase.savePlot(outputWrapeer, fig, outputPath = "highlightRPFRegions")
        plt.close(fig)

def parseArguments():
    """ Parse arguments to this module. """
    # Setup command line parser
    parser = argparse.ArgumentParser(description = "Creating plot to illustrate Reaction Plane Fit fit ranges.")
    required = parser.add_argument_group("required arguments")
    required.add_argument("-f", "--inputFile", metavar="inputFile",
                         type=str,
                         help="Path to input ROOT filename.")
    required.add_argument("-i", "--inputHistName", metavar="histName",
                         type=str,
                         help="Name of hist.")
    parser.add_argument("-o", "--outputPath", metavar="outputPath",
                        type=str, default = ".",
                        help="Path to where the printed hist should be saved.")
    parser.add_argument("-p", "--printingExtensions", action="store",
                        nargs="*", metavar="extensions",
                        default=["pdf"],
                        help="Printing extensions for saving plots. Can specify more than one. Do not include the period.")

    # Parse arguments
    args = parser.parse_args()

    return (args.inputFile, args.inputHistName, args.outputPath, args.printingExtensions)

def runFromTerminal():
    (inputFile, histName, outputPath, printingExtensions) = parseArguments()
    plotRPFRegions(inputFile = inputFile,
            histName = histName,
            outputPrefix = outputPath,
            printingExtensions = printingExtensions)

if __name__ == "__main__":
    runFromTerminal()
