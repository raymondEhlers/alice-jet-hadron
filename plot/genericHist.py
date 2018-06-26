#!/usr/bin/env python

# Py2/3
from builtins import range
from future.utils import iteritems

import sys
import re
import logging
# Setup logger
logger = logging.getLogger(__name__)

import jetH.base.params as params
import jetH.plot.base as plotBase

import matplotlib.pyplot as plt
import matplotlib.colors
plt.rc('text', usetex=True)
import matplotlib.ticker
# For 3D plots
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

import numpy as np

import rootpy.ROOT as ROOT
import rootpy
import rootpy.plotting.root2matplotlib as rplt
import root_numpy

class HistPlotter(object):
    def __init__(self,
            histNames = None,
            hist = None,
            hists = None,
            outputName = "",
            title = None,
            automaticTitleFromName = False,
            exactNameMatch = False,
            xLabel = None,
            yLabel = None,
            zLabel = None,
            xLimits = None,
            yLimits = None,
            textLabel = None,
            scientificNotationOnAxis = "",
            logy = False,
            logz = False,
            surface = False,
            usePColorMesh = False,
            stepPlot = True):
        # A list of dictionaries, with key histName and value histTitle
        if histNames is None:
            histNames = {}
        self.histNames = histNames
        if hists is None:
            hists = []
        self.hists = hists
        # Store the hist in the list if it was passed
        if hist:
            self.hists.append(hist)
        self.outputName = outputName
        self.title = title
        # Convert hist name to title by splitting on camel case
        self.automaticTitleFromName = automaticTitleFromName
        # If an exact hist name should be required
        self.exactNameMatch = exactNameMatch
        self.xLabel = xLabel
        self.yLabel = yLabel
        self.zLabel = zLabel
        self.xLimits = xLimits
        self.yLimits = yLimits
        self.textLabel = textLabel
        self.scientificNotationOnAxis = scientificNotationOnAxis
        self.logy = logy
        self.logz = logz
        self.surface = surface
        self.usePColorMesh = usePColorMesh
        self.stepPlot = stepPlot

    def getFirstHist(self):
        return next(iter(self.hists))

    def applyHistSettings(self, ax):
        self.applyAxisSettings(ax)

        if self.logy:
            logger.debug("Setting logy")
            ax.set_yscale("log")

        self.applyHistLimits(ax)
        self.applyHistTitles(ax)

    def applyAxisSettings(self, ax):
        # Do not apply useMathText to the axis tick format!
        # It is imcompatible with using latex randering
        # If for some reason latex is turned off, one can use the lines below to only apply the option to axes which
        # support it (most seem to work fine with it)
        #logger.debug("x: {}, y: {}".format(ax.get_xaxis().get_major_formatter(), ax.get_yaxis().get_major_formatter()))
        #if isinstance(ax.get_xaxis().get_major_formatter(), matplotlib.ticker.ScalarFormatter) and isinstance(ax.get_yaxis().get_major_formatter(), matplotlib.ticker.ScalarFormatter):
        #ax.ticklabel_format(axis = "both", useMathText = True)

        if self.scientificNotationOnAxis != "":
            # See: https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.ticklabel_format.html
            # (0,0) means apply to all numbers of the axis
            # axis could be x, y, or both
            ax.ticklabel_format(axis = self.scientificNotationOnAxis, style = "sci", scilimits = (0,0))

    def postDrawOptions(self, ax):
        pass

    def applyHistTitles(self, ax):
        # Title sanity check - we should only set one
        if self.automaticTitleFromName and self.title:
            logger.critical("Set both automatic title extraction and an actual title. Please select only one setting")
            sys.exit(1)

        # Figure title
        if self.automaticTitleFromName:
            tempTitle = self.getFirstHist().GetTitle()
            # Remove the common leading "h" in a hist name
            if tempTitle[0] == "h":
                tempTitle = tempTitle[1:]
            # For the regex, see: https://stackoverflow.com/a/43898219
            title = re.sub('([a-z])([A-Z])', r'\1 \2', tempTitle)
        else:
            title = self.title if self.title else self.getFirstHist().GetTitle()

        if title:
            logger.debug("Set title: {}".format(title))
            ax.set_title(title)

        # Axis labels
        labelMap = {"x" : (self.xLabel, ROOT.TH1.GetXaxis, ax.set_xlabel),
                "y" : (self.yLabel, ROOT.TH1.GetYaxis, ax.set_ylabel)}
        for axisName, (val, axis, applyTitle) in iteritems(labelMap):
            if val:
                label = val
            else:
                label = axis(self.getFirstHist()).GetTitle()
                # Convert any "#" (from ROOT) to "\" for latex
                label = label.replace("#", "\\")
                # Convert "%" -> "\%" to ensure that latex runs successfully
                label = label.replace("%", r"\%")
                # Guessing that units start with the last "(", which is usually the case
                foundLatex = label.find("_") > -1 or label.find("^") > -1 or label.find('\\') > -1
                # Apply latex equation ("$") up to the units
                unitsLocation = label.rfind("(")
                logger.debug("Found latex: {}, label: \"{}\"".format(foundLatex, label))
                if foundLatex:
                    if unitsLocation > -1:
                        label = "$" + label[:unitsLocation-1] + "$ " + label[unitsLocation:]
                    else:
                        label = "$" + label + "$"
            logger.debug("Apply {} axis title with label \"{}\", axis {}, and applyTitle function {}".format(axisName, label, axis, applyTitle))
            applyTitle(label)

    def applyHistLimits(self, ax):
        if not self.xLimits is None:
            logger.debug("Setting x limits of {}".format(self.xLimits))
            ax.set_xlim(self.xLimits)
        if not self.yLimits is None:
            logger.debug("Setting y limits of {}".format(self.yLimits))
            ax.set_ylim(self.yLimits)

    def addTextLabels(self, ax, obj):
        """

        Available properties include:
            - "cellLabel"
            - "clusterLabel"
            - "trackLabel"
            - "textLocation" : [x, y]. Default location is top center
            - "textAlignment" : option for multialignment
            - "fontSize"
            - aliceLabelType (set in the main config)
        """
        if self.textLabel is not None:
            text = ""
            text += params.aliceLabel[obj.aliceLabelType].str()
            if obj.taskLabel.str() != "":
                # We don't want a new line here - we just want to continue it
                text += " " + obj.taskLabel.str()

            text += "\n" + params.systemLabel(energy = obj.collisionEnergy,
                    system = obj.collisionSystem,
                    activity = obj.eventActivity)
            propertyLabels = []
            if self.textLabel.get("cellLabel", False):
                # Handled separately because it is long enough that it has to be
                # on a separate line
                text += "\nCell $E_{\mathrm{seed}} = 100$ MeV, $E_{\mathrm{cell}} = 50$ MeV"
            if self.textLabel.get("clusterLabel", False):
                propertyLabels.append("$E_{cluster} > 300$ MeV")
            if self.textLabel.get("trackLabel", False):
                propertyLabels.append("$p_{T,track} > 150\:\mathrm{MeV/\mathit{c}}$")
            if len(propertyLabels) > 0:
                text += "\n" + ", ".join(propertyLabels)
            #logger.debug("text: {}".format(text))

            # The default location is top center
            ax.text(*self.textLabel.get("textLocation", [0.5, 0.92]), s = text,
                    horizontalalignment = "center",
                    verticalalignment = "center",
                    multialignment = self.textLabel.get("textAlignment", "left"),
                    fontsize = self.textLabel.get("fontSize", 12.5),
                    transform = ax.transAxes)

    def plot(self, obj, outputName = ""):
        # Make the plots
        fig, ax = plt.subplots(figsize=(8,6))

        # Draw the hist
        if self.getFirstHist().InheritsFrom(ROOT.TH2.Class()):
            if len(self.hists) > 1:
                logger.critical("Too many hists are included for a 2D hist. Obj contains {}".format(self.hists))
                sys.exit(1)

            # We can do a few different types of plots:
            #   - Surface plot
            #   - Image plot (like colz), created using:
            #       - imshow() by default
            #       - pcolormesh() as an option
            # NOTE: seaborn heatmap is not really as flexible here, so we'll handle it manually instead.
            #       Since the style is already applied, there isn't really anything lost

            # Retrieve the array corresponding to the data
            # NOTE: The data is transposed from what is normally expected. Apparently this is done to match up with numpy axis conventions
            #       We will have to transpose the data when we go to plot it.
            # NOTE: binEdges is an array which contains edges for each axis. x is 0.
            (histArray, binEdges) = root_numpy.hist2array(self.getFirstHist(), return_edges=True)
            # Set all 0s to nan to get similar behavior to ROOT.In ROOT, it will basically ignore 0s. This is especially important
            # for log plots. Matplotlib doesn't handle 0s as well, since it attempts to plot them and then will throw exceptions
            # when the log is taken.
            # By setting to nan, mpl basically ignores them similar to ROOT
            # NOTE: This requires a few special functions later which ignore nan
            histArray[histArray == 0] = np.nan

            # Define and fill kwargs
            kwargs = {}

            # Set the z axis normalization via this parituclar function
            # Will be called when creating the arugments
            normalizationFunction = matplotlib.colors.Normalize
            if self.logz:
                normalizationFunction = matplotlib.colors.LogNorm
            # Evalute
            kwargs["norm"] = normalizationFunction(vmin = np.nanmin(histArray), vmax = np.nanmax(histArray))
            logger.debug("min: {}, max: {}".format(np.nanmin(histArray), np.nanmax(histArray)))
            # Colormap is the default from sns.heatmap
            kwargs["cmap"] = plotBase.prepareColormap(sns.cm.rocket)
            # Label is included so we could use a legend if we want
            kwargs["label"] = self.getFirstHist().GetTitle()

            logger.debug("kwargs: {}".format(kwargs))

            if self.surface:
                logger.debug("Plotting surface")
                # Need to retrieve a special 3D axis for the surface plot
                ax = plt.axes(projection = "3d")

                # For the surface plot, we want to specify (X,Y) as bin centers. Edges of surfaces
                # will be at these points.
                # NOTE: There are n-1 faces for n points, so not every value will be represented by a face.
                #       However, the location of the points at the bin centers is still correct!
                hist = self.getFirstHist()
                xRange = np.array([hist.GetXaxis().GetBinCenter(i) for i in range(1, hist.GetXaxis().GetNbins()+1)])
                yRange = np.array([hist.GetYaxis().GetBinCenter(i) for i in range(1, hist.GetYaxis().GetNbins()+1)])
                X, Y = np.meshgrid(xRange, yRange)

                # Create the plot
                axFromPlot = ax.plot_surface(X, Y, histArray.T, **kwargs)
            else:
                # Assume an image plot if we don't explicitly select surface
                logger.debug("Plotting as an image")

                # Determine the x, y for the plot.
                #
                # pcolormesh is somewhat like a hist in that (X,Y) should define bin edges. So we need (X,Y) to
                # define lower bin edges, with the last element in the array corresponding to the lower edge of the
                # next (n+1) bin (which is the same as the upper edge of the nth bin).
                #
                # imshow also takes advantage of these limits to determine the extent, but only in a limited way,
                # as it just takes the min and max of each range.
                #
                # NOTE: The addition of epsilon to the max is extremely important! Otherwise, the x and y ranges will
                #       be one bin short since arange is not inclusive. This could also be reolved by using linspace,
                #       but I think this approach is perfectly fine.
                # NOTE: This epsilon is smaller than the one in JetHUtils because we are sometimes dealing with small times (~ns).
                #       The other value is larger because (I seem to recall) that smaller values didn't always place nice with ROOT,
                #       but it is fine here, since we're working with numpy.
                # NOTE: This should be identical to taking the min and max of the axis using TAxis.GetXmin() and TAxis.GetXmax(),
                #       but I prefer this approach.
                epsilon = 1e-9
                xRange = np.arange(np.amin(binEdges[0]), np.amax(binEdges[0]) + epsilon, self.getFirstHist().GetXaxis().GetBinWidth(1))
                yRange = np.arange(np.amin(binEdges[1]), np.amax(binEdges[1]) + epsilon, self.getFirstHist().GetYaxis().GetBinWidth(1))
                X, Y = np.meshgrid(xRange, yRange)

                # Plot with either imshow or pcolormesh
                # Anecdotally, I think pcolormesh is a bit more flexible, but imshow seems to be much faster.
                # According to https://stackoverflow.com/a/21169703, either one can be fine with the proper options.
                # So the plan is to stick with imshow unless pcolormesh is needed for some reason
                if self.usePColorMesh:
                    logger.debug("Plotting with pcolormesh")

                    axFromPlot = plt.pcolormesh(X, Y, histArray.T, **kwargs)
                else:
                    logger.debug("Plotting with imshow ")

                    # This imshow is quite similar to rplt.imshow (it is locaed in plotting.root2matplotlib.imshow)
                    # which can be seen here: https://github.com/rootpy/rootpy/blob/master/rootpy/plotting/root2matplotlib.py#L743
                    # However, we don't copy it directly because we want to set the 0s to nan so they won't be plotted.
                    extent = [np.amin(X), np.amax(X),
                              np.amin(Y), np.amax(Y)]
                    #logger.debug("Extent: {}, binEdges[1]: {}".format(extent, binEdges[1]))
                    axFromPlot = plt.imshow(histArray.T,
                            extent = extent,
                            interpolation = "nearest",
                            aspect = "auto",
                            origin = "lower",
                            **kwargs)

            # Draw the colorbar based on the drawn axis above.
            label = self.zLabel if self.zLabel else self.getFirstHist().GetZaxis().GetTitle()
            # NOTE: The mappable argument must be the separate axes, not the overall one
            # NOTE: This can cause the warning:
            #       '''
            #       matplotlib/colors.py:1031: RuntimeWarning: invalid value encountered in less_equal
            #           mask |= resdat <= 0"
            #       '''
            #       The warning is due to the nan we introduced above. It can safely be ignored
            #       See: https://stackoverflow.com/a/34955622
            #       (Could suppress, but I don't feel it's necessary at the moment)
            fig.colorbar(axFromPlot, label = label)

            # See: https://stackoverflow.com/a/42092305
            #axFromPlot.collections[0].colorbar.set_label(label)

        elif self.getFirstHist().InheritsFrom(ROOT.TH1.Class()):
            # Set colors before plotting because it's not handled automatically
            colorPaletteArgs = {}
            if len(self.hists) > 6:
                # The default seaborn color palette is 6 - beyond that, we need to set the palette ourselves
                # Resources include: https://seaborn.pydata.org/generated/seaborn.color_palette.html
                # and https://seaborn.pydata.org/tutorial/color_palettes.html#palette-tutorial
                # NOTE: "Blues_d" is also a nice scheme for sequential data
                colorPaletteArgs["palette"] = "hls"
                colorPaletteArgs["n_colors"] = len(self.hists)
                logger.debug("Color palette args: {}".format(colorPaletteArgs))
            currentColorPalette = iter(sns.color_palette(**colorPaletteArgs))

            for hist in self.hists:
                logger.debug("Plotting hist: {}, title: {}".format(hist.GetName(), hist.GetTitle()))
                # Convert to more usable form
                (histArray, binEdges) = root_numpy.hist2array(hist, return_edges=True)
                # NOTE: Could remove 0s when taking logs if necessary here

                # NOTE: We plot by hand instead of using rplt.hist() so that we can have better control over the plotting options
                if self.stepPlot:
                    # Step plots look like histograms. They plot some value for some width
                    # According to the documentation, it is basically a thin wrapper around plt.plot() with some modified options
                    # An additional 0 neds to be appended to the end of the data so the arrays are the same length (this appears
                    # to be imposed as an mpl requirement), but it is not meaningful. binEdges[0] corresponds to x axis bin edges
                    # (which are the only relevant ones here)
                    plt.step(binEdges[0], np.append(histArray, [0]), where = "post", label = hist.GetTitle(), color = next(currentColorPalette))
                else:
                    # A more standard plot, with each value at the bin center
                    binCenters = np.array([hist.GetXaxis().GetBinCenter(iBin) for iBin in xrange(1, hist.GetXaxis().GetNbins()+1)])
                    plt.plot(binCenters, histArray, label = hist.GetTitle(), color = next(currentColorPalette))

            # Only plot the legend for 1D hists where it may be stacked. For 2D hists,
            # we will only plot one quantity, so the legend isn't really necessary.
            plt.legend(loc="best")

        # Apply the options
        # Need to apply these here because rplt messes with them!
        self.applyHistSettings(ax)

        # Apply post drawing options
        self.postDrawOptions(ax)

        self.addTextLabels(ax, obj)

        # Final plotting options
        plt.tight_layout()

        # If a name was not passed (ie ""), then use whatever name was given.
        # It is stil possible that that name could be empty, but that should be
        # corrected by the user.
        outputName = outputName if outputName != "" else self.outputName
        if outputName == "":
            logger.critical("No output name passed or set for the object! Please set a name.")
            sys.exit(1)

        # Save and close the figure
        plotBase.savePlot(obj, fig, outputName)
        plt.close(fig)

