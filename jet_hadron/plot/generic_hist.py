#!/usr/bin/env python

""" Generic histograms plotting module.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.ticker
# For 3D plots
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401. Needed for 3D plots, even if not directly called.
import re
import seaborn as sns
from typing import Dict, Union

import rootpy.ROOT as ROOT
import root_numpy

from pachyderm import histogram

from jet_hadron.base import params
from jet_hadron.plot import base as plotBase

# Setup logger
logger = logging.getLogger(__name__)
# Use latex in matplotlib.
plt.rc('text', usetex=True)

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
            ax.ticklabel_format(axis = self.scientificNotationOnAxis, style = "sci", scilimits = (0, 0))

    def postDrawOptions(self, ax):
        pass

    def applyHistTitles(self, ax):
        # Title sanity check - we should only set one
        if self.automaticTitleFromName and self.title:
            raise ValueError("Set both automatic title extraction and an actual title. Please select only one setting")

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
        labelMap = {"x": (self.xLabel, ROOT.TH1.GetXaxis, ax.set_xlabel),
                    "y": (self.yLabel, ROOT.TH1.GetYaxis, ax.set_ylabel)}
        for axisName, (val, axis, applyTitle) in labelMap.items():
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
                        label = "$" + label[:unitsLocation - 1] + "$ " + label[unitsLocation:]
                    else:
                        label = "$" + label + "$"
            logger.debug("Apply {} axis title with label \"{}\", axis {}, and applyTitle function {}".format(axisName, label, axis, applyTitle))
            applyTitle(label)

    def applyHistLimits(self, ax):
        if self.xLimits is not None:
            logger.debug("Setting x limits of {}".format(self.xLimits))
            ax.set_xlim(self.xLimits)
        if self.yLimits is not None:
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
            - aliceLabel (set in the main config)
        """
        if self.textLabel is not None:
            text = ""
            text += str(obj.alice_label)
            if str(obj.task_label) != "":
                # We don't want a new line here - we just want to continue it
                text += " " + str(obj.task_label)

            text += "\n" + params.system_label(energy = obj.collision_energy,
                                               system = obj.collision_system,
                                               activity = obj.event_activity)
            propertyLabels = []
            if self.textLabel.get("cellLabel", False):
                # Handled separately because it is long enough that it has to be
                # on a separate line
                text += "\n" + r"Cell $E_{\mathrm{seed}} = 100$ MeV, $E_{\mathrm{cell}} = 50$ MeV"
            if self.textLabel.get("clusterLabel", False):
                propertyLabels.append(r"$E_{cluster} > 300$ MeV")
            if self.textLabel.get("trackLabel", False):
                propertyLabels.append(r"$p_{T,track} > 150\:\mathrm{MeV/\mathit{c}}$")
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

    def plot(self, obj, outputName: str = "") -> None:
        # Make the plots
        fig, ax = plt.subplots(figsize=(8, 6))

        # Draw the hist
        if isinstance(self.getFirstHist(), ROOT.TH2):
            self.plot_2D_hists(fig = fig, ax = ax)
        elif isinstance(self.getFirstHist(), ROOT.TH1):
            self.plot_1D_hists()

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
            raise ValueError("No output name passed or set for the object! Please set a name.")

        # Save and close the figure
        plotBase.savePlot(obj, fig, outputName)
        plt.close(fig)

    def plot_2D_hists(self, fig, ax) -> None:
        """ Plot a 2D hist using matplotlib.

        Note:
            This is is restricted to plotting only one hist (because how would you plot multiple
            2D hists on one canvas)?

        Args:
            fig: Figure from matplotlib
            ax: Axis from matplotlib.
        Raises:
            ValueError: If more than one histogram is stored in the ``hists``, which doesn't make sense for
                2D hist, which doesn't make sense for 2D hists.
        """
        if len(self.hists) > 1:
            raise ValueError(self.hists, "Too many hists are included for a 2D hist. Should only be one!")

        # We can do a few different types of plots:
        #   - Surface plot
        #   - Image plot (like colz), created using:
        #       - imshow() by default
        #       - pcolormesh() as an option
        # NOTE: seaborn heatmap is not really as flexible here, so we'll handle it manually instead.
        #       Since the style is already applied, there isn't really anything lost

        # Retrieve the array corresponding to the data
        # NOTE: The data is transposed from what is normally expected. Apparently this is done to match
        #       up with numpy axis conventions. We will have to transpose the data when we go to plot it.
        #       We continue using this root_numpy convention even after swtiching to pachyderm for consistency.
        X, Y, hist_array = histogram.get_array_from_hist2D(
            hist = self.getFirstHist(),
            set_zero_to_NaN = True,
            return_bin_edges = not self.surface
        )
        (test_hist_array, test_bin_edges) = root_numpy.hist2array(self.getFirstHist(), return_edges=True)
        test_hist_array[test_hist_array == 0] = np.nan
        #test_hist_array = test_hist_array.T

        assert np.allclose(hist_array, test_hist_array, equal_nan = True)
        assert np.allclose(hist_array.T, test_hist_array.T, equal_nan = True)
        assert np.isclose(np.nanmin(hist_array), np.nanmin(test_hist_array))
        assert np.isclose(np.nanmax(hist_array), np.nanmax(test_hist_array))

        # Define and fill kwargs
        kwargs = {}

        # Set the z axis normalization via this parituclar function
        # Will be called when creating the arugments
        normalizationFunction = matplotlib.colors.Normalize
        if self.logz:
            normalizationFunction = matplotlib.colors.LogNorm
        # Evalute
        kwargs["norm"] = normalizationFunction(vmin = np.nanmin(hist_array), vmax = np.nanmax(hist_array))
        logger.debug("min: {}, max: {}".format(np.nanmin(hist_array), np.nanmax(hist_array)))
        # Colormap is the default from sns.heatmap
        kwargs["cmap"] = plotBase.prepareColormap(sns.cm.rocket)
        # Label is included so we could use a legend if we want
        kwargs["label"] = self.getFirstHist().GetTitle()

        logger.debug("kwargs: {}".format(kwargs))

        if self.surface:
            logger.debug("Plotting surface")
            # Need to retrieve a special 3D axis for the surface plot
            ax = plt.axes(projection = "3d")

            test_hist = self.getFirstHist()
            xRange = np.array(
                [test_hist.GetXaxis().GetBinCenter(i) for i in range(1, test_hist.GetXaxis().GetNbins() + 1)]
            )
            yRange = np.array(
                [test_hist.GetYaxis().GetBinCenter(i) for i in range(1, test_hist.GetYaxis().GetNbins() + 1)]
            )
            test_X, test_Y = np.meshgrid(xRange, yRange)

            assert np.allclose(X, test_X, equal_nan = True)
            assert np.allclose(Y, test_Y, equal_nan = True)

            # For the surface plot, we want to specify (X,Y) as bin centers. Edges of surfaces
            # will be at these points.
            # NOTE: There are n-1 faces for n points, so not every value will be represented by a face.
            #       However, the location of the points at the bin centers is still correct!
            # Create the plot
            axFromPlot = ax.plot_surface(X, Y, hist_array.T, **kwargs)
        else:
            # Assume an image plot if we don't explicitly select surface
            logger.debug("Plotting as an image")

            # pcolormesh is somewhat like a hist in that (X,Y) should define bin edges. So we need (X,Y)
            # to define lower bin edges, with the last element in the array corresponding to the lower
            # edge of the next (n+1) bin (which is the same as the upper edge of the nth bin).
            #
            # imshow also takes advantage of these limits to determine the extent, but only in a limited
            # way, as it just takes the min and max of each range.
            #

            epsilon = 1e-9
            xRange = np.arange(
                np.amin(test_bin_edges[0]),
                np.amax(test_bin_edges[0]) + epsilon,
                self.getFirstHist().GetXaxis().GetBinWidth(1)
            )
            yRange = np.arange(
                np.amin(test_bin_edges[1]),
                np.amax(test_bin_edges[1]) + epsilon,
                self.getFirstHist().GetYaxis().GetBinWidth(1)
            )
            test_X, test_Y = np.meshgrid(xRange, yRange)

            assert np.allclose(X, test_X, equal_nan = True)
            assert np.allclose(Y, test_Y, equal_nan = True)

            # Plot with either imshow or pcolormesh
            # Anecdotally, I think pcolormesh is a bit more flexible, but imshow seems to be much faster.
            # According to https://stackoverflow.com/a/21169703, either one can be fine with the proper
            # options. So the plan is to stick with imshow unless pcolormesh is needed for some reason
            if self.usePColorMesh:
                logger.debug("Plotting with pcolormesh")

                axFromPlot = plt.pcolormesh(X, Y, hist_array.T, **kwargs)
            else:
                logger.debug("Plotting with imshow ")

                # This imshow is quite similar to rplt.imshow (it is locaed in
                # ``plotting.root2matplotlib.imshow``), which can be seen here:
                # https://github.com/rootpy/rootpy/blob/master/rootpy/plotting/root2matplotlib.py#L743
                # However, we don't copy it directly because we want to set the 0s to nan so they won't
                # be plotted.
                extent = [np.amin(X), np.amax(X),
                          np.amin(Y), np.amax(Y)]

                test_extent = [np.amin(test_X), np.amax(test_X),
                               np.amin(test_Y), np.amax(test_Y)]

                assert np.allclose(extent, test_extent)

                #logger.debug("Extent: {}, binEdges[1]: {}".format(extent, binEdges[1]))
                axFromPlot = plt.imshow(hist_array.T,
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

    def plot_1D_hists(self) -> None:
        """ Plot (a collection of) 1D histograms. """
        # TODO: Update to use object oriented `ax` instead of just `plt`.
        # Set colors before plotting because it's not handled automatically
        color_palette_args: Dict[str, Union[str, int]] = {}
        if len(self.hists) > 6:
            # The default seaborn color palette is 6 - beyond that, we need to set the palette ourselves
            # Resources include: https://seaborn.pydata.org/generated/seaborn.color_palette.html
            # and https://seaborn.pydata.org/tutorial/color_palettes.html#palette-tutorial
            # NOTE: "Blues_d" is also a nice scheme for sequential data
            # NOTE: "husl" is preferred to "hls" because it is even in color space AND in perception!
            color_palette_args["palette"] = "husl"
            color_palette_args["n_colors"] = len(self.hists)
            logger.debug(f"Color palette args: {color_palette_args}")
        currentColorPalette = iter(sns.color_palette(**color_palette_args))

        for hist in self.hists:
            logger.debug("Plotting hist: {}, title: {}".format(hist.GetName(), hist.GetTitle()))
            # Convert to more usable form
            hist1D = histogram.Histogram1D.from_existing_hist(hist)
            # NOTE: Could remove 0s here if necessary when taking logs

            # NOTE: We plot by hand instead of using rplt.hist() so that we can have better control over
            #       the plotting options.
            if self.stepPlot:
                # Step plots look like histograms. They plot some value for some width
                # According to the documentation, it is basically a thin wrapper around plt.plot() with
                # some modified options. An additional 0 neds to be appended to the end of the data so
                # the arrays are the same length (this appears to be imposed as an mpl requirement), but
                # it is not meaningful.
                plt.step(
                    hist1D.bin_edges, np.append(hist1D.y, [0]),
                    where = "post",
                    label = hist.GetTitle(),
                    color = next(currentColorPalette)
                )
            else:
                # A more standard plot, with each value at the bin center
                plt.plot(hist1D.x, hist1D.y, label = hist.GetTitle(), color = next(currentColorPalette))

        # Only plot the legend for 1D hists where it may be stacked. For 2D hists,
        # we will only plot one quantity, so the legend isn't really necessary.
        plt.legend(loc="best")
