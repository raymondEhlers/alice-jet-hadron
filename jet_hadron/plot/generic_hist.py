#!/usr/bin/env python

""" Generic histograms plotting module.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

from dataclasses import dataclass, InitVar
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.ticker
# For 3D plots
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401. Needed for 3D plots, even if not directly called.
import os
import re
import seaborn as sns
from typing import Any, Dict, List, Tuple, Union

import rootpy.ROOT as ROOT

from pachyderm import histogram
from pachyderm import yaml

from jet_hadron.base import analysis_objects
from jet_hadron.base import params
from jet_hadron.base.typing_helpers import Hist
from jet_hadron.plot import base as plot_base

# Setup logger
logger = logging.getLogger(__name__)
# Use latex in matplotlib.
plt.rc('text', usetex = True)

@dataclass
class HistPlotter:
    """ Handles generic plotting of histograms via matplotlib.

    Note:
        We ignore the typing on some of these values because we really except them to be
        ``None`` if they are not set. It's not the cleanest design, but it's fine for
        our purposes.
    """
    hist_names: List[Dict[str, str]] = None  # type: ignore
    hist: InitVar[Hist] = None
    hists: List[Hist] = None  # type: ignore
    output_name: str = ""
    title: str = ""
    automatic_title_from_name: bool = False
    exact_name_match: bool = False
    x_label: str = ""
    y_label: str = ""
    z_label: str = ""
    x_limits: Tuple[float, float] = None  # type: ignore
    y_limits: Tuple[float, float] = None  # type: ignore
    text_label: Dict[str, Any] = None  # type: ignore
    scientific_notation_on_axis: str = ""
    logy: bool = False
    logz: bool = False
    surface: bool = False
    use_pcolor_mesh: bool = False
    step_plot: bool = True
    processing: Dict[str, Any] = None  # type: ignore

    def __post_init__(self, hist: Hist):
        """ Final object initialization. """
        # Dataclass fields don't always seem to play well with YAML. So we just do our final initialization here.
        if self.hist_names is None:
            self.hist_names = [{}]
        if self.hists is None:
            self.hists = []
        if self.processing is None:
            self.processing = {}
        # Store the hist in the list if it was passed
        if hist:
            self.hists.append(hist)

    def __repr__(self) -> str:
        """ Representation of the object. """
        values = ("{k!s} = {v!r}".format(k = k, v = v) for k, v in self.__dict__.items())
        return "{}({})".format(self.__class__.__name__, ", ".join(values))

    @classmethod
    def from_yaml(cls, constructor: yaml.Constructor, data: yaml.ruamel.yaml.nodes.MappingNode) -> "HistPlotter":
        """ Create object from YAML using the constructor so default values will be specified.

        It appears that when YAML creates an object, it doesn't pass arguments to the constructor, but rather
        creates an empty object and then specifies the state of the particular variables specified. This is fine
        except that it bypasses default values. To utilize default values, we need to take the arguments for
        the object as a mapping and then pass them to the object constructor.

        Unfortunately, this obvious approach doesn't work in all cases. In particular, when used, merge keys at
        the next level are silently dropped after the mapping is constructed. I don't know why. To get around this,
        we specify all arguments under the "kwargs" key, and then just pass those into the constructor. This works
        as expected. So other than being an ugly hack, it's fine.
        """
        # First convert the object from a node to an easy to use mapping.
        # We use the SafeConstructor to avoid needing to pass a poorly documented ``maptyp`` argument.
        #logger.debug(f"Using representer, {data}")
        data = yaml.ruamel.yaml.constructor.SafeConstructor.construct_mapping(constructor, data, deep = True)
        #logger.debug(f"mapping after construction: {data}")
        # Now that we have our mapping, ensure that only the kwargs are passed so that no other information is lost.
        if "kwargs" not in data or not data["kwargs"] or len(data) > 1:
            raise ValueError("Must only pass arguments under the 'kwargs' key.")

        # Finally construct the object using the kwargs that were extracted via the mapping.
        return cls(**data["kwargs"])

    def get_first_hist(self) -> ROOT.TH1:
        return next(iter(self.hists))

    def apply_hist_settings(self, ax: matplotlib.axes.Axes) -> None:
        self.apply_axis_settings(ax)

        if self.logy:
            logger.debug("Setting logy")
            ax.set_yscale("log")

        self.apply_hist_limits(ax)
        self.apply_hist_titles(ax)

    def apply_axis_settings(self, ax: matplotlib.axes.Axes) -> None:
        # Do not apply useMathText to the axis tick format!
        # It is incompatible with using latex rendering
        # If for some reason latex is turned off, one can use the lines below to only apply the option to axes which
        # support it (most seem to work fine with it)
        #logger.debug(f"x: {ax.get_xaxis().get_major_formatter()}, y: {ax.get_yaxis().get_major_formatter()}")
        #if isinstance(ax.get_xaxis().get_major_formatter(), matplotlib.ticker.ScalarFormatter) and isinstance(ax.get_yaxis().get_major_formatter(), matplotlib.ticker.ScalarFormatter):
        #ax.ticklabel_format(axis = "both", useMathText = True)

        if self.scientific_notation_on_axis != "":
            # See: https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.ticklabel_format.html
            # (0,0) means apply to all numbers of the axis
            # axis could be x, y, or both
            ax.ticklabel_format(axis = self.scientific_notation_on_axis, style = "sci", scilimits = (0, 0))

    def post_draw_options(self, ax: matplotlib.axes.Axes) -> None:
        pass

    def apply_hist_titles(self, ax: matplotlib.axes.Axes) -> None:
        # Title sanity check - we should only set one
        if self.automatic_title_from_name and self.title:
            raise ValueError("Set both automatic title extraction and an actual title. Please select only one setting")

        # Figure title
        if self.automatic_title_from_name:
            temp_title = self.get_first_hist().GetTitle()
            # Remove the common leading "h" in a hist name
            if temp_title[0] == "h":
                temp_title = temp_title[1:]
            # For the regex, see: https://stackoverflow.com/a/43898219
            title = re.sub('([a-z])([A-Z])', r'\1 \2', temp_title)
        else:
            title = self.title if self.title else self.get_first_hist().GetTitle()

        if title:
            logger.debug(f"Set title: {title}")
            ax.set_title(title)

        # Axis labels
        label_map = {
            "x": (self.x_label, ROOT.TH1.GetXaxis, ax.set_xlabel),
            "y": (self.y_label, ROOT.TH1.GetYaxis, ax.set_ylabel),
        }
        for axis_name, (val, axis, apply_title) in label_map.items():
            if val:
                label = val
            else:
                label = axis(self.get_first_hist()).GetTitle()
                # Convert any "#" (from ROOT) to "\" for latex
                label = label.replace("#", "\\")
                # Convert "%" -> "\%" to ensure that latex runs successfully
                label = label.replace("%", r"\%")
                # Guessing that units start with the last "(", which is usually the case
                found_latex = label.find("_") > -1 or label.find("^") > -1 or label.find('\\') > -1
                # Apply latex equation ("$") up to the units
                units_location = label.rfind("(")
                logger.debug(f"Found latex: {found_latex}, label: \"{label}\"")
                if found_latex:
                    if units_location > -1:
                        label = "$" + label[:units_location - 1] + "$ " + label[units_location:]
                    else:
                        label = "$" + label + "$"
            logger.debug(f"Apply {axis_name} axis title with label \"{label}\", axis {axis}, and apply_title function {apply_title}")
            apply_title(label)

    def apply_hist_limits(self, ax: matplotlib.axes.Axes) -> None:
        if self.x_limits is not None:
            logger.debug(f"Setting x limits of {self.x_limits}")
            ax.set_xlim(self.x_limits)
        if self.y_limits is not None:
            logger.debug(f"Setting y limits of {self.y_limits}")
            ax.set_ylim(self.y_limits)

    def add_text_labels(self, ax: matplotlib.axes.Axes, obj: analysis_objects.JetHBase) -> None:
        """ Add text labels to a plot.

        Available properties include:
            - "cellLabel"
            - "clusterLabel"
            - "trackLabel"
            - "textLocation" : [x, y]. Default location is top center
            - "textAlignment" : option for multialignment
            - "fontSize"
            - aliceLabel (set in the main config)

        These properties are selected via the component configuration.

        Args:
            ax: Axis on which the text labels will be added.
            obj: Object which contains the analysis parameters.
        """
        if self.text_label is not None:
            text = ""
            text += str(obj.alice_label)
            # Only add the task label if the display string is not empty.
            if str(obj.task_label) != "":
                # We don't want a new line here - we just want to continue it
                text += " " + str(obj.task_label)

            text += "\n" + params.system_label(energy = obj.collision_energy,
                                               system = obj.collision_system,
                                               activity = obj.event_activity)
            property_labels = []
            if self.text_label.get("cellLabel", False):
                # Handled separately because it is long enough that it has to be
                # on a separate line
                text += "\n" + r"Cell $E_{\mathrm{seed}} = 100$ MeV, $E_{\mathrm{cell}} = 50$ MeV"
            if self.text_label.get("clusterLabel", False):
                property_labels.append(r"$E_{cluster} > 300$ MeV")
            if self.text_label.get("trackLabel", False):
                property_labels.append(r"$p_{T,track} > 150\:\mathrm{MeV/\mathit{c}}$")
            if len(property_labels) > 0:
                text += "\n" + ", ".join(property_labels)
            #logger.debug("text: {}".format(text))

            # The default location is top center
            ax.text(*self.text_label.get("textLocation", [0.5, 0.92]), s = text,
                    horizontalalignment = "center",
                    verticalalignment = "center",
                    multialignment = self.text_label.get("textAlignment", "left"),
                    fontsize = self.text_label.get("fontSize", 12.5),
                    transform = ax.transAxes)

    def plot(self, obj: analysis_objects.JetHBase, output_name: str = "") -> None:
        # Ensure that the output directory is available.
        path = os.path.join(obj.output_prefix, os.path.dirname(output_name))
        if not os.path.exists(path):
            os.makedirs(path)

        # Make the plots
        fig, ax = plt.subplots(figsize=(8, 6))

        # Draw the hist
        if isinstance(self.get_first_hist(), ROOT.TH2):
            self.plot_2D_hists(fig = fig, ax = ax)
        elif isinstance(self.get_first_hist(), ROOT.TH1):
            self.plot_1D_hists(fig = fig, ax = ax)
        else:
            raise ValueError(f"Histogram must be 1D or 2D. Type provided: {type(self.get_first_hist())}")

        # Apply the options
        # Need to apply these here because rplt messes with them!
        self.apply_hist_settings(ax)

        # Apply post drawing options
        self.post_draw_options(ax)
        self.add_text_labels(ax, obj)

        # Final plotting options
        plt.tight_layout()

        # Determine the final output path
        # Use ``self.output_name`` for the end of the output name if it's set. If it's not set, then we just
        # use the plot configuration key name passed in via ``output_name``
        final_output_name = output_name
        if self.output_name:
            final_output_name = os.path.join(os.path.dirname(final_output_name), self.output_name)
        logger.debug(f"final_output_name: {final_output_name}, self.output_name: {self.output_name}")

        # Save and close the figure
        plot_base.save_plot(obj, fig, final_output_name)
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
        # NOTE: The convention for the array is arr[x_index][y_index]. Apparently this matches up with
        #       numpy axis conventions (?). We will have to transpose the data when we go to plot it.
        #       This is the same convention as used in root_numpy.
        X, Y, hist_array = histogram.get_array_from_hist2D(
            hist = self.get_first_hist(),
            set_zero_to_NaN = True,
            return_bin_edges = not self.surface
        )

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
        kwargs["cmap"] = plot_base.prepare_colormap(sns.cm.rocket)
        # Label is included so we could use a legend if we want
        kwargs["label"] = self.get_first_hist().GetTitle()

        logger.debug("kwargs: {}".format(kwargs))

        if self.surface:
            logger.debug("Plotting surface")
            # Need to retrieve a special 3D axis for the surface plot
            ax = plt.axes(projection = "3d")

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
            # Plot with either imshow or pcolormesh
            # Anecdotally, I think pcolormesh is a bit more flexible, but imshow seems to be much faster.
            # According to https://stackoverflow.com/a/21169703, either one can be fine with the proper
            # options. So the plan is to stick with imshow unless pcolormesh is needed for some reason
            if self.use_pcolor_mesh:
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

                #logger.debug("Extent: {}, binEdges[1]: {}".format(extent, binEdges[1]))
                axFromPlot = plt.imshow(hist_array.T,
                                        extent = extent,
                                        interpolation = "nearest",
                                        aspect = "auto",
                                        origin = "lower",
                                        **kwargs)

        # Draw the colorbar based on the drawn axis above.
        label = self.z_label if self.z_label else self.get_first_hist().GetZaxis().GetTitle()
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

    def plot_1D_hists(self, fig, ax) -> None:
        """ Plot (a collection of) 1D histograms.

        Args:
            fig: Figure from matplotlib
            ax: Axis from matplotlib.
        Raises:
            ValueError: If more than one histogram is stored in the ``hists``, which doesn't make sense for
                2D hist, which doesn't make sense for 2D hists.
        """
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
            if self.step_plot:
                # Step plots look like histograms. They plot some value for some width
                # According to the documentation, it is basically a thin wrapper around plt.plot() with
                # some modified options. An additional 0 neds to be appended to the end of the data so
                # the arrays are the same length (this appears to be imposed as an mpl requirement), but
                # it is not meaningful.
                ax.step(
                    hist1D.bin_edges, np.append(hist1D.y, [0]),
                    where = "post",
                    label = hist.GetTitle(),
                    color = next(currentColorPalette)
                )
            else:
                # A more standard plot, with each value at the bin center
                ax.plot(hist1D.x, hist1D.y, label = hist.GetTitle(), color = next(currentColorPalette))

        # Only plot the legend for 1D hists where it may be stacked. For 2D hists,
        # we will only plot one quantity, so the legend isn't really necessary.
        ax.legend(loc="best")
