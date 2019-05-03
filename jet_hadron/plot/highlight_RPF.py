#!/usr/bin/env python

""" Highlight Reaction Plane Fit fit regions.

The highlighted regions correspond to the signal and background regions.

Note:
    This doesn't quite follow the traditional split between functions and plots
    because it also works as a standalone executable. The standalone functionality
    is located at the bottom of the module.

Dependencies:
  - numpy
  - matplotlib
  - seaborn
  - pachyderm

.. codeauthor:: Raymond Ehlers <raymond.ehlers@yale.edu>, Yale University
"""

import argparse
import logging
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# Needed for 3D plots
import mpl_toolkits.mplot3d
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401. Needed for 3D plots, even if not directly called.
import seaborn as sns
from typing import cast, Dict, List, Optional, Sequence, Tuple

from pachyderm import histogram

from jet_hadron.base import analysis_objects
from jet_hadron.plot import base as plot_base

logger = logging.getLogger(__name__)

# Typing helpers
# (R, G, B, A)
Color = Tuple[float, float, float, float]

##########
# Plotting
##########
def convert_color_to_max_255(color: float) -> int:
    """ Convert color to be in the range [0,255]. """
    return int(round(color * 255))

def convert_color_to_max_1(color: float) -> float:
    """ Convert color to be in the range [0,1]. """
    return color / 255.0

def overlay_colors(foreground: Color, background: Color) -> Color:
    """ Combine two colors together using an "overlay" method.

    Implemented using the formula from [colorblendy](http://colorblendy.com/). Specifically, see
    static/js/colorlib.js:blend_filters in the colorblendy github.

    Note:
        These formulas are based on colors from [0, 255], so we need to scale our [0, 1] colors
        up and down.

    Args:
        foreground: Foreground (in-front) (R, G, B, A), where each value is between [0, 1].
        background: Background (below) (R, G, B, A), where each value is between [0, 1].
    Returns:
        (R, G, B, A) overlay of the foreground and background colors.
    """
    output = []
    for fg, bg in zip(foreground, background):
        # Need to scale up to the 255 scale to use the formula
        fg = convert_color_to_max_255(fg)
        bg = convert_color_to_max_255(bg)
        overlay_color = (2 * fg * bg / 255.) if bg < 128. else (255 - 2 * (255 - bg) * (255 - fg) / 255.)
        # We then need to scale the color to be between 0 and 1.
        overlay_color = convert_color_to_max_1(overlay_color)
        output.append(overlay_color)

    output_color = cast(Color, tuple(output))
    return output_color

def screen_colors(foreground: Color, background: Color) -> Color:
    """ Combine two colors together using a "screen" method.

    Implemented using the formula from [colorblendy](http://colorblendy.com/). Specifically, see
    static/js/colorlib.js:blend_filters in the colorblendy github.

    Note:
        These formulas are based on colors from [0, 255], so we need to scale our [0, 1] colors
        up and down.

    Args:
        foreground: Foreground (in-front) (R, G, B, A), where each value is between [0, 1].
        background: Background (below) (R, G, B, A), where each value is between [0, 1].
    Returns:
        (R, G, B, A) overlay of the foreground and background colors.
    """
    output = []
    for fg, bg in zip(foreground, background):
        # Need to scale up to the 255 scale to use the formula
        fg = convert_color_to_max_255(fg)
        bg = convert_color_to_max_255(bg)
        screen_color = 255 - (((255 - fg) * (255 - bg)) >> 8)
        # We then need to scale the color to be between 0 and 1.
        converted_screen_color = convert_color_to_max_1(screen_color)
        output.append(converted_screen_color)

    output_color = cast(Color, tuple(output))
    return output_color

def mathematical_blending(foreground: Color, background: Color) -> Color:
    """ Use mathematical blending as described in https://stackoverflow.com/a/29321264 .

    Appears to look similar to ``screen_colors(...)``

    Args:
        foreground: Foreground (in-front) (R, G, B, A), where each value is between [0, 1].
        background: Background (below) (R, G, B, A), where each value is between [0, 1].
    Returns:
        (R, G, B, A) overlay of the foreground and background colors.
    """
    output = []
    for i, (fg, bg) in enumerate(zip(foreground, background)):
        # Need to scale up to the 255 scale to use the formula
        fg = convert_color_to_max_255(fg)
        bg = convert_color_to_max_255(bg)
        # Blending parameter
        t = 0.5
        if i <= 2:
            # R, G, or B value
            screen_color = np.sqrt((1 - t) * fg ** 2 + t * bg ** 2)
        else:
            # Alpha
            screen_color = (1 - t) * fg + t * bg

        # We then need to scale the color to be between 0 and 1.
        screen_color = convert_color_to_max_1(screen_color)
        output.append(screen_color)

    output_color = cast(Color, tuple(output))
    return output_color

def highlight_region_of_surface(surf: mpl_toolkits.mplot3d.art3d.Poly3DCollection,
                                highlight_colors: Color, X: np.ndarray, Y: np.ndarray,
                                phi_range: Tuple[float, float], eta_range: Tuple[float, float],
                                use_edge_colors: bool = False, use_color_overlay: bool = False,
                                use_color_screen: bool = False, use_mathematical_blending: bool = False) -> np.ndarray:
    """ Highlight a region of a surface plot.

    The colors of the selected region are changed from their default value on the surface plot to the selected
    highliht colors.  Adapted from: https://stackoverflow.com/a/5276486

    Args:
        surf: Surface on which regions will be highlighted.
        highlight_colors: The highlight color in the form (r,g,b, alpha).
        X: X bin centers.
        Y: Y bin centers.
        phi_range: Min, max phi of highlight region.
        eta_range: Min, max eta of highlight region.
        use_edge_colors: Use edge colors instead of face colors. Default: False
        use_color_overlay: Combine the highlight color and existing color using ``overlay_colors(...)``. Default: False
        use_color_screen: Combine the highlight color and existing color using ``screen_colors(...)``. Default: False
        use_mathematical_blending: Combine the highlight color and the existing color using
            ``mathematical_blending(...)``. Default: False
    Return:
        Indices selected for highlighting
    """
    # Get tuple values
    phi_min, phi_max = phi_range
    eta_min, eta_max = eta_range

    # define a region to highlight
    highlight = (X > phi_min) & (X < phi_max) & (Y > eta_min) & (Y < eta_max)
    # Get the original colors so they can be updated.
    # NOTE: Using get_facecolors() requires a workaround which copies 3d colors to 2d colors.
    #       Otherwise, the colors be retrieved directly via: `colors = surf._facecolors3d`.
    #       The equivalent also applies to edge colors
    if use_edge_colors:
        colors = surf.get_edgecolors()
    else:
        colors = surf.get_facecolors()
    #logger.debug(f"colors.shape: {colors.shape}, highlight: {highlight.shape}")
    #logger.debug(f"colors: {colors}")

    # Update the colors with the highlight color.
    # The colors are stored as a list for some reason so get the flat indicies
    logger.debug(f"max idx: {np.max(np.where(highlight[:-1, :-1].flat)[0])}")
    for idx in np.where(highlight[:-1, :-1].flat)[0]:
        background = cast(Color, tuple(colors[idx]))
        # Modify the color of each index one-by-one
        if use_color_overlay:
            colors[idx] = list(overlay_colors(foreground = highlight_colors, background = background))
        elif use_color_screen:
            colors[idx] = list(screen_colors(foreground = highlight_colors, background = background))
        elif use_mathematical_blending:
            colors[idx] = list(mathematical_blending(foreground = highlight_colors, background = background))
        else:
            colors[idx] = list(highlight_colors)

    # Reset the colors
    if use_edge_colors:
        surf.set_edgecolors(colors)
    else:
        surf.set_facecolors(colors)

    return highlight

def surface_plot_for_highlighting(ax: matplotlib.axes.Axes, X: np.ndarray, Y: np.ndarray,
                                  hist_array: np.ndarray, colormap: matplotlib.colors.ListedColormap,
                                  colorbar: bool = False) -> mpl_toolkits.mplot3d.art3d.Poly3DCollection:
    """ Plot a surface with the passed arguments for use with highlighting a region.

    This function could be generally useful for plotting surfaces. However, it should
    be noted that the arguments are specifically optimized for plotting a highlighted
    region.

    Args:
        X: X bin centers.
        Y: Y bin centers.
        hist_array: Histogram data as 2D array.
        colormap: Colormap used to map the data to colors.
        colorbar: True if a colorbar should be added.
    Returns:
        Value returned by the surface plot.
    """
    # NOTE: {r,c}count tells the surf plot how many times to sample the data. By using len(),
    #       we ensure that every point is plotted.
    # NOTE: Cannot just use norm and cmap as args to plot_surface, as this seems to return colors only based
    #       on value, instead of based on (x, y). To work around this, we calculate the norm separately,
    #       and then use that to set the facecolors manually, which appears to set the colors based on
    #       (x, y) position. Inspired by https://stackoverflow.com/a/42927880
    norm = matplotlib.colors.Normalize(vmin = np.min(hist_array), vmax = np.max(hist_array))
    surf = ax.plot_surface(X, Y, hist_array.T,
                           facecolors = colormap(norm(hist_array.T)),
                           #norm = matplotlib.colors.Normalize(vmin = np.min(hist_array), vmax = np.max(hist_array)),
                           #cmap = sns.cm.rocket,
                           rcount = len(hist_array.T[:, 0]),
                           ccount = len(hist_array.T[0]))

    if colorbar:
        # NOTE: Cannot use surf directly, because it doesn't contain the mapping from data to color
        #       Instead, we basically handle it by hand.
        m = matplotlib.cm.ScalarMappable(cmap = colormap, norm = surf.norm)
        m.set_array(hist_array.T)
        ax.colorbar(m)

    # Need to manually update the 2d colors based on the 3d colors to be able to use `get_facecolors()`.
    # This workaround is from: https://github.com/matplotlib/matplotlib/issues/4067
    surf._facecolors2d = surf._facecolors3d
    surf._edgecolors2d = surf._edgecolors3d

    return surf

def countour_plot_for_highlighting(ax: matplotlib.axes.Axes, X: np.ndarray, Y: np.ndarray,
                                   hist_array: np.ndarray, colormap: matplotlib.colors.ListedColormap,
                                   level_step: float = 0.002) -> matplotlib.axes.Axes.contour:
    """ Plot a contour for the region highlighting plot.

    Included as an option because ROOT more or less includes one, but it doesn't really
    have an impact on the output, so it can probably be ignored.

    Args:
        X: X bin centers.
        Y: Y bin centers.
        hist_array: Histogram data as 2D array.
        colormap: Colormap used to map the data to colors.
        level_step: Step in z between each contour
    Returns:
        Value returned by the contour plot.
    """
    contour = ax.contour(X, Y, hist_array.T,
                         zdir = 'z',
                         alpha = 1,
                         cmap = colormap,
                         levels = np.arange(np.min(hist_array), np.max(hist_array), level_step))

    return contour

def plot_RPF_fit_regions(hist: Tuple[np.ndarray, np.ndarray, np.ndarray], highlight_regions: Sequence["HighlightRegion"],
                         colormap: matplotlib.colors.ListedColormap = sns.cm.rocket,
                         use_transparency_in_highlights: bool = False,
                         plot_contonur: bool = False,
                         view_angle: Tuple[float, float] = (35, 225),
                         **highlight_args: bool) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """ Highlight the RPF fit range corresponding to the selected highlight_regions

    Recommended colormap options include:
        - seaborn heatmap: `sns.cm.rocket`
        - ROOT kBird (apparently basically the same as "parula" in Matlab...), "ROOT_kBird"
        - Better matplotlib default colormaps: "viridis" looks promising. Other options
          are described here: https://bids.github.io/colormap/

    Args:
        hist (numpy.ndarray, numpy.ndarray, numpy.ndarray): Tuple of (x bin centers, y bin centers, array
            of hist data).  Useful to use in conjunction with ``histogram.get_array_from_hist2D(hist)``.
        highlight_regions (list): ``HighlightRegion`` objects corresponding to the signal and background regions to
            be highlighted.
        colormap (colormap or str): Either a colormap or the name of a colormap to retrieve via `plt.get_cmap(colormap)`.
            Default: seaborn.cm.rocket.
        use_transparency_in_highlights (bool): If true, allow for transparency in the highlight colors by plotting the
            surface twice.  Only the colors of the upper surface will be modified when highlighting. Default: False.
        plot_contonur (bool): If true, also plot a contour. Included as an option because ROOT seems to use it, but
            it doesn't seem to have an impact on the image here. Default: False.
        view_angle (float, float): Set the view angle of the output, in the form
            of (elevation, azimuth). Default: (35, 225).
        highlight_args (dict): Additional arguments for how colors highlighting the surface are blended. Passed to
            highlight_region_of_surface(), so see the arguments there.
    Returns:
        tuple: (fig, ax)
    """
    X, Y, hist_array = hist
    # Set view angle after the function definition because python3 doesn't support
    # tuple parameter unpacking in functions
    #if view_angle is None:
    #    view_angle = None

    # Setup plot
    fig = plt.figure(figsize=(10, 7.5))
    # We need to create the axis in a special manner to be able to use 3d projections such
    # as surface or contour.
    ax = plt.axes(projection='3d')

    # Configure the colormap
    # If passed a name instead of a colormap directly, retrieve the colormap via mpl
    if isinstance(colormap, str):
        colormap = plt.get_cmap(colormap)
    # Set bad values to white with not transparency
    colormap = plot_base.prepare_colormap(colormap)

    # Plot surface(s)
    surf = surface_plot_for_highlighting(ax, X, Y, hist_array, colormap = colormap)
    # Surface to be used for highlighting the regions
    highlight_surf = surf
    if use_transparency_in_highlights:
        surf2 = surface_plot_for_highlighting(ax, X, Y, hist_array, colormap = colormap)
        highlight_surf = surf2

    if plot_contonur:
        countour_plot_for_highlighting(ax, X, Y, hist_array, colormap = colormap)

    # Draw highlights
    for region in highlight_regions:
        region.draw_highlights(highlight_surf, X, Y, **highlight_args)

    # Add labels
    ax.set_xlabel(r"$\Delta\varphi$")
    ax.set_ylabel(r"$\Delta\eta$")
    # Create entries for the highlight regions
    handles = []
    for region in highlight_regions:
        handles.append(matplotlib.patches.Patch(color = region.color, label = region.label))
    ax.legend(handles = handles, frameon = False)

    # View options
    ax.view_init(*view_angle)
    plt.tight_layout()

    return (fig, ax)

###################
# Highlight regions
###################
class HighlightRegion:
    """ Manages regions for highlighting.

    Args:
        label (str): Label for the highlighted region(s)
        color (tuple): The highlight color in the form (r,g,b, alpha).
    """
    def __init__(self, label: str, color: Color):
        self.label = label
        self.color = color
        self.regions: List[List[Tuple[float, float]]] = []

    def add_highlight_region(self, phi_range: Tuple[float, float], eta_range: Tuple[float, float]) -> None:
        """ Add a region to this highlight object.

        Args:
            phi_range: Tuple of min, max phi of highlight region.
            eta_range: Tuple of min, max eta of highlight region.
        Returns:
            None. The `regions` list is modified.
        """
        self.regions.append([phi_range, eta_range])

    def draw_highlights(self, highlight_surf: mpl_toolkits.mplot3d.art3d.Poly3DCollection,
                        X: np.ndarray, Y: np.ndarray, **kwargs: bool) -> None:
        """ Highlight the seleteced region(s) on the passed surface.

        It will loop over all selected regions for this highlight object.

        Args:
            highlight_surf (Poly3DCollection): Surface on which regions will be highlighted.
            X (numpy.ndarray): X bin centers.
            Y (numpy.ndarray): Y bin centers.
            kwargs (dict): Additional options for highlight_region_of_surface().
        Returns:
            None. The regions are highlighted.
        """
        for region in self.regions:
            # 0 is the phi tuple, while 1 is the eta tuple
            highlight_region_of_surface(highlight_surf, self.color, X, Y, region[0], region[1], **kwargs)

def define_highlight_regions() -> List[HighlightRegion]:
    """ Define regions to highlight.

    The user should modify or override this function if they want to define different ranges. By default,
    we highlight.

    Args:
        None
    Returns:
        HighlightRegion objects, suitably defined for highlighting the signal and background regions.
    """
    # Select the highlighted regions.
    highlight_regions = []
    # NOTE: The edge color is still that of the colormap, so there is still a hint of the origin
    #       colormap, although the facecolors are replaced by selected highlight colors

    # Signal
    signal_color = (1, 0, 0, 1.0,)
    signal_region = HighlightRegion("Signal region", signal_color)
    signal_region.add_highlight_region((-np.pi / 2, 3.0 * np.pi / 2), (-0.6, 0.6))
    highlight_regions.append(signal_region)

    # Background
    background_color = (0, 1, 0, 1.0,)
    background_phi_range = (-np.pi / 2, np.pi / 2)
    background_region = HighlightRegion("Background region", background_color)
    background_region.add_highlight_region(background_phi_range, (-1.2, -0.8))
    background_region.add_highlight_region(background_phi_range, ( 0.8,  1.2))  # noqa: E201, E241
    highlight_regions.append(background_region)

    return highlight_regions

###################################################
# Standalone functionality
#
# All standalone functionality is below this point.
###################################################
def plot_RPF_regions(input_file: str, hist_name: str, output_prefix: str = ".", printing_extensions: Optional[List[str]] = None) -> None:
    """ Main entry point for stand-alone highlight plotting functionality.

    If this is being used as a library, call ``plot_RPF_fit_regions(...)`` directly instead.

    Args:
        input_file (str): Path to the input file.
        hist_name (str): Name of the histogram to be highlighted.
        output_prefix (str): Directory where the output file should be stored. Default: "."
        printing_extensions (list): Printing extensions to be used. Default: None, which corresponds
            to printing to ``.pdf``.
    Returns:
        None.
    """
    # Argument validation
    if printing_extensions is None:
        printing_extensions = [".pdf"]

    # Basic setup
    # Create logger
    logging.basicConfig(level=logging.DEBUG)
    # Quiet down the matplotlib logging
    logging.getLogger("matplotlib").setLevel(logging.INFO)

    # Retrieve hist
    with histogram.RootOpen(filename = input_file, mode = "READ") as f:
        hist = f.Get(hist_name)
        hist.SetDirectory(0)

    # Increase the size of the fonts, etc
    with sns.plotting_context(context = "notebook", font_scale = 1.5):
        # See the possible arguments to highlight_region_of_surface(...)
        # For example, to turn on the color overlay, it would be:
        #highlight_args = {"use_color_screen" : True}
        highlight_args: Dict[str, bool] = {}
        # Call plotting functions
        fig, ax = plot_RPF_fit_regions(
            histogram.get_array_from_hist2D(hist),
            highlight_regions = define_highlight_regions(),
            colormap = "ROOT_kBird", view_angle = (35, 225),
            **highlight_args,
        )

        # Modify axis labels
        # Set the distance from axis to label in pixels.
        # Having to set this by hand isn't ideal, but tight_layout doesn't work as well for 3D plots.
        ax.xaxis.labelpad = 12
        # Visually, delta eta looks closer
        ax.yaxis.labelpad = 15

        # If desired, add additional labeling here.
        # Probably want to use, for examxple, `ax.text2D(0.1, 0.9, "label text", transform = ax.transAxes)`
        # This would place the text in the upper left corner (it's like NDC)

        # Save and finish up
        # The figure will be saved at ``output_prefix/output_path.printing_extension``
        output_wrapper = analysis_objects.PlottingOutputWrapper(output_prefix = output_prefix, printing_extensions = printing_extensions)
        plot_base.save_plot(output_wrapper, fig, output_name = "highlightRPFRegions")
        plt.close(fig)

def parse_arguments() -> Tuple[str, str, str, List[str]]:
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

def run_from_terminal() -> None:
    """ Entry point when running from the terminal.

    Running from the terminal implies that arguments need to be parsed.
    """
    (input_file, hist_name, output_path, printing_extensions) = parse_arguments()
    plot_RPF_regions(
        input_file = input_file, hist_name = hist_name,
        output_prefix = output_path, printing_extensions = printing_extensions
    )

if __name__ == "__main__":
    run_from_terminal()
