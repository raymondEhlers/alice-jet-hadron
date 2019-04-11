#!/usr/bin/env python

""" Extracted values plotting module.

Includes quantities such as widths and yields.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import logging
from pachyderm import utils
import matplotlib.pyplot as plt
import numpy as np
import ROOT
from typing import Dict, Optional, TYPE_CHECKING, Union

from jet_hadron.base import analysis_config
from jet_hadron.base import analysis_objects
from jet_hadron.base import labels
from jet_hadron.base import params
from jet_hadron.plot import base as plot_base
# Careful to watch out for import loops...
from jet_hadron.analysis import fit as fitting

if TYPE_CHECKING:
    from jet_hadron.analysis import correlations

logger = logging.getLogger(__name__)

# Typing helpers
Correlations = Union["correlations.DeltaPhiSignalDominated",
                     "correlations.DeltaEtaNearSide", "correlations.DeltaEtaAwaySide"]

def delta_eta_with_gaussian(analysis: "correlations.Correlations") -> None:
    """ Plot the subtracted delta eta near-side. """
    # Setup
    fig, ax = plt.subplots(figsize = (8, 6))

    # Plot only the near side for now because the away-side doesn't have a gaussian shape
    # Of the form (attribute_name, mean)
    attribute_names = [
        ("near_side", 0.0),
    ]
    for attribute_name, mean in attribute_names:
        # Setup
        # Correlation
        correlation: Union["correlations.DeltaEtaNearSide"] = \
            getattr(analysis.correlation_hists_delta_eta_subtracted, attribute_name)
        # Extracted width
        extracted_width: analysis_objects.ExtractedObservable = getattr(analysis.widths_delta_eta, attribute_name)

        # Plot the data.
        h = correlation.hist
        ax.errorbar(
            h.x, h.y, yerr = h.errors,
            marker = "o", linestyle = "",
            label = f"{correlation.type.display_str()}",
        )

        # Plot the fit
        logger.debug(f"mean: {type(mean)}, width: {type(extracted_width.value)}")
        gauss = fitting.gaussian(h.x, mu = mean, sigma = extracted_width.value)
        fit_plot = ax.plot(
            h.x, gauss,
            label = fr"Gaussian fit: $\mu = $ {mean:.2f}, $\sigma = $ {extracted_width.value:.2f}",
        )
        # Fill in the error band.
        error = extracted_width.error * np.ones(len(h.x))
        ax.fill_between(
            h.x, gauss - error, gauss + error,
            facecolor = fit_plot[0].get_color(), alpha = 0.5,
        )

        # Labels.
        ax.set_xlabel(labels.make_valid_latex_string(correlation.axis.display_str()))
        ax.set_ylabel(labels.make_valid_latex_string(labels.delta_eta_axis_label()))
        jet_pt_label = labels.jet_pt_range_string(analysis.jet_pt)
        track_pt_label = labels.track_pt_range_string(analysis.track_pt)
        ax.set_title(fr"Subtracted 1D ${correlation.axis.display_str()}$,"
                     f" {analysis.reaction_plane_orientation.display_str()} event plane orient.,"
                     f" {jet_pt_label}, {track_pt_label}")
        ax.legend(loc = "upper right")

        # Final adjustments
        fig.tight_layout()
        # Save plot and cleanup
        plot_base.save_plot(analysis.output_info, fig,
                            f"jetH_delta_eta_{analysis.identifier}_width_{attribute_name}_fit")
        # Reset for the next iteration of the loop
        ax.clear()

    # Final cleanup.
    plt.close(fig)

def delta_phi_with_gaussians(analysis: "correlations.Correlations") -> None:
    """ Plot the subtracted delta phi correlation with gaussian fits to the near and away side. """
    # Setup
    fig, ax = plt.subplots(figsize = (8, 6))
    correlation = analysis.correlation_hists_delta_phi_subtracted.signal_dominated
    h = correlation.hist

    # First we plot the data
    ax.errorbar(
        h.x, h.y, yerr = h.errors,
        marker = "o", linestyle = "",
        label = f"{correlation.type.display_str()}",
    )

    # Of the form (attribute_name, mean)
    delta_phi_regions = [
        ("near_side", 0.0),
        ("away_side", np.pi),
    ]
    for attribute_name, mean in delta_phi_regions:
        # Setup
        # Extracted width
        extracted_width: analysis_objects.ExtractedObservable = getattr(analysis.widths_delta_phi, attribute_name)
        # Convert the attribute name to display better. Ex: "near_side" -> "Near side"
        attribute_display_name = attribute_name.replace("_", " ").capitalize()

        # Plot the fit
        gauss = fitting.gaussian(h.x, mu = mean, sigma = extracted_width.value)
        fit_plot = ax.plot(
            h.x, gauss,
            label = fr"{attribute_display_name} gaussian fit: $\mu = $ {mean:.2f}"
                    fr", $\sigma = $ {extracted_width.value:.2f}",
        )
        # Fill in the error band.
        error = extracted_width.error * np.ones(len(h.x))
        ax.fill_between(
            h.x, gauss - error, gauss + error,
            facecolor = fit_plot[0].get_color(), alpha = 0.5,
        )

    # Labels.
    ax.set_xlabel(labels.make_valid_latex_string(correlation.axis.display_str()))
    ax.set_ylabel(labels.make_valid_latex_string(labels.delta_phi_axis_label()))
    jet_pt_label = labels.jet_pt_range_string(analysis.jet_pt)
    track_pt_label = labels.track_pt_range_string(analysis.track_pt)
    ax.set_title(fr"Subtracted 1D ${correlation.axis.display_str()}$,"
                 f" {analysis.reaction_plane_orientation.display_str()} event plane orient.,"
                 f" {jet_pt_label}, {track_pt_label}")
    ax.legend(loc = "upper right")

    # Final adjustments
    fig.tight_layout()
    # Save plot and cleanup
    plot_base.save_plot(analysis.output_info, fig,
                        f"jetH_delta_phi_{analysis.identifier}_width_signal_dominated_fit")
    plt.close(fig)

def plot_extracted_values(manager: "correlations.CorrelationsManager") -> None:
    """ Plot extracted values. """
    # Setup
    fig, ax = plt.subplots(figsize = (8, 6))
    ep_colors = {
        params.ReactionPlaneOrientation.inclusive: "black",
        params.ReactionPlaneOrientation.in_plane: "C0",  # blue
        params.ReactionPlaneOrientation.mid_plane: "C2",  # green
        params.ReactionPlaneOrientation.out_of_plane: "C3",  # red
    }

    value_attribute_name = "widths_delta_phi.near_side"

    # These are both used for labeling purposes and are identical for all analyses that are iterated over.
    analysis_identifier: Optional[str] = None
    jet_pt: Optional[analysis_objects.JetPtBin] = None
    for displace_index, ep_orientation in enumerate(manager.selected_iterables["reaction_plane_orientation"]):
        # Store the values to be plotted
        values: Dict[analysis_objects.PtBin, analysis_objects.ExtractedObservable] = {}
        for key_index, analysis in \
                analysis_config.iterate_with_selected_objects(manager.analyses, reaction_plane_orientation = ep_orientation):
            # Store each extracted value.
            values[analysis.track_pt] = utils.recursive_getattr(analysis, value_attribute_name)
            # These are both used for labeling purposes and are identical for all analyses that are iterated over.
            if analysis_identifier is None:
                analysis_identifier = analysis.identifier
            if jet_pt is None:
                jet_pt = analysis.jet_pt

        # Plot the values
        bin_centers = np.array([k.bin_center for k in values])
        bin_centers = bin_centers + displace_index * 0.05
        ax.errorbar(
            bin_centers, [v.value for v in values.values()], yerr = [v.error for v in values.values()],
            label = ep_orientation.display_str(), color = ep_colors[ep_orientation], marker = "o",
        )

    # Help out mypy...
    assert analysis_identifier is not None
    assert jet_pt is not None

    # Labels.
    # TODO: ALICE, Extraction range
    ax.set_xlabel(labels.make_valid_latex_string(labels.track_pt_display_label()))
    ax.set_ylabel("Width")
    ax.set_title(f"Near-side widths for {labels.jet_pt_range_string(jet_pt)}")
    ax.legend(loc = "upper right")

    # Final adjustments
    fig.tight_layout()
    # Save plot and cleanup
    plot_base.save_plot(manager.output_info, fig,
                        f"jetH_delta_phi_{analysis.identifier}_widths")
    plt.close(fig)

def plotYields(jetH):
    """ Plot extracted yields. """
    for yields, rangeLimits, tag in [(jetH.yieldsAS, (5e-3, 3), "yieldsAS"),
                                     (jetH.yieldsNS, (5e-3, 5), "yieldsNS"),
                                     (jetH.yieldsDEtaNS, (5e-3, 2), "yieldsDEtaNS")]:
        parameters = (r"p_{\mathrm{T}}^{assoc}", r"$\mathrm{d}N/\mathrm{d}p_{\mathrm{T}} (\mathrm{GeV}/c)^{-1}$", rangeLimits, tag)
        plotExtractedValues(jetH, yields, parameters)

def plotWidths(jetH):
    """ Plot extracted widths. """
    for widths, rangeLimits, tag, yAxisLabel in [(jetH.widthsAS, (0, 2.5), "widthsAS", "Away-side width"),
                                                 (jetH.widthsNS, (0, 2.5), "widthsNS", "Near-side width"),
                                                 (jetH.widthsDEtaNS, (0, 2.5), "widthsDEtaNS", "Near-side width")]:
        parameters = (r"p_{\mathrm{T}}^{assoc}", yAxisLabel, rangeLimits, tag)
        plotExtractedValues(jetH, widths, parameters)

def createTGraphsFromExtractedValues(jetH, values):
    """ Create new TGraphs from the extracted values.

    Returns: (OrderedDict): One TGraph filled with the corresponding values per jet pt bin.
    """
    graphs = {}
    for jetPtBin in params.iterateOverJetPtBins():
        graphs[jetPtBin] = ROOT.TGraphErrors(len(params.trackPtBins) - 1)
        # Disable title
        graphs[jetPtBin].SetTitle("")

    for observable in values.values():
        # Center points in the bin
        trackPtBin = observable.trackPtBin
        halfBinWidth = (params.trackPtBins[trackPtBin + 1] - params.trackPtBins[trackPtBin]) / 2.0
        offset = 0.07 * observable.jetPtBin
        binCenterPoint = params.trackPtBins[trackPtBin] + halfBinWidth + offset
        logger.debug("binCenterPoint: {}".format(binCenterPoint))

        graphs[observable.jetPtBin].SetPoint(observable.trackPtBin, binCenterPoint, observable.value)
        # Second argument simply sets the x error to 0, since we don't want to see that bar.
        graphs[observable.jetPtBin].SetPointError(observable.trackPtBin, 0., observable.error)

    return graphs

def plotExtractedValues(jetH, values, parameters):
    """ Plot extracted via using a TGraphErrors. """
    # Colors from different event-plane orientations in the semi-central analysis.
    #         Black,       Blue,       Green, Red
    colors = [ROOT.kBlack, ROOT.kBlue - 7, 8, ROOT.kRed - 4]

    (xAxisTitle, yAxisTitle, (minRange, maxRange), tag) = parameters

    # Create graphs
    graphs = createTGraphsFromExtractedValues(values)

    # Plot and save graphs
    canvas = ROOT.TCanvas("extractedValues", "extractedValues")
    if "yields" in tag:
        canvas.SetLogy()

    # Create legend
    #legend = createYieldsAndWidthsLegend(location, plotType, yieldLimit, collisionSystem = collisionSystem)
    legend = createExtractedValuesLegend(jetH.collisionSystem, tag)

    # NOTE: i is equivalent to the key of the graph dicts. Either is fine.
    firstDraw = False
    for i, graph in enumerate(graphs.values()):
        # Style
        graph.SetLineColor(colors[i + 1])
        graph.SetMarkerColor(colors[i + 1])
        graph.SetLineWidth(1)
        #graph.SetMarkerSize(1)
        graph.SetMarkerStyle(ROOT.kFullCircle)

        # Handle first draw carefully
        if not firstDraw:
            graph.GetXaxis().SetTitle(xAxisTitle)
            graph.GetYaxis().SetTitle(yAxisTitle)
            # Set viewable range
            #graph.GetYaxis().SetRangeUser(minRange, maxRange)
            # Only draw the axis on the first draw call
            graph.Draw("AP")
            firstDraw = True
        else:
            graph.Draw("P")

        # Add legend entry
        legend.AddEntry(graph, params.generateJetPtRangeString(i), "LEP")

    legend.Draw("same")

    # Save plot
    plot_base.save_plot(jetH, canvas, tag)

def createExtractedValuesLegend(collisionSystem, tag):
    """ Create legends for extracted value plots. """
    if "yields" in tag:
        leg = ROOT.TLegend(0.12, 0.12, 0.5, 0.4)
    elif "widths" in tag:
        leg = ROOT.TLegend(0.5, 0.55, 0.89, 0.87)
    leg.SetFillColorAlpha(0, 0)
    leg.SetBorderSize(0)
    leg.SetTextSize(0.03)
    leg.AddEntry("", "{0} #sqrt{{s_{{NN}}}} = 2.76 TeV{1}".format("Pb--Pb" if collisionSystem == params.CollisionSystem.kPbPb else "pp #otimes Pb--Pb", ", 0-10%" if collisionSystem == params.CollisionSystem.kPbPb else ""), "")
    leg.AddEntry("", "Anti-k_{T} full jets, R=0.2", "")

    # TODO: Add extraction ranges

    return leg

# TODO: Merge and refactor with the above
def PlotWidthsNew(jetH, widths):
    for location, paramData in widths.items():
        # Define axes for plot
        fig, ax = plt.subplots()

        tempX = []
        tempWidths = []
        tempErrors = []
        for (jetPtBin, trackPtBin), observable in paramData.items():
            # Skip first bin, which is fit very poorly
            if trackPtBin == 0:
                continue

            halfBinWidth = (params.trackPtBins[trackPtBin + 1] - params.trackPtBins[trackPtBin]) / 2.0
            binCenterPoint = params.trackPtBins[trackPtBin] + halfBinWidth

            logger.debug("location: {}, jetPtBin: {}, trackPtBin: {}, X: {}, width: {},"
                         " error: {}".format(location.upper(),
                                             jetPtBin,
                                             trackPtBin,
                                             binCenterPoint,
                                             observable.value,
                                             observable.error))
            tempX.append(binCenterPoint)
            tempWidths.append(observable.value)
            tempErrors.append(observable.error)

        ax.errorbar(tempX, tempWidths, yerr = tempErrors, marker = "o", label = "{} Widths".format(location.upper()))
        ax.set_xlabel(r"$p_{\mathrm{T}}^{\mathrm{assoc}}$")
        ax.set_ylabel(r"$\sigma_{AS}$")

        # Tight the plotting up
        fig.tight_layout()
        # Then adjust spacing between subplots
        # Must go second so it isn't reset by tight_layout()
        # Tuned for "paper" context
        fig.subplots_adjust(hspace = 0, wspace = 0.05, bottom = 0.12, left = 0.1)

        plt.legend(loc="best")

        # Save plot
        # TODO: Define this name in the class!
        plot_base.save_plot(jetH, fig, "widths{}RPF".format(location.upper()))

        # Cleanup
        plt.close(fig)
