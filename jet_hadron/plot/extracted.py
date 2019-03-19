#!/usr/bin/env python

""" Extracted values plotting module.

Includes quantities such as widths and yields.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import logging

# Import plotting packages
# Use matplotlib in some cases
import matplotlib.pyplot as plt
# And use ROOT in others
import ROOT

from jet_hadron.base import params as params
from jet_hadron.plot import base as plot_base

logger = logging.getLogger(__name__)

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
