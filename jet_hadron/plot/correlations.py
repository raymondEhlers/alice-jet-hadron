#!/usr/bin/env python

""" Correlations plotting module.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

# Py2/3
from future.utils import iteritems
from future.utils import itervalues

import logging
import numpy as np

# Import plotting packages
# Use matplotlib in some cases
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import seaborn as sns
# And use ROOT in others
import rootpy.ROOT as ROOT

from pachyderm import histogram

from jet_hadron.base import params
from jet_hadron.plot import base as plotBase
from jet_hadron.plot import highlight_RPF

# Setup logger
logger = logging.getLogger(__name__)

def plot2DCorrelations(jetH):
    """ Plot the 2D correlations. """
    canvas = ROOT.TCanvas("canvas2D", "canvas2D")

    # Iterate over 2D hists
    for histCollection in jetH.hists2D:
        for name, observable in iteritems(histCollection):
            # Retrieve hist and plot

            # We don't want to scale the mixed event hist because we already determined the normalization
            if "mixed" in name:
                hist = observable.hist
            else:
                hist = observable.hist.createScaledByBinWidthHist()

            # We don't need the title with all of the labeling
            hist.SetTitle("")

            # Draw plot
            hist.Draw("surf2")

            # Label axes
            hist.GetXaxis().CenterTitle(True)
            hist.GetXaxis().SetTitleSize(0.08)
            hist.GetXaxis().SetLabelSize(0.06)
            hist.GetYaxis().CenterTitle(True)
            hist.GetYaxis().SetTitleSize(0.08)
            # If I remove this, it looks worse, even though this is not supposed to do anything
            hist.GetYaxis().SetTitleOffset(1.2)
            hist.GetYaxis().SetLabelSize(0.06)
            hist.GetZaxis().CenterTitle(True)
            hist.GetZaxis().SetTitleSize(0.06)
            hist.GetZaxis().SetLabelSize(0.05)
            hist.GetZaxis().SetTitleOffset(0.8)
            canvas.SetLeftMargin(0.13)

            if "mixed" in name:
                hist.GetZaxis().SetTitle(r"$a(\Delta\varphi,\Delta\eta)$")
                hist.GetZaxis().SetTitleOffset(0.9)
            else:
                zTitle = r"$1/\mathrm{N}_{\mathrm{trig}}\mathrm{d^{2}N}%(label)s/\mathrm{d}\Delta\varphi\mathrm{d}\Delta\eta$"
                if "corr" in name:
                    zTitle = zTitle % {"label": ""}
                else:
                    zTitle = zTitle % {"label": r"_{\mathrm{raw}}"}
                    # Decrease size so it doesn't overlap with the other labels
                    hist.GetZaxis().SetTitleSize(0.05)

                hist.GetZaxis().SetTitle(zTitle)

            # Add labels
            # PDF DOES NOT WORK HERE: https://root-forum.cern.ch/t/latex-sqrt-problem/17442/15
            # Instead, print to EPS and then convert to PDF
            aliceLabel = jetH.aliceLabel.str()
            systemLabel = params.systemLabel(energy = jetH.collisionEnergy,
                                             system = jetH.collisionSystem,
                                             activity = jetH.eventActivity)
            (jetFinding, constituentCuts, leadingHadron, jetPt) = params.jetPropertiesLabel(observable.jetPtBin)
            assocPt = params.generateTrackPtRangeString(observable.trackPtBin)
            #logger.debug("label: {}, systemLabel: {}, constituentCuts: {}, leadingHadron: {}, jetPt: {}, assocPt: {}".format(aliceLabel, systemLabel, constituentCuts, leadingHadron, jetPt, assocPt))

            tex = ROOT.TLatex()
            tex.SetTextSize(0.04)
            # Upper left side
            tex.DrawLatexNDC(.03, .96, aliceLabel)
            tex.DrawLatexNDC(.005, .91, systemLabel)
            tex.DrawLatexNDC(.005, .86, jetPt)
            tex.DrawLatexNDC(.005, .81, jetFinding)

            # Upper right side
            tex.DrawLatexNDC(.67, .96, assocPt)
            tex.DrawLatexNDC(.73, .91, constituentCuts)
            tex.DrawLatexNDC(.75, .86, leadingHadron)

            # Reproduce ROOT problems with the below. Plot in ROOT on a canvas will look fine, but
            # when printed to PDF, will display the raw latex (for most symbols, but not all)
            #text = ROOT.TLatex()
            #text.DrawLatexNDC(.1, .3, r"Hello")
            #text.DrawLatexNDC(.1, .4, "#sqrt{test}")
            ## Visual corruption shows up with a "\"
            #text.DrawLatexNDC(.1, .5, "\sqrt{test}")
            ## This one doesn't work, but the others do!
            #text.DrawLatexNDC(.1, .6, "#mathrm{test}")
            #text.DrawLatexNDC(.1, .7, "\mathrm{test}")

            # Save plot
            plotBase.saveCanvas(jetH, canvas, observable.hist.GetName())

            # Draw as colz to view more precisely
            hist.Draw("colz")
            plotBase.saveCanvas(jetH, canvas, observable.hist.GetName() + "colz")

            canvas.Clear()

def plot1DCorrelations(jetH):
    canvas = ROOT.TCanvas("canvas1D", "canvas1D")

    for histCollection in jetH.hists1D:
        for name, observable in iteritems(histCollection):
            # Draw the 1D histogram.
            # NOTE: that we don't want to scale the histogram here by the bin width because we've already done that!
            observable.hist.Draw("")
            plotBase.saveCanvas(jetH, canvas, observable.hist.GetName())

def plot1DCorrelationsWithFits(jetH):
    canvas = ROOT.TCanvas("canvas1D", "canvas1D")

    histsWithFits = [[jetH.dPhi, jetH.dPhiFit], [jetH.dPhiSubtracted, jetH.dPhiSubtractedFit],
                     [jetH.dEtaNS, jetH.dEtaNSFit], [jetH.dEtaNSSubtracted, jetH.dEtaNSSubtractedFit]]

    for histCollection, fitCollection in histsWithFits:
        for (name, observable), fit in zip(iteritems(histCollection), itervalues(fitCollection)):
            # Create scaled hist and plot it
            observable.hist.Draw("")
            fit.Draw("same")
            plotBase.saveCanvas(jetH, canvas, observable.hist.GetName())

def mixedEventNormalization(jetH,
                            # For labeling purposes
                            histName, etaLimits, jetPtTitle, trackPtTitle,
                            # Basic data
                            linSpace,      peakFindingArray,  # noqa: E241
                            linSpaceRebin, peakFindingArrayRebin,
                            # CWT
                            peakLocations, peakLocationsRebin,
                            # Moving Average
                            maxMovingAvg, maxMovingAvgRebin,
                            # Smoothed gaussian
                            linSpaceResample, smoothedArray, maxSmoothedMovingAvg,
                            # Linear fits
                            maxLinearFit1D, maxLinearFit1DRebin,
                            maxLinearFit2D, maxLinearFit2DRebin):

    # Make the actual plot
    fig, ax = plt.subplots()
    # Add additional y margin at the bottom so the legend will fit a bit better
    # Cannot do asyemmtric padding via `ax.set_ymargin()`, so we'll do it by hand
    # See: https://stackoverflow.com/a/42804403
    dataMin = min(peakFindingArray.min(), peakFindingArrayRebin.min())
    dataMax = max(peakFindingArray.max(), peakFindingArrayRebin.max())
    yMin = dataMin - 0.5 * (dataMax - dataMin)
    yMax = dataMax + 0.12 * (dataMax - dataMin)
    ax.set_ylim(yMin, yMax)

    # Can either plot the hist or the array
    # Hist based on: https://stackoverflow.com/a/8553614
    #ax.hist(linSpace, weights=peakFindingArray, bins=len(peakFindingArray))
    # If plotting the hist, it's best to set the y axis limits to make it easier to view
    #ax.set_ylim(ymin=.95*min(peakFindingArray), ymax=1.05*max(peakFindingArray))
    # Plot array
    ax.plot(linSpace, peakFindingArray, label="ME")
    ax.plot(linSpaceRebin, peakFindingArrayRebin, label = "ME rebin")
    # Peak finding
    # Set zorder of 10 to ensure that the stars are always visible
    plotArrayPeak = ax.plot(linSpace[peakLocations], peakFindingArray[peakLocations], marker="*", markersize=10, linestyle="None", label = "CWT", zorder=10)
    plotArrayRebinPeak = ax.plot(linSpaceRebin[peakLocationsRebin], peakFindingArrayRebin[peakLocationsRebin], marker="*", markersize=10, linestyle="None", label = "CWT rebin", zorder=10)
    # Moving average
    ax.axhline(maxMovingAvg, color = plotArrayPeak[0].get_color(), label = r"Mov. avg. (size $\pi$)")
    ax.axhline(maxMovingAvgRebin, color = plotArrayRebinPeak[0].get_color(), linestyle = "--", label = "Mov. avg. rebin")
    # Gaussian
    # Use a mask so the range doesn't get extremely distorted when the interpolation drops around the edges
    mask = np.where(np.logical_and(linSpaceResample > -0.3 * np.pi, linSpaceResample < 1.3 * np.pi))
    plotGaussian = ax.plot(linSpaceResample[mask], smoothedArray[mask], label = "Gauss. smooth")
    ax.axhline(maxSmoothedMovingAvg, color = plotGaussian[0].get_color(), linestyle = "--", label ="Gauss. mov. avg")
    #ax.axhline(maxSmoothed, color = plotGaussian[0].get_color(), linestyle = ":", label = "Gauss. max")

    # Linear fits
    ax.axhline(maxLinearFit1D, color = "g", label = "1D fit")
    ax.axhline(maxLinearFit1DRebin, color = "g", linestyle = "--", label = "1D fit rebin")
    ax.axhline(maxLinearFit2D, color = "b", label = "2D fit")
    ax.axhline(maxLinearFit2DRebin, color = "b", linestyle = "--", label = "2D fit rebin")

    etaLimitsLabel = AnchoredText(r"|$\Delta\eta$|<{}".format(etaLimits[1]), loc=2, frameon=False)
    ax.add_artist(etaLimitsLabel)

    # Legend and Labels for the plot
    #ax.set_ymargin = 0.01
    ax.legend(loc="lower left", ncol=3)
    #ax.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
    #          ncol=3, mode="expand", borderaxespad=0)
    ax.set_title("ME norm. for {}, {}".format(jetPtTitle, trackPtTitle))
    ax.set_ylabel(r"$\Delta N/\Delta\varphi$")
    ax.set_xlabel(r"$\Delta\varphi$")

    #plt.tight_layout()
    plotBase.savePlot(jetH, fig, histName)
    # Close the figure
    plt.close(fig)

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
    palette = sns.color_palette()

    # Signal
    # Blue used for the signal data color
    # NOTE: Blue really doesn't look good with ROOT_kBird, so for that case, the
    #       signal fit color, seaborn green, should be used.
    signalColor = palette[0] + (1.0,)
    signalRegion = highlight_RPF.highlightRegion("Signal dom. region,\n" + r"$|\Delta\eta|<0.6$", signalColor)
    signalRegion.addHighlightRegion((-np.pi / 2, 3.0 * np.pi / 2), (-0.6, 0.6))
    highlightRegions.append(signalRegion)

    # Background
    # Red used for background data color
    backgroundColor = palette[2] + (1.0,)
    backgroundPhiRange = (-np.pi / 2, np.pi / 2)
    backgroundRegion = highlight_RPF.highlightRegion("Background dom. region,\n" + r"$0.8<|\Delta\eta|<1.2$", backgroundColor)
    backgroundRegion.addHighlightRegion(backgroundPhiRange, (-1.2, -0.8))
    backgroundRegion.addHighlightRegion(backgroundPhiRange, ( 0.8,  1.2))  # noqa: E201, E241
    highlightRegions.append(backgroundRegion)

    return highlightRegions

def plotRPFFitRegions(jetH, jetPtBin = 1, trackPtBin = 4):
    """ Plot showing highlighted RPF fit regions.

    Args:
        jetH (JetHAnalysis.JetHAnalysis): Main analysis object.
        jetPtBin (int): Jet pt bin of the hist to be plotted.
        trackPtBin (int): Track pt bin of the hist to be plotted.
    """
    # Retrieve the hist to be plotted
    # Here we selected the corrected 2D correlation
    # Bins are currently selected arbitrarily
    observable = jetH.signal2D[jetH.histNameFormat2D.format(jetPtBin = jetPtBin, trackPtBin = trackPtBin, tag = "corr")]

    with sns.plotting_context(context = "notebook", font_scale = 1.5):
        # Perform the plotting
        # TODO: Determmine if color overlays are better here!
        (fig, ax) = highlight_RPF.plotRPFFitRegions(histogram.getArrayFromHist2D(observable.hist.hist),
                                                    highlightRegions = defineHighlightRegions(),
                                                    useColorOverlay = False)

        # Add additional labeling
        # Axis
        # Needed to fix zaxis rotation. See: https://stackoverflow.com/a/21921168
        ax.zaxis.set_rotate_label(False)
        ax.set_zlabel(r"$1/\mathrm{N}_{\mathrm{trig}}\mathrm{d^{2}N}/\mathrm{d}\Delta\varphi\mathrm{d}\Delta\eta$", rotation=90)
        # Set the distance from axis to label in pixels.
        # This is not ideal, but clearly tight_layout doesn't work as well for 3D plots
        ax.xaxis.labelpad = 12
        # Visually, dEta looks closer
        ax.yaxis.labelpad = 15
        ax.zaxis.labelpad = 12
        # Overall
        aliceLabel = jetH.aliceLabel.str()
        systemLabel = params.systemLabel(energy = jetH.collisionEnergy,
                                         system = jetH.collisionSystem,
                                         activity = jetH.eventActivity)
        (jetFinding, constituentCuts, leadingHadron, jetPt) = params.jetPropertiesLabel(observable.jetPtBin)
        assocPt = params.generateTrackPtRangeString(observable.trackPtBin)

        # Upper left side
        upperLeftText = ""
        upperLeftText += aliceLabel
        upperLeftText += "\n" + systemLabel
        upperLeftText += "\n" + jetPt
        upperLeftText += "\n" + jetFinding

        # Upper right side
        upperRightText = ""
        upperRightText += leadingHadron
        upperRightText += "\n" + constituentCuts
        upperRightText += "\n" + assocPt

        # Need a different text function since we have a 3D axis
        ax.text2D(0.01, 0.99, upperLeftText,
                  horizontalalignment = "left",
                  verticalalignment = "top",
                  multialignment = "left",
                  transform = ax.transAxes)
        ax.text2D(0.00, 0.00, upperRightText,
                  horizontalalignment = "left",
                  verticalalignment = "bottom",
                  multialignment = "left",
                  transform = ax.transAxes)

        # Finish up
        plotBase.savePlot(jetH, fig, "highlightRPFRegions")
        plt.close(fig)

