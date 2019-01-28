#!/usr/bin/env python

""" General plotting module.

Contains brief plotting functions which don't belong elsewhere

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import logging

from jet_hadron.plot import base as plot_base

logger = logging.getLogger(__name__)

####
# General plots
####
def plotGeneralAnalysisHistograms(jetH):
    """ Plot general analysis histograms related to the JetH analysis. """
    import ROOT
    canvas = ROOT.TCanvas("canvas", "canvas")
    # Hists
    drawAndSaveGeneralHist(jetH, canvas = canvas, hists = jetH.generalHists1D)
    # 2D hists
    drawAndSaveGeneralHist(jetH, canvas = canvas, hists = jetH.generalHists2D, drawOpt = "colz")

def drawAndSaveGeneralHist(jetH, canvas, hists, drawOpt = ""):
    """ Simple helper to draw a histogram and save it out. """
    for name, hist in hists.items():
        logger.debug("name: {}, hist: {}".format(name, hist))
        if hist:
            logger.info("Drawing general hist {}".format(name))
            # Set logy for pt hists
            resetLog = False
            if "Pt" in name:
                canvas.SetLogy(True)
                resetLog = True

            hist.Draw(drawOpt)
            plot_base.save_plot(jetH, canvas, name)

            if resetLog:
                canvas.SetLogy(False)
        else:
            logger.info("Skipping hist {} because it doesn't exist".format(name))

#####
# Spectra
#####
def plotTriggerJetSpectra(jetH):
    """ Plot the trigger jet spectra. """
    # Use the general histogrmms as a proxy, since usually if we want them, then we
    # also want the trigger spectra
    if jetH.taskConfig["processingOptions"]["generalHistograms"]:
        import ROOT
        canvas = ROOT.TCanvas("canvas", "canvas")
        canvas.SetLogy(True)
        jetH.triggerJetPt[jetH.histNameFormatTrigger].hist.Draw()
        plot_base.save_plot(jetH, canvas, jetH.histNameFormatTrigger)

