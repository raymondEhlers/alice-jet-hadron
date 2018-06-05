#!/usr/bin/env python

######################
# General plotting module
#
# Contains brief plotting functions which don't belong elsewhere
######################

import jetH.plot.base as plotBase

# Use matplitlib for some plots
import matplotlib.pyplot as plt
# And use ROOT in others
import rootpy.ROOT as ROOT

####
# General plots
####
def plotGeneralAnalysisHistograms(jetH):
    """ Plot general analysis histograms related to the JetH analysis. """
    canvas = ROOT.TCanvas("canvas", "canvas")
    # Hists
    drawAndSaveGeneralHist(jetH, canvas = canvas, hists = jetH.generalHists1D)
    # 2D hists
    drawAndSaveGeneralHist(jetH, canvas = canvas, hists = jetH.generalHists2D, drawOpt = "colz")

def drawAndSaveGeneralHist(jetH, canvas, hists, drawOpt = ""):
    """ Simple helper to draw a histogram and save it out. """
    for name, hist in hists.iteritems():
        logger.debug("name: {}, hist: {}".format(name, hist))
        if hist:
            logger.info("Drawing general hist {}".format(name))
            # Set logy for pt hists
            resetLog = False
            if "Pt" in name:
                canvas.SetLogy(True)
                resetLog = True

            hist.Draw(drawOpt)
            plotBase.saveCanvas(jetH, canvas, name)

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
    if jetH.config["processingOptions"]["generalHistograms"]:
        canvas = ROOT.TCanvas("canvas", "canvas")
        canvas.SetLogy(True)
        jetH.triggerJetPt[jetH.histNameFormatTrigger].hist.Draw()
        plotBase.saveCanvas(jetH, canvas, jetH.histNameFormatTrigger)

