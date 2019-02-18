#!/usr/bin/env python

""" General plotting module.

Contains brief plotting functions which don't belong elsewhere

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import logging

from jet_hadron.plot import base as plot_base

logger = logging.getLogger(__name__)

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

