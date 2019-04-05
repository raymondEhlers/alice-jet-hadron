#!/usr/bin/env python

""" Main jet-hadron correlations analysis module

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import coloredlogs
import copy
#import ctypes
import dataclasses
from dataclasses import dataclass
import enlighten
import enum
import IPython
import logging
import os
import numpy as np
import pprint
import sys
from typing import Any, cast, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from pachyderm import generic_class
from pachyderm import generic_config
from pachyderm import histogram
from pachyderm import projectors
from pachyderm.projectors import HistAxisRange
from pachyderm import utils
from pachyderm.utils import epsilon
from pachyderm import yaml

import reaction_plane_fit as rpf
from reaction_plane_fit import fit as rpf_fit
from reaction_plane_fit import three_orientations

from jet_hadron.base import analysis_config
from jet_hadron.base import analysis_objects
from jet_hadron.base import labels
from jet_hadron.base import params
from jet_hadron.base.typing_helpers import Hist
from jet_hadron.plot import generic_hist as plot_generic_hist
from jet_hadron.plot import correlations as plot_correlations
from jet_hadron.plot import fit as plot_fit
from jet_hadron.plot import extracted as plot_extracted
from jet_hadron.analysis import correlations_helpers
from jet_hadron.analysis import fit as fitting
from jet_hadron.analysis import generic_tasks

import ROOT

# Setup logger
logger = logging.getLogger(__name__)

# Run in batch mode
ROOT.gROOT.SetBatch(True)

this_module = sys.modules[__name__]

class JetHCorrelationSparse(enum.Enum):
    """ Defines the axes in the Jet-Hadron THn Sparses. """
    centrality = 0
    jet_pt = 1
    track_pt = 2
    delta_eta = 3
    delta_phi = 4
    leading_jet = 5
    jet_hadron_deltaR = 6
    reaction_plane_orientation = 7

    # Handle YAML serialization
    to_yaml = classmethod(yaml.enum_to_yaml)
    from_yaml = classmethod(yaml.enum_from_yaml)

class JetHTriggerSparse(enum.Enum):
    """ Define the axes in the Jet-Hadron Trigger Sparse. """
    centrality = 0
    jet_pt = 1
    reaction_plane_orientation = 2

    # Handle YAML serialization
    to_yaml = classmethod(yaml.enum_to_yaml)
    from_yaml = classmethod(yaml.enum_from_yaml)

class JetHCorrelationSparseProjector(projectors.HistProjector):
    """ Projector for THnSparse into 2D histograms.

    Note:
        This class isn't really necessary, but it makes further customization straightforward if
        it is found to be necessary, so we keep it around.
    """
    ...

class JetHCorrelationProjector(projectors.HistProjector):
    """ Projector for the jet-hadron 2D correlation hists to 1D correlation hists. """
    def get_hist(self, observable: "CorrelationObservable2D", **kwargs: Any) -> Hist:
        """ Retrieve the histogram from the observable. """
        return observable.hist

class JetHAnalysis(analysis_objects.JetHBase):
    """ Main jet-hadron analysis task. """
    def __init__(self, *args, **kwargs):
        """ """
        # Initialize the base class
        super().__init__(*args, **kwargs)

        # Bookkeeping
        self.ranProjections = False
        self.ranFitting = False
        self.ranPostFitProcessing = False

        # Input histograms
        self.inputHists = []

        # Projectors
        self.sparseProjectors = []
        self.correlationProjectors = []

        # General histograms
        self.generalHists1D = {}
        self.generalHists2D = {}

        self.generalHists = [self.generalHists1D, self.generalHists2D]

        # Trigger jet pt. Used for calculating N_trig
        self.triggerJetPt = {}

        # 2D correlations
        self.rawSignal2D = {}
        self.mixedEvents2D = {}
        self.signal2D = {}

        # All 2D hists
        # NOTE: We can't use an iterator here because we rely on them being separate for initializing from ROOT files.
        #       Would look something like:
        #self.hists2D = itertools.chain(self.rawSignal2D, self.mixedEvents2D, self.signal2D)
        self.hists2D = [self.rawSignal2D, self.mixedEvents2D, self.signal2D]

        # 1D correlations
        self.dPhi = {}
        self.dPhiArray = {}
        self.dPhiSubtracted = {}
        self.dPhiSubtractedArray = {}
        self.dPhiSideBand = {}
        self.dPhiSideBandArray = {}
        self.dEtaNS = {}
        self.dEtaNSArray = {}
        self.dEtaNSSubtracted = {}
        self.dEtaNSSubtractedArray = {}

        # All 1D hists
        self.hists1D = [self.dPhi, self.dPhiSubtracted, self.dPhiSideBand, self.dEtaNS, self.dEtaNSSubtracted]
        # Standard
        self.hists1DStandard = [self.dPhi, self.dPhiSideBand, self.dEtaNS]
        # Standard hist array
        self.hists1DStandardArray = [self.dPhiArray, self.dPhiSideBandArray, self.dEtaNSArray]
        # Subtracted
        self.hists1DSubtracted = [self.dPhiSubtracted, self.dEtaNSSubtracted]
        # Subtracted hist array
        self.hists1DSubtractedArray = [self.dPhiSubtractedArray, self.dEtaNSSubtractedArray]

        # 1D fits
        self.dPhiFit = {}
        self.dPhiSideBandFit = {}
        self.dPhiSubtractedFit = {}
        self.dEtaNSFit = {}
        self.dEtaNSSubtractedFit = {}

        # All 1D fits
        self.fits1D = [self.dPhiFit, self.dPhiSideBandFit, self.dPhiSubtractedFit, self.dEtaNSFit, self.dEtaNSSubtractedFit]
        # Standard
        self.fits1DStandard = [self.dPhiFit, self.dPhiSideBandFit, self.dEtaNSFit]
        # Subtracted
        self.fits1DSubtracted = [self.dPhiSubtractedFit, self.dEtaNSSubtractedFit]

        # Yields
        self.yieldsNS = {}
        self.yieldsAS = {}
        self.yieldsDEtaNS = {}

        # All yields
        self.yields = [self.yieldsNS, self.yieldsAS, self.yieldsDEtaNS]

        # Widths
        self.widthsNS = {}
        self.widthsAS = {}
        self.widthsDEtaNS = {}

        # All widths
        self.widths = [self.widthsNS, self.widthsAS, self.widthsDEtaNS]

    def fitBackgroundDominatedRegion(self):
        """ Fit the background dominated 1D correlations. """
        fitFunctions = {params.collisionSystem.pp: fitting.fitDeltaPhiBackground,
                        params.collisionSystem.PbPb: fitting.fitDeltaPhiBackground}
        self.fit1DCorrelations(hists = self.dPhiSideBand, fits = self.dPhiSideBandFit, fitFunction = fitFunctions[self.collisionSystem])

    def fitSignalRegion(self):
        """ Fit the signal dominated 1D correlations. """
        fitFunctions = {params.collisionSystem.pp: fitting.fitDeltaPhi,
                        params.collisionSystem.PbPb: fitting.ReactionPlaneFit}
        self.fit1DCorrelations(hists = self.dPhi, fits = self.dPhiFit, fitFunction = fitFunctions[self.collisionSystem])

    def fitDEtaCorrelations(self):
        """ Fit the dEta near-side correlation. """
        self.fit1DCorrelations(hists = self.dEtaNS, fits = self.dEtaNSFit, fitFunction = fitting.fitDeltaEta)

    def fit1DCorrelations(self, hists, fits, fitFunction):
        """ Fit the selected 1D correlations. """
        fitOptions = self.config["fitOptions"]

        for name, observable in hists.items():
            # Fit the unsubtracted delta phi hist
            logger.debug("Fitting observable {} (named: {}) with function {}".format(observable, name, fitFunction))
            fit = fitFunction(observable.hist, trackPtBin = observable.trackPtBin, zyam = fitOptions["zyam"], disableVN = fitOptions["disableVN"], setFixedVN = fitOptions["fixedVN"])
            fits["{}_fit".format(name)] = fit

    def subtractSignalRegion(self):
        """ Subtract the fit from dPhi 1D signal correlations. """
        self.subtract1DCorrelations(hists = self.dPhi, signalFits = self.dPhiFit,
                                    backgroundFits = self.dPhiSideBandFit,
                                    subtractedHists = self.dPhiSubtracted,
                                    subtractedFits = self.dPhiSubtractedFit)

    def subtractDEtaNS(self):
        """ Subtract the fit from dEta NS correlatons. """
        self.subtract1DCorrelations(hists = self.dEtaNS, signalFits = self.dEtaNSFit,
                                    backgroundFits = self.dEtaNSFit,
                                    subtractedHists = self.dEtaNSSubtracted,
                                    subtractedFits = self.dEtaNSSubtractedFit)

    def subtract1DCorrelations(self, hists, signalFits, backgroundFits, subtractedHists, subtractedFits):
        """ Subtract the fit from seleected 1D correlations. """
        for ((name, observable), signalFit, bgFit) in zip(hists.items(), signalFits.values(), backgroundFits.values()):
            logger.debug("name: {}, observable: {}, signalFit: {}, bgFit: {}".format(name, observable, signalFit, bgFit))
            # Create a clone of the dPhi hist to subtract
            subtractedHist = observable.hist.Clone("{}_subtracted".format(name))

            # Fit background
            #bgFit = fitDeltaPhiBackground(dPhi, trackPtBin = trackPtBin, zyam = zyam, disableVN = disableVN, setFixedVN = setFixedVN)
            # Get the minimum from the fit to be less sensitive to fluctuations
            bg = signalFit.GetMinimum()
            #bgFit.SetParameter(0, bg)

            # Remove background
            # Already have the background fits, so remove it
            subtractedHist.Add(bgFit, -1)
            subtractedHists["{}_subtracted".format(name)] = CorrelationObservable1D(
                jetPtBin = observable.jetPtBin,
                trackPtBin = observable.trackPtBin,
                correlationType = observable.correlationType,
                axis = observable.axis,
                hist = analysis_objects.HistContainer(subtractedHist)
            )

            # Create subtracted fit from previous fits
            # TODO: This should be improved! This should actually do the fit!
            subtractedFit = signalFit.Clone("{}_subtracted_fit".format(name))
            # For now, manually zero out the backgroud in the fit, which is what the above subtraction does for the hist
            # We need to subtract the bg fit from the pedistal, because they don't always agree
            # (sometimes the pedistal is smaller than a flat background - the reason is unclear!)
            subtractedFit.SetParameter(6, subtractedFit.GetParameter(6) - bg)
            #printFitParameters(bgFit)
            #printFitParameters(signalFit)
            #printFitParameters(subtractedFit)

            # Store the subtracted fit
            subtractedFits[subtractedFit.GetName()] = subtractedFit

    def extractWidths(self):
        """ Extract widths from the fits. """
        parameters = {"NS": [2, self.widthsNS, self.dPhiSubtracted, self.dPhiSubtractedFit],
                      "AS": [5, self.widthsAS, self.dPhiSubtracted, self.dPhiSubtractedFit],
                      "dEtaNS": [2, self.widthsDEtaNS, self.dEtaNSSubtracted, self.dEtaNSSubtractedFit]}

        for location, (parameterNumber, widths, hists, fits) in parameters.items():
            for (name, observable), fit in zip(hists.items(), fits.values()):
                widths["{}_width".format(name)] = analysis_objects.ExtractedObservable(
                    jetPtBin = observable.jetPtBin,
                    trackPtBin = observable.trackPtBin,
                    value = fit.GetParameter(parameterNumber),
                    error = fit.GetParError(parameterNumber)
                )

    def writeToRootFile(self, output_observable, mode = "UPDATE"):
        """ Write output list to a file """
        ...

    def write1DCorrelations(self):
        """ Write the 1D Correlations to file. """
        # Write the ROOT files
        self.writeToRootFile(self.hists1DStandard)

    def writeSubtracted1DCorrelations(self):
        """ Write the subtracted 1D Correlations to file. """
        self.writeToRootFile(self.hists1DSubtracted)

    def writeSubtractedArray1DCorrelations(self):
        """ Write 1D subtracted correlations stored in hist arrays. """
        self.writeToRootFile(self.hists1DSubtractedArray)

    def write1DFits(self):
        """ Write the 1D fits to file. """
        self.writeToRootFile(self.fits1DStandard)

    def writeSubtracted1DFits(self):
        """ Write the subtracted 1D fits to file. """
        self.writeToRootFile(self.fits1DSubtracted)

    def InitFromRootFile(self, GeneralHists = False, TriggerJetSpectra = False, Correlations2D = False, Correlations1D = False, Correlations1DArray = False, Correlations1DSubtracted = False, Correlations1DSubtractedArray = False, Fits1D = False, exitOnFailure = True):
        """ Initialize the JetHAnalysis object from a saved ROOT file. """
        filename = os.path.join(self.output_prefix, self.output_filename)

        # TODO: Depending on this ROOT file when opening a YAML file is not really right.
        #       However, it's fine for now because the ROOT file should almost always exist

        with histogram.RootOpen(filename = filename, mode = "READ") as f:
            if GeneralHists:
                logger.critical("General hists are not yet implemented!")
                sys.exit(1)

            if TriggerJetSpectra:
                # We only have one trigger jet hist at the moment, but by using the same approach
                # as for the correlations, it makes it straightforward to generalize later if needed
                histName = self.histNameFormatTrigger
                hist = f.Get(histName)
                if hist:
                    #self.triggerJetPt[histName] = analysis_objects.Observable(hist = analysis_objects.HistContainer(hist))
                    # Ensure that the hist doesn't disappear when the file closes
                    hist.SetDirectory(0)
                else:
                    # We usually are explicitly ask for a hist, so we should be aware if it doesn't exist!
                    logger.critical("Failed to retrieve hist {0} from the ROOT input file!".format(histName))
                    sys.exit(1)

            # Loop over the expected names and insert as possible.
            for (jetPtBin, trackPtBin) in params.iterateOverJetAndTrackPtBins(self.config):
                if Correlations2D:
                    for (storedDict, tag) in zip(self.hists2D, ["raw", "mixed", "corr"]):
                        # 2D hists
                        histName = self.histNameFormat2D.format(jetPtBin = jetPtBin, trackPtBin = trackPtBin, tag = tag)
                        hist = f.Get(histName)
                        if hist:
                            storedDict[histName] = analysis_objects.CorrelationObservable(
                                jetPtBin = jetPtBin,
                                trackPtBin = trackPtBin,
                                hist = analysis_objects.HistContainer(hist)
                            )
                            # Ensure that the hist doesn't disappear when the file closes
                            hist.SetDirectory(0)
                        else:
                            # We usually are explicitly ask for a hist, so we should be aware if it doesn't exist!
                            logger.critical("Failed to retrieve hist {0} from the ROOT input file!".format(histName))
                            sys.exit(1)

                if Correlations1D or Correlations1DSubtracted or Correlations1DArray or Correlations1DSubtractedArray:
                    # Form of [dict list, nameFormat, correlationType, axis]
                    # Normal format is self.histNameFormatDPhi.format(jetPtBin = jetPtBin, trackPtBin = trackPtBin, tag = correlationType.str())
                    hists1D = []
                    retrieveArray = False
                    if Correlations1D:
                        hists1D.append([self.dPhi,         self.histNameFormatDPhi, analysis_objects.CorrelationType.signal_dominated,     analysis_objects.CorrelationAxis.delta_phi])  # noqa: E241
                        hists1D.append([self.dPhiSideBand, self.histNameFormatDPhi, analysis_objects.CorrelationType.background_dominated, analysis_objects.CorrelationAxis.delta_phi])  # noqa: E241
                        hists1D.append([self.dEtaNS,       self.histNameFormatDEta, analysis_objects.CorrelationType.near_side,            analysis_objects.CorrelationAxis.delta_eta])  # noqa: E241
                    if Correlations1DSubtracted:
                        hists1D.append([self.dPhiSubtracted,   self.histNameFormatDPhiSubtracted, analysis_objects.CorrelationType.signal_dominated,     analysis_objects.CorrelationAxis.delta_phi])  # noqa: E241
                        hists1D.append([self.dEtaNSSubtracted, self.histNameFormatDEtaSubtracted, analysis_objects.CorrelationType.near_side,            analysis_objects.CorrelationAxis.delta_eta])  # noqa: E241
                    if Correlations1DArray:
                        logger.debug("Correlations1DArray hists")
                        retrieveArray = True
                        hists1D.append([self.dPhiArray,         self.histNameFormatDPhiArray, analysis_objects.CorrelationType.signal_dominated,     analysis_objects.CorrelationAxis.delta_phi])  # noqa: E241
                        hists1D.append([self.dPhiSideBandArray, self.histNameFormatDPhiArray, analysis_objects.CorrelationType.background_dominated, analysis_objects.CorrelationAxis.delta_phi])  # noqa: E241
                        hists1D.append([self.dEtaNS,            self.histNameFormatDEtaArray, analysis_objects.CorrelationType.near_side,            analysis_objects.CorrelationAxis.delta_eta])  # noqa: E241
                    if Correlations1DSubtractedArray:
                        retrieveArray = True
                        hists1D.append([self.dPhiSubtractedArray,   self.histNameFormatDPhiSubtractedArray, analysis_objects.CorrelationType.signal_dominated, analysis_objects.CorrelationAxis.delta_phi])  # noqa: E241
                        hists1D.append([self.dEtaNSSubtractedArray, self.histNameFormatDEtaSubtractedArray, analysis_objects.CorrelationType.near_side,        analysis_objects.CorrelationAxis.delta_eta])  # noqa: E241

                    for (storedDict, nameFormat, correlationType, axis) in hists1D:
                        # 1D hists
                        histName = nameFormat.format(jetPtBin = jetPtBin, trackPtBin = trackPtBin, tag = correlationType.str())
                        # Should only be assigned if we successfully retrieve the hist
                        actualHistName = None
                        logger.debug("histName: {}".format(histName))
                        if retrieveArray:
                            logger.debug("Retrieving array: ({}, {})".format(jetPtBin, trackPtBin))
                            hist = analysis_objects.HistArray.initFromYAML(prefix = self.outputPrefix, objType = correlationType, jetPtBin = jetPtBin, trackPtBin = trackPtBin)
                            if hist:
                                actualHistName = hist.outputFilename.format(type = correlationType.str(),
                                                                            jetPtBin = jetPtBin,
                                                                            trackPtBin = trackPtBin)
                        else:
                            hist = f.Get(histName)
                            if hist:
                                actualHistName = hist.GetName()
                                # Ensure that the hist doesn't disappear when the file closes
                                hist.SetDirectory(0)
                                # Create a hist container
                                hist = analysis_objects.HistContainer(hist)

                        if hist:
                            storedDict[histName] = analysis_objects.CorrelationObservable1D(
                                jetPtBin = jetPtBin,
                                trackPtBin = trackPtBin,
                                correlationType = correlationType,
                                axis = axis,
                                hist = hist
                            )
                        else:
                            # By default, we explicitly ask for a hist, so we should be aware if it doesn't
                            # exist!
                            logger.critical(f"Failed to retrieve hist {histName} from the ROOT input file!")
                            if exitOnFailure:
                                sys.exit(1)
                            else:
                                continue

                        logger.debug("Init hist observable: {}, hist: {}, name: {}".format(storedDict[histName], storedDict[histName].hist, actualHistName))

            if Fits1D:
                logger.critical("1D fits is not yet implemented!")
                sys.exit(1)

    def measureMixedEventNormalization(self, mixedEvent, jetPtBin, trackPtBin):
        """ Determine normalization of the mixed event. """
        ...

    #################################
    # Utility functions for the class
    #################################
    def generateTeXToIncludeFiguresInAN(self):
        """ Generate latex to put into AN. """
        outputText = ""

        # Define the histograms that should be included
        # Of the form (name, tag, caption)
        histsToWriteOut = {}
        # Raw
        histsToWriteOut["raw"] = (self.histNameFormat2D, "raw", r"Raw correlation function with the efficiency correction $\epsilon(\pT{},\eta{})$ applied, but before acceptance correction via the mixed events. This correlation is for $%(jetPtLow)s < \pTJet{} < %(jetPtHigh)s$ \gevc{} and $%(trackPtLow)s < \pTAssoc{} < %(trackPtHigh)s$ \gevc{}.")
        # Mixed
        histsToWriteOut["mixed"] = (self.histNameFormat2D, "mixed", r"Mixed event correlation for $%(jetPtLow)s < \pTJet{} < %(jetPtHigh)s$ \gevc{} and $%(trackPtLow)s < \pTAssoc{} < %(trackPtHigh)s$ \gevc{}. Note that this correlation has already been normalized to unity at the region of maximum efficiency.")
        # Corrected 2D
        histsToWriteOut["corr"] = (self.histNameFormat2D, "corr", r"Acceptance corrected correlation for $%(jetPtLow)s < \pTJet{} < %(jetPtHigh)s$ \gevc{} and $%(trackPtLow)s < \pTAssoc{} < %(trackPtHigh)s$ \gevc{}.")
        # Mixed event norm
        histsToWriteOut["mixedEventNorm"] = (self.histNameFormat2D, "mixed_peakFindingHist", r"Mixed event normalization comparison for a variety of possible functions to find the maximum. This mixed event corresponds to $%(jetPtLow)s < \pTJet{} < %(jetPtHigh)s$ \gevc{} and $%(trackPtLow)s < \pTAssoc{} < %(trackPtHigh)s$ \gevc{}.")
        # dPhi correlations
        # TODO: Depend on the type of fit here instead of assuming signal dominated
        histsToWriteOut["dPhiCorrelations"] = (self.fitNameFormat, analysis_objects.CorrelationType.signal_dominated.str(), r"\dPhi{} correlation with the all angles signal and event plane dependent background fit components. This correlation corresponding to $%(jetPtLow)s < \pTJet{} < %(jetPtHigh)s$ \gevc{} and $%(trackPtLow)s < \pTAssoc{} < %(trackPtHigh)s$ \gevc{}.")
        # TODO: Add comparisons to Joel, but probably best to do in an entirely separate section
        histsToWriteOut["joelComparisonSubtracted"] = ("joelComparison_jetPt{jetPtBin}_trackPt{trackPtBin}_{tag}", analysis_objects.CorrelationType.signal_dominated.str() + "_subtracted", r"Subtracted \dPhi{} correlation comparing correlations from this analysis and those produced using the semi-central analysis code described in \cite{jetHEventPlaneAN}. Error bars correspond to statistical errors and error bands correspond to the error on the fit. This correlation corresponding to $%(jetPtLow)s < \pTJet{} < %(jetPtHigh)s$ \gevc{} and $%(trackPtLow)s < \pTAssoc{} < %(trackPtHigh)s$ \gevc{}.")

        # Define the overall template
        figTemplate = r"""
\begin{figure}
\centering
\includegraphics[width=.9\textwidth]{images/%(collisionSystem)s/%(reaction_plane_orientation)s/%(name)s.eps}
\caption{%(description)s}
\label{fig:%(name)s}
\end{figure}"""

        # Iterate over the available pt bins
        for jetPtBin, trackPtBin in params.iterateOverJetAndTrackPtBins(self.config):

            # Section output
            out = ""

            # Add section description
            descriptionDict = {"jetPtLow": params.jetPtBins[jetPtBin], "jetPtHigh": params.jetPtBins[jetPtBin + 1], "trackPtLow": params.trackPtBins[trackPtBin], "trackPtHigh": params.trackPtBins[trackPtBin + 1]}
            out += "\n"
            out += r"\subsubsection{$%(jetPtLow)s < \pTJet{} < %(jetPtHigh)s$, $%(trackPtLow)s < \pTAssoc{} < %(trackPtHigh)s$}" % descriptionDict

            # Iterate over the hists to be included
            binDict = {"jetPtBin": jetPtBin, "trackPtBin": trackPtBin}
            for (name, tag, description) in histsToWriteOut.values():
                # Define name
                baseDict = {"tag": tag}
                baseDict.update(binDict)
                name = name.format(**baseDict)
                # Fill in description
                descriptionDict.update({"name": name})
                descriptionDict.update(baseDict)
                description = description % descriptionDict

                # Define values needed for figure template
                figDict = {"reaction_plane_orientation": self.reaction_plane_orientation.filenameStr(),
                           "description": description,
                           "collisionSystem": self.collisionSystem.str()}
                figDict.update(descriptionDict)

                # Fill template and add to output string
                out += figTemplate % figDict
            # clearpage due to "too many floats". See: https://tex.stackexchange.com/a/46514
            out += "\n" + r"\clearpage{}" + "\n"

            # This is quite verbose, but can be useful for debugging
            #logger.debug("jetPtBin: {}, trackPtBin: {}, out: {}".format(jetPtBin, trackPtBin, out))

            outputText += out

        with open(os.path.join(self.outputPrefix, "resultsCombined.tex"), "wb") as f:
            f.write(outputText.encode())

    @staticmethod
    def postProjectionProcessing1DCorrelation(observable, normalizationFactor, rebinFactor, titleLabel, jetPtBin, trackPtBin):
        """ Basic post processing tasks for a new 1D correlation observable. """
        ...

    @classmethod
    def run(cls, configFilename, selectedAnalysisOptions):
        """ Main driver function to create, process, and plot task hists.

        Args:
            configFilename (str): Filename of the yaml config.
            selectedAnalysisOptions (params.selectedAnalysisOptions): Selected analysis options.
        Returns:
            nested tuple: Tuple of nested analysis objects as described in analysis_config.constructFromConfigurationFile(...).
        """
        # Basic setup
        # Load reasonable style...
        #ROOT.gROOT.ProcessLine(".x {0}".format(os.path.expandvars("${MYINSTALL}/include/readableStyle.h")))
        #ROOT.gROOT.ProcessLine(".x readableStyle.h")
        # Create logger
        logging.basicConfig(level=logging.DEBUG)
        # Quiet down the matplotlib logging
        logging.getLogger("matplotlib").setLevel(logging.INFO)
        # Turn off stats box
        ROOT.gStyle.SetOptStat(0)

        # Construct analysis tasks
        (selectedOptionNames, analyses) = cls.constructFromConfigurationFile(
            configFilename = configFilename,
            selectedAnalysisOptions = selectedAnalysisOptions
        )

        # Run the analysis
        logger.info("About to run the jet-h correlations analysis")
        for keys, jetH in generic_config.unrollNestedDict(analyses):
            # Print the jet-h analysis task selected analysis options
            opts = ["{name}: \"{value}\"".format(name = name, value = value.str()) for name, value in zip(selectedOptionNames, keys)]
            logger.info("Processing jet-h correlations task {} with options:\n\t{}".format(jetH.taskName, "\n\t".join(opts)))

            jetH.generateTeXToIncludeFiguresInAN()

            logger.info("Running analysis through projecting 1D correlations")
            jetH.runProjections()

            #plot_general.plotTriggerJetSpectra(jetH)

        # Need all of the event plane angles to perform the RP Fit
        logger.info("Running 1D fits")
        epFit = JetHAnalysis.runFitting(analyses)

        return (analyses, epFit)

    def runProjections(self):
        """ Steering function for plotting Jet-H histograms. """
        # Get configuration
        processing_options = self.taskConfig["processing_options"]

        # Only need to check if file exists for the first if statement because we cannot get past there without somehow having some hists
        file_exists = os.path.isfile(os.path.join(self.outputPrefix, self.outputFilename))

        # NOTE: Only normalize hists when plotting, and then only do so to a copy!
        #       The exceptions are the 2D correlations, which are normalized by n_trig for the raw correlation and the maximum efficiency
        #       for the mixed events. They are excepted because we don't have a purpose for such unnormalized hists.
        if processing_options["generate2DCorrelations"] or not file_exists:
            ...
        else:
            ...

        if processing_options["plot2DCorrelations"]:
            ...

        if processing_options["generate1DCorrelations"]:
            # Write the properly scaled projections
            self.write1DCorrelations()

            if processing_options["plot1DCorrelations"]:
                logger.info("Plotting 1D correlations")
                plot_correlations.plot1DCorrelations(self)

            # Ensure that the next step in the chain is run
            processing_options["fit1DCorrelations"] = True
        else:
            # Initialize the 1D correlations from the file
            logger.info("Loading 1D correlations from file")
            self.InitFromRootFile(Correlations1D = True)
            self.InitFromRootFile(Correlations1DArray = True, exitOnFailure = False)

        self.ranProjections = True

    @staticmethod
    def runFitting(analyses):
        """ Run pp fitting. """
        # TODO: TEMP
        self = None
        # ENDTEMP

        # processing_options["fit1DCorrelations"]
        if False:
            # Run the fitting code
            epFit = None

            # Handle pp in the standard way
            # Fit the 1D correlations
            logger.info("Fitting the background domintation region")
            self.fitBackgroundDominatedRegion()
            logger.info("Fitting signal region")
            self.fitSignalRegion()
            #self.fitSignalRegion(hists = hists, jetH = jetH, outputPrefix = outputPrefix)
            self.fitDEtaCorrelations()

            # Write output
            self.write1DFits()

            # Ensure that the next step in the chain is run
            #processing_options["subtract1DCorrelations"] = True
        else:
            # Initialize the fits from the file
            self.InitFromRootFile(Fits1D = True)

        # Note that the step was completed
        for _, jetH in generic_config.unrollNestedDict(analyses):
            jetH.ranFitting = True

        return epFit

    @staticmethod
    def subtractEPHists(analyses, epFit):
        ...

    def postFitProcessing(self):
        """ PP post processing """
        # Ensure that the previous step was run
        if not self.ranFitting:
            logger.critical("Must run the fitting step before subtracting correlations!")
            sys.exit(1)

        # Get processing options
        processing_options = self.taskConfig["processing_options"]

        # Subtracted fit functions from the correlations
        if processing_options["subtract1DCorrelations"]:
            # Subtract fit functions
            logger.info("Subtracting fit functions.")
            logger.info("Subtracting side-band fit from signal region.")
            self.subtractSignalRegion()
            logger.info("Subtracting fit from near-side dEta.")
            self.subtractDEtaNS()

            # Write output
            self.writeSubtracted1DCorrelations()
            self.writeSubtracted1DFits()

            if processing_options["plot1DCorrelationsWithFits"]:
                logger.info("Plotting 1D correlations with fits")
                plot_correlations.plot1DCorrelationsWithFits(self)

            # Ensure that the next step in the chain is run
            processing_options["extractWidths"] = True
        else:
            # Initialize subtracted hists from the file
            pass

        # Note that the step was completed
        self.ranPostFitProcessing = True

def printFitParameters(fit):
    """ Print out all of the fit parameters. """
    outputParameters = []
    for i in range(0, fit.GetNpar()):
        parameter = fit.GetParameter(i)
        parameterName = fit.GetParName(i)
        lowerLimit = ROOT.Double(0.0)
        upperLimit = ROOT.Double(0.0)
        fit.GetParLimits(i, lowerLimit, upperLimit)

        outputParameters.append("{0}: {1} = {2} from {3} - {4}".format(i, parameterName, parameter, lowerLimit, upperLimit))

    pprint.pprint(outputParameters)
    #logger.debug("subtractedFitParameters: {0}".format([param for param in subtractedFit.GetParameters()]))

class PlotGeneralHistograms(generic_tasks.PlotTaskHists):
    """ Task to plot general task hists, such as centrality, Z vertex, very basic QA spectra, etc.

    Note:
        This class inherits from the base class just to add the possibility of disabling the
        task based on the configuration.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Only run if it's enabled.
        self.enabled = self.task_config["enabled"]

    def setup(self) -> None:
        if self.enabled:
            super().setup()
        else:
            logger.info("General hists disabled. Skipping setup.")

    def run(self, *args, **kwargs) -> bool:
        if self.enabled:
            return super().run(*args, **kwargs)
        else:
            logger.info("General hists disabled. Skipping running.")
            return False

class GeneralHistogramsManager(generic_tasks.TaskManager):
    """ Manager for plotting general histograms. """
    def construct_tasks_from_configuration_file(self) -> analysis_config.ConstructedObjects:
        return analysis_config.construct_from_configuration_file(
            task_name = "GeneralHists",
            config_filename = self.config_filename,
            selected_analysis_options = self.selected_analysis_options,
            additional_possible_iterables = {"task_label": None, "pt_hard_bin": None},
            additional_classes_to_register = [plot_generic_hist.HistPlotter],
            obj = PlotGeneralHistograms,
        )

@dataclass
class CorrelationObservable2D(analysis_objects.Observable):
    type: str
    # In principle, we could create an enum here, but it's only one value, so it's not worth it.
    axis: str = "delta_eta_delta_phi"
    analysis_identifier: Optional[str] = None

    @property
    def name(self) -> str:
        # If the analysis identifier isn't specified, we preserved the field for it to be filled in later.
        analysis_identifier = self.analysis_identifier
        if self.analysis_identifier is None:
            analysis_identifier = "{analysis_identifier}"
        return f"jetH_{self.axis}_{analysis_identifier}_{self.type}"

_2d_correlations_histogram_information = {
    "correlation_hists_2d.raw": CorrelationObservable2D(hist = None, type = "raw"),
    "correlation_hists_2d.mixed_event": CorrelationObservable2D(hist = None, type = "mixed_event"),
    "correlation_hists_2d.signal": CorrelationObservable2D(hist = None, type = "signal"),
}

@dataclass
class CorrelationHistograms2D:
    raw: CorrelationObservable2D
    mixed_event: CorrelationObservable2D
    signal: CorrelationObservable2D

    def __iter__(self) -> Iterator[Tuple[str, Hist]]:
        # NOTE: dataclasses.asdict(...) is recursive, so it's far
        #       too aggressive for our purposes!
        for k, v in vars(self).items():
            yield k, v

@dataclass
class NumberOfTriggersObservable(analysis_objects.Observable):
    """ Simple container for the spectra used to determine the number of triggers.

    Note:
        We don't include an identifier for the name because we project the entire spectra
        and then select subsets of the range later. We will overwrite this object unnecessarily,
        but that should have minimal impact on the file size.
    """
    @property
    def name(self) -> str:
        return "jetH_number_of_triggers"

_number_of_triggers_histogram_information: Mapping[str, analysis_objects.Observable] = {
    "number_of_triggers_observable": NumberOfTriggersObservable(hist = None),
}

@dataclass
class CorrelationObservable1D(analysis_objects.Observable):
    type: analysis_objects.CorrelationType
    axis: analysis_objects.CorrelationAxis
    analysis_identifier: Optional[str] = None

    @property
    def name(self) -> str:
        # If the analysis identifier isn't specified, we preserved the field for it to be filled in later.
        analysis_identifier = self.analysis_identifier
        if self.analysis_identifier is None:
            analysis_identifier = "{analysis_identifier}"
        return f"jetH_{self.axis}_{analysis_identifier}_{self.type}"

@dataclass
class DeltaPhiObservable(CorrelationObservable1D):
    axis: analysis_objects.CorrelationAxis = analysis_objects.CorrelationAxis.delta_phi

@dataclass
class DeltaPhiSignalDominated(DeltaPhiObservable):
    type: analysis_objects.CorrelationType = analysis_objects.CorrelationType.signal_dominated

@dataclass
class DeltaPhiBackgroundDominated(DeltaPhiObservable):
    type: analysis_objects.CorrelationType = analysis_objects.CorrelationType.background_dominated

@dataclass
class DeltaEtaObservable(CorrelationObservable1D):
    axis: analysis_objects.CorrelationAxis = analysis_objects.CorrelationAxis.delta_eta

@dataclass
class DeltaEtaNearSide(DeltaEtaObservable):
    type: analysis_objects.CorrelationType = analysis_objects.CorrelationType.near_side

@dataclass
class DeltaEtaAwaySide(DeltaEtaObservable):
    type: analysis_objects.CorrelationType = analysis_objects.CorrelationType.away_side

_1d_correlations_histogram_information: Mapping[str, CorrelationObservable1D] = {
    "correlation_hists_delta_phi.signal_dominated": DeltaPhiSignalDominated(hist = None),
    "correlation_hists_delta_phi.background_dominated": DeltaPhiBackgroundDominated(hist = None),
    "correlation_hists_delta_phi_subtracted.signal_dominated": DeltaPhiSignalDominated(hist = None),
    "correlation_hists_delta_phi_subtracted.background_dominated": DeltaPhiBackgroundDominated(hist = None),
    "correlation_hists_delta_eta.near_side": DeltaEtaNearSide(hist = None),
    "correlation_hists_delta_eta.away_side": DeltaEtaAwaySide(hist = None),
    "correlation_hists_delta_eta_subtracted.near_side": DeltaEtaNearSide(hist = None),
    "correlation_hists_delta_eta_subtracted.away_side": DeltaEtaAwaySide(hist = None),
}

@dataclass
class CorrelationHistogramsDeltaPhi:
    signal_dominated: DeltaPhiSignalDominated
    background_dominated: DeltaPhiBackgroundDominated

    def __iter__(self) -> Iterator[Tuple[str, DeltaPhiObservable]]:
        # NOTE: dataclasses.asdict(...) is recursive, so it's far
        #       too aggressive for our purposes!
        for k, v in vars(self).items():
            yield k, v

@dataclass
class CorrelationHistogramsDeltaEta:
    near_side: DeltaEtaNearSide
    away_side: DeltaEtaAwaySide

    def __iter__(self) -> Iterator[Tuple[str, DeltaEtaObservable]]:
        # NOTE: dataclasses.asdict(...) is recursive, so it's far
        #       too aggressive for our purposes!
        for k, v in vars(self).items():
            yield k, v

@dataclass
class PedestalFitResult:
    value: float
    error: float

@dataclass
class DeltaEtaFitObjects:
    near_side: PedestalFitResult
    away_side: PedestalFitResult

    def __iter__(self) -> Iterator[Tuple[str, PedestalFitResult]]:
        for k, v in vars(self).items():
            yield k, v

@dataclass
class CorrelationYields:
    near_side: analysis_objects.ExtractedObservable
    away_side: analysis_objects.ExtractedObservable

    def __iter__(self) -> Iterator[Tuple[str, analysis_objects.ExtractedObservable]]:
        for k, v in vars(self).items():
            yield k, v

@dataclass
class CorrelationWidths:
    near_side: analysis_objects.ExtractedObservable
    away_side: analysis_objects.ExtractedObservable

    def __iter__(self) -> Iterator[Tuple[str, analysis_objects.ExtractedObservable]]:
        for k, v in vars(self).items():
            yield k, v

class Correlations(analysis_objects.JetHReactionPlane):
    """ Main correlations analysis object.

    Args:
        jet_pt_bin: Jet pt bin.
        track_pt_bin: Track pt bin.
    Attributes:
        jet_pt: Jet pt bin.
        track_pt: Track pt bin.
        ...
    """
    def __init__(self, jet_pt_bin: analysis_objects.JetPtBin, track_pt_bin: analysis_objects.TrackPtBin, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Basic information
        # Analysis parameters
        self.jet_pt = jet_pt_bin
        self.track_pt = track_pt_bin
        # Identifier information
        self.jet_pt_identifier = "jetPtBiased" if self.config["constituent_cut_biased_jets"] else "jetPt"
        self.jet_pt_identifier += f"_{self.jet_pt.min}_{self.jet_pt.max}"
        self.track_pt_identifier = f"trackPt_{self.track_pt.min}_{self.track_pt.max}"
        self.identifier = f"{self.jet_pt_identifier}_{self.track_pt_identifier}"

        # Pt hard bins are optional.
        self.pt_hard_bin = kwargs.get("pt_hard_bin", None)
        if self.pt_hard_bin:
            self.train_number = self.pt_hard_bin.train_number
            self.input_filename = self.input_filename.format(pt_hard_bin_train_number = self.train_number)
            self.output_prefix = self.output_prefix.format(pt_hard_bin_train_number = self.train_number)
        # Validate output filename
        if not self.output_filename.endswith(".root"):
            self.output_filename += ".root"

        self.input_hists: Dict[str, Any] = {}
        # For convenience since it is frequently accessed.
        self.processing_options = self.task_config["processingOptions"]
        # Status information
        self.ran_projections: bool = False
        self.ran_fitting: bool = False
        self.ran_post_fit_processing: bool = False

        # Relevant histograms
        # We need a field use with replace to successfully copy the dataclass. We just want a clean copy,
        # (and apparently using replace is strongly preferred for a dataclass compared to copying)
        # so we replace the hist (which is already None) with None and we get a copy of the dataclass.
        self.number_of_triggers_observable: analysis_objects.Observable = dataclasses.replace(
            _number_of_triggers_histogram_information["number_of_triggers_observable"], hist = None,
        )
        # Apparently using dataclass replace to copy and modify a dataclass is preferred to
        # copying the class and changing a value. So we use the replace function.
        self.correlation_hists_2d: CorrelationHistograms2D = CorrelationHistograms2D(
            raw = dataclasses.replace(
                _2d_correlations_histogram_information["correlation_hists_2d.raw"],
                analysis_identifier = self.identifier
            ),
            mixed_event = dataclasses.replace(
                _2d_correlations_histogram_information["correlation_hists_2d.mixed_event"],
                analysis_identifier = self.identifier
            ),
            signal = dataclasses.replace(
                _2d_correlations_histogram_information["correlation_hists_2d.signal"],
                analysis_identifier = self.identifier
            ),
        )
        self.correlation_hists_delta_phi: CorrelationHistogramsDeltaPhi = CorrelationHistogramsDeltaPhi(
            signal_dominated = dataclasses.replace(
                cast(
                    DeltaPhiSignalDominated,
                    _1d_correlations_histogram_information["correlation_hists_delta_phi.signal_dominated"]
                ),
                analysis_identifier = self.identifier,
            ),
            background_dominated = dataclasses.replace(
                cast(
                    DeltaPhiBackgroundDominated,
                    _1d_correlations_histogram_information["correlation_hists_delta_phi.background_dominated"]
                ),
                analysis_identifier = self.identifier,
            ),
        )
        self.correlation_hists_delta_eta: CorrelationHistogramsDeltaEta = CorrelationHistogramsDeltaEta(
            near_side = dataclasses.replace(
                cast(DeltaEtaNearSide, _1d_correlations_histogram_information["correlation_hists_delta_eta.near_side"]),
                analysis_identifier = self.identifier,
            ),
            away_side = dataclasses.replace(
                cast(DeltaEtaAwaySide, _1d_correlations_histogram_information["correlation_hists_delta_eta.away_side"]),
                analysis_identifier = self.identifier,
            ),
        )
        self.correlation_hists_delta_phi_subtracted: CorrelationHistogramsDeltaPhi = CorrelationHistogramsDeltaPhi(
            signal_dominated = dataclasses.replace(
                cast(
                    DeltaPhiSignalDominated,
                    _1d_correlations_histogram_information["correlation_hists_delta_phi_subtracted.signal_dominated"]
                ),
                analysis_identifier = self.identifier,
            ),
            background_dominated = dataclasses.replace(
                cast(
                    DeltaPhiBackgroundDominated,
                    _1d_correlations_histogram_information["correlation_hists_delta_phi_subtracted.background_dominated"]
                ),
                analysis_identifier = self.identifier,
            ),
        )
        self.correlation_hists_delta_eta_subtracted: CorrelationHistogramsDeltaEta = CorrelationHistogramsDeltaEta(
            near_side = dataclasses.replace(
                cast(
                    DeltaEtaNearSide,
                    _1d_correlations_histogram_information["correlation_hists_delta_eta_subtracted.near_side"]
                ),
                analysis_identifier = self.identifier,
            ),
            away_side = dataclasses.replace(
                cast(
                    DeltaEtaAwaySide,
                    _1d_correlations_histogram_information["correlation_hists_delta_eta_subtracted.away_side"]
                ),
                analysis_identifier = self.identifier,
            ),
        )
        # Yields
        self.yields_delta_phi: CorrelationYields = CorrelationYields(
            near_side = analysis_objects.ExtractedObservable(-1, -1),
            away_side = analysis_objects.ExtractedObservable(-1, -1),
        )
        self.yields_delta_eta: CorrelationYields = CorrelationYields(
            near_side = analysis_objects.ExtractedObservable(-1, -1),
            away_side = analysis_objects.ExtractedObservable(-1, -1),
        )
        # Widths
        self.widths_delta_phi: CorrelationWidths = CorrelationWidths(
            near_side = analysis_objects.ExtractedObservable(-1, -1),
            away_side = analysis_objects.ExtractedObservable(-1, -1),
        )
        self.widths_delta_eta: CorrelationWidths = CorrelationWidths(
            near_side = analysis_objects.ExtractedObservable(-1, -1),
            away_side = analysis_objects.ExtractedObservable(-1, -1),
        )

        # Fit object
        self.fit_object: rpf.fit.FitComponent
        self.fit_objects_delta_eta: DeltaEtaFitObjects = DeltaEtaFitObjects(
            near_side = PedestalFitResult(value = -1, error = -1),
            away_side = PedestalFitResult(value = -1, error = -1),
        )

        # Other relevant analysis information
        self.number_of_triggers: int = 0

        # Projectors
        self.sparse_projectors: List[JetHCorrelationSparseProjector] = []
        self.correlation_projectors: List[JetHCorrelationProjector] = []

        # Useful information
        # These values are only half the range (ie only the positive values).
        self.signal_dominated_eta_region = analysis_objects.AnalysisBin(
            params.SelectedRange(
                *self.task_config["deltaEtaRanges"]["signalDominated"]
            )
        )
        self.background_dominated_eta_region = analysis_objects.AnalysisBin(
            params.SelectedRange(
                *self.task_config["deltaEtaRanges"]["backgroundDominated"]
            )
        )
        self.near_side_phi_region = analysis_objects.AnalysisBin(
            params.SelectedRange(
                *self.task_config["deltaPhiRanges"]["nearSide"]
            )
        )
        # Interpret these values as defined relative to pi.
        away_side_values = self.task_config["deltaPhiRanges"]["awaySide"]
        away_side_values = [np.pi + val for val in away_side_values]
        self.away_side_phi_region = analysis_objects.AnalysisBin(
            params.SelectedRange(
                *away_side_values
            )
        )

    def __iter__(self) -> Iterator[analysis_objects.Observable]:
        """ Iterate over the histograms in the correlations analysis object.

        Returns:
            The observable object, which contains the histogram.
        """
        all_hists_info: Mapping[str, analysis_objects.Observable] = {
            **_2d_correlations_histogram_information,
            **_number_of_triggers_histogram_information,
            **_1d_correlations_histogram_information,
        }
        for attribute_name, observable in all_hists_info.items():
            yield observable

    def _write_2d_correlations(self) -> None:
        """ Write 2D correlations to output file. """
        self._write_hists_to_root_file(hists = self.correlation_hists_2d)

    def _write_number_of_triggers_hist(self) -> None:
        """ Write trigger jet spectra to file. """
        # This dict construction is a hack, but it's convenient since it mirrors the structure of the other objects.
        self._write_hists_to_root_file(hists = {"ignore_key": self.number_of_triggers_observable}.items())

    def _write_1d_correlations(self):
        """ Write 1D correlations to file. """
        logger.debug("Writing 1D delta phi correlations")
        self._write_hists_to_root_file(hists = self.correlation_hists_delta_phi)
        logger.debug("Writing 1D delta eta correlations")
        self._write_hists_to_root_file(hists = self.correlation_hists_delta_eta)

    def _write_hists_to_root_file(self, hists: Iterable[Tuple[str, analysis_objects.Observable]],
                                  mode: str = "UPDATE") -> None:
        """ Write the provided histograms to a ROOT file. """
        filename = os.path.join(self.output_prefix, self.output_filename)
        directory_name = os.path.dirname(filename)
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)

        logger.info(f"Saving correlations to {filename}")
        # Then actually iterate through and save the hists.
        with histogram.RootOpen(filename = filename, mode = mode):
            for _, observable in hists:
                hist = observable.hist
                # Only write the histogram if it's valid. It's possible that it's still ``None``.
                if hist:
                    logger.debug(f"Writing hist {hist} with name {observable.name}")
                    hist.Write(observable.name)

    def _init_2d_correlations_hists_from_root_file(self) -> None:
        """ Initialize 2D correlation hists. """
        self._init_hists_from_root_file(hists = self.correlation_hists_2d)

    def _init_number_of_triggers_hist_from_root_file(self) -> None:
        """ Write number of triggers hists. """
        # This dict construction is a hack, but it's convenient since it mirrors the structure of the other objects.
        self._init_hists_from_root_file(hists = {"ignore_key": self.number_of_triggers_observable}.items())

    def _init_1d_correlations_hists_from_root_file(self) -> None:
        """ Initialize 1D correlation hists. """
        self._init_hists_from_root_file(hists = self.correlation_hists_delta_phi)
        self._init_hists_from_root_file(hists = self.correlation_hists_delta_eta)

    def _init_hists_from_root_file(self, hists: Iterable[Tuple[str, analysis_objects.Observable]]) -> None:
        """ Initialize processed histograms from a ROOT file. """
        # We want to initialize from our saved hists - they will be at the output_prefix.
        filename = os.path.join(self.output_prefix, self.output_filename)
        with histogram.RootOpen(filename = filename, mode = "READ") as f:
            for _, observable in hists:
                logger.debug(f"Looking for hist {observable.name}")
                h = f.Get(observable.name)
                if not h:
                    h = None
                else:
                    # Detach it from the file so we can store it for later use.
                    h.SetDirectory(0)
                logger.debug(f"Initializing hist {h} to be stored in {observable}")
                observable.hist = h

    def _setup_sparse_projectors(self) -> None:
        """ Setup the THnSparse projectors.

        The created projectors are added to the ``sparse_projectors`` list.
        """
        # Helper which defines the full axis range
        full_axis_range = {
            "min_val": HistAxisRange.apply_func_to_find_bin(None, 1),
            "max_val": HistAxisRange.apply_func_to_find_bin(ROOT.TAxis.GetNbins),
        }

        # Define common axes
        # NOTE: The axis will be changed a copy below when necessary (ie for the trigger, since the axes are different).
        # Centrality axis
        centrality_cut_axis = HistAxisRange(
            axis_type = JetHCorrelationSparse.centrality,
            axis_range_name = "centrality",
            min_val = HistAxisRange.apply_func_to_find_bin(
                ROOT.TAxis.FindBin, self.event_activity.value_range.min + epsilon
            ),
            max_val = HistAxisRange.apply_func_to_find_bin(
                ROOT.TAxis.FindBin, self.event_activity.value_range.max - epsilon
            ),
        )
        # Event plane selection
        if self.reaction_plane_orientation == params.ReactionPlaneOrientation.inclusive:
            reaction_plane_axis_range = full_axis_range
            logger.debug("Using full EP angle range")
        else:
            reaction_plane_axis_range = {
                "min_val": projectors.HistAxisRange.apply_func_to_find_bin(
                    None,
                    self.reaction_plane_orientation.value.bin
                ),
                "max_val": projectors.HistAxisRange.apply_func_to_find_bin(
                    None,
                    self.reaction_plane_orientation.value.bin
                ),
            }
            logger.debug(f"Using selected EP angle range {self.reaction_plane_orientation.name}")
        reaction_plane_orientation_cut_axis = HistAxisRange(
            axis_type = JetHCorrelationSparse.reaction_plane_orientation,
            axis_range_name = "reaction_plane",
            **reaction_plane_axis_range,
        )
        # delta_phi full axis
        delta_phi_axis = HistAxisRange(
            axis_type = JetHCorrelationSparse.delta_phi,
            axis_range_name = "delta_phi",
            **full_axis_range,
        )
        # delta_eta full axis
        delta_eta_axis = HistAxisRange(
            axis_type = JetHCorrelationSparse.delta_eta,
            axis_range_name = "delta_eta",
            **full_axis_range,
        )
        # Jet pt axis
        jet_pt_axis = HistAxisRange(
            axis_type = JetHCorrelationSparse.jet_pt,
            axis_range_name = f"jet_pt{self.jet_pt.min}-{self.jet_pt.max}",
            min_val = HistAxisRange.apply_func_to_find_bin(
                ROOT.TAxis.FindBin, self.jet_pt.range.min + epsilon
            ),
            max_val = HistAxisRange.apply_func_to_find_bin(
                ROOT.TAxis.FindBin, self.jet_pt.range.max - epsilon
            )
        )
        # Track pt axis
        track_pt_axis = HistAxisRange(
            axis_type = JetHCorrelationSparse.track_pt,
            axis_range_name = f"track_pt{self.track_pt.min}-{self.track_pt.max}",
            min_val = HistAxisRange.apply_func_to_find_bin(
                ROOT.TAxis.FindBin, self.track_pt.range.min + epsilon
            ),
            max_val = HistAxisRange.apply_func_to_find_bin(
                ROOT.TAxis.FindBin, self.track_pt.range.max - epsilon
            )
        )

        ###########################
        # Trigger projector
        #
        # Note that it has no jet pt or trigger pt dependence.
        # We will select jet pt ranges later when determining n_trig
        ###########################
        projection_information: Dict[str, Any] = {}
        trigger_projector = JetHCorrelationSparseProjector(
            observable_to_project_from = self.input_hists["fhnTrigger"],
            output_observable = self.number_of_triggers_observable,
            output_attribute_name = "hist",
            projection_name_format = self.number_of_triggers_observable.name,
            projection_information = projection_information
        )
        # Take advantage of existing centrality and event plane object, but need to copy and modify the axis type
        if self.collision_system != params.CollisionSystem.pp:
            trigger_centrality_cut_axis = copy.deepcopy(centrality_cut_axis)
            trigger_centrality_cut_axis.axis_type = JetHTriggerSparse.centrality
            trigger_projector.additional_axis_cuts.append(trigger_centrality_cut_axis)
        trigger_reaction_plane_orientation_cut_axis = copy.deepcopy(reaction_plane_orientation_cut_axis)
        trigger_reaction_plane_orientation_cut_axis.axis_type = JetHTriggerSparse.reaction_plane_orientation
        trigger_projector.additional_axis_cuts.append(trigger_reaction_plane_orientation_cut_axis)
        # No projection dependent cut axes
        trigger_projector.projection_dependent_cut_axes.append([])
        # Projection axis
        trigger_projector.projection_axes.append(
            HistAxisRange(
                axis_type = JetHTriggerSparse.jet_pt,
                axis_range_name = "jet_pt",
                **full_axis_range
            )
        )
        self.sparse_projectors.append(trigger_projector)

        ###########################
        # Raw signal projector
        ###########################
        projection_information = {}
        raw_signal_projector = JetHCorrelationSparseProjector(
            observable_to_project_from = self.input_hists["fhnJH"],
            output_observable = self.correlation_hists_2d.raw,
            output_attribute_name = "hist",
            projection_name_format = self.correlation_hists_2d.raw.name,
            projection_information = projection_information,
        )
        if self.collision_system != params.CollisionSystem.pp:
            raw_signal_projector.additional_axis_cuts.append(centrality_cut_axis)
        raw_signal_projector.additional_axis_cuts.append(reaction_plane_orientation_cut_axis)
        raw_signal_projector.additional_axis_cuts.append(jet_pt_axis)
        raw_signal_projector.additional_axis_cuts.append(track_pt_axis)
        raw_signal_projector.projection_dependent_cut_axes.append([])
        # Projection Axes
        raw_signal_projector.projection_axes.append(delta_phi_axis)
        raw_signal_projector.projection_axes.append(delta_eta_axis)
        self.sparse_projectors.append(raw_signal_projector)

        ###########################
        # Mixed Event projector
        ###########################
        projection_information = {}
        mixed_event_projector = JetHCorrelationSparseProjector(
            observable_to_project_from = self.input_hists["fhnMixedEvents"],
            output_observable = self.correlation_hists_2d.mixed_event,
            output_attribute_name = "hist",
            projection_name_format = self.correlation_hists_2d.mixed_event.name,
            projection_information = projection_information,
        )
        if self.collision_system != params.CollisionSystem.pp:
            mixed_event_projector.additional_axis_cuts.append(centrality_cut_axis)
        # According to Joel's AN (Fig 11), there is no dependence on EP orientation for mixed events.
        # So we only perform this projection if selected in order to improve our mixed event statistics.
        if self.task_config["mixed_events_with_EP_dependence"]:
            mixed_event_projector.additional_axis_cuts.append(reaction_plane_orientation_cut_axis)
        mixed_event_projector.additional_axis_cuts.append(jet_pt_axis)
        # At higher pt, tracks are straight enough that the detector acceptance doesn't change much
        # with increasing pt. According to Joel's AN (fig 13), we can just merge them together above
        # 2 GeV. The figure shows that the ME is roughly flat (note that there is a constant offset,
        # so it must be scaled somewhat differently).
        if self.task_config["use_broader_high_pt_mixed_events"] and self.track_pt.min >= 2.0:
            # Select from 2.0 to the maximum (10.0)
            mixed_event_projector.additional_axis_cuts.append(
                HistAxisRange(
                    axis_type = JetHCorrelationSparse.track_pt,
                    axis_range_name = f"track_pt2.0-10.0",
                    min_val = HistAxisRange.apply_func_to_find_bin(
                        ROOT.TAxis.FindBin, 2.0 + epsilon
                    ),
                    max_val = HistAxisRange.apply_func_to_find_bin(
                        ROOT.TAxis.GetNbins
                    )
                )
            )
        else:
            mixed_event_projector.additional_axis_cuts.append(track_pt_axis)
        mixed_event_projector.projection_dependent_cut_axes.append([])
        # Projection Axes
        mixed_event_projector.projection_axes.append(delta_phi_axis)
        mixed_event_projector.projection_axes.append(delta_eta_axis)
        self.sparse_projectors.append(mixed_event_projector)

    def _setup_projectors(self):
        """ Setup the projectors for the analysis. """
        # NOTE: It's best to define the projector right before utilizing it. Here, this runs as the last
        #       step of the setup, and then these projectors are executed immediately.
        #       This is the best practice because we can only define the projectors for single objects once
        #       the histogram that it will project from exists. If it doesn't yet exist, the projector will
        #       fail because it stores the value (ie the hist) at the time of the projector definition.
        self._setup_sparse_projectors()

    def _determine_number_of_triggers(self) -> int:
        """ Determine the number of triggers for the specific analysis parameters. """
        return correlations_helpers.determine_number_of_triggers(
            hist = self.number_of_triggers_observable.hist,
            jet_pt = self.jet_pt,
        )

    def setup(self, input_hists):
        """ Setup the correlations object. """
        # Setup the input hists and projectors
        super().setup(input_hists = input_hists)

    def _post_creation_processing_for_2d_correlation(self, hist: Hist, normalization_factor: float, title_label: str, rebin_factors: Tuple[int, int] = None) -> None:
        """ Perform post creation processing for 2D correlations. """
        correlations_helpers.post_projection_processing_for_2d_correlation(
            hist = hist, normalization_factor = normalization_factor, title_label = title_label,
            jet_pt = self.jet_pt, track_pt = self.track_pt, rebin_factors = rebin_factors,
        )

    def _compare_mixed_event_normalization_options(self, mixed_event: Hist) -> None:
        """ Compare mixed event normalization options. """
        eta_limits = self.task_config["mixedEventNormalizationOptions"].get("etaLimits", [-0.3, 0.3])

        # Create the comparison
        (
            # Basic data
            peak_finding_hist, lin_space, peak_finding_hist_array, lin_space_rebin, peak_finding_hist_array_rebin,
            # CWT
            peak_locations, peak_locations_rebin,
            # Moving Average
            max_moving_avg, max_moving_avg_rebin,
            # Smoothed gaussian
            lin_space_resample, smoothed_array, max_smoothed_moving_avg,
            # Linear fits
            max_linear_fit_1d, max_linear_fit_1d_rebin, max_linear_fit_2d, max_linear_fit_2d_rebin,
        ) = correlations_helpers.compare_mixed_event_normalization_options(
            mixed_event = mixed_event, eta_limits = eta_limits,
        )

        # Plot the comparison
        plot_correlations.mixed_event_normalization(
            self.output_info,
            # For labeling purposes
            output_name = f"mixed_event_normalization_{self.identifier}", eta_limits = eta_limits,
            jet_pt_title = labels.jet_pt_range_string(self.jet_pt),
            track_pt_title = labels.track_pt_range_string(self.track_pt),
            # Basic data
            lin_space = lin_space, peak_finding_hist_array = peak_finding_hist_array,
            lin_space_rebin = lin_space_rebin, peak_finding_hist_array_rebin = peak_finding_hist_array_rebin,
            # CWT
            peak_locations = peak_locations, peak_locations_rebin = peak_locations_rebin,
            # Moving Average
            max_moving_avg = max_moving_avg, max_moving_avg_rebin = max_moving_avg_rebin,
            # Smoothed gaussian
            lin_space_resample = lin_space_resample,
            smoothed_array = smoothed_array, max_smoothed_moving_avg = max_smoothed_moving_avg,
            # Linear fits
            max_linear_fit_1d = max_linear_fit_1d, max_linear_fit_1d_rebin = max_linear_fit_1d_rebin,
            max_linear_fit_2d = max_linear_fit_2d, max_linear_fit_2d_rebin = max_linear_fit_2d_rebin,
        )

    def _measure_mixed_event_normalization(self, mixed_event: Hist, delta_phi_rebin_factor: int = 1) -> float:
        """ Measure the mixed event normalization. """
        # See the note on the selecting the eta_limits in `correlations_helpers.measure_mixed_event_normalization(...)`
        eta_limits = self.task_config["mixedEventNormalizationOptions"].get("etaLimits", [-0.3, 0.3])
        return correlations_helpers.measure_mixed_event_normalization(
            mixed_event = mixed_event,
            eta_limits = eta_limits,
            delta_phi_rebin_factor = delta_phi_rebin_factor,
        )

    def _create_2d_raw_and_mixed_correlations(self) -> None:
        """ Generate raw and mixed event 2D correlations. """
        # Project the histograms
        # Includes the trigger, raw signal 2D, and mixed event 2D hists
        for projector in self.sparse_projectors:
            projector.project()

        # Determine number of triggers for the analysis.
        self.number_of_triggers = self._determine_number_of_triggers()
        rebin_factors = self.task_config.get("2d_rebin_factors", None)

        # Raw signal hist post processing.
        self._post_creation_processing_for_2d_correlation(
            hist = self.correlation_hists_2d.raw.hist,
            normalization_factor = self.number_of_triggers,
            title_label = "Raw signal",
            rebin_factors = rebin_factors,
        )

        # Compare mixed event normalization options
        # We must do this before scaling the mixed event (otherwise we will get the wrong scaling values.)
        if self.task_config["mixedEventNormalizationOptions"].get("compareOptions", False):
            self._compare_mixed_event_normalization_options(
                mixed_event = self.correlation_hists_2d.mixed_event.hist
            )

        # Normalize and post process the mixed event observable
        mixed_event_normalization_factor = self._measure_mixed_event_normalization(
            mixed_event = self.correlation_hists_2d.mixed_event.hist,
            delta_phi_rebin_factor = rebin_factors[0] if rebin_factors else 1,
        )
        self._post_creation_processing_for_2d_correlation(
            hist = self.correlation_hists_2d.mixed_event.hist,
            normalization_factor = mixed_event_normalization_factor,
            title_label = "Mixed event",
            rebin_factors = rebin_factors,
        )

    def _create_2d_signal_correlation(self) -> None:
        """ Create 2D signal correlation for raw and mixed correlations.

        This method is intentionally decoupled for creating the raw and mixed event hists so that the
        THnSparse can be swapped out when desired.
        """
        # The signal correlation is the raw signal divided by the mixed events
        self.correlation_hists_2d.signal.hist = self.correlation_hists_2d.raw.hist.Clone(
            self.correlation_hists_2d.signal.name
        )
        self.correlation_hists_2d.signal.hist.Divide(self.correlation_hists_2d.mixed_event.hist)

        self._post_creation_processing_for_2d_correlation(
            hist = self.correlation_hists_2d.signal.hist,
            normalization_factor = 1.0,
            title_label = "Correlation",
        )

    def _run_2d_projections(self) -> None:
        """ Run the correlations 2D projections. """
        # Only need to check if file exists for this if statement because we cannot get past there
        # without somehow having some hists
        file_exists = os.path.isfile(os.path.join(self.output_prefix, self.output_filename))

        # NOTE: Only normalize hists when plotting, and then only do so to a copy!
        #       The exceptions are the 2D correlations, which are normalized by n_trig for the raw correlation
        #       and the maximum efficiency for the mixed events. They are excepted because we don't have a
        #       purpose for such unnormalized hists.
        if self.processing_options["generate2DCorrelations"] or not file_exists:
            # Create the correlations by utilizing the projectors
            logger.info("Projecting 2D correlations")
            self._create_2d_raw_and_mixed_correlations()
            # Create the signal correlation
            self._create_2d_signal_correlation()

            # Write the correlations
            self._write_2d_correlations()
            # Write triggers
            self._write_number_of_triggers_hist()

            # Ensure we execute the next step
            self.processing_options["generate1DCorrelations"] = True
        else:
            # Initialize the 2D correlations from the file
            logger.info("Loading 2D correlations and trigger jet spectra from file")
            self._init_2d_correlations_hists_from_root_file()
            self._init_number_of_triggers_hist_from_root_file()

        # Plotting
        if self.processing_options["plot2DCorrelations"]:
            logger.info("Plotting 2D correlations")
            plot_correlations.plot_2d_correlations(self)
            logger.info("Plotting RPF example region")
            if self.processing_options["plotRPFHighlights"]:
                plot_correlations.plot_RPF_fit_regions(
                    self,
                    filename = f"highlight_RPF_regions_{self.identifier}"
                )

    def _setup_1d_projectors(self) -> None:
        """ Setup 2D -> 1D correlation projectors.

        The created projectors are added to the ``sparse_projectors`` list.
        """
        # Helper which defines the full axis range
        full_axis_range = {
            "min_val": HistAxisRange.apply_func_to_find_bin(None, 1),
            "max_val": HistAxisRange.apply_func_to_find_bin(ROOT.TAxis.GetNbins)
        }

        ###########################
        # delta_phi signal
        ###########################
        projection_information: Dict[str, Any] = {}
        delta_phi_signal_projector = JetHCorrelationProjector(
            observable_to_project_from = self.correlation_hists_2d.signal,
            output_observable = self.correlation_hists_delta_phi.signal_dominated,
            output_attribute_name = "hist",
            projection_name_format = self.correlation_hists_delta_phi.signal_dominated.name,
            projection_information = projection_information,
        )
        # Select signal dominated region in eta
        # Could be a single range, but this is conceptually clearer when compared to the background
        # dominated region. Need to do this as projection dependent cuts because it is selecting different
        # ranges on the same axis
        delta_phi_signal_projector.projection_dependent_cut_axes.append([
            HistAxisRange(
                axis_type = analysis_objects.CorrelationAxis.delta_eta,
                axis_range_name = "negative_eta_signal_dominated",
                min_val = HistAxisRange.apply_func_to_find_bin(
                    ROOT.TAxis.FindBin, -1 * self.signal_dominated_eta_region.max + epsilon,
                ),
                max_val = HistAxisRange.apply_func_to_find_bin(
                    ROOT.TAxis.FindBin, -1 * self.signal_dominated_eta_region.min - epsilon,
                ),
            )
        ])
        delta_phi_signal_projector.projection_dependent_cut_axes.append([
            HistAxisRange(
                axis_type = analysis_objects.CorrelationAxis.delta_eta,
                axis_range_name = "Positive_eta_signal_dominated",
                min_val = HistAxisRange.apply_func_to_find_bin(
                    ROOT.TAxis.FindBin, self.signal_dominated_eta_region.min + epsilon,
                ),
                max_val = HistAxisRange.apply_func_to_find_bin(
                    ROOT.TAxis.FindBin, self.signal_dominated_eta_region.max - epsilon,
                ),
            )
        ])
        delta_phi_signal_projector.projection_axes.append(
            HistAxisRange(
                axis_type = analysis_objects.CorrelationAxis.delta_phi,
                axis_range_name = "delta_phi",
                **full_axis_range
            )
        )
        self.correlation_projectors.append(delta_phi_signal_projector)

        ###########################
        # delta_phi Background dominated
        ###########################
        projection_information = {}
        delta_phi_background_projector = JetHCorrelationProjector(
            observable_to_project_from = self.correlation_hists_2d.signal,
            output_observable = self.correlation_hists_delta_phi.background_dominated,
            output_attribute_name = "hist",
            projection_name_format = self.correlation_hists_delta_phi.background_dominated.name,
            projection_information = projection_information,
        )
        # Select background dominated region in eta
        # Redundant to find the index, but it helps check that it is actually in the list!
        # Need to do this as projection dependent cuts because it is selecting different ranges
        # on the same axis
        delta_phi_background_projector.projection_dependent_cut_axes.append([
            HistAxisRange(
                axis_type = analysis_objects.CorrelationAxis.delta_eta,
                axis_range_name = "negative_eta_background_dominated",
                min_val = HistAxisRange.apply_func_to_find_bin(
                    ROOT.TAxis.FindBin, -1 * self.background_dominated_eta_region.max + epsilon,
                ),
                max_val = HistAxisRange.apply_func_to_find_bin(
                    ROOT.TAxis.FindBin, -1 * self.background_dominated_eta_region.min - epsilon,
                ),
            )
        ])
        delta_phi_background_projector.projection_dependent_cut_axes.append([
            HistAxisRange(
                axis_type = analysis_objects.CorrelationAxis.delta_eta,
                axis_range_name = "positive_eta_background_dominated",
                min_val = HistAxisRange.apply_func_to_find_bin(
                    ROOT.TAxis.FindBin, self.background_dominated_eta_region.min + epsilon,
                ),
                max_val = HistAxisRange.apply_func_to_find_bin(
                    ROOT.TAxis.FindBin, self.background_dominated_eta_region.max - epsilon,
                ),
            )
        ])
        delta_phi_background_projector.projection_axes.append(
            HistAxisRange(
                axis_type = analysis_objects.CorrelationAxis.delta_phi,
                axis_range_name = "delta_phi",
                **full_axis_range,
            )
        )
        self.correlation_projectors.append(delta_phi_background_projector)

        ###########################
        # delta_eta NS
        ###########################
        projection_information = {}
        delta_eta_ns_projector = JetHCorrelationProjector(
            observable_to_project_from = self.correlation_hists_2d.signal,
            output_observable = self.correlation_hists_delta_eta.near_side,
            output_attribute_name = "hist",
            projection_name_format = self.correlation_hists_delta_eta.near_side.name,
            projection_information = projection_information,
        )
        # Select near side in delta phi
        delta_eta_ns_projector.additional_axis_cuts.append(
            HistAxisRange(
                axis_type = analysis_objects.CorrelationAxis.delta_phi,
                axis_range_name = "deltaPhiNearSide",
                min_val = HistAxisRange.apply_func_to_find_bin(
                    ROOT.TAxis.FindBin, self.near_side_phi_region.min + epsilon
                ),
                max_val = HistAxisRange.apply_func_to_find_bin(
                    ROOT.TAxis.FindBin, self.near_side_phi_region.max - epsilon
                ),
            )
        )
        # No projection dependent cut axes
        delta_eta_ns_projector.projection_dependent_cut_axes.append([])
        delta_eta_ns_projector.projection_axes.append(
            HistAxisRange(
                axis_type = analysis_objects.CorrelationAxis.delta_eta,
                axis_range_name = "delta_eta",
                **full_axis_range
            )
        )
        self.correlation_projectors.append(delta_eta_ns_projector)

        ###########################
        # delta_eta AS
        ###########################
        projection_information = {}
        delta_eta_as_projector = JetHCorrelationProjector(
            observable_to_project_from = self.correlation_hists_2d.signal,
            output_observable = self.correlation_hists_delta_eta.away_side,
            output_attribute_name = "hist",
            projection_name_format = self.correlation_hists_delta_eta.away_side.name,
            projection_information = projection_information,
        )
        # Select away side in delta phi
        delta_eta_as_projector.additional_axis_cuts.append(
            HistAxisRange(
                axis_type = analysis_objects.CorrelationAxis.delta_phi,
                axis_range_name = "deltaPhiAwaySide",
                min_val = HistAxisRange.apply_func_to_find_bin(
                    ROOT.TAxis.FindBin, self.away_side_phi_region.min + epsilon
                ),
                max_val = HistAxisRange.apply_func_to_find_bin(
                    ROOT.TAxis.FindBin, self.away_side_phi_region.max - epsilon
                ),
            )
        )
        # No projection dependent cut axes
        delta_eta_as_projector.projection_dependent_cut_axes.append([])
        delta_eta_as_projector.projection_axes.append(
            HistAxisRange(
                axis_type = analysis_objects.CorrelationAxis.delta_eta,
                axis_range_name = "delta_eta",
                **full_axis_range
            )
        )
        self.correlation_projectors.append(delta_eta_as_projector)

    def _create_1d_correlations(self) -> None:
        # Project the histograms
        # Includes the delta phi signal dominated, delta phi background dominated, and delta eta near side
        for projector in self.correlation_projectors:
            projector.project()

        # Post process and scale
        for correlations in [self.correlation_hists_delta_phi, self.correlation_hists_delta_eta]:
            # Help out mypy...
            assert isinstance(correlations, (CorrelationHistogramsDeltaPhi, CorrelationHistogramsDeltaEta))
            for name, observable in correlations:
                logger.debug(f"name: {name}, observable: {observable}")
                logger.info(f"Post projection processing of 1D correlation: {observable.axis}, {observable.type}")

                # Determine normalization factor
                # However, it is then important that we report the ranges in which we measure!
                # NOTE: We calculate the values very explicitly to try to ensure that any changes in
                #       values will be noticed quickly.
                if observable.axis == analysis_objects.CorrelationAxis.delta_phi:
                    ranges = {
                        analysis_objects.CorrelationType.signal_dominated: self.signal_dominated_eta_region,
                        analysis_objects.CorrelationType.background_dominated: self.background_dominated_eta_region,
                    }
                    r = ranges[observable.type]
                    # Ranges are multiplied by 2 because the ranges are symmetric and the stored values
                    # only cover the positive range.
                    normalization_factor = (r.max - r.min) * 2.
                elif observable.axis == analysis_objects.CorrelationAxis.delta_eta:
                    ranges = {
                        analysis_objects.CorrelationType.near_side: self.near_side_phi_region,
                        analysis_objects.CorrelationType.away_side: self.away_side_phi_region,
                    }
                    r = ranges[observable.type]
                    normalization_factor = r.max - r.min
                else:
                    raise ValueError(f"Unrecognized observable axis: {observable.axis}")

                # Determine the rebin factor, which depends on the observable axis.
                rebin_factor = self.task_config.get(f"1d_rebin_factor_{observable.axis}", 1)

                # Post process and scale
                title_label = rf"${observable.axis.display_str()}$, {observable.type.display_str()}"
                correlations_helpers.post_creation_processing_for_1d_correlations(
                    hist = observable.hist,
                    normalization_factor = normalization_factor,
                    rebin_factor = rebin_factor,
                    title_label = title_label,
                    axis_label = observable.axis.display_str(),
                    jet_pt = self.jet_pt,
                    track_pt = self.track_pt,
                )

    def _post_1d_projection_scaling(self):
        """ Perform post-projection scaling to avoid needing to scale the fit functions later. """
        # Since the histograms are always referencing the same root object, the stored hists
        # will also be updated.
        for hists in [self.correlation_hists_delta_phi, self.correlation_hists_delta_eta]:
            for _, observable in hists:
                logger.debug(f"hist: {observable}")
                correlations_helpers.scale_by_bin_width(observable.hist)

    def _compare_to_other_hist(self,
                               our_hist: Hist, their_hist: Hist,
                               title: str, x_label: str, y_label: str,
                               output_name: str) -> None:
        # Convert for simplicity
        if not isinstance(our_hist, histogram.Histogram1D):
            our_hist = histogram.Histogram1D.from_existing_hist(our_hist)
        if not isinstance(their_hist, histogram.Histogram1D):
            their_hist = histogram.Histogram1D.from_existing_hist(their_hist)

        # Create a ratio plot
        # We want to take their hist and divide it by ours.
        ratio = their_hist / our_hist

        # Make the comparison.
        plot_correlations.comparison_1d(
            output_info = self.output_info,
            our_hist = our_hist,
            their_hist = their_hist,
            ratio = ratio,
            title = title,
            x_label = x_label,
            y_label = y_label,
            output_name = output_name,
        )

    def _compare_unsubtracted_1d_signal_correlation_to_joel(self) -> None:
        """ Compare Joel's unsubtracted delta phi signal region correlations to mine. """
        comparison_hists = correlations_helpers.get_joels_comparison_hists(
            track_pt = self.track_pt,
            path = self.task_config["joelsCorrelationsFilePath"]
        )
        # Define map by hand because it's out of our control.
        map_to_joels_hist_names = {
            params.ReactionPlaneOrientation.inclusive: "all",
            params.ReactionPlaneOrientation.in_plane: "in",
            params.ReactionPlaneOrientation.mid_plane: "mid",
            params.ReactionPlaneOrientation.out_of_plane: "out",
        }

        # Example hist name for all orientations: "allReconstructedSignalwithErrorsNOMnosub"
        joel_hist_name = map_to_joels_hist_names[self.reaction_plane_orientation]
        joel_hist_name += "ReconstructedSignalwithErrorsNOMnosub"

        self._compare_to_other_hist(
            our_hist = self.correlation_hists_delta_phi.signal_dominated.hist,
            their_hist = comparison_hists[joel_hist_name],
            title = f"Unsubtracted 1D: ${self.correlation_hists_delta_phi.signal_dominated.axis.display_str()}$,"
                    f" {self.reaction_plane_orientation.display_str()} event plane orient.,"
                    f" {labels.jet_pt_range_string(self.jet_pt)}, {labels.track_pt_range_string(self.track_pt)}",
            x_label = r"$\Delta\varphi$",
            y_label = r"$\mathrm{d}N/\mathrm{d}\varphi$",
            output_name = f"jetH_delta_phi_{self.identifier}_joel_comparison_unsub",
        )

    def _run_1d_projections(self) -> None:
        """ Run the 2D -> 1D projections. """
        if self.processing_options["generate1DCorrelations"]:
            # Setup the projectors here.
            logger.info("Setting up 1D correlations projectors.")
            self._setup_1d_projectors()

            # Project in 1D
            logger.info("Projecting 1D correlations")
            self._create_1d_correlations()

            # Perform post-projection scaling to avoid needing to scale the fit functions later
            logger.info("Performing post projection histogram scaling")
            self._post_1d_projection_scaling()

            # Write the properly scaled projections
            self._write_1d_correlations()

            # Ensure that the next step in the chain is run
            self.processing_options["fit1DCorrelations"] = True
        else:
            # Initialize the 1D correlations from the file
            logger.info("Loading 1D correlations from file")
            self._init_1d_correlations_hists_from_root_file()

        # Plot the correlations
        if self.processing_options["plot1DCorrelations"]:
            if self.collision_energy == params.CollisionEnergy.two_seven_six and self.event_activity == params.EventActivity.central:
                logger.info("Comparing unsubtracted correlations to Joel's.")
                self._compare_unsubtracted_1d_signal_correlation_to_joel()
            else:
                logger.info("Skipping comparison with Joel since we're not analyzing the right system.")
            logger.info("Plotting 1D correlations")
            plot_correlations.plot_1d_correlations(self, self.processing_options["plot1DCorrelationsWithROOT"])
            plot_correlations.delta_eta_unsubtracted(
                hists = self.correlation_hists_delta_eta,
                jet_pt = self.jet_pt, track_pt = self.track_pt,
                reaction_plane_orientation = self.reaction_plane_orientation,
                identifier = self.identifier,
                output_info = self.output_info,
            )

    def run_projections(self) -> None:
        """ Run all analysis steps through projectors. """
        self._run_2d_projections()
        self._run_1d_projections()

        # Store that we've completed this step.
        self.ran_projections = True

    def fit_delta_eta_correlations(self) -> None:
        """ Fit a pedestal to the background dominated region of the delta eta correlations. """
        attribute_names = ["near_side", "away_side"]
        for attribute_name in attribute_names:
            correlation = getattr(self.correlation_hists_delta_eta, attribute_name)
            constant, error = fitting.fit_pedestal_to_delta_eta_background_dominated_region(
                h = histogram.Histogram1D.from_existing_hist(correlation.hist),
                fit_range = self.background_dominated_eta_region.range,
            )

            # Store the result
            fit_result = PedestalFitResult(value = constant, error = error)
            setattr(self.fit_objects_delta_eta, attribute_name, fit_result)

    def subtract_background_fit_function_from_signal_dominated(self) -> None:
        """ Subtract the background function extract from a fit from the signal dominated hist.

        Args:
            None.
        Returns:
            None. The subtracted hist is stored.
        """
        # We want to subtract the signal dominated hist from the background function.
        # We want to do the same thing regardless of whether an object contributed to the signal
        # dominated or background dominated portion of the fit.
        signal_dominated = self.correlation_hists_delta_phi.signal_dominated
        signal_dominated_hist = histogram.Histogram1D.from_existing_hist(signal_dominated.hist)
        # Evaluate the hist and the fit at the same x locations.
        x = signal_dominated_hist.x
        fit_hist = histogram.Histogram1D(
            bin_edges = signal_dominated_hist.bin_edges,
            y = self.fit_object.evaluate_background(x),
            errors_squared = self.fit_object.calculate_background_function_errors(x) ** 2,
        )
        self.correlation_hists_delta_phi_subtracted.signal_dominated.hist = signal_dominated_hist - fit_hist

    def compare_subtracted_1d_signal_correlation_to_joel(self) -> None:
        """ Compare subtracted 1D signal correlation hists to Joel.

        Args:
            None.
        Returns:
            None. The comparison will be plotted.
        """
        comparison_hists = correlations_helpers.get_joels_comparison_hists(
            track_pt = self.track_pt,
            path = self.task_config["joelsCorrelationsFilePath"]
        )
        # Define map by hand because it's out of our control.
        map_to_joels_hist_names = {
            params.ReactionPlaneOrientation.inclusive: "all",
            params.ReactionPlaneOrientation.in_plane: "in",
            params.ReactionPlaneOrientation.mid_plane: "mid",
            params.ReactionPlaneOrientation.out_of_plane: "out",
        }

        # Example hist name for all orientations: "allReconstructedSignalwithErrorsNOMnosub"
        joel_hist_name = map_to_joels_hist_names[self.reaction_plane_orientation]
        joel_hist_name += "ReconstructedSignalwithErrorsNOM"

        self._compare_to_other_hist(
            our_hist = self.correlation_hists_delta_phi_subtracted.signal_dominated.hist,
            their_hist = comparison_hists[joel_hist_name],
            title = f"Subtracted 1D: ${self.correlation_hists_delta_phi.signal_dominated.axis.display_str()}$,"
                    f" {self.reaction_plane_orientation.display_str()} event plane orient.,"
                    f" {labels.jet_pt_range_string(self.jet_pt)}, {labels.track_pt_range_string(self.track_pt)}",
            x_label = r"$\Delta\varphi$",
            y_label = r"$\mathrm{d}N/\mathrm{d}\varphi$",
            output_name = f"jetH_delta_phi_{self.identifier}_joel_comparison_sub",
        )

    def subtract_delta_eta_correlations(self) -> None:
        """ Subtract the pedestal from the delta eta correlations.

        For now, we subtract the near-side fit from the away-side because it's not clear what
        should be done for the away side given the eta swing.

        Args:
            None.
        Returns:
            None. The subtracted hist is stored.
        """
        attribute_names = ["near_side", "away_side"]
        # We will use the near-side pedestal for _both_ the near-side and away-side
        fit_object = self.fit_objects_delta_eta.near_side
        for attribute_name in attribute_names:
            # Retrieve the hist
            correlation = getattr(self.correlation_hists_delta_eta, attribute_name)
            correlation_hist = histogram.Histogram1D.from_existing_hist(correlation.hist)

            # Determine the pedestal representing the background.
            background_hist = histogram.Histogram1D(
                bin_edges = correlation_hist.bin_edges,
                y = fit_object.value * np.ones(len(correlation_hist.x)),
                errors_squared = (fit_object.error * np.ones(len(correlation_hist.x))) ** 2,
            )

            # Subtract and store the output
            subtracted_hist = correlation_hist - background_hist
            utils.recursive_setattr(self.correlation_hists_delta_eta_subtracted, f"{attribute_name}.hist", subtracted_hist)

    def _extract_yield_from_hist(self, hist: histogram.Histogram1D,
                                 central_value: float, yield_limit: float) -> analysis_objects.ExtractedObservable:
        """ Helper function to actually extract a yield from a histogram.

        Yields are extracted within central_value +/- yield_limit.

        Args:
            hist: Histogram from which the yield should be extracted.
            central_value: Central value from which the yield should be integrated.
            yield_limit: Distance from the central value to include in the yield.
        Returns:
            Extracted observable containing the yield and the error on the yield.
        """
        # Integrate the histogram to get the yield.
        yield_value, yield_error = hist.integral(
            min_value = central_value - yield_limit + epsilon, max_value = central_value + yield_limit - epsilon,
        )

        # Scale by track pt bin width
        track_pt_bin_width = self.track_pt.max - self.track_pt.min
        yield_value /= track_pt_bin_width
        yield_error /= track_pt_bin_width

        # Store the yield in an observable
        observable = analysis_objects.ExtractedObservable(value = yield_value, error = yield_error)
        return observable

    def extract_yields(self) -> None:
        """ Extract and store near-side and away-side yields. """
        # Delta phi yields
        logger.debug("Extracting delta phi yields.")
        # Of the form (attribute_name, central_value)
        delta_phi_regions = [
            ("near_side", 0),
            ("away_side", np.pi),
        ]
        for attribute_name, central_value in delta_phi_regions:
            observable = self._extract_yield_from_hist(
                hist = self.correlation_hists_delta_phi_subtracted.signal_dominated.hist,
                central_value = central_value,
                yield_limit = self.task_config["delta_phi_yield_limit"],
            )
            # Store the extract yield
            logger.debug(f"Extracted {attribute_name} yield: {observable.value}, error: {observable.error}")
            setattr(self.yields_delta_phi, attribute_name, observable)

        # Delta eta yields
        logger.debug("Extracting delta eta yields.")
        # Of the form (attribute_name, central_value)
        delta_eta_regions = [
            ("near_side", 0),
            ("away_side", 0),
        ]
        for attribute_name, central_value in delta_eta_regions:
            observable = self._extract_yield_from_hist(
                hist = utils.recursive_getattr(self.correlation_hists_delta_eta_subtracted, f"{attribute_name}.hist"),
                central_value = central_value,
                yield_limit = self.task_config["delta_eta_yield_limit"],
            )
            setattr(self.yields_delta_eta, attribute_name, observable)

    def _retrieve_widths_from_RPF(self) -> bool:
        """ Helper function to actually extract and store widths from the RP fit. """
        logger.debug("Attempting to extract widths from the RPF fit.")
        regions = ["near_side", "away_side"]
        # Retrieve the widths parameter and it's error
        for region in regions:
            # Need to convert "near_side" -> "ns" to retrieve the parameters
            short_name = "".join([s[0] for s in region.split("_")])
            width_value = self.fit_object.fit_result.values_at_minimum.get(f"{short_name}_sigma", None)
            width_error = self.fit_object.fit_result.errors_on_parameters.get(f"{short_name}_sigma", None)
            # Only attempt to store the width if we were able to extract it.
            if width_value is None or width_error is None:
                logger.debug(f"Could not extract width or error from RPF for {self.identifier}, {self.reaction_plane_orientation}")
                return False
            # Help out mypy...
            assert width_value is not None and width_error is not None
            logger.debug(f"Extracted {region} width: {width_value}, error: {width_error}")

            # Store the output
            observable = analysis_objects.ExtractedObservable(value = width_value, error = width_error)
            setattr(self.widths_delta_phi, region, observable)

        return True

    def _fit_and_extract_delta_phi_widths(self) -> None:
        """ Extract delta phi near-side and away-side widths via a gaussian fit.

        The widths are extracted by fitting the subtracted delta phi corerlations to gaussians.
        """
        # Setup
        # Of the form (attribute_name, mean, initial_width)
        delta_phi_regions = [
            ("near_side", fitting.GaussianFitInputs(
                mean = 0, initial_width = 0.15,
                fit_range = self.near_side_phi_region.range,
            )),
            ("away_side", fitting.GaussianFitInputs(
                mean = np.pi, initial_width = 0.3,
                fit_range = self.away_side_phi_region.range,
            )),
        ]
        subtracted = self.correlation_hists_delta_phi_subtracted.signal_dominated

        # Fit and extract the widths.
        for attribute_name, inputs in delta_phi_regions:
            width, width_error = fitting.fit_gaussian_to_histogram(
                h = subtracted.hist, inputs = inputs,
            )

            # Store the result
            observable = analysis_objects.ExtractedObservable(
                value = width,
                error = width_error,
            )
            setattr(self.widths_delta_phi, attribute_name, observable)

    def _fit_and_extract_delta_eta_widths(self) -> None:
        """ Extract delta eta near-side and away-side widths via a gaussian fit.

        The widths are extracted by fitting the subtracted delta eta corerlations to gaussians.
        """
        # Setup
        # Of the form (attribute_name, mean, initial_width)
        delta_eta_regions = [
            ("near_side", fitting.GaussianFitInputs(
                mean = 0, initial_width = 0.15,
                fit_range = self.signal_dominated_eta_region.range,
            )),
            ("away_side", fitting.GaussianFitInputs(
                mean = np.pi, initial_width = 0.3,
                fit_range = self.signal_dominated_eta_region.range,
            )),
        ]

        # Fit and extract the widths.
        for attribute_name, inputs in delta_eta_regions:
            subtracted = getattr(self.correlation_hists_delta_eta_subtracted, attribute_name)
            width, error = fitting.fit_gaussian_to_histogram(
                h = subtracted.hist, inputs = inputs,
            )

            # Store the result
            observable = analysis_objects.ExtractedObservable(
                value = width,
                error = error,
            )
            setattr(self.widths_delta_eta, attribute_name, observable)

    def extract_widths(self) -> None:
        """ Extract and store near-side and away-side widths. """
        # Delta phi
        # Attempt to retrieve the widths from the RPF.
        extracted_from_RPF = self._retrieve_widths_from_RPF()
        if not extracted_from_RPF:
            logger.debug("Extracting widths via Gaussian fits")
            self._fit_and_extract_delta_phi_widths()

        # Delta eta
        # We will never extract these from the RPF, so we always need to run this.
        self._fit_and_extract_delta_eta_widths()

class CorrelationsManager(generic_class.EqualityMixin):
    def __init__(self, config_filename: str, selected_analysis_options: params.SelectedAnalysisOptions, **kwargs):
        self.config_filename = config_filename
        self.selected_analysis_options = selected_analysis_options
        self.task_name = "CorrelationsManager"
        # Retrieve YAML config for manager configuration
        # NOTE: We don't store the overridden selected_analysis_options because in principle they depend
        #       on the selected task. In practice, such options are unlikely to vary between the manager
        #       and the analysis tasks. However, the validation cannot handle the overridden options
        #       (because the leading hadron bias enum is converting into the object). So we just use
        #       the overridden option in formatting the output prefix (where it is required to determine
        #       the right path), and then passed the non-overridden values to the analysis objects.
        self.config, overridden_selected_analysis_options = analysis_config.read_config_using_selected_options(
            task_name = self.task_name,
            config_filename = self.config_filename,
            selected_analysis_options = self.selected_analysis_options
        )
        # Determine the formatting options needed for the output prefix
        formatting_options = analysis_config.determine_formatting_options(
            task_name = self.task_name, config = self.config,
            selected_analysis_options = overridden_selected_analysis_options
        )
        # Additional helper variables
        self.task_config = self.config[self.task_name]
        self.output_info = analysis_objects.PlottingOutputWrapper(
            # Format to ensure that the selected analysis options are filled in.
            output_prefix = self.config["outputPrefix"].format(**formatting_options),
            printing_extensions = self.config["printingExtensions"],
        )
        # For convenience since it is frequently accessed.
        self.processing_options = self.task_config["processing_options"]

        # Create the actual analysis objects.
        self.analyses: Mapping[Any, Correlations]
        self.selected_iterables: Dict[str, Sequence[Any]]
        (self.key_index, self.selected_iterables, self.analyses) = self.construct_correlations_from_configuration_file()

        # Store the fits.
        # We explicitly deselected the reaction plane orientation, because the main fit object doesn't
        # depend on it.
        self.fit_key_index = analysis_config.create_key_index_object(
            "FitKeyIndex",
            iterables = {k: v for k, v in self.selected_iterables.items() if k != "reaction_plane_orientation"},
        )
        self.fit_objects: Dict[Any, rpf_fit.ReactionPlaneFit] = {}

        # General histograms
        self.general_histograms = GeneralHistogramsManager(
            config_filename = self.config_filename,
            selected_analysis_options = self.selected_analysis_options
        )

        # Keep track of processing progress
        self._progress_manager = enlighten.get_manager()

    def construct_correlations_from_configuration_file(self) -> analysis_config.ConstructedObjects:
        """ Construct Correlations objects based on iterables in a configuration file. """
        return analysis_config.construct_from_configuration_file(
            task_name = "Correlations",
            config_filename = self.config_filename,
            selected_analysis_options = self.selected_analysis_options,
            additional_possible_iterables = {"pt_hard_bin": None, "jet_pt_bin": None, "track_pt_bin": None},
            obj = Correlations,
        )

    def setup(self) -> None:
        """ Setup the correlations manager. """
        # Retrieve input histograms (with caching).
        input_hists: Dict[str, Any] = {}
        with self._progress_manager.counter(total = len(self.analyses),
                                            desc = "Setting up:",
                                            unit = "analysis objects") as setting_up:
            for key_index, analysis in analysis_config.iterate_with_selected_objects(self.analyses):
                # We should now have all RP orientations.
                # We are effectively caching the values here.
                if not input_hists:
                    input_hists = histogram.get_histograms_in_file(filename = analysis.input_filename)
                logger.debug(f"{key_index}")
                # Setup input histograms and projectors.
                analysis.setup(input_hists = input_hists)
                # Keep track of progress
                setting_up.update()

    def _fit_delta_eta_correlations(self) -> None:
        """ Fit the delta eta correlations. """
        with self._progress_manager.counter(total = len(self.analyses),
                                            desc = "Fitting:",
                                            unit = "delta eta correlations") as fitting:
            for key_index, analysis in analysis_config.iterate_with_selected_objects(self.analyses):
                if self.processing_options["fit_correlations"]:
                    # Fit a pedestal to the background dominated eta region
                    # The result is stored in the analysis object.
                    analysis.fit_delta_eta_correlations()
                else:
                    # TODO: Load from file.
                    ...

                if self.processing_options["plot_delta_eta_fit"]:
                    plot_fit.delta_eta_fit(analysis)

                # Update progress
                fitting.update()

    def _reaction_plane_fit(self) -> None:
        """ Fit the delta phi correlations using the reaction plane fit. """
        number_of_fits = int(len(self.analyses) / len(self.selected_iterables["reaction_plane_orientation"]))
        with self._progress_manager.counter(total = number_of_fits,
                                            desc = "Reaction plane fitting:",
                                            unit = "delta phi hists") as fitting:
            resolution_parameters = self.task_config["reaction_plane_fit"]["resolution_parameters"]
            # To successfully fit, we need all histograms from a given reaction plane orientation.
            for ep_analyses in \
                    analysis_config.iterate_with_selected_objects_in_order(
                        analysis_objects = self.analyses,
                        analysis_iterables = self.selected_iterables,
                        selection = "reaction_plane_orientation",
                    ):
                # We will keep track of the inclusive analysis so we can easily access some analysis parameters.
                inclusive_analysis: Correlations
                # Setup the input data
                input_hists: rpf.fit.InputData = {
                    "signal": {},
                    "background": {},
                }
                for key_index, analysis in ep_analyses:
                    # Sanity checks
                    if analysis.ran_projections is False:
                        raise ValueError("Hists must be projected before running the fit.")

                    # Setup the input data
                    if analysis.reaction_plane_orientation == params.ReactionPlaneOrientation.inclusive:
                        inclusive_analysis = analysis
                    key = str(analysis.reaction_plane_orientation)
                    # Include the signal for inclusive orientations, but background for others.
                    if analysis.reaction_plane_orientation == params.ReactionPlaneOrientation.inclusive:
                        input_hists["signal"][key] = analysis.correlation_hists_delta_phi.signal_dominated
                    else:
                        input_hists["background"][key] = analysis.correlation_hists_delta_phi.background_dominated

                # Determine the key index for the fit object.
                # We want all iterables except the one that we selected on (the reaction plane orientations).
                fit_key_index = self.fit_key_index(**{k: v for k, v in key_index if k != "reaction_plane_orientation"})

                # Determine the user arguments.
                user_arguments = self.task_config["reaction_plane_fit"].get("fit_params", {}) \
                    .get(inclusive_analysis.jet_pt_identifier, {}) \
                    .get(inclusive_analysis.track_pt_identifier, {}).get("args", {})
                use_log_likelihood = self.task_config["reaction_plane_fit"].get("fit_params", {}) \
                    .get(inclusive_analysis.jet_pt_identifier, {}) \
                    .get(inclusive_analysis.track_pt_identifier, {}).get("use_log_likelihood", False)

                # Setup the fit
                logger.debug(f"Performing RPF for {inclusive_analysis.jet_pt_identifier}, {inclusive_analysis.track_pt_identifier}")
                fit_type = self.task_config["reaction_plane_fit"]["fit_type"]
                FitFunction = getattr(three_orientations, fit_type)
                fit_obj = FitFunction(
                    resolution_parameters = resolution_parameters,
                    use_log_likelihood = use_log_likelihood,
                    signal_region = analysis.signal_dominated_eta_region,
                    background_region = analysis.background_dominated_eta_region,
                    #use_minos = True,
                )

                # Now, perform the fit (or load in the fit result).
                rpf_filename = os.path.join(self.output_info.output_prefix, f"RPFitResult_{inclusive_analysis.identifier}.yaml")
                if self.processing_options["fit_correlations"]:
                    # Perform the fit.
                    fit_success, fit_data, _ = fit_obj.fit(
                        data = input_hists,
                        user_arguments = user_arguments,
                    )

                    # This should already be caught, but we handle it for good measure
                    if not fit_success:
                        raise RuntimeError(f"Fit failed for {inclusive_analysis.identifier}")

                    # Write out the fit results
                    logger.debug(f"Writing RPF to {rpf_filename}")
                    fit_obj.write_fit_results(filename = rpf_filename)
                else:
                    # Load from file.
                    logger.debug(f"Loading RPF from {rpf_filename}")
                    fit_obj.read_fit_results(filename = rpf_filename)

                # Store the fit results in the manager.
                # This main object has access to the entire result.
                self.fit_objects[fit_key_index] = fit_obj
                # Store the results relevant to each component in the individual analysis.
                for index, fit_component in fit_obj.components.items():
                    for key_index, analysis in ep_analyses:
                        if str(key_index.reaction_plane_orientation) in index.orientation:
                            analysis.fit_object = fit_component

                # Plot the result
                if self.processing_options["plot_RPF"]:
                    # Main fit plot
                    plot_fit.plot_RP_fit(
                        rp_fit = fit_obj,
                        inclusive_analysis = inclusive_analysis,
                        ep_analyses = ep_analyses,
                        output_info = self.output_info,
                        output_name = f"{fit_type}_{inclusive_analysis.identifier}",
                    )

                    # Covariance matrix
                    plot_fit.rpf_covariance_matrix(
                        fit_obj.fit_result,
                        output_info = self.output_info,
                        identifier = f"{fit_type}_{inclusive_analysis.identifier}",
                    )
                    # Correlation matrix
                    plot_fit.rpf_correlation_matrix(
                        fit_obj.fit_result,
                        output_info = self.output_info,
                        identifier = f"{fit_type}_{inclusive_analysis.identifier}",
                    )

                # Update progress
                for key_index, analysis in ep_analyses:
                    analysis.ran_fitting = True
                fitting.update()

        if self.processing_options["plot_RPF"]:
            # Fit parameters
            plot_fit.fit_parameters_vs_assoc_pt(
                fit_objects = self.fit_objects,
                selected_analysis_options = self.selected_analysis_options,
                output_info = self.output_info,
            )

            # Signal dominated with background function
            for key_index, analysis in analysis_config.iterate_with_selected_objects(self.analyses):
                plot_fit.signal_dominated_with_background_function(analysis)

    def fit(self) -> bool:
        """ Fit the stored correlations. """
        # Fit the delta eta correlations
        self._fit_delta_eta_correlations()
        # Fit the delta phi correlations using the reaction plane fit.
        self._reaction_plane_fit()
        return True

    def _subtract_reaction_plane_fits(self) -> None:
        """ Subtract the reaction plane fit from the delta phi correlations."""
        with self._progress_manager.counter(total = len(self.analyses),
                                            desc = "Subtracting fit from signal dominated hists:",
                                            unit = "delta phi hists") as subtracting:
            for ep_analyses, rp_fit in \
                    zip(analysis_config.iterate_with_selected_objects_in_order(
                        analysis_objects = self.analyses,
                        analysis_iterables = self.selected_iterables,
                        selection = "reaction_plane_orientation",
                    ),
                    self.fit_objects
                    ):
                # Subtract the background function from the signal dominated hist.
                inclusive_analysis: Correlations
                for key_index, analysis in ep_analyses:
                    # Sanity check
                    if not analysis.ran_fitting:
                        raise RuntimeError("Must run the fitting before subtracting!")

                    # Subtract
                    if self.processing_options["subtract_correlations"]:
                        analysis.subtract_background_fit_function_from_signal_dominated()
                    else:
                        # TODO: Load from file.
                        ...

                    # We will keep track of the inclusive analysis so we can easily access some analysis parameters.
                    if analysis.reaction_plane_orientation == params.ReactionPlaneOrientation.inclusive:
                        inclusive_analysis = analysis

                    if self.processing_options["plot_subtracted_correlations"]:
                        plot_fit.fit_subtracted_signal_dominated(analysis = analysis)
                        # Compare to Joel
                        if analysis.collision_energy == params.CollisionEnergy.two_seven_six \
                                and analysis.event_activity == params.EventActivity.central:
                            logger.info("Comparing subtracted correlations to Joel's.")
                            analysis.compare_subtracted_1d_signal_correlation_to_joel()
                        else:
                            logger.info("Skipping comparison with Joel since we're not analyzing the right system.")

                # Plot all RP fit angles together
                if self.processing_options["plot_subtracted_correlations"]:
                    fit_type = self.task_config["reaction_plane_fit"]["fit_type"]
                    plot_fit.rp_fit_subtracted(
                        ep_analyses = ep_analyses,
                        inclusive_analysis = inclusive_analysis,
                        output_info = self.output_info,
                        output_name = f"{fit_type}_{inclusive_analysis.identifier}",
                    )

                # Update progress
                for key_index, analysis in ep_analyses:
                    analysis.ran_post_fit_processing = True
                    subtracting.update()

    def _subtract_delta_eta_fits(self) -> None:
        """ Subtract the fits from the delta eta correlations. """
        with self._progress_manager.counter(total = len(self.analyses),
                                            desc = "Subtracting:",
                                            unit = "delta eta correlations") as subtracting:
            for key_index, analysis in analysis_config.iterate_with_selected_objects(self.analyses):
                if self.processing_options["subtract_correlations"]:
                    # Fit a pedestal to the background dominated eta region
                    # The result is stored in the analysis object.
                    analysis.subtract_delta_eta_correlations()
                else:
                    # TODO: Load from file.
                    ...

                if self.processing_options["plot_subtracted_correlations"]:
                    plot_fit.delta_eta_fit_subtracted(analysis)

                # Update progress
                subtracting.update()

    def subtract_fits(self) -> bool:
        """ Subtract the fits from the analysis histograms. """
        self._subtract_reaction_plane_fits()
        self._subtract_delta_eta_fits()

        return True

    def extract_yields(self) -> bool:
        """ Extract yields from analysis objects. """
        if self.processing_options["extract_yields"]:
            with self._progress_manager.counter(total = len(self.analyses),
                                                desc = "Extractin' yields:",
                                                unit = "delta phi hists") as extracting:
                for key_index, analysis in analysis_config.iterate_with_selected_objects(self.analyses):
                    # Ensure that the previous step was run
                    if not analysis.ran_post_fit_processing:
                        raise RuntimeError("Must run the post fit processing step before extracting yields!")

                    # Extract and store the yields.
                    analysis.extract_yields()

                    # Update progress
                    extracting.update()
        else:
            # TODO: Load yields from file.
            ...

        # Plot
        if self.processing_options["plot_yields"]:
            plot_extracted.plotYields(self)

        return True

    def extract_widths(self) -> bool:
        """ Extract widths from analysis objects. """
        if self.processing_options["extract_widths"]:
            with self._progress_manager.counter(total = len(self.analyses),
                                                desc = "Extractin' widths:",
                                                unit = "delta phi hists") as extracting:
                for key_index, analysis in analysis_config.iterate_with_selected_objects(self.analyses):
                    # Ensure that the previous step was run
                    if not analysis.ran_post_fit_processing:
                        raise RuntimeError("Must run the post fit processing step before extracting widths!")

                    # Extract and store the yields.
                    analysis.extract_widths()

                    # Update progress
                    extracting.update()
        else:
            # TODO: Load widths from file.
            ...

        # Plot
        if self.processing_options["plot_widths"]:
            plot_extracted.plotWidths(self)

        return True

    def run(self) -> bool:
        """ Run the analysis in the correlations manager. """
        # Analysis steps:
        # 1. Setup the correlations objects.
        # 2. Run the general histograms (if enabled.)
        # 3. Project, normalize, and plot the correlations down to 1D.
        # 4. Fit and plot the correlations.
        # 5. Subtract the fits from the correlations.
        # 6. Extract and plot the yields.
        # 7. Extract and plot the widths.
        steps = 7
        with self._progress_manager.counter(total = steps,
                                            desc = "Overall processing progress:",
                                            unit = "") as overall_progress:
            # First setup the correlations
            self.setup()
            overall_progress.update()

            # Run the general hists
            self.general_histograms.run()
            overall_progress.update()

            # First analysis step
            with self._progress_manager.counter(total = len(self.analyses),
                                                desc = "Projecting:",
                                                unit = "analysis objects") as projecting:
                for key_index, analysis in analysis_config.iterate_with_selected_objects(self.analyses):
                    analysis.run_projections()
                    # Keep track of progress
                    projecting.update()
            overall_progress.update()

            # Fitting
            self.fit()
            overall_progress.update()

            # Subtract the fits
            self.subtract_fits()
            overall_progress.update()

            # Extract yields
            self.extract_yields()
            overall_progress.update()

            # Extract widths
            self.extract_widths()
            overall_progress.update()

        # Disable enlighten so that it won't mess with any later steps (such as exploration with IPython)
        # Otherwise, IPython will act very strangely and is basically impossible to use.
        self._progress_manager.stop()

        return True

def write_analyses(manager: CorrelationsManager, output_filename: str) -> None:
    """ Write analyses to file via YAML. """
    # Need to register all ROOT histograms so that we can write them.
    root_classes_needed_for_yaml = [
        ROOT.TH1F,
        ROOT.TH2F,
        ROOT.TH1D,
        ROOT.TH2D,
        ROOT.THnSparseF,
    ]
    # NOTE: May need KeyIndex...
    #KeyIndex = next(iter(manager.analyses))

    # Register the necessary modules and classes
    y = yaml.yaml(
        modules_to_register = [
            histogram,
            projectors,
            HistAxisRange,
            this_module,
        ],
        classes_to_register = [
            #KeyIndex,
            *root_classes_needed_for_yaml,
        ],
    )

    analyses = list(manager.analyses.values())

    with open(output_filename, "w") as f:
        y.dump(analyses, f)

def run_from_terminal():
    """ Driver function for running the correlations analysis. """
    # Basic setup
    coloredlogs.install(
        level = logging.DEBUG,
        fmt = "%(asctime)s %(name)s:%(lineno)d %(levelname)s %(message)s"
    )
    # Quiet down the matplotlib logging
    logging.getLogger("matplotlib").setLevel(logging.INFO)
    # Quiet down pachyderm
    logging.getLogger("pachyderm").setLevel(logging.INFO)
    # Quiet down reaction_plane_fit
    logging.getLogger("reaction_plane_fit").setLevel(logging.INFO)

    # Turn off stats box
    ROOT.gStyle.SetOptStat(0)

    # Setup the analysis
    (config_filename, terminal_args, additional_args) = analysis_config.determine_selected_options_from_kwargs(
        task_name = "Correlations"
    )
    selected_analysis_options, _ = analysis_config.validate_arguments(selected_args = terminal_args)
    analysis_manager = CorrelationsManager(
        config_filename = config_filename,
        selected_analysis_options = selected_analysis_options,
    )
    # Finally run the analysis.
    analysis_manager.run()

    # Quiet down IPython.
    logging.getLogger("parso").setLevel(logging.INFO)
    # Embed IPython to allow for some additional exploration
    IPython.embed()

    # Return the manager for convenience.
    return analysis_manager

if __name__ == "__main__":
    run_from_terminal()

