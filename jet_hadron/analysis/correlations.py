#!/usr/bin/env python

""" Main jet-hadron correlations analysis module

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import copy
import ctypes
import dataclasses
from dataclasses import dataclass
import enum
#import IPython
import logging
import os
import pprint
import math
import sys
from typing import Any, Dict, List, Mapping, Optional, Type
import warnings

from pachyderm import generic_class
from pachyderm import generic_config
from pachyderm import histogram
from pachyderm import projectors
from pachyderm.projectors import HistAxisRange
from pachyderm import utils
from pachyderm.utils import epsilon

from jet_hadron.base import analysis_config
from jet_hadron.base import analysis_objects
from jet_hadron.base import params
from jet_hadron.base.typing_helpers import Hist
from jet_hadron.plot import general as plot_general
from jet_hadron.plot import correlations as plot_correlations
from jet_hadron.plot import fit as plot_fit
from jet_hadron.plot import extracted as plot_extracted
from jet_hadron.analysis import correlations_helpers
from jet_hadron.analysis import fit as fitting
from jet_hadron.analysis import generic_tasks

import ROOT
# Tell ROOT to ignore command line options so args are passed to python
# NOTE: Must be immediately after import ROOT!
ROOT.PyConfig.IgnoreCommandLineOptions = True

# Handle rootpy warning
warnings.filterwarnings(action='ignore', category=RuntimeWarning, message=r'creating converter for unknown type "_Atomic\(bool\)"')

# Setup logger
logger = logging.getLogger(__name__)

# Run in batch mode
ROOT.gROOT.SetBatch(True)

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

class JetHTriggerSparse(enum.Enum):
    """ Define the axes in the Jet-Hadron Trigger Sparse. """
    centrality = 0
    jet_pt = 1
    reaction_plane_orientation = 2

class JetHCorrelationSparseProjector(projectors.HistProjector):
    """ Projector for THnSparse into 2D histograms.

    Note:
        This class isn't really necessary, but it makes further customization straightforward if
        it is found to be necessary, so we keep it around.
    """
    ...

class JetHCorrelationProjector(projectors.HistProjector):
    """ Projector for the Jet-h 2D correlation hists to 1D correlation hists. """
    def projection_name(self, **kwargs):
        """ Define the projection name for the Jet-H response matrix projector """
        track_pt = kwargs["track_pt"]
        jet_pt = kwargs["jet_pt"]
        logger.info(f"Projecting hist name: {self.projection_name_format.format(track_pt_bin = track_pt.bin, jet_pt_bin = jet_pt.bin, **kwargs)}")
        return self.projection_name_format.format(track_pt_bin = track_pt.bin, jet_pt_bin = jet_pt.bin, **kwargs)

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

    def retrieveInputHists(self):
        """ Run general setup tasks. """
        ...

    def assignGeneralHistsFromDict(self, histDict, outputDict):
        """ Simple helper to assign hists named in a dict to an output dict. """
        for name, histName in histDict.items():
            # NOTE: The hist may not always exist, so we return None if it doesn't!
            outputDict[name] = self.inputHists.get(histName, None)

    def generalHistograms(self):
        """ Process some general histograms such as centrality, Z vertex, very basic QA spectra, etc. """
        ...
        # Get configuration
        processing_options = self.taskConfig["processing_options"]

        if processing_options["generalHistograms"]:
            # 1D hists
            hists1D = {
                "zVertex": "fHistZVertex",
                "centrality": "fHistCentrality",
                "eventPlane": "fHistEventPlane",
                "jetMatchingSameEvent": "fHistJetMatchingSameEventCuts",
                "jetMatchingMixedEvent": "fHistJetMatchingMixedEventCuts",
                # The following track and jet pt hists look better with logy
                # They will be selected by the "Pt" in their name to apply logy in the plotting function
                "trackPt": "fHistTrackPt",
                # Just 0-10%
                "jetPt": "fHistJetPt_0",
                "jetPtBias": "fHistJetPtBias_0"
            }
            self.assignGeneralHistsFromDict(histDict = hists1D, outputDict = self.generalHists1D)
            # 2D hists
            # All of the jets in "jetEtaPhi" are in the EMCal
            hists2D = {
                "jetEtaPhi": "fHistJetEtaPhi",
                # Eta-phi of charged hadrons used for jet-hadron correlations
                "jetHEtaPhi": "fHistJetHEtaPhi"
            }
            self.assignGeneralHistsFromDict(histDict = hists2D, outputDict = self.generalHists2D)

            # Plots all hists
            plot_general.plotGeneralAnalysisHistograms(self)

            # Save out the hists
            self.writeGeneralHistograms()

    def generate2DCorrelationsTHnSparse(self):
        """ Generate raw and mixed event 2D correlations. """
        ...

    def generate2DSignalCorrelation(self):
        """ Generate 2D signal correlation.

        Intentionally decoupled for creating the raw and mixed event hists so that the THnSparse can be swapped out
        when desired.
        """
        ...

    def generate1DCorrelations(self):
        """ Generate 1D Correlation by projecting 2D correlations. """
        ...

    def post1DProjectionScaling(self):
        """ Perform post projection scalings.

        In particular, scale the 1D hists by their bin widths so no further scaling is necessary.
        """
        ...
        for hists in self.hists1DStandard:
            #logger.debug("len(hists): {}, hists.keys(): {}".format(len(hists), hists.keys()))
            for name, observable in hists.items():
                scaleFactor = observable.hist.calculateFinalScaleFactor()
                #logger.info("Post projection scaling of hist {} with scale factor 1/{}".format(name, 1/scaleFactor))
                observable.hist.Scale(scaleFactor)

    def generateSparseProjectors(self):
        """ Generate sparse projectors """
        ...

    def generateCorrelationProjectors(self):
        """ Generate correlation projectors (2D -> 1D) """
        ...

    def convert1DRootHistsToArray(self, inputHists, outputHists):
        """ Convert requested 1D hists to hist array format. """
        for observable in inputHists.values():
            outputHistName = self.histNameFormatDPhiArray.format(jetPtBin = observable.jetPtBin, trackPtBin = observable.trackPtBin, tag = observable.correlationType.str())
            histArray = analysis_objects.HistArray.initFromRootHist(observable.hist.hist)
            outputHists[outputHistName] = analysis_objects.CorrelationObservable1D(
                jetPtBin = observable.jetPtBin,
                trackPtBin = observable.trackPtBin,
                correlationType = observable.correlationType,
                axis = observable.axis,
                hist = histArray
            )

    def convertDPhiHists(self):
        """ Convert dPhi hists to hist arrays. """
        self.convert1DRootHistsToArray(self.dPhi, self.dPhiArray)
        self.convert1DRootHistsToArray(self.dPhiSideBand, self.dPhiSideBandArray)

    @staticmethod
    def fitCombinedSignalAndBackgroundRegion(analyses):
        """ Driver for the apply the reaction plane fit to the signal and background regions. """
        # Define EP fit object to manage the RPF
        epFit = fitting.JetHEPFit(analyses)

        # Setup and perform the fit
        epFit.DefineFits()
        epFit.PerformFit()

        # Fit errors
        epFit.DetermineFitErrors()

        return epFit

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
            subtractedHists["{}_subtracted".format(name)] = analysis_objects.CorrelationObservable1D(
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

    def extractYields(self, yieldLimit):
        # Yields
        #extractYields(dPhiSubtracted, yields, yieldErrors, yieldLimit, iJetPtBin, iTrackPtBin)
        parameters = {"NS": [0, self.yieldsNS, self.dPhiSubtracted, self.dPhiSubtractedFit],
                      "AS": [math.pi, self.yieldsAS, self.dPhiSubtracted, self.dPhiSubtractedFit],
                      "dEtaNS": [0, self.yieldsDEtaNS, self.dEtaNSSubtracted, self.dEtaNSSubtractedFit]}

        for location, (centralValue, yields, hists, fits) in parameters.items():
            for (name, observable), fit in zip(hists.items(), fits.values()):
                # Extract yield
                min_val = observable.hist.GetXaxis().FindBin(centralValue - yieldLimit + utils.epsilon)
                max_val = observable.hist.GetXaxis().FindBin(centralValue + yieldLimit - utils.epsilon)
                yieldError = ctypes.c_double(0)
                yieldValue = observable.hist.IntegralAndError(min_val, max_val, yieldError, "width")

                # Convert ctype back to python type for convenience
                yieldError = yieldError.value

                # Scale by track pt bin width
                trackPtBinWidth = self.track_pt.range.max - self.track_pt.range.min
                yieldValue /= trackPtBinWidth
                yieldError /= trackPtBinWidth

                # Store yield
                yields[f"{name}_yield"] = analysis_objects.ExtractedObservable(
                    jetPtBin = observable.jetPtBin,
                    trackPtBin = observable.trackPtBin,
                    value = yieldValue,
                    error = yieldError
                )

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

    def writeHistsToYAML(self, output_observable, mode = "wb"):
        """ Write hist to YAML file. """

        logger.info("Saving hist arrays!")

        #for histCollection in output_observable:
        #    for name, observable in histCollection.items():
        #        if isinstance(observable, analysis_objects.Observable):
        #            hist = observable.hist
        #        else:
        #            hist = observable

        #        hist.saveToYAML(prefix = self.outputPrefix,
        #                        objType = observable.correlationType,
        #                        jetPtBin = observable.jetPtBin,
        #                        trackPtBin = observable.trackPtBin,
        #                        fileAccessMode = mode)

    def writeGeneralHistograms(self):
        """ Write general histograms to file. """
        self.writeToRootFile(self.generalHists)

    def writeTriggerJetSpectra(self):
        """ Write trigger jet spectra to file. """
        self.writeToRootFile([self.triggerJetPt])

    def write2DCorrelations(self):
        """ Write the 2D Correlations to file. """
        ...

    def write1DCorrelations(self):
        """ Write the 1D Correlations to file. """
        # Write the ROOT files
        self.writeToRootFile(self.hists1DStandard)
        # Write the yaml files
        self.writeHistsToYAML(self.hists1DStandardArray)

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

    def writeYields(self):
        """ Write yields to a YAML file. """
        pass

    def writeWidths(self):
        """ Write widths to a YAML file. """
        pass

    def writeExtractedValuesToYAML(self, values, extractedValueLabel):
        """ Write extracted values to a YAML file. """
        pass

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
                        hists1D.append([self.dPhi,         self.histNameFormatDPhi, analysis_objects.JetHCorrelationType.signal_dominated,     analysis_objects.JetHCorrelationAxis.delta_phi])  # noqa: E241
                        hists1D.append([self.dPhiSideBand, self.histNameFormatDPhi, analysis_objects.JetHCorrelationType.background_dominated, analysis_objects.JetHCorrelationAxis.delta_phi])  # noqa: E241
                        hists1D.append([self.dEtaNS,       self.histNameFormatDEta, analysis_objects.JetHCorrelationType.near_side,            analysis_objects.JetHCorrelationAxis.delta_eta])  # noqa: E241
                    if Correlations1DSubtracted:
                        hists1D.append([self.dPhiSubtracted,   self.histNameFormatDPhiSubtracted, analysis_objects.JetHCorrelationType.signal_dominated,     analysis_objects.JetHCorrelationAxis.delta_phi])  # noqa: E241
                        hists1D.append([self.dEtaNSSubtracted, self.histNameFormatDEtaSubtracted, analysis_objects.JetHCorrelationType.near_side,            analysis_objects.JetHCorrelationAxis.delta_eta])  # noqa: E241
                    if Correlations1DArray:
                        logger.debug("Correlations1DArray hists")
                        retrieveArray = True
                        hists1D.append([self.dPhiArray,         self.histNameFormatDPhiArray, analysis_objects.JetHCorrelationType.signal_dominated,     analysis_objects.JetHCorrelationAxis.delta_phi])  # noqa: E241
                        hists1D.append([self.dPhiSideBandArray, self.histNameFormatDPhiArray, analysis_objects.JetHCorrelationType.background_dominated, analysis_objects.JetHCorrelationAxis.delta_phi])  # noqa: E241
                        hists1D.append([self.dEtaNS,            self.histNameFormatDEtaArray, analysis_objects.JetHCorrelationType.near_side,            analysis_objects.JetHCorrelationAxis.delta_eta])  # noqa: E241
                    if Correlations1DSubtractedArray:
                        retrieveArray = True
                        hists1D.append([self.dPhiSubtractedArray,   self.histNameFormatDPhiSubtractedArray, analysis_objects.JetHCorrelationType.signal_dominated, analysis_objects.JetHCorrelationAxis.delta_phi])  # noqa: E241
                        hists1D.append([self.dEtaNSSubtractedArray, self.histNameFormatDEtaSubtractedArray, analysis_objects.JetHCorrelationType.near_side,        analysis_objects.JetHCorrelationAxis.delta_eta])  # noqa: E241

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
        histsToWriteOut["dPhiCorrelations"] = (self.fitNameFormat, analysis_objects.JetHCorrelationType.signal_dominated.str(), r"\dPhi{} correlation with the all angles signal and event plane dependent background fit components. This correlation corresponding to $%(jetPtLow)s < \pTJet{} < %(jetPtHigh)s$ \gevc{} and $%(trackPtLow)s < \pTAssoc{} < %(trackPtHigh)s$ \gevc{}.")
        # TODO: Add comparisons to Joel, but probably best to do in an entirely separate section
        histsToWriteOut["joelComparisonSubtracted"] = ("joelComparison_jetPt{jetPtBin}_trackPt{trackPtBin}_{tag}", analysis_objects.JetHCorrelationType.signal_dominated.str() + "_subtracted", r"Subtracted \dPhi{} correlation comparing correlations from this analysis and those produced using the semi-central analysis code described in \cite{jetHEventPlaneAN}. Error bars correspond to statistical errors and error bands correspond to the error on the fit. This correlation corresponding to $%(jetPtLow)s < \pTJet{} < %(jetPtHigh)s$ \gevc{} and $%(trackPtLow)s < \pTAssoc{} < %(trackPtHigh)s$ \gevc{}.")

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

    @staticmethod
    def printProperty(name, val):
        """ Convenience method to pretty print a property. """
        logger.info("    {name}: {val}".format(name = name, val = val))

    @staticmethod
    def constructFromConfigurationFile(configFilename, selectedAnalysisOptions):
        """ Helper function to construct jet-h correlation analysis objects.

        Args:
            configFilename (str): Filename of the yaml config.
            selectedAnalysisOptions (params.SelectedAnalysisOptions): Selected analysis options.
        Returns:
            nested tuple: Tuple of nested analysis objects as described in analysis_config.constructFromConfigurationFile(...).
        """
        return analysis_config.constructFromConfigurationFile(
            taskName = "JetHAnalysis",
            configFilename = configFilename,
            selectedAnalysisOptions = selectedAnalysisOptions,
            obj = JetHAnalysis
        )

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

            jetH.retrieveInputHists()
            jetH.generateTeXToIncludeFiguresInAN()

            logger.info("Creating general histograms")
            jetH.generalHistograms()

            logger.info("Running analysis through projecting 1D correlations")
            jetH.runProjections()

            plot_general.plotTriggerJetSpectra(jetH)

        # TEMP
        #IPython.embed()
        #sys.exit(0)
        # ENDTEMP

        # Need all of the event plane angles to perform the RP Fit
        logger.info("Running 1D fits")
        epFit = JetHAnalysis.runFitting(analyses)

        if epFit:
            logger.info("Running post EP fit processing")
            logger.info("Subtracting EP hists")
            # TODO: Ensure that the new analyses object is handled properly!
            # TODO: This does more than it the name implies, so it should be renamed!
            JetHAnalysis.subtractEPHists(analyses, epFit)
        else:
            # Finish the analysis
            for keys, task in generic_config.unrollNestedDict(analyses):
                logger.info(f"Running {keys} post fit analysis")
                jetH.postFitProcessing()
                jetH.yieldsAndWidths()

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
            # First generate the projectors
            logger.info("Generating 1D projectors")
            self.generateCorrelationProjectors()
            # Project in 1D
            logger.info("Projecting 1D correlations")
            self.generate1DCorrelations()

            # Perform post-projection scalings to avoid needing to scale the fit functions later
            logger.info("Performing post projection histogram scalings")
            self.post1DProjectionScaling()

            # Create hist arrays
            self.convertDPhiHists()

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
        # TODO: TEMP
        self = None
        # ENDTEMP
        # Ensure that the previous step was run
        for _, jetH in generic_config.unrollNestedDict(analyses):
            if not jetH.ranProjections:
                raise RuntimeError("Must run the projection step before fitting!")

        # Get the first analysis so we can get configuration options, etc
        _, firstAnalysis = next(generic_config.unrollNestedDict(analyses))
        processing_options = firstAnalysis.taskConfig["processing_options"]

        # Run the fitting code
        if processing_options["fit1DCorrelations"]:
            if firstAnalysis.collisionSystem == params.collisionSystem.PbPb:
                # Run the combined fit over the analyses
                epFit = JetHAnalysis.fitCombinedSignalAndBackgroundRegion(analyses)

                if processing_options["plot1DCorrelationsWithFits"]:
                    # Plot the result
                    plot_fit.PlotRPF(epFit)
            else:
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
                processing_options["subtract1DCorrelations"] = True
        else:
            # TODO: This isn't well defined because self isn't defined.
            #       Need to think more about how to handle it properly.
            # Initialize the fits from the file
            self.InitFromRootFile(Fits1D = True)

        # Note that the step was completed
        for _, jetH in generic_config.unrollNestedDict(analyses):
            jetH.ranFitting = True

        return epFit

    @staticmethod
    def subtractEPHists(analyses, epFit):
        # TODO: Refactor this function!
        # Ensure that the previous step was run
        for _, jetH in generic_config.unrollNestedDict(analyses):
            if not jetH.runFitting:
                raise RuntimeError("Must run the fitting step before subtraction!")

        # Get the first analysis so we can get configuration options, etc
        _, firstAnalysis = next(generic_config.unrollNestedDict(analyses))
        processing_options = firstAnalysis.taskConfig["processing_options"]

        # Subtracted fit functions from the correlations
        if processing_options["subtract1DCorrelations"]:
            logger.info("Subtracting EP dPhi hists")
            epFit.SubtractEPHists()

            if processing_options["plotSubtracted1DCorrelations"]:
                plot_fit.PlotSubtractedEPHists(epFit)

            logger.info("Comparing to Joel")
            plot_fit.CompareToJoel(epFit)

            logger.info("Plotting widths")
            widths = epFit.RetrieveWidths()
            plot_extracted.PlotWidthsNew(firstAnalysis, widths)

    def postFitProcessing(self):
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

    def yieldsAndWidths(self):
        # Ensure that the previous step was run
        if not self.ranPostFitProcessing:
            logger.critical("Must run the post fit processing step before extracting yields and widths!")
            sys.exit(1)

        # Get processing options
        processing_options = self.taskConfig["processing_options"]

        # Extract yields
        if processing_options["extractYields"]:
            # Setup yield limits
            # 1.0 is the value from the semi-central analysis.
            yieldLimit = self.config.get("yieldLimit", 1.0)

            logger.info("Extracting yields with yield limit: {}".format(yieldLimit))
            self.extractYields(yieldLimit = yieldLimit)
            #logger.info("jetH AS yields: {}".format(self.yieldsAS))

            # Plot
            if processing_options["plotYields"]:
                plot_extracted.plotYields(self)

            processing_options["extractWidths"] = True
        else:
            # Initialize yields from the file
            pass

        # Extract widths
        if processing_options["extractWidths"]:
            # Extract widths
            logger.info("Extracting widths from the fits.")
            self.extractWidths()
            #logger.info("jetH AS widths: {}".format(self.widthsAS))

            # Plot
            if processing_options["plotWidths"]:
                plot_extracted.plotWidths(self)
        else:
            # Initialize widths from the file
            pass

        ## Inclusive plots
        #logger.info("Plotting inclusive jet and track spectra!")
        #logger.info("Plotting inclusive 2D correlations!")

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

#def runFromTerminal():
#    (configFilename, terminalArgs, additionalArgs) = analysis_config.determineSelectedOptionsFromKwargs(taskName = "correlations analysis")
#    analyses = JetHAnalysis.run(
#        configFilename = configFilename,
#        selectedAnalysisOptions = terminalArgs
#    )
#
#    return analyses

#class GeneralHistograms(analysis_objects.JetHBase):
#    """ Some very general histograms such as centrality, Z vertex, very basic QA spectra, etc. """
#    def __init__(self, *args, **kwargs):
#        super().__init__(*args, **kwargs)
#
#        # Basic information
#        self.input_hists: Dict[str, Any] = {}
#
#        # Histograms
#        self.z_vertex: Hist
#        self.centrality: Hist
#        self.event_plane: Hist
#        self.jet_matching_same_event: Hist
#        self.jet_matching_mixed_event : Hist
#        self.track_pt: Hist
#        self.jet_pt: Hist
#        self.jet_pt_bias: Hist
#        # 2D hists
#        self.jet_eta_phi: Hist
#        self.jetH_eta_phi: Hist
#

class PlotGeneralHistograms(generic_tasks.PlotTaskHists):
    """ Task to plot general task hists.

    Note:
        This current doesn't have any embedding specific functionality. It is created for clarity and to
        encourage extension in the future.
    """
    ...

class GeneralHistogramsManager(generic_tasks.TaskManager):
    """ Manager for plotting general histograms. """
    def construct_tasks_from_configuration_file(self) -> analysis_config.ConstructedObjects:
        return analysis_config.construct_from_configuration_file(
            task_name = "GeneralHists",
            config_filename = self.config_filename,
            selected_analysis_options = self.selected_analysis_options,
            additional_possible_iterables = {"task_label": None, "pt_hard_bin": None},
            obj = PlotGeneralHistograms,
        )

@dataclass
class CorrelationHistograms2D:
    raw: Hist
    mixed_event: Hist
    signal: Hist

    def asdict(self) -> Dict[str, Hist]:
        return dataclasses.asdict(self)

@dataclass
class CorrelationHistogramsDeltaPhi:
    signal_dominated: Optional[analysis_objects.CorrelationObservable1D]
    background_dominated: Optional[analysis_objects.CorrelationObservable1D]
    axis = analysis_objects.JetHCorrelationAxis.delta_phi

    def asdict(self) -> Dict[str, Hist]:
        return dataclasses.asdict(self)

@dataclass
class CorrelationHistogramsDeltaEta:
    near_side: Optional[analysis_objects.CorrelationObservable1D]
    away_side: Optional[analysis_objects.CorrelationObservable1D]
    axis = analysis_objects.JetHCorrelationAxis.delta_eta

    def asdict(self) -> Dict[str, Hist]:
        return dataclasses.asdict(self)

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
    # Properties
    # Define as static variables since they don't depend on any particular instance
    hist_name_format = "jetH%(label)s_jetPt{jet_pt_bin}_trackPt{track_pt_bin}_{tag}"
    hist_name_format_2d = hist_name_format % {"label": "DEtaDPhi"}
    # Standard 1D hists
    hist_name_format_delta_phi = hist_name_format % {"label": "DPhi"}
    hist_name_format_delta_phi_array = hist_name_format % {"label": "DPhi"} + "Array"
    hist_name_format_delta_eta = hist_name_format % {"label": "DEta"}
    hist_name_format_delta_eta_array = hist_name_format % {"label": "DEta"} + "Array"
    # Subtracted 1D hists
    hist_name_format_delta_phi_subtracted = hist_name_format_delta_phi + "_subtracted"
    hist_name_format_delta_phi_subtracted_array = hist_name_format_delta_phi_array + "_subtracted"
    hist_name_format_delta_eta_subtracted = hist_name_format_delta_eta + "_subtracted"
    hist_name_format_delta_eta_subtracted_array = hist_name_format_delta_eta_array + "_subtracted"

    # These is nothing here to format - it's just the jet spectra
    # However, the variable name will stay the same for clarity
    hist_name_format_trigger = "jetH_trigger_pt"
    fit_name_format = hist_name_format % {"label": "Fit"}

    def __init__(self, jet_pt_bin: analysis_objects.JetPtBin, track_pt_bin: analysis_objects.TrackPtBin, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Analysis parameters
        self.jet_pt = jet_pt_bin
        self.track_pt = track_pt_bin
        # Pt hard bins are optional.
        self.pt_hard_bin = kwargs.get("pt_hard_bin", None)
        if self.pt_hard_bin:
            self.train_number = self.pt_hard_bin.train_number
            self.input_filename = self.input_filename.format(pt_hard_bin_train_number = self.train_number)

        # Basic information
        self.input_hists: Dict[str, Any] = {}
        self.projectors: List[Type[projectors.HistProjectors]] = []
        # For convenience since it is frequently accessed.
        self.processing_options = self.task_config["processingOptions"]
        # Status information
        self.ran_projections: bool = False

        # Relevant histograms
        self.number_of_triggers_hist: Hist = None
        self.correlation_hists_2d: CorrelationHistograms2D = CorrelationHistograms2D(None, None, None)
        self.correlation_hists_delta_phi: CorrelationHistogramsDeltaPhi = CorrelationHistogramsDeltaPhi(None, None)
        self.correlation_hists_delta_eta: CorrelationHistogramsDeltaEta = CorrelationHistogramsDeltaEta(None, None)

        # Other relevant analysis information
        self.number_of_triggers: int = 0

        # Projectors
        self.sparse_projectors: List[JetHCorrelationSparseProjector] = []
        self.correlation_projectors: List[JetHCorrelationProjector] = []

    def _write_2d_correlations(self) -> None:
        """ Write 2D correlatinos to output file. """
        hist_names = ["raw", "mixed_event", "signal"]
        self._write_to_root_file(hists = self.correlation_hists_2d, hist_names = hist_names)

    def _write_trigger_jet_spectra(self):
        """ Write trigger jet spectra to file. """
        self._write_to_root_file(hists = self, hist_names = ["number_of_triggers_hist"])

    def _write_to_root_file(self, hists, hist_names: List[str], mode: str = "UPDATE") -> None:
        """ Write output list to a file """
        filename = os.path.join(self.output_prefix, self.output_filename)

        logger.info(f"Saving correlations to {filename}")

        with histogram.RootOpen(filename = filename, mode = mode):
            for hist_name in hist_names:
                hist = getattr(hists, hist_name)
                if hist:
                    hist.Write()

    def _init_from_root_file(self, **kwargs) -> bool:
        raise NotImplementedError("Need to move from the previous implementation.")

    def _setup_sparse_projectors(self) -> None:
        """ Setup the THnSparse projectors.

        The created projectors are added to the ``sparse_projectors`` list.
        """
        # Helper which defines the full axis range
        full_axis_range = {
            "min_val": HistAxisRange.apply_func_to_find_bin(None, 1),
            "max_val": HistAxisRange.apply_func_to_find_bin(ROOT.TAxis.GetNbins)
        }

        # Define common axes
        # Centrality axis
        centrality_cut_axis = HistAxisRange(
            axis_type = JetHCorrelationSparse.centrality,
            axis_range_name = "centrality",
            min_val = HistAxisRange.apply_func_to_find_bin(ROOT.TAxis.FindBin, 0 + epsilon),
            max_val = HistAxisRange.apply_func_to_find_bin(ROOT.TAxis.FindBin, 10 - epsilon),
        )
        # Event plane selection
        if self.reaction_plane_orientation == params.ReactionPlaneOrientation.all:
            reaction_plane_axis_range = full_axis_range
            logger.info("Using full EP angle range")
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
            logger.info(f"Using selected EP angle range {self.reaction_plane_orientation.name}")
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
            axis_range_name = f"jet_pt{self.jet_pt.bin}",
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
            axis_range_name = f"track_pt{self.track_pt.bin}",
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
            output_observable = self,
            output_attribute_name = "number_of_triggers_hist",
            projection_name_format = self.hist_name_format_trigger,
            projection_information = projection_information
        )
        # Take advantage of existing centrality and event plane object, but need to copy and modify the axis type
        if self.collision_system != params.CollisionSystem.pp:
            trigger_centrality_cut_axis = copy.deepcopy(centrality_cut_axis)
            trigger_centrality_cut_axis.axis_type = JetHTriggerSparse.centrality
            trigger_projector.additional_axis_cuts.append(trigger_centrality_cut_axis)
        if reaction_plane_orientation_cut_axis:
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

        # Jet and track pt bin dependent cuts
        projection_information = {"jet_pt_bin": self.jet_pt.bin, "track_pt_bin": self.track_pt.bin}

        ###########################
        # Raw signal projector
        ###########################
        projection_information["tag"] = "raw"
        raw_signal_projector = JetHCorrelationSparseProjector(
            observable_to_project_from = self.input_hists["fhnJH"],
            output_observable = self.correlation_hists_2d,
            output_attribute_name = "raw",
            projection_name_format = self.hist_name_format_2d,
            projection_information = projection_information,
        )
        if self.collision_system != params.CollisionSystem.pp:
            raw_signal_projector.additional_axis_cuts.append(centrality_cut_axis)
        if reaction_plane_orientation_cut_axis:
            raw_signal_projector.additional_axis_cuts.append(reaction_plane_orientation_cut_axis)
        # TODO: Do these projectors really need projection dependent cut axes?
        #       It seems like additionalAxisCuts would be sufficient.
        projection_dependent_cut_axes = [jet_pt_axis, track_pt_axis]
        # NOTE: We are passing a list to the list of cuts. Therefore, the two cuts defined above will be
        #       applied on the same projection!
        raw_signal_projector.projection_dependent_cut_axes.append(projection_dependent_cut_axes)
        # Projection Axes
        raw_signal_projector.projection_axes.append(delta_phi_axis)
        raw_signal_projector.projection_axes.append(delta_eta_axis)
        self.sparse_projectors.append(raw_signal_projector)

        ###########################
        # Mixed Event projector
        ###########################
        # TODO: Use a broader range of pt for mixed events like Joel?
        projection_information["tag"] = "mixed"
        mixed_event_projector = JetHCorrelationSparseProjector(
            observable_to_project_from = self.input_hists["fhnMixedEvents"],
            output_observable = self.correlation_hists_2d,
            output_attribute_name = "mixed_event",
            projection_name_format = self.hist_name_format_2d,
            projection_information = projection_information,
        )
        if self.collision_system != params.CollisionSystem.pp:
            mixed_event_projector.additional_axis_cuts.append(centrality_cut_axis)
        if reaction_plane_orientation_cut_axis:
            mixed_event_projector.additional_axis_cuts.append(reaction_plane_orientation_cut_axis)
        projection_dependent_cut_axes = [jet_pt_axis, track_pt_axis]
        # NOTE: We are passing a list to the list of cuts. Therefore, the two cuts defined above will be
        #       applied on the same projection!
        mixed_event_projector.projection_dependent_cut_axes.append(projection_dependent_cut_axes)
        # Projection Axes
        mixed_event_projector.projection_axes.append(delta_phi_axis)
        mixed_event_projector.projection_axes.append(delta_eta_axis)
        self.sparse_projectors.append(mixed_event_projector)

    def _setup_projectors(self):
        """ Setup the projectors for the analysis. """
        # NOTE: It's best to define the projector right before utilizing it. Here, this runs as the last
        #       step of the setup, and then these projectors are executed immeidately.
        #       This is the best practice because we can only define the projectors for single objects once
        #       the histogram that it will project from exists. If it doesn't yet exist, the projector will
        #       fail because it stores the value (ie the hist) at the time of the projector definition.
        self._setup_sparse_projectors()

    def _determine_number_of_triggers(self) -> int:
        """ Determine the number of triggers for the specific analysis parameters. """
        return correlations_helpers.determine_number_of_triggers(
            hist = self.number_of_triggers_hist,
            jet_pt = self.jet_pt,
        )

    def setup(self, input_hists):
        """ Setup the correlations object. """
        # Setup the input hists and projectors
        super().setup(input_hists = input_hists)

    def _post_creation_processing_for_2d_correlation(self, hist: Hist, normalization_factor: float, title_label: str) -> None:
        """ Perform post creation processing for 2D correlations. """
        correlations_helpers.post_projection_processing_for_2d_correlation(
            hist = hist, normalization_factor = normalization_factor, title_label = title_label,
            jet_pt = self.jet_pt, track_pt = self.track_pt,
        )

    def _compoare_mixed_event_normalization_options(self, mixed_event: Hist) -> None:
        """ Compare mixed event normalization options. """
        eta_limits = self.task_config["mixedEventNormalizationOptions"].get("etaLimits", [-0.3, 0.3])

        # Create the comparison
        (
            # Basic data
            peak_finding_hist,
            lin_space, peak_finding_hist_array,
            lin_space_rebin, peak_finding_hist_array_rebin,
            # CWT
            peak_locations,
            peak_locations_rebin,
            # Moving Average
            max_moving_avg,
            max_moving_avg_rebin,
            # Smoothed gaussian
            lin_space_resample,
            smoothed_array,
            max_smoothed_moving_avg,
            # Linear fits
            max_linear_fit_1d,
            max_linear_fit_1d_rebin,
            max_linear_fit_2d,
            max_linear_fit_2d_rebin,
        ) = correlations_helpers.compare_mixed_event_normalization_options(
            mixed_event = mixed_event, eta_limits = eta_limits,
        )

        # Plot the comparison
        plot_correlations.mixed_event_normalization(
            self,
            # For labeling purposes
            hist_name = peak_finding_hist.GetName(),
            eta_limits = eta_limits,
            jet_pt_title = params.generate_jet_pt_range_string(self.jet_pt),
            track_pt_title = params.generate_track_pt_range_string(self.track_pt),
            # Basic data
            lin_space = lin_space,
            peak_finding_hist_array = peak_finding_hist_array,
            lin_space_rebin = lin_space_rebin,
            peak_finding_hist_array_rebin = peak_finding_hist_array_rebin,
            # CWT
            peak_locations = peak_locations,
            peak_locations_rebin = peak_locations_rebin,
            # Moving Average
            max_moving_avg = max_moving_avg,
            max_moving_avg_rebin = max_moving_avg_rebin,
            # Smoothed gaussian
            lin_space_resample = lin_space_resample,
            smoothed_array = smoothed_array,
            max_smoothed_moving_avg = max_smoothed_moving_avg,
            # Linear fits
            max_linear_fit_1d = max_linear_fit_1d,
            max_linear_fit_1d_rebin = max_linear_fit_1d_rebin,
            max_linear_fit_2d = max_linear_fit_2d,
            max_linear_fit_2d_rebin = max_linear_fit_2d_rebin,
        )

    def _measure_mixed_event_normalization(self, mixed_event: Hist) -> float:
        """ Measure the mixed event normalization. """
        # See the note on the selecting the eta_limits in `correlations_helpers.measure_mixed_event_normalization(...)`
        eta_limits = self.task_config["mixedEventNormalizationOptions"].get("etaLimits", [-0.3, 0.3])
        return correlations_helpers.measure_mixed_event_normalization(
            mixed_event = mixed_event,
            eta_limits = eta_limits,
        )

    def _create_2d_raw_and_mixed_correlations(self) -> None:
        """ Generate raw and mixed event 2D correlations. """
        # Project the histograms
        # Includes the trigger, raw signal 2D, and mixed event 2D hists
        for projector in self.sparse_projectors:
            projector.project()

        # Determine number of triggers for the analysis.
        self.number_of_triggers = self._determine_number_of_triggers()

        # Raw signal hist post processing.
        self._post_creation_processing_for_2d_correlation(
            hist = self.correlation_hists_2d.raw,
            normalization_factor = self.number_of_triggers,
            title_label = "Raw signal",
        )

        # Compare mixed event normalization options
        # We must do this before scaling the mixed event (otherwise we will get the wrong scaling values.)
        if self.task_config["mixedEventNormalizationOptions"].get("compareOptions", False):
            self._compoare_mixed_event_normalization_options(
                mixed_event = self.correlation_hists_2d.mixed_event
            )

        # Normalize and post process the mixed event observable
        mixed_event_normalization_factor = self._measure_mixed_event_normalization(
            mixed_event = self.correlation_hists_2d.mixed_event
        )
        self._post_creation_processing_for_2d_correlation(
            hist = self.correlation_hists_2d.mixed_event,
            normalization_factor = mixed_event_normalization_factor,
            title_label = "Mixed event",
        )

    def _create_2d_signal_correlation(self) -> None:
        """ Create 2D signal correlation for raw and mixed correlations.

        This method is intentionally decoupled for creating the raw and mixed event hists so that the
        THnSparse can be swapped out when desired.
        """
        # The signal correlation is the raw signal divided by the mixed events
        self.correlation_hists_2d.signal = self.correlation_hists_2d.raw.Clone(
            self.hist_name_format_2d.format(jet_pt_bin = self.jet_pt.bin, track_pt_bin = self.track_pt.bin, tag = "corr")
        )
        self.correlation_hists_2d.signal.Divide(self.correlation_hists_2d.mixed_event)

        self._post_creation_processing_for_2d_correlation(
            hist = self.correlation_hists_2d.signal,
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
            self._write_trigger_jet_spectra()

            # Ensure we execute the next step
            self.processing_options["generate1DCorrelations"] = True
        else:
            # Initialize the 2D correlations from the file
            logger.info("Loading 2D correlations and trigger jet spectra from file")
            self._init_from_root_file(correlations_2d = True)
            self._init_from_root_file(trigger_jet_spectra = True)

        if self.processing_options["plot2DCorrelations"]:
            logger.info("Plotting 2D correlations")
            plot_correlations.plot_2d_correlations(self)
            logger.info("Plotting RPF example region")
            if self.processing_options["plotRPFHighlights"]:
                plot_correlations.plot_RPF_fit_regions(
                    self,
                    filename = "highlight_RPF_regions_jetPt{self.jet_pt.bin}_trackPt{self.track_pt.bin}"
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
        projection_information = {
            "correlation_type": analysis_objects.JetHCorrelationType.signal_dominated,
            "axis": analysis_objects.JetHCorrelationAxis.delta_phi,
            "jet_pt": self.jet_pt,
            "track_pt": self.track_pt,
        }
        projection_information["tag"] = str(projection_information["correlation_type"])  # type: ignore
        delta_phi_signal_projector = JetHCorrelationProjector(
            observable_to_project_from = self.correlation_hists_2d.signal,
            output_observable = self.correlation_hists_delta_phi,
            output_attribute_name = "signal_dominated",
            projection_name_format = self.hist_name_format_delta_phi,
            projection_information = projection_information,
        )
        # Select signal dominated region in eta
        # Could be a single range, but this is conceptually clearer when compared to the background
        # dominated region. Need to do this as projection dependent cuts because it is selecting different
        # ranges on the same axis
        delta_phi_signal_projector.projection_dependent_cut_axes.append([
            HistAxisRange(
                axis_type = analysis_objects.JetHCorrelationAxis.delta_eta,
                axis_range_name = "negative_eta_signal_dominated",
                min_val = HistAxisRange.apply_func_to_find_bin(
                    ROOT.TAxis.FindBin, -1 * params.eta_bins[params.eta_bins.index(0.6)] + epsilon
                ),
                max_val = HistAxisRange.apply_func_to_find_bin(
                    ROOT.TAxis.FindBin, -1 * params.eta_bins[params.eta_bins.index(0)] - epsilon
                ),
            )
        ])
        delta_phi_signal_projector.projection_dependent_cut_axes.append([
            HistAxisRange(
                axis_type = analysis_objects.JetHCorrelationAxis.delta_eta,
                axis_range_name = "Positive_eta_signal_dominated",
                min_val = HistAxisRange.apply_func_to_find_bin(
                    ROOT.TAxis.FindBin, params.eta_bins[params.eta_bins.index(0)] + epsilon
                ),
                max_val = HistAxisRange.apply_func_to_find_bin(
                    ROOT.TAxis.FindBin, params.eta_bins[params.eta_bins.index(0.6)] - epsilon
                ),
            )
        ])
        delta_phi_signal_projector.projection_axes.append(
            HistAxisRange(
                axis_type = analysis_objects.JetHCorrelationAxis.delta_phi,
                axis_range_name = "delta_phi",
                **full_axis_range
            )
        )
        self.correlation_projectors.append(delta_phi_signal_projector)

        ###########################
        # delta_phi Background dominated
        ###########################
        projection_information = {
            "correlation_type": analysis_objects.JetHCorrelationType.background_dominated,
            "axis": analysis_objects.JetHCorrelationAxis.delta_phi,
            "jet_pt": self.jet_pt,
            "track_pt": self.track_pt,
        }
        projection_information["tag"] = str(projection_information["correlation_type"])  # type: ignore
        delta_phi_background_projector = JetHCorrelationProjector(
            observable_to_project_from = self.correlation_hists_2d.signal,
            output_observable = self.correlation_hists_delta_phi,
            output_attribute_name = "background_dominated",
            projection_name_format = self.hist_name_format_delta_phi,
            projection_information = projection_information,
        )
        # Select background dominated region in eta
        # Redundant to find the index, but it helps check that it is actually in the list!
        # Need to do this as projection dependent cuts because it is selecting different ranges
        # on the same axis
        delta_phi_background_projector.projection_dependent_cut_axes.append([
            HistAxisRange(
                axis_type = analysis_objects.JetHCorrelationAxis.delta_eta,
                axis_range_name = "negative_eta_background_dominated",
                min_val = HistAxisRange.apply_func_to_find_bin(
                    ROOT.TAxis.FindBin, -1 * params.eta_bins[params.eta_bins.index(1.2)] + epsilon
                ),
                max_val = HistAxisRange.apply_func_to_find_bin(
                    ROOT.TAxis.FindBin, -1 * params.eta_bins[params.eta_bins.index(0.8)] - epsilon
                ),
            )
        ])
        delta_phi_background_projector.projection_dependent_cut_axes.append([
            HistAxisRange(
                axis_type = analysis_objects.JetHCorrelationAxis.delta_eta,
                axis_range_name = "positive_eta_background_dominated",
                min_val = HistAxisRange.apply_func_to_find_bin(
                    ROOT.TAxis.FindBin, params.eta_bins[params.eta_bins.index(0.8)] + epsilon
                ),
                max_val = HistAxisRange.apply_func_to_find_bin(
                    ROOT.TAxis.FindBin, params.eta_bins[params.eta_bins.index(1.2)] - epsilon
                ),
            )
        ])
        delta_phi_background_projector.projection_axes.append(
            HistAxisRange(
                axis_type = analysis_objects.JetHCorrelationAxis.delta_phi,
                axis_range_name = "delta_phi",
                **full_axis_range,
            )
        )
        self.correlation_projectors.append(delta_phi_background_projector)

        ###########################
        # delta_eta NS
        ###########################
        projection_information = {
            "correlation_type": analysis_objects.JetHCorrelationType.near_side,
            "axis": analysis_objects.JetHCorrelationAxis.delta_eta,
            "jet_pt": self.jet_pt,
            "track_pt": self.track_pt,
        }
        projection_information["tag"] = str(projection_information["correlation_type"])  # type: ignore
        delta_eta_ns_projector = JetHCorrelationProjector(
            observable_to_project_from = self.correlation_hists_2d.signal,
            output_observable = self.correlation_hists_delta_eta,
            output_attribute_name = "near_side",
            projection_name_format = self.hist_name_format_delta_eta,
            projection_information = projection_information,
        )
        # Select near side in delta phi
        delta_eta_ns_projector.additional_axis_cuts.append(
            HistAxisRange(
                axis_type = analysis_objects.JetHCorrelationAxis.delta_phi,
                axis_range_name = "deltaPhiNearSide",
                min_val = HistAxisRange.apply_func_to_find_bin(ROOT.TAxis.FindBin, params.phi_bins[params.phi_bins.index(-1. * math.pi / 2.)] + epsilon),
                max_val = HistAxisRange.apply_func_to_find_bin(ROOT.TAxis.FindBin, params.phi_bins[params.phi_bins.index(1. * math.pi / 2.)] - epsilon)
            )
        )
        # No projection dependent cut axes
        delta_eta_ns_projector.projection_dependent_cut_axes.append([])
        delta_eta_ns_projector.projection_axes.append(
            HistAxisRange(
                axis_type = analysis_objects.JetHCorrelationAxis.delta_eta,
                axis_range_name = "delta_eta",
                **full_axis_range
            )
        )
        self.correlation_projectors.append(delta_eta_ns_projector)

    def _create_1d_correlations(self) -> None:
        # Project the histograms
        # Includes the delta phi signal dominated, delta phi background dominated, and delta eta near side
        for projector in self.correlation_projectors:
            projector.project()

        # Post process and scale
        # TODO: Add in delta_eta hists
        for correlation_axis, hists in [(analysis_objects.JetHCorrelationAxis.delta_phi, self.correlation_hists_delta_phi.asdict())]:
            for name, hist in hists.items():
                logger.info(f"Post projection processing of 1D correlation {name}")

                # Determine normalization factor
                # We only apply this so we don't unnecessarily scale the signal region.
                # However, it is then important that we report the eta range in which we measure!
                normalization_factor = 1.
                # TODO: IMPORTANT: Remove hard code here and restore proper scaling!
                # Scale is dependent on the signal and background range
                # Since this is hard-coded, it is calculated very explicitly so it will
                # be caught if the values are modified.
                # Ranges are multiplied by 2 because the ranges are symmetric
                signal_min_val = params.eta_bins[params.eta_bins.index(0.0)]
                signal_max_val = params.eta_bins[params.eta_bins.index(0.6)]
                signal_range = (signal_max_val - signal_min_val) * 2.
                background_min_val = params.eta_bins[params.eta_bins.index(0.8)]
                background_max_val = params.eta_bins[params.eta_bins.index(1.2)]
                background_range = (background_max_val - background_min_val) * 2.

                ################
                # If we wanted to plug into the projectors (it would take some work), we could do something like:
                ## Determine the min and max values
                #axis = projector.axis(hist)
                #min_val = rangeSet.min_val(axis)
                #max_val = rangeSet.max_val(axis)
                ## Determine the projection range for proper scaling.
                #projectionRange += (axis.GetBinUpEdge(max_val) - axis.GetBinLowEdge(min_val))
                ################
                # Could also consider trying to get the projector directly and apply it to a hist
                ################

                # TODO: Improve the quality of this check. This is too fragile.
                if "background" in name:
                    # Scale by (signal region)/(background region)
                    # NOTE: Will be applied as `1/normalization_factor`, so the value is the inverse
                    #normalization_factor = background_range/signal_range
                    normalization_factor = background_range
                    logger.debug(f"Scaling background by normalization_factor {normalization_factor}")
                else:
                    normalization_factor = signal_range
                    logger.debug(f"Scaling signal by normalization_factor {normalization_factor}")

                # Post process and scale
                rebin_factor = 2
                title_label = f"{name}, {str(correlation_axis)}"
                self._post_creation_processing_for_1d_correlation(
                    hist = hist,
                    normalization_factor = normalization_factor,
                    rebin_factor = rebin_factor,
                    title_label = title_label,
                    axis_label = params.use_label_with_root(correlation_axis.display_str()),
                )

    def _post_creation_processing_for_1d_correlation(self, hist: Hist,
                                                     normalization_factor: float,
                                                     rebin_factor: int,
                                                     title_label: str,
                                                     axis_label: str):
        """ Basic post processing tasks for a new 1D correlation. """
        correlations_helpers.post_creation_processing_for_1d_correlations(
            hist = hist,
            normalization_factor = normalization_factor,
            rebin_factor = rebin_factor,
            title_label = title_label,
            axis_label = axis_label,
            jet_pt = self.jet_pt,
            track_pt = self.track_pt,
        )

    def _post_1d_projection_scaling(self):
        """ Perform post-projection scaling to avoid needing to scale the fit functions later. """
        # Since the histograms are always referencing the same root object, the stored hists
        # will also be updated.
        # TODO: Include back the delta_eta hists.
        #for hists in [self.correlation_hists_delta_phi, self.correlation_hists_delta_eta]:
        for hists in [self.correlation_hists_delta_phi]:
            for hist in hists.asdict().values():
                logger.debug(f"hist: {hist}")
                correlations_helpers.scale_by_bin_width(hist)

    def _compare_to_other_hist(self, our_hist: Hist, their_hist: Hist):
        # Validation
        for h in [our_hist, their_hist]:
            if not isinstance(h, histogram.Histogram1D):
                h = histogram.Histogram1D.from_existing_hist(h)

        # Make the comparison.
        ...

    def _compare_unsubtracted_1d_signal_correlation_to_joel(self):
        """ Compare Joel's unsubtracted delta phi signal region correlations to mine. """
        map_to_joels_hist_names = {
            params.ReactionPlaneOrientation.all: "all",
            params.ReactionPlaneOrientation.in_plane: "in",
            params.ReactionPlaneOrientation.mid_plane: "mid",
            params.ReactionPlaneOrientation.out_of_plane: "out",
        }

        # Example hist name for all orientations: "allReconstructedSignalwithErrorsNOMnosub"
        joel_hist_name = map_to_joels_hist_names[self.reaction_plane_orientation]
        joel_hist_name += "ReconstructedSignalwithErrorsNOMnosub"

        self._compare_to_other_hist(
            our_hist = self.correlation_hists_delta_phi.signal_dominated,
            theirs = self.comparison_hists[joel_hist_name]
        )

    def _compare_to_joel(self):
        """ Compare 1D correlations against Joel's produced correlations. """
        # Need minus 1 just to conform with convention.
        # TODO: Update
        comparison_track_pt_bin = self.track_pt.bin - 1
        comparison_filename = f"RPF_sysScaleCorrelations{comparison_track_pt_bin}rebinX2bg.root"
        comparison_filename = os.path.join(self.processing_options["joelsCentralCorrelations"], comparison_filename)
        comparison_hists = histogram.get_histograms_in_file(filename = comparison_filename)
        logger.debug(f"{comparison_hists}")

    def _convert_1d_correlations(self):
        """ Convert 1D correlations to Histograms. """
        ...

    def _run_1d_projections(self):
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

            # TODO: Move the methods below back above this line
            if self.processing_options["plot1DCorrelations"]:
                logger.info("Plotting 1D correlations")
                plot_correlations.plot_1d_correlations(self)

            # TODO: Can only compare after fit...
            #self._compare_to_joel()

            # TODO: Uncomment
            # Create hist arrays
            #self._convert_1d_correlations()

            # Write the properly scaled projections
            #self.write1DCorrelations()

            # Ensure that the next step in the chain is run
            self.processing_options["fit1DCorrelations"] = True
        else:
            # Initialize the 1D correlations from the file
            logger.info("Loading 1D correlations from file")
            self._init_from_root_file(correlations_1d = True)
            self._init_from_root_file(correlations_1d_array = True, exitOnFailure = False)

        self.ran_projections = True

    def run_projections(self):
        """ Run all analysis steps through projectors. """
        self._run_2d_projections()
        self._run_1d_projections()

class CorrelationsManager(generic_class.EqualityMixin):
    def __init__(self, config_filename: str, selected_analysis_options: params.SelectedAnalysisOptions, **kwargs):
        self.config_filename = config_filename
        self.selected_analysis_options = selected_analysis_options

        # Create the actual analysis objects.
        self.analyses: Mapping[Any, Correlations]
        (self.key_index, self.selected_iterables, self.analyses) = self.construct_correlations_from_configuration_file()

        # General histograms
        self.general_histograms = GeneralHistogramsManager(
            config_filename = self.config_filename,
            selected_analysis_options = self.selected_analysis_options
        )

    def construct_correlations_from_configuration_file(self) -> analysis_config.ConstructedObjects:
        """ Construct Correlations objects based on iterables in a configuration file. """
        return analysis_config.construct_from_configuration_file(
            task_name = "Correlations",
            config_filename = self.config_filename,
            selected_analysis_options = self.selected_analysis_options,
            additional_possible_iterables = {"pt_hard_bin": None, "jet_pt_bin": None, "track_pt_bin": None},
            obj = Correlations,
        )

    def setup(self):
        """ Setup the correlations manager. """
        # Retrieve input histograms (with caching).
        input_hists = {}
        for key_index, analysis in analysis_config.iterate_with_selected_objects(self.analyses):
            # We should now have all RP orientations.
            # We are effectively caching the values here.
            if not input_hists:
                input_hists = histogram.get_histograms_in_file(filename = analysis.input_filename)
            logger.debug(f"{key_index}")
            # Setup input histograms and projctors.
            analysis.setup(input_hists = input_hists)

    def run(self) -> bool:
        """ Run the analysis in the correlations manager. """
        # First setup the correlations
        self.setup()

        # Run the general hists
        #self.general_histograms.run()

        # First analysis step
        for key_index, analysis in analysis_config.iterate_with_selected_objects(self.analyses):
            analysis.run_projections()

        # Fitting
        # To successfully fit, we need all histograms from a given reaction plane orientation.
        # TODO: Investigate whether mypy is right here...
        #for jet_pt in self.selected_analysis_options["jet_pt_bin"]:  # type: ignore
        #    for track_pt in self.selected_analysis_options["track_pt_bin"]:  # type: ignore
        #        logger.debug(f"Selection: jet_pt: {jet_pt}, track_pt: {track_pt}")
        #        for key_index, analysis in \
        #                analysis_config.iterate_with_selected_objects(self.analyses, jet_pt_bin = jet_pt, track_pt_bin = track_pt):
        #            logger.debug(f"{key_index}")

        return True

def run_from_terminal():
    """ Driver function for running the correlations analysis. """
    # Basic setup
    logging.basicConfig(level = logging.DEBUG)
    # Quiet down the matplotlib logging
    logging.getLogger("matplotlib").setLevel(logging.INFO)
    # Quiet down pachyderm
    logging.getLogger("pachyderm").setLevel(logging.INFO)

    # Turn off stats box
    ROOT.gStyle.SetOptStat(0)

    # Setup the analysis
    (config_filename, terminal_args, additional_args) = analysis_config.determine_selected_options_from_kwargs(
        task_name = "Correlations"
    )
    analysis_manager = CorrelationsManager(
        config_filename = config_filename,
        selected_analysis_options = terminal_args
    )
    # Finally run the analysis.
    analysis_manager.run()

    # Return it for convenience.
    return analysis_manager

if __name__ == "__main__":
    run_from_terminal()

