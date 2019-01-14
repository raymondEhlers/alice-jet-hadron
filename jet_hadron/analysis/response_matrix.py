#!/usr/bin/env python

""" Create the response matrix, with proper scaling by pt hard bins

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import argparse
import collections
import ctypes
import enum
import IPython
import logging
import numpy as np
import os
import pprint
import re
import ruamel.yaml as yaml
import seaborn as sns
import sys

from pachyderm import histogram
from pachyderm import projectors
from pachyderm.projectors import HistAxisRange
from pachyderm import utils

from jet_hadron.base import analysis_objects
from jet_hadron.base.params import ReactionPlaneOrientation

import ROOT
# Tell ROOT to ignore command line options so args are passed to python
# NOTE: Must be immediately after import ROOT!
ROOT.PyConfig.IgnoreCommandLineOptions = True

# Configure ROOT
# Run in batch mode
ROOT.gROOT.SetBatch(True)
# Disable stats box
ROOT.gStyle.SetOptStat(False)

# Setup logger
logger = logging.getLogger(__name__)

class JetResponseMakerMatchingSparse(enum.Enum):
    """ Defines the axes in the AliJetResponseMaker fMatching THnSparse. """
    kDetLevelJetPt = 0
    kPartLevelJetPt = 1
    kProjectionDistance = 4
    kDetLevelLeadingParticle = 7
    kPartLevelLeadingParticle = 8
    kDetLevelReactionPlaneOrientation = 9
    kPartLevelReactionPlaneOrientation = 10

class JetResponseMakerJetsSparse(enum.Enum):
    """ Defines the axes in the AliJetResponseMaker fJets THnSparse """
    kPhi = 0
    kEta = 1
    kJetPt = 2
    kJetArea = 3
    # Different if the event plane is included in the output or not!!
    kReactionPlaneOrientation = 4
    kLeadingParticlePP = 4
    kLeadingParticlePbPb = 5

class EventActivity(enum.Enum):
    kUndefined = -1
    kCentral = 0
    kSemiCentral = 2
    kPeripheral = 3
    kpp = 0

    def __str__(self):
        """ Turns kSemiCentral into "semi_central" """
        tempStr = self.filenameStr()
        tempStr = tempStr[:1].lower() + tempStr[1:]
        return tempStr

    def filenameStr(self):
        """ Turns kSemiCentral into "SemiCentral" """
        return self.name.replace("k", "", 1)

    def displayStr(self):
        """ Turns kSemiCentral into "Semi-Central". """
        tempStr = self.filenameStr()
        tempList = re.findall('[A-Z][^A-Z]*', tempStr)
        return "-".join(tempList)

class RMNormalizationType(enum.Enum):
    """ Selects the type of normalization to apply to the RM """
    kNone = 0
    kNormalizeEachDetectorBin = 1
    kNormalizeEachTruthBin = 2

    def __str__(self):
        """ Return the name of the value with the appended "k". This is just a convenience function """
        if self.name == "kNone":
            tempStr = self.name.replace("kNone", "no", 1)
        else:
            tempStr = self.name.replace("kNormalizeEach", "", 1)
        tempStr = tempStr[:1].lower() + tempStr[1:] + "Normalization"
        return str(tempStr)

class JetHResponseMatrixProjector(projectors.HistProjector):
    """ Projector for the Jet-h response matrix THnSparse. """
    def ProjectionName(self, **kwargs):
        """ Define the projection name for the JetH RM projector """
        ptHardBin = kwargs["inputKey"]
        hist = kwargs["inputHist"]
        logger.debug("Projecting pt hard bin: {0}, hist: {1}, projectionName: {2}".format(ptHardBin, hist.GetName(), self.projectionNameFormat.format(ptHardBin = ptHardBin)))
        return self.projectionNameFormat.format(ptHardBin = ptHardBin)

    def OutputKeyName(self, inputKey, outputHist, *args, **kwargs):
        """ Retrun the input key, which is the pt hard bin"""
        return inputKey

class JetHResponseMatrix(object):
    """ Jet H response matrix. """

    def __init__(self, configFile, productionRootFile, collisionSystem = None, useFloatHists = None, outputPath = None, responseMatrixBaseName = None, clusterBias = None, *args, **kwargs):
        # Report if unexpected values are passed (although this isn't necessary a problem)
        if args:
            for arg in args:
                logger.warning("Received unexpected ordered arg {}".format(args))
        if kwargs:
            for key, val in kwargs.iteritems():
                logger.warning("Received unexpected keyword arg {}: {}".format(key, val))

        # Required values
        # Load configuration
        if not configFile:
            raise ValueError("Must pass a valid config file!")
        with open(configFile, "rb") as f:
            self.config = yaml.safe_load(f)
        self.productionRootFile = productionRootFile

        if collisionSystem is None:
            collisionSystem = self.config.get("collisionSystem", str(analysis_objects.CollisionSystem.kNA))
        if collisionSystem[:1] != "k":
            collisionSystem = "k" + collisionSystem
        self.collisionSystem = analysis_objects.CollisionSystem[collisionSystem]

        if useFloatHists is None:
            useFloatHists = self.config.get("useFloatHists", False)
        self.useFloatHists = useFloatHists

        if responseMatrixBaseName is None:
            responseMatrixBaseName = self.config.get("responseMatrixBaseName", "JESCorrection")
        self.responseMatrixBaseName = responseMatrixBaseName

        if clusterBias is None:
            clusterBias = self.config.get("clusterBias", 6)
        self.clusterBias = clusterBias

        # Only configurable via the config file
        self.afterEventSelection = self.config.get("scaleFactorsFromAfterEventSelection", False)
        self.eventPlaneSelection = self.config.get("eventPlaneSelection", None)
        if self.eventPlaneSelection:
            # Get the proper value from the enum
            # The enum translates this to the bin number
            self.eventPlaneSelection = ReactionPlaneOrientation[self.eventPlaneSelection]

        self.rmNormalizationType = RMNormalizationType[self.config.get("rmNormalizationType", "kNone")]
        self.eventActivity = EventActivity[self.config.get("eventActivity")]

        # Usually it is a pt hard bin production
        # If not, we need to handle it a bit more carefully
        self.ptHardBinProduction = self.config.get("ptHardBinProduction", True)

        if outputPath is None:
            outputPath = self.config.get("outputPath", "")
            outputPath = outputPath.format(eventActivity = str(self.eventActivity),
                                           rmNormalizationType = str(self.rmNormalizationType))
        # Create the default path if value is not set
        if outputPath == "":
            # Path to current directory
            # Using __file__ to ensure that it can be run from anywhere
            currentDirectory = os.path.dirname(os.path.realpath(__file__))
            outputPath = os.path.join(currentDirectory, "output", "responseMatrix", str(self.collisionSystem), str(self.rmNormalizationType))

        if not os.path.exists(outputPath):
            os.makedirs(outputPath)

        self.outputPath = outputPath

        # Define histograms
        self.hists = {"responseMatrix": None, "responseMatrixErrors": None,
                      "responseMatrixPtHard": collections.OrderedDict(),
                      "responseMatrixPtHardSparse": collections.OrderedDict(),
                      "sampleTaskJetSpectraPartLevel": None,
                      "sampleTaskJetSpectraPartLevelPtHard": collections.OrderedDict(),
                      "unmatchedJetSpectraPartLevel": None,
                      "unmatchedJetSpectraPartLevelPtHard": collections.OrderedDict(),
                      "unmatchedPartLevelJetsPtHardSparse": collections.OrderedDict(),
                      "jetSpectraPartLevel": None,
                      "jetSpectraPartLevelPtHard": collections.OrderedDict(),
                      "partLevelJetsPtHardSparse": collections.OrderedDict(),
                      "unmatchedJetSpectraDetLevel": None,
                      "unmatchedJetSpectraDetLevelPtHard": collections.OrderedDict(),
                      "unmatchedDetLevelJetsPtHardSparse": collections.OrderedDict(),
                      "jetSpectraDetLevel": None,
                      "jetSpectraDetLevelPtHard": collections.OrderedDict(),
                      "detLevelJetsPtHardSparse": collections.OrderedDict(),
                      "ptHardSpectra": None, "ptHardSpectraAfterEventSelection": None,
                      "ptHardSpectraPtHard": collections.OrderedDict(),
                      "ptHardSpectraAfterEventSelectionPtHard": collections.OrderedDict(),
                      "crossSection": None, "crossSectionAfterEventSelection": None,
                      "crossSectionPtHard": collections.OrderedDict(),
                      "crossSectionAfterEventSelectionPtHard": collections.OrderedDict(),
                      "nTrials": None, "nTrialsAfterEventSelection": None,
                      "nTrialsPtHard": collections.OrderedDict(),
                      "nTrialsAfterEventSelectionPtHard": collections.OrderedDict(),
                      "nEvents": None, "nEventsAfterEventSelection": None,
                      "nEventsPtHard": collections.OrderedDict(),
                      "nEventsAfterEventSelectionPtHard": collections.OrderedDict()}

        # Scale factors
        self.scaleFactors = collections.OrderedDict()
        # Projectors
        self.projectors = []
        self.postScaleAndMergeProjectors = []

    def Analyze(self):
        """ Run the main analysis. """
        self.Initialize()
        self.GetHists()
        if self.ptHardBinProduction:
            self.GetScaleFactors()
        self.Process(scaleHists = True)

    def SaveAndPrint(self):
        """ Save and print the hists. """
        self.PlotHists(plotIndividualPtHardBins = True)
        self.SaveRootFile()

    def Initialize(self):
        """ Setup the Response Matrix. Includes definition of projectors, etc. """
        # Define projectors

        # Helper range
        fullAxisRange = {"minVal": HistAxisRange.ApplyFuncToFindBin(None, 1),
                         "maxVal": HistAxisRange.ApplyFuncToFindBin(ROOT.TAxis.GetNbins)}

        #################
        # Response matrix
        #################
        responseMatrixProjector = JetHResponseMatrixProjector(
            observable_dict = self.hists["responseMatrixPtHard"],
            observables_to_project_from = self.hists["responseMatrixPtHardSparse"],
            projectionNameFormat = "responseMatrixPtHard_{ptHardBin}"
        )
        responseMatrixProjector.additionalAxisCuts.append(
            HistAxisRange(
                axisType = JetResponseMakerMatchingSparse.kDetLevelLeadingParticle,
                axisRangeName = "detLevelLeadingParticle",
                minVal = HistAxisRange.ApplyFuncToFindBin(ROOT.TAxis.FindBin, self.clusterBias),
                maxVal = HistAxisRange.ApplyFuncToFindBin(ROOT.TAxis.GetNbins)
            )
        )
        if self.eventPlaneSelection:
            logger.debug("self.eventPlaneSelection.value: {0}".format(self.eventPlaneSelection.value))
            if self.eventPlaneSelection == ReactionPlaneOrientation.kAll:
                eventPlaneAxisRange = fullAxisRange
                logger.info("Using full EP angle range")
            else:
                eventPlaneAxisRange = {
                    "minVal": HistAxisRange.ApplyFuncToFindBin(None, self.eventPlaneSelection.value),
                    "maxVal": HistAxisRange.ApplyFuncToFindBin(None, self.eventPlaneSelection.value)
                }
                logger.info(f"Using selected EP angle range {self.eventPlaneSelection.name}")

            eventPlaneSelectionProjectorAxis = HistAxisRange(
                axisType = JetResponseMakerMatchingSparse.kDetLevelReactionPlaneOrientation,
                axisRangeName = "detLevelReactionPlaneOrientation",
                **eventPlaneAxisRange
            )
            responseMatrixProjector.additionalAxisCuts.append(eventPlaneSelectionProjectorAxis)

        # No additional cuts for the projection dependent axes
        responseMatrixProjector.projectionDependentCutAxes.append([])
        responseMatrixProjector.projectionAxes.append(
            HistAxisRange(
                axisType = JetResponseMakerMatchingSparse.kDetLevelJetPt,
                axisRangeName = "detLevelJetPt",
                **fullAxisRange
            )
        )
        responseMatrixProjector.projectionAxes.append(
            HistAxisRange(
                axisType = JetResponseMakerMatchingSparse.kPartLevelJetPt,
                axisRangeName = "partLevelJetPt",
                **fullAxisRange
            )
        )
        # Save the projector for later use
        self.projectors.append(responseMatrixProjector)

        ###################
        # Unmatched part level jet pt
        ###################
        unmatchedPartLevelJetSpectraProjector = JetHResponseMatrixProjector(
            observable_dict = self.hists["unmatchedJetSpectraPartLevelPtHard"],
            observables_to_project_from = self.hists["unmatchedPartLevelJetsPtHardSparse"],
            projectionNameFormat = "unmatchedJetSpectraPartLevelPtHard_{ptHardBin}"
        )
        # Can't apply a leading cluster cut on part level, since we don't have clusters
        unmatchedPartLevelJetSpectraProjector.projectionDependentCutAxes.append([])
        unmatchedPartLevelJetSpectraProjector.projectionAxes.append(
            HistAxisRange(
                axisType = JetResponseMakerJetsSparse.kJetPt,
                axisRangeName = "unmatchedPartLevelJetSpectra",
                **fullAxisRange
            )
        )
        # Save the projector for later use
        self.projectors.append(unmatchedPartLevelJetSpectraProjector)

        ###################
        # (Matched) Part level jet pt
        ###################
        partLevelJetSpectraProjector = JetHResponseMatrixProjector(
            observable_dict = self.hists["jetSpectraPartLevelPtHard"],
            observables_to_project_from = self.hists["responseMatrixPtHardSparse"],
            projectionNameFormat = "jetSpectraPartLevelPtHard_{ptHardBin}"
        )
        if self.eventPlaneSelection:
            partLevelJetSpectraProjector.additionalAxisCuts.append(eventPlaneSelectionProjectorAxis)
        # Can't apply a leading cluster cut on part level, since we don't have clusters
        partLevelJetSpectraProjector.projectionDependentCutAxes.append([])
        partLevelJetSpectraProjector.projectionAxes.append(
            HistAxisRange(
                axisType = JetResponseMakerMatchingSparse.kPartLevelJetPt,
                axisRangeName = "partLevelJetSpectra",
                **fullAxisRange
            )
        )
        # Save the projector for later use
        self.projectors.append(partLevelJetSpectraProjector)

        ##################
        # Unmatched det level jet pt
        ##################
        unmatchedDetLevelJetSpectraProjector = JetHResponseMatrixProjector(
            observable_dict = self.hists["unmatchedJetSpectraDetLevelPtHard"],
            observables_to_project_from = self.hists["unmatchedDetLevelJetsPtHardSparse"],
            projectionNameFormat = "unmatchedJetSpectraDetLevelPtHard_{ptHardBin}"
        )
        unmatchedDetLevelJetSpectraProjector.additionalAxisCuts.append(
            HistAxisRange(
                axisType = JetResponseMakerJetsSparse.kLeadingParticlePbPb if self.collisionSystem == analysis_objects.CollisionSystem.kPbPb else JetResponseMakerJetsSparse.kLeadingParticlePP,
                axisRangeName = "unmatchedDetLevelLeadingParticle",
                minVal = HistAxisRange.ApplyFuncToFindBin(ROOT.TAxis.FindBin, self.clusterBias),
                maxVal = HistAxisRange.ApplyFuncToFindBin(ROOT.TAxis.GetNbins)
            )
        )
        unmatchedDetLevelJetSpectraProjector.projectionDependentCutAxes.append([])
        unmatchedDetLevelJetSpectraProjector.projectionAxes.append(
            HistAxisRange(
                axisType = JetResponseMakerJetsSparse.kJetPt,
                axisRangeName = "unmatchedDetLevelJetSpectra",
                **fullAxisRange
            )
        )
        # Save the projector for later use
        self.projectors.append(unmatchedDetLevelJetSpectraProjector)

        ##################
        # (Matched) Det level jet pt
        ##################
        detLevelJetSpectraProjector = JetHResponseMatrixProjector(
            observable_dict = self.hists["jetSpectraDetLevelPtHard"],
            observables_to_project_from = self.hists["responseMatrixPtHardSparse"],
            projectionNameFormat = "jetSpectraDetLevelPtHard_{ptHardBin}"
        )
        detLevelJetSpectraProjector.additionalAxisCuts.append(
            HistAxisRange(
                axisType = JetResponseMakerMatchingSparse.kDetLevelLeadingParticle,
                axisRangeName = "detLevelLeadingParticle",
                minVal = HistAxisRange.ApplyFuncToFindBin(ROOT.TAxis.FindBin, self.clusterBias),
                maxVal = HistAxisRange.ApplyFuncToFindBin(ROOT.TAxis.GetNbins)
            )
        )
        if self.eventPlaneSelection:
            detLevelJetSpectraProjector.additionalAxisCuts.append(eventPlaneSelectionProjectorAxis)
        detLevelJetSpectraProjector.projectionDependentCutAxes.append([])
        detLevelJetSpectraProjector.projectionAxes.append(
            HistAxisRange(axisType = JetResponseMakerMatchingSparse.kDetLevelJetPt,
                          axisRangeName = "detLevelJetSpectra", **fullAxisRange)
        )
        # Save the projector for later use
        self.projectors.append(detLevelJetSpectraProjector)

    def GetHists(self):
        """ Get the histogarms needed to create the response matrix. """
        self.RetrieveHists()

    def GetScaleFactors(self):
        """ Handle retrieving scale factors. """
        # Attempt to extract scale factors automatically
        success = self.GetPredefinedScaleFactors()
        if success:
            logger.info("Using predefined scale factors")
        else:
            logger.info("Extracting scale factors {0}".format("after event selection" if self.afterEventSelection else ""))
            success = self.ExtractScaleFactors()
            if success:
                logger.info("Using extracted scale factors")

        # If we somehow got the scale factors, relative scaling to account for the different number of events in a particular analysis
        # results file should be applied
        if success:
            logger.debug("Scale factors before relative scaling: {0}".format(self.scaleFactors))
            self.ExtractAndApplyRelativePtHardBinScaleFactors()
            logger.debug("Scale factors  after relative scaling: {0}".format(self.scaleFactors))
            return

        # If we were unsuccessful, then fail
        logger.critical("Could not get pre-defined or extracted scale factors! Please check the train output or manually define scale factors")
        sys.exit(1)

    def GetPredefinedScaleFactors(self):
        """ Attempt to extract scale factors from the configuration. """
        # If unsuccessful, see if it's already been defined
        scaleFactors = self.config.get("scaleFactors", None)
        if scaleFactors:
            # Assign scale factors from the configuration
            # NOTE: that this assumes that the scale factors are ordered by pt hard bin (which is not so unreasonable)
            for i, factor in enumerate(scaleFactors):
                logger.debug("ptHardBin: {0}, scaleFactor: {1}".format(i + 1, factor))
                self.scaleFactors[str(i + 1)] = factor
            return True

        return False

    def Process(self, scaleHists = True):
        """ Main processing of the response matrix """
        self.SetSumW2()
        self.ProjectSparses()

        #if not testMatrix:
        if scaleHists:
            # Properly format matrix name
            self.ScaleAndMergeSelectedHists()

        self.ProjectPostScaledAndMergedHists()

        #if (not testMatrix) or (testMatrix and testMatrixWeight == 1):
        if self.rmNormalizationType != RMNormalizationType.kNone:
            # Normalize response matrix
            normalizeResponseMatrix(hist = self.hists["responseMatrix"], rmNormalizationType = self.rmNormalizationType)

            # Check that the normalizaiton was successful
            checkNormalization(hist = self.hists["responseMatrix"], rmNormalizationType = self.rmNormalizationType, useFloatHists = self.useFloatHists)
        else:
            # Skip normalization, since we have a desired weight
            pass

        # Create response matrix errors
        self.CreateResponseMatrixErrors()

    def SetSumW2(self):
        """ Set SumW2 on the THn hists before using them for anything. """

        for hists in [self.hists["unmatchedDetLevelJetsPtHardSparse"], self.hists["unmatchedPartLevelJetsPtHardSparse"], self.hists["responseMatrixPtHardSparse"]]:
            for hist in hists.itervalues():
                hist.Sumw2()

    def ProjectPostScaledAndMergedHists(self):
        """ Project histograms after scaling and merging pt hard hists, but before any overall normalization is applied. """
        # Initialize the dicts for the input and output
        inputDict = {"partSpectraProjection": self.hists["responseMatrix"]}
        outputDict = {}

        # Define the projector
        # This is admittedly a bit more complicated than a standard projection, but it ensures that we
        # are using consistent methods throughout, which should reduce bugs.
        partSpectraProjector = JetHResponseMatrixProjector(
            observable_dict = outputDict,
            observables_to_project_from = inputDict,
            projectionNameFormat = "partSpectraProjection"
        )
        partSpectraProjector.additionalAxisCuts.append(HistAxisRange(
            axisType = projectors.TH1AxisType.xAxis,
            axisRangeName = "partSpectraProjectionDetLimits",
            minVal = HistAxisRange.ApplyFuncToFindBin(ROOT.TAxis.FindBin, 20),
            maxVal = HistAxisRange.ApplyFuncToFindBin(ROOT.TAxis.FindBin, 40))
        )
        partSpectraProjector.projectionDependentCutAxes.append([])
        partSpectraProjector.projectionAxes.append(HistAxisRange(
            axisType = projectors.TH1AxisType.yAxis,
            axisRangeName = "partSpectraProjection",
            minVal = HistAxisRange.ApplyFuncToFindBin(None, 1),
            maxVal = HistAxisRange.ApplyFuncToFindBin(ROOT.TAxis.GetNbins))
        )

        # Perform the actual projection
        partSpectraProjector.Project()

        # Retrieve projection output and scale properly
        self.hists["partSpectraProjection"] = next(outputDict.itervalues())
        # Scale because we project over 20 1 GeV bins
        self.hists["partSpectraProjection"].Scale(1.0 / 20.0)
        self.hists["partSpectraProjection"].SetDirectory(0)
        logger.debug("N jets in partSpectraProjection: {}".format(self.hists["partSpectraProjection"].Integral()))

    def PlotHists(self, plotIndividualPtHardBins = True, responseMatrixSaveName = "responseMatrix"):
        """ Plot histograms related to the response matrix and its creation. """
        canvas = ROOT.TCanvas("canvas", "canvas")

        # Plot spectra
        if plotIndividualPtHardBins:
            # So it can be restored later
            normalRightMargin = canvas.GetRightMargin()
            # Smaller on right since it is empty
            canvas.SetRightMargin(0.03)
            # Samller title area
            canvas.SetTopMargin(0.08)

            # Log Y
            canvas.SetLogy(1)

            # Pt hard binning
            ptHardBinning = self.config.get("ptHardBinning", [])

            if self.ptHardBinProduction:
                # Pt Hard Spectra
                self.hists["ptHardSpectra"].SetTitle("#mathit{p}_{#mathrm{T}} Hard Spectra")
                self.hists["ptHardSpectra"].GetXaxis().SetTitle("#mathit{p}_{#mathrm{T}}^{Hard}")
                self.hists["ptHardSpectra"].GetYaxis().SetTitle("#frac{dN}{d#mathit{p}_{#mathrm{T}}^{Hard}}")
                self.hists["ptHardSpectra"].GetYaxis().SetTitleOffset(1.2)
                plot1DPtHardHists(self.hists["ptHardSpectra"], self.hists["ptHardSpectraPtHard"], canvas, outputPath = self.outputPath, ptHardBinning = ptHardBinning)
                # Pt Hard Spectra after event selection
                self.hists["ptHardSpectraAfterEventSelection"].SetTitle("#mathit{p}_{#mathrm{T}} Hard Spectra After Event Selection")
                self.hists["ptHardSpectraAfterEventSelection"].GetXaxis().SetTitle("#mathit{p}_{#mathrm{T}}^{Hard}")
                self.hists["ptHardSpectraAfterEventSelection"].GetYaxis().SetTitle("#frac{dN}{d#mathit{p}_{#mathrm{T}}^{Hard}}")
                self.hists["ptHardSpectraAfterEventSelection"].GetYaxis().SetTitleOffset(1.2)
                plot1DPtHardHists(self.hists["ptHardSpectraAfterEventSelection"], self.hists["ptHardSpectraAfterEventSelectionPtHard"], canvas, outputPath = self.outputPath, ptHardBinning = ptHardBinning)
            # Sample Task Jet Spectra - Part Level
            self.hists["sampleTaskJetSpectraPartLevel"].SetTitle("Sample task particle level jet #mathit{p}_{#mathrm{T}}")
            self.hists["sampleTaskJetSpectraPartLevel"].GetXaxis().SetTitle("#mathit{p}_{#mathrm{T,jet}}^{part}")
            self.hists["sampleTaskJetSpectraPartLevel"].GetYaxis().SetTitle("#frac{dN}{d#mathit{p}_{#mathrm{T}}}")
            self.hists["sampleTaskJetSpectraPartLevel"].GetYaxis().SetTitleOffset(1.2)
            self.hists["sampleTaskJetSpectraPartLevel"].GetXaxis().SetRangeUser(0, 150)
            plot1DPtHardHists(self.hists["sampleTaskJetSpectraPartLevel"], self.hists["sampleTaskJetSpectraPartLevelPtHard"], canvas, outputPath = self.outputPath, ptHardBinning = ptHardBinning)
            # Unmatched Jet Spectra - Part Level
            self.hists["unmatchedJetSpectraPartLevel"].SetTitle("Unmatched particle level jet #mathit{p}_{#mathrm{T}}")
            self.hists["unmatchedJetSpectraPartLevel"].GetXaxis().SetTitle("#mathit{p}_{#mathrm{T,jet}}^{part}")
            self.hists["unmatchedJetSpectraPartLevel"].GetYaxis().SetTitle("#frac{dN}{d#mathit{p}_{#mathrm{T}}}")
            self.hists["unmatchedJetSpectraPartLevel"].GetYaxis().SetTitleOffset(1.2)
            plot1DPtHardHists(self.hists["unmatchedJetSpectraPartLevel"], self.hists["unmatchedJetSpectraPartLevelPtHard"], canvas, outputPath = self.outputPath, ptHardBinning = ptHardBinning)
            # (Matched) Jet Spectra - Part Level
            self.hists["jetSpectraPartLevel"].SetTitle("Particle level jet #mathit{p}_{#mathrm{T}}")
            self.hists["jetSpectraPartLevel"].GetXaxis().SetTitle("#mathit{p}_{#mathrm{T,jet}}^{part}")
            self.hists["jetSpectraPartLevel"].GetYaxis().SetTitle("#frac{dN}{d#mathit{p}_{#mathrm{T}}}")
            self.hists["jetSpectraPartLevel"].GetYaxis().SetTitleOffset(1.2)
            plot1DPtHardHists(self.hists["jetSpectraPartLevel"], self.hists["jetSpectraPartLevelPtHard"], canvas, outputPath = self.outputPath, ptHardBinning = ptHardBinning)
            # Unmatched Jet Spectra - Det Level
            self.hists["unmatchedJetSpectraDetLevel"].SetTitle("Unmatched detector level jet #mathit{p}_{#mathrm{T}}")
            self.hists["unmatchedJetSpectraDetLevel"].GetXaxis().SetTitle("#mathit{p}_{#mathrm{T,jet}}^{det}")
            self.hists["unmatchedJetSpectraDetLevel"].GetYaxis().SetTitle("#frac{dN}{d#mathit{p}_{#mathit{T}}}")
            self.hists["unmatchedJetSpectraDetLevel"].GetYaxis().SetTitleOffset(1.2)
            plot1DPtHardHists(self.hists["unmatchedJetSpectraDetLevel"], self.hists["unmatchedJetSpectraDetLevelPtHard"], canvas, outputPath = self.outputPath, ptHardBinning = ptHardBinning)
            # (Matched) Jet Spectra - Det Level
            self.hists["jetSpectraDetLevel"].SetTitle("Detector level jet #mathit{p}_{#mathrm{T}}")
            self.hists["jetSpectraDetLevel"].GetXaxis().SetTitle("#mathit{p}_{#mathrm{T,jet}}^{det}")
            self.hists["jetSpectraDetLevel"].GetYaxis().SetTitle("#frac{dN}{d#mathit{p}_{#mathrm{T}}}")
            self.hists["jetSpectraDetLevel"].GetYaxis().SetTitleOffset(1.2)
            plot1DPtHardHists(self.hists["jetSpectraDetLevel"], self.hists["jetSpectraDetLevelPtHard"], canvas, outputPath = self.outputPath, ptHardBinning = ptHardBinning)
            canvas.SetLogy(0)
            canvas.Clear()

            # Restore the right margin for the scale on the 2D plots
            canvas.SetRightMargin(normalRightMargin)

        # Plot response matrix
        canvas.SetLogz(1)
        responseMatrix = self.hists["responseMatrix"]
        logger.debug("Response matrix n jets: {}".format(responseMatrix.Integral()))
        responseMatrix.SetTitle("Response Matrix")
        responseMatrix.GetXaxis().SetTitle("#mathit{p}_{#mathrm{T,jet}}^{det} (GeV/#it{c})")
        responseMatrix.GetYaxis().SetTitle("#mathit{p}_{#mathrm{T,jet}}^{part} (GeV/#it{c})")
        responseMatrix.Draw("colz")
        minVal = ctypes.c_double(0)
        maxVal = ctypes.c_double(0)
        responseMatrix.GetMinimumAndMaximum(minVal, maxVal)
        # * 1.1 to put it slightly above the max value
        # minVal doesn't work here, because there are some entries at 0
        responseMatrix.GetZaxis().SetRangeUser(10e-7, maxVal.value * 1.1)
        canvas.SaveAs(os.path.join(self.outputPath, "{0}.pdf".format(responseMatrixSaveName)))
        canvas.SetLogz(0)

        # Truth spectra projection
        proj = self.hists["partSpectraProjection"]
        #proj.Scale(1.0/proj.Integral(0, 100))
        proj.SetTitle("")
        proj.GetXaxis().SetTitle("#mathit{p}_{#mathrm{T,jet}}^{part} (GeV/#it{c})")
        proj.GetYaxis().SetTitle("#frac{dN}{d#mathit{p}_{#mathrm{T}}}")

        # Draw ALICE Information
        latex = []
        latex.append(ROOT.TLatex(.65, .8, "ALICE"))
        latex.append(ROOT.TLatex(.58, .72, "20 < #mathit{p}_{#mathrm{T}}^{det} < 40 GeV/#it{c}"))

        # Draw
        proj.Draw()
        for el in latex:
            el.SetNDC()
            el.Draw()
        # Linear
        canvas.SetLogy(0)
        canvas.SaveAs(os.path.join(self.outputPath, "{0}.pdf".format("partPtLimitedRangeProjection")))
        # Log
        canvas.SetLogy(1)
        canvas.SaveAs(os.path.join(self.outputPath, "{0}.pdf".format("partPtLimitedRangeProjectionLog")))
        # Reset
        canvas.SetLogy(0)

        # Plot response matrix errors
        responseMatrixErrors = self.hists["responseMatrixErrors"]
        responseMatrixErrors.SetTitle("Response Matrix Relative Statistical Errors")
        responseMatrixErrors.GetXaxis().SetTitle("#mathit{p}_{#mathrm{T,jet}}^{det}")
        responseMatrixErrors.GetYaxis().SetTitle("#mathit{p}_{#mathrm{T,jet}}^{part}")
        responseMatrixErrors.Draw("colz")
        canvas.SaveAs(os.path.join(self.outputPath, "{0}Errors.pdf".format(responseMatrixSaveName)))

    def SaveRootFile(self):
        """ Save the processed histograms to a ROOT file. """
        logger.info("Saving root file to {0}".format(os.path.join(self.outputPath, "{0}.root".format(self.responseMatrixBaseName))))
        outputFilename = os.path.join(self.outputPath, "{0}.root".format(self.responseMatrixBaseName))
        with histogram.RootOpen(outputFilename, "RECREATE") as fOut:  # noqa: F841
            self.hists["responseMatrix"].Write()
            self.hists["responseMatrixErrors"].Write()
            if not self.productionRootFile:
                if self.ptHardBinProduction:
                    self.hists["ptHardSpectra"].Write()
                self.hists["partSpectraProjection"].Write()
                self.hists["sampleTaskJetSpectraPartLevel"].Write()
                self.hists["jetSpectraPartLevel"].Write()
                self.hists["jetSpectraDetLevel"].Write()

                for dictOfHists in [self.hists["ptHardSpectraPtHard"], self.hists["sampleTaskJetSpectraPartLevelPtHard"], self.hists["jetSpectraPartLevelPtHard"], self.hists["jetSpectraDetLevelPtHard"], self.hists["responseMatrixPtHard"]]:
                    for ptHardBin, hist in dictOfHists.iteritems():
                        hist.Write()

    def InitFromRootFile(self):
        """ Retrieve processed histograms from an existing output file. """
        inputFilename = os.path.join(self.outputPath, "{0}.root".format(self.responseMatrixBaseName))
        logger.info("Loading histograms from ROOT file located at \"{}\"".format(inputFilename))
        with histogram.RootOpen(inputFilename, "READ") as f:
            histNames = {"responseMatrix": self.GetResponseMatrixName(), "responseMatrixErrors": None}
            for dictName, histName in histNames.iteritems():
                if not histName:
                    histName = dictName
                logger.debug("dictName: {}, histName: {}".format(dictName, histName))
                hist = f.Get(histName)
                hist.SetDirectory(0)
                self.hists[dictName] = hist
            if not self.productionRootFile:
                histNames = {"partSpectraProjection": None, "ptHardSpectra": None, "sampleTaskJetSpectraPartLevel": None, "jetSpectraPartLevel": None, "jetSpectraDetLevel": None}
                for dictName, histName in histNames.iteritems():
                    if not histName:
                        histName = dictName
                    logger.debug("dictName: {}, histName: {}".format(dictName, histName))
                    hist = f.Get(histName)
                    hist.SetDirectory(0)
                    self.hists[histName] = hist
                # Pt hard hists
                # Remove "partSpectraProjection", which doesn't have the corresponding pt hard hists
                del histNames["partSpectraProjection"]
                # Add or change the name for the individual pt hard spectra
                histNames["responseMatrix"] = None
                histNames["ptHardSpectra"] = "fHist"  # Leave out "PtHard" here since it will be added back in below
                for dictName, histName in histNames.iteritems():
                    if not histName:
                        histName = dictName
                    dictName = "{dictName}PtHard".format(dictName = dictName)
                    name = "{histName}PtHard".format(histName = histName)
                    logger.debug("dictName: {}, histName: {}".format(dictName, histName))

                    for ptHardBin in self.ExtractPtHardBinsFromDirectoryPaths():
                        logger.debug("name: {}_{}".format(name, ptHardBin))
                        hist = f.Get("{name}_{ptHardBin}".format(name = name, ptHardBin = ptHardBin))
                        hist.SetDirectory(0)
                        self.hists[dictName][ptHardBin] = hist
                        # Set the name back to the epxected form
                        #self.hists[name][ptHardBin].SetName("{name}_{ptHardBin}".format(name = name, ptHardBin = ptHardBin))

    def Print(self):
        """ Print the properties of the class. """
        logger.info("{name} Properties:".format(name = self.__class__.__name__))
        # See: https://stackoverflow.com/a/1398059
        members = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        for member in members:
            printProperty(member, getattr(self, member))

    ############################
    # More complicated functions
    ############################
    def ExtractPtHardBinsFromDirectoryPaths(self):
        """ Retrieves the pt hard bin from the directory structure. """
        # Skip if it's not a pt hard bin production
        # We will return one bin at pt hard bin "0" so that the code can proceed as normal
        # with the understanding that bin "0" is not pt hard binned and it contains all of the
        # available data
        if not self.ptHardBinProduction:
            return [0]

        # Configuration value to extract from
        inputPath = self.config["taskConfiguration"]["inputPath"]

        # Get pt hard directories
        # Remove anything after "ptHardBin" when initially looking for valid pt hard bins
        tempInputPath = inputPath.format(ptHardBin = "removeBeforeHere")
        ptHardDirectories = [name for name in os.listdir(tempInputPath[:tempInputPath.find("removeBeforeHere")]) if os.path.isdir(inputPath.format(ptHardBin = name)) or os.path.islink(inputPath.format(ptHardBin = name))]
        if not ptHardDirectories:
            logger.critical("Could not find pt hard directories!")
            sys.exit(1)
        # Sort the pt hard bins to ensure that it is ordered by [1, 2, 3, 4, ...] instead of [1, 10, 2, 3, ...]
        ptHardDirectories = sorted(ptHardDirectories, key=int)

        ptHardBins = []
        for ptHardBin in ptHardDirectories:
            # We only want a single digit. If not, attempt to extract just the digit.
            if not ptHardBin.isdigit():
                logger.info("Pt hard bin {0} is not just a digit. Attempt to extract it.")
                ptHardBin = [hardBin for hardBin in ptHardBin.split() if hardBin.isdigit()]
                if ptHardBin.isdigit():
                    logger.info("Extracted pt hard bin {0}. Check that this is the proper value!")
                else:
                    logger.critical("Failed to extract pt hard bin from {0}".format(ptHardBin))
                    sys.exit(1)

            ptHardBins.append(ptHardBin)

        return ptHardBins

    def RetrieveHists(self):
        """ Retrieve hists from the AnalysisResults.root files """
        # Get config
        taskConfiguration = self.config["taskConfiguration"]
        inputPath = taskConfiguration["inputPath"]

        # Extract the available pt hard bins
        ptHardBins = self.ExtractPtHardBinsFromDirectoryPaths()

        for ptHardBin in ptHardBins:
            logger.info("Processing pt hard bin {0}".format(ptHardBin))
            # Setup
            inputFilename = taskConfiguration["inputFilename"]
            filename = os.path.join(inputPath.format(ptHardBin = ptHardBin), inputFilename)
            logger.info("Accessing file at {0}".format(filename))

            # Get the event count before event selection
            pythiaInfoTaskName = taskConfiguration.get("pythiaInfoTaskName", None)
            if pythiaInfoTaskName:
                logger.info("Pythia task name: {0}".format(pythiaInfoTaskName))
                pythiaInfo = analysis_objects.getHistogramsInList(filename, pythiaInfoTaskName)
                pythiaInfoAfterEventSelection = analysis_objects.getHistogramsInList(filename, taskConfiguration["pythiaInfoAfterEventSelectionTaskName"])
            # Get the response maker
            responseMakerHists = analysis_objects.getHistogramsInList(filename, taskConfiguration["responseMakerTaskName"])
            # Get the particle level sample task (to check the jet spectra independently)
            sampleTaskParticleLevel = analysis_objects.getHistogramsInList(filename, taskConfiguration["sampleTaskParticleLevelTaskName"])
            sampleTaskParticleLevelJetsName = taskConfiguration.get("sampleTaskParticleLevelJetsName", "truthJets_AKTFullR020_mcparticles_pT3000_pt_scheme")
            # Get the sample task (to check the jet spectra independently)
            #sampleTaskDetLevel = analysis_objects.getHistogramsInList(filename, taskConfiguration["sampleTaskDetLevelTaskName"])

            if pythiaInfoTaskName:
                # Get N events
                self.hists["nEventsPtHard"][ptHardBin] = pythiaInfo["fHistEventCount"]
                self.hists["nEventsAfterEventSelectionPtHard"][ptHardBin] = pythiaInfoAfterEventSelection["fHistEventCount"]

                # Get cross section
                self.hists["crossSectionPtHard"][ptHardBin] = pythiaInfo["fHistXsection"]
                self.hists["crossSectionAfterEventSelectionPtHard"][ptHardBin] = pythiaInfoAfterEventSelection["fHistXsectionAfterSel"]

                # Get N trials
                self.hists["nTrialsPtHard"][ptHardBin] = pythiaInfo["fHistTrials"]
                self.hists["nTrialsAfterEventSelectionPtHard"][ptHardBin] = pythiaInfoAfterEventSelection["fHistTrialsAfterSel"]

                # Get pt hard spectra
                hist = pythiaInfo["fHistPtHard"]
                hist.SetName("{histName}_{ptHardBin}".format(histName = hist.GetName(), ptHardBin = ptHardBin))
                self.hists["ptHardSpectraPtHard"][ptHardBin] = hist
                # Need to check for pp case, since the name is not changed after event selection
                hist = pythiaInfoAfterEventSelection.get("fHistPtHardAfterSel", None)
                if not hist:
                    hist = pythiaInfoAfterEventSelection.get("fHistPtHard", None)
                hist.SetName("{histName}_{ptHardBin}".format(histName = hist.GetName(), ptHardBin = ptHardBin))
                self.hists["ptHardSpectraAfterEventSelectionPtHard"][ptHardBin] = hist

            # Get hists from Response Maker
            self.hists["unmatchedDetLevelJetsPtHardSparse"][ptHardBin] = responseMakerHists["fHistJets1"]
            self.hists["unmatchedPartLevelJetsPtHardSparse"][ptHardBin] = responseMakerHists["fHistJets2"]
            self.hists["responseMatrixPtHardSparse"][ptHardBin] = responseMakerHists["fHistMatching"]

            # Get the particle level spectra from the sample task
            sampleTaskParticleLevelJets = sampleTaskParticleLevel[sampleTaskParticleLevelJetsName]
            hist = sampleTaskParticleLevelJets["histJetPt_{}".format(self.eventActivity.value)]
            hist.SetName("{histName}_{ptHardBin}".format(histName = "sampleTaskJetSpectraPartLevelPtHard", ptHardBin = ptHardBin))
            self.hists["sampleTaskJetSpectraPartLevelPtHard"][ptHardBin] = hist

        logger.debug("Hists after retrieval:")
        logger.debug(pprint.pformat(self.hists))

    def ExtractScaleFactors(self):
        """ Extract scale factor from the xsec and trails histograms """
        # Following strategy here: https://twiki.cern.ch/twiki/bin/viewauth/ALICE/PPEventNormalisation
        xSecListName = "crossSection"
        nTrialsListName = "nTrials"
        if self.afterEventSelection:
            xSecListName += "AfterEventSelection"
            nTrialsListName += "AfterEventSelection"
        # Always need to append this
        xSecListName += "PtHard"
        nTrialsListName += "PtHard"

        # The keys are simply the pt hard bin, and any of the histograms could have been selected equally
        for ptHardBin, xSecHist, nTrialsHist in zip(self.hists[xSecListName].keys(), self.hists[xSecListName].values(), self.hists[nTrialsListName].values()):
            logger.info("Extracting scale factor for pt hard bin {0}".format(ptHardBin))
            logger.debug("xSecHist: {0}, entries: {1}; nTrials: {2}, entries: {3}".format(xSecHist, xSecHist.GetEntries(), nTrialsHist, nTrialsHist.GetEntries()))
            # +2 is due to the hist binning starting at -1 and the hist being exlcusive on the upper edge of the bin
            # Thus, pt hard 5 ends up in the 5-6 bin.
            # Multiply the cross section by number of events because the histogram stores the mean cross section (it's a TProfile)
            xSec = xSecHist.GetBinContent(int(ptHardBin) + 1) * xSecHist.GetEntries()
            nTrials = nTrialsHist.GetBinContent(int(ptHardBin) + 1)

            # Specially handle pt hard bin 9 in pp, where it spills over into pt hard bin 10 due to the default binning.
            # Thus, we need to merge the two values together. It is easiest to do it here.
            # (Note that pt hard bin 10 in LHC15g1a doens't matched up to the LHC12a15e_fix pt hard binning)
            if self.collisionSystem == analysis_objects.CollisionSystem.kpp and ptHardBin == "9":
                logger.warning("Handling special case for pt hard bin 9 in LHC15g1a! Careful if this isn't expected!")
                # To merge cross sections, they need to be averaged
                xSec = (xSecHist.GetBinContent(int(ptHardBin) + 1) + xSecHist.GetBinContent(int(ptHardBin) + 2)) / 2. * xSecHist.GetEntries()
                nTrials = nTrialsHist.GetBinContent(int(ptHardBin) + 1) + nTrialsHist.GetBinContent(int(ptHardBin) + 2)

            logger.debug("xSec: {0}, nTrials: {1}".format(xSec, nTrials))
            if nTrials > 0:
                scaleFactor = xSec / nTrials
            else:
                logger.error("nTrials is 0! Cannot calculate it, so setting to 0 (ie won't contribute)!")
                scaleFactor = 0

            self.scaleFactors[ptHardBin] = scaleFactor

        # Return true if all scale factors are non-zero
        returnValue = all(val for val in self.scaleFactors if val != 0)
        logger.debug("Return value: {0}".format(returnValue))
        return returnValue

    def ExtractAndApplyRelativePtHardBinScaleFactors(self):
        """ Get relative scaling for each pt hard bin and scale the scale factors by each relative value """
        nTotalEvents = 0.
        nEventsPtHard = collections.OrderedDict()
        for ptHardBin in self.scaleFactors.iterkeys():
            # Get the number of accepted events (bin 1)
            nEvents = self.hists["nEventsPtHard"][ptHardBin].GetBinContent(1)
            nEventsPtHard[ptHardBin] = nEvents
            nTotalEvents += nEvents

        nEventsAvg = nTotalEvents / len(self.scaleFactors)

        for ptHardBin, scaleFactor in self.scaleFactors.iteritems():
            self.scaleFactors[ptHardBin] = scaleFactor * nEventsAvg / nEventsPtHard[ptHardBin]

    def ProjectSparses(self):
        """ Perform the actual THnSparse projections. """
        # Perform the various projections
        for projector in self.projectors:
            projector.Project()

    def GetResponseMatrixName(self):
        """ Determine response matrix name based on the selected cluster bias. """
        if self.clusterBias > 0:
            responseMatrixName = "{0}_Clus{1:.2f}".format(self.responseMatrixBaseName, self.clusterBias)
        else:
            responseMatrixName = self.responseMatrixBaseName
        return responseMatrixName

    def ScaleAndMergeSelectedHists(self):
        """ Direct the scaling and merging of pt hard separated hists. """
        responseMatrixName = self.GetResponseMatrixName()

        # Tuple is of the form (ptHardHistsName, mergeHistName, outputHistName)
        histsToScaleAndMerge = [("unmatchedJetSpectraPartLevelPtHard", "unmatchedJetSpectraPartLevel", "unmatchedJetSpectraPartLevel"),
                                ("jetSpectraPartLevelPtHard", "jetSpectraPartLevel", "jetSpectraPartLevel"),
                                ("unmatchedJetSpectraDetLevelPtHard", "unmatchedJetSpectraDetLevel", "unmatchedJetSpectraDetLevel"),
                                ("jetSpectraDetLevelPtHard", "jetSpectraDetLevel", "jetSpectraDetLevel"),
                                ("sampleTaskJetSpectraPartLevelPtHard", "sampleTaskJetSpectraPartLevel", "sampleTaskJetSpectraPartLevel"),
                                ("responseMatrixPtHard", responseMatrixName, "responseMatrix")]
        if self.ptHardBinProduction:
            # Handle explicit pt hard bins
            histsToScaleAndMerge.append(("ptHardSpectraPtHard", "ptHardSpectra", "ptHardSpectra"))
            histsToScaleAndMerge.append(("ptHardSpectraAfterEventSelectionPtHard", "ptHardSpectraAfterEventSelection", "ptHardSpectraAfterEventSelection"))

        for ptHardHistsName, mergedHistName, outputHistName in histsToScaleAndMerge:
            logger.debug("ptHardHistsName: {0}".format(ptHardHistsName))
            logger.debug("outputHistName before: {0}".format(outputHistName))
            if self.ptHardBinProduction:
                self.ScaleHists(ptHardHistsName)
            self.MergeHists(ptHardHistsName, mergedHistName, outputHistName)
            logger.debug("outputHistName  after: {0}, hist: {1}".format(outputHistName, self.hists[outputHistName]))

    def ScaleHists(self, ptHardHistsName):
        """ Scale pt hard separated hists by the scale factor """
        # Create cloned hist
        ptHardHists = self.hists[ptHardHistsName]
        if not isinstance(ptHardHists, collections.Iterable):
            raise ValueError("Must be an iterable of histograms to scale! Passed hists: {0}".format(ptHardHists))

        for ptHardBin, ptHardHist in ptHardHists.iteritems():
            # Check for outliers
            # This is _not_ the same when an angle is selected compared to when processing all angles together unless performed carefully!!
            # Fortunately, the function handles this properly
            removeOutliers(ptHardHist, limit = 1.5, outputPath = self.OutputPathWithoutAngleInName())

            # Scale hists
            scaleFactor = self.scaleFactors[ptHardBin]
            logger.info("Scaling hist {0} of ptHardBin {1} with scale factor {2}".format(ptHardHist.GetName(), ptHardBin, scaleFactor))
            ptHardHist.Scale(scaleFactor)

    def OutputPathWithoutAngleInName(self):
        """ Utility function to remove the EP angle from the name. """
        return self.outputPath[:self.outputPath.index(str(self.rmNormalizationType)) + len(str(self.rmNormalizationType))]

    def MergeHists(self, ptHardHistsName, mergedHistName, outputHistName):
        """ Add pt hard bin separated hists into a signle histogram """
        ptHardHists = self.hists[ptHardHistsName]
        if not isinstance(ptHardHists, collections.Iterable):
            raise ValueError("Must be an iterable of histograms to scale! Passed hists: {0}".format(ptHardHists))

        logger.debug("ptHardHists: {}".format(ptHardHists))
        hist = ptHardHists.itervalues().next().Clone(mergedHistName)
        # Reset so we can just Add() all hists without worrying which hist is being processed
        hist.Reset()
        # However, we must ensure that Sumw2 is still set!
        hist.Sumw2()
        logger.debug("hist: {0}".format(hist))

        for i, ptHardHist in enumerate(ptHardHists.itervalues()):
            # Add hist
            hist.Add(ptHardHist)

        logger.debug("Output hist: {0}".format(hist))
        self.hists[outputHistName] = hist

    def CreateResponseMatrixErrors(self):
        """ Fill response matrix errors hist based on the errors in the response matrix """
        # Get response matrix
        responseMatrix = self.hists["responseMatrix"]
        # Clone response matrix so that it automatically has the same limits
        responseMatrixErrors = responseMatrix.Clone("responseMatrixErrors")
        # Reset so that we can fill it with the errors
        responseMatrixErrors.Reset()

        # Fill response matrix errors
        # Careful with GetXaxis().GetFirst() -> The range can be restricted by SetRange()
        for x in range(1, responseMatrix.GetXaxis().GetNbins() + 1):
            for y in range(1, responseMatrix.GetYaxis().GetNbins() + 1):
                fillValue = responseMatrix.GetBinError(x, y)
                #if fillValue > 1:
                #    logger.debug("Error > 1 before scaling: {0}, bin content: {1}, bin error: {2}, ({3}, {4})".format(fillValue, responseMatrix.GetBinContent(x, y), responseMatrix.GetBinError(x,y), x, y))
                if responseMatrix.GetBinContent(x, y) > 0:
                    if responseMatrix.GetBinContent(x, y) < responseMatrix.GetBinError(x, y):
                        logger.error("Bin content < bin error. bin content: {0}, bin error: {1}, ({2}, {3})".format(responseMatrix.GetBinContent(x, y), responseMatrix.GetBinError(x, y), x, y))
                    fillValue = fillValue / responseMatrix.GetBinContent(x, y)
                else:
                    if responseMatrix.GetBinError(x, y) > analysis_objects.epsilon:
                        logger.warning("No bin content, but associated error is non-zero. Content: {0}, error: {1}".format(responseMatrix.GetBinContent(x, y), responseMatrix.GetBinError(x, y)))
                if fillValue > 1:
                    logger.error("Error > 1 after scaling: {0}, bin content: {1}, bin error: {2}, ({3}, {4})".format(fillValue, responseMatrix.GetBinContent(x, y), responseMatrix.GetBinError(x, y), x, y))

                # Fill hist
                binNumber = responseMatrixErrors.Fill(responseMatrixErrors.GetXaxis().GetBinCenter(x), responseMatrixErrors.GetYaxis().GetBinCenter(y), fillValue)

                # Check to ensure that we filled where we expected
                if binNumber != responseMatrixErrors.GetBin(x, y):
                    logger.error("Mismatch between fill bin number ({0}) and GetBin() ({1})".format(binNumber, responseMatrixErrors.GetBin(x, y)))

        # Save the output
        self.hists["responseMatrixErrors"] = responseMatrixErrors

    #####################
    # Execution functions
    #####################
    @staticmethod
    def createResponseMatrixAnalysis(jetHArgs):
        """ Create different response matrix objects based on the arguments.

        NOTE:
            This is a bit strange because it creates objects that depend on this
            class. It's located here just for organization, so it should be fine.
        """
        # Create the object
        JetHResponse = JetHResponseMatrix(**jetHArgs)

        JetHResponse.Print()

        if jetHArgs.get("initFromRootFile", False):
            JetHResponse.InitFromRootFile()
        else:
            JetHResponse.Analyze()
            JetHResponse.SaveAndPrint()

        return (JetHResponse, True)

    @staticmethod
    def run(args):
        """ Main entry point for the response matrix. """
        # Create logger
        logging.basicConfig(level=logging.DEBUG)
        # Quiet down the matplotlib logging
        logging.getLogger("matplotlib").setLevel(logging.INFO)

        # Argument validation
        args = validateArguments(args)

        # TODO: Create a constructFromConfigurationFile() ?

        # Print out configuration
        logger.info("Arguments:")
        for key, val in args.iteritems():
            printProperty(key, val)

        # Define arguments
        #jetHArgs = {"configFile": args.configFile, "clusterBias": args.clusterBias, "productionRootFile": args.productionRootFile, "useFloatHists": args.useFloatHists, "collisionSystem": args.collisionSystem}
        jetHArgs = args

        JetHResponseEP = collections.OrderedDict()
        JetHResponse = None
        if jetHArgs.get("runAllAngles", False):
            # NOTE: To get outlier removal correct, it's best to start with all angles, as it has the best statistics.
            #       Alternatively, it can be run twice to confirm that the proper values were removed.

            # This must be configured here - it's too confusing otherwise
            # Perhaps could take from configFile if needed in the future
            baseConfigPath = "responseMatrix/responseMatrixSemiCentral{epAngle}.yaml"
            for epAngle in ReactionPlaneOrientation:
                logger.info("Processing event plane angle {}".format(str(epAngle)))
                jetHArgs["configFile"] = baseConfigPath.format(epAngle = epAngle.filenameStr())

                (jetHResponse, success) = JetHResponseMatrix.createResponseMatrixAnalysis(jetHArgs)

                if success:
                    JetHResponseEP[epAngle] = jetHResponse
                else:
                    logger.error("Jet-H response for EP angle {} failed!".format(str(epAngle)))

            plotParticleSpectraProjection(JetHResponseEP)

            checkAgreementOfAllAnglesVsSumOfAngles(JetHResponseEP)

            packageMatricesIntoOneFile(JetHResponseEP)
        else:
            (JetHResponse, success) = JetHResponseMatrix.createResponseMatrixAnalysis(jetHArgs)

        # Careful, this convention is a bit counter-intuitive. In this case,
        # False as the default value will ensure that IPython is embedded by default.
        if not jetHArgs.get("nonInteractive", False):
            IPython.embed()

###################
# Utility functions
###################
def printProperty(name, val):
    """ Convenience method to pretty print a property. """
    logger.info("    {name}: {val}".format(name = name, val = val))

def accessSetOfValuesAssociatedWithABin(hist, binOfInterest, rmNormalizationType, scaleFactor = -1):
    """ Access a set of bins associated with a particular bin value in the other axis. For example, if the hist
    looks like this graphically:

    a b c
    d e f
    g h i <- values (here and above)
    1 2 3 <- bin number

    then in the case of accessing a set of y bins associated with an x bin,
    for example, x bin 2, it would return values h, e, and b.

    Note:
        This function assumes 2D hists, but that should be a perfectly reasonable assumption here.
    """
    # Initial configuration
    axis = None
    getBin = None
    if rmNormalizationType == RMNormalizationType.kNormalizeEachDetectorBin:
        axis = ROOT.TH1.GetYaxis

        def GetBin(hist, binOfInterest, index):
            return hist.GetBin(binOfInterest, index)
        getBin = GetBin
    elif rmNormalizationType == RMNormalizationType.kNormalizeEachTruthBin:
        axis = ROOT.TH1.GetXaxis

        def GetBin(hist, binOfInterest, index):
            return hist.GetBin(index, binOfInterest)
        getBin = GetBin
    else:
        raise ValueError("RM Normalization value {0} not recognized".format(rmNormalizationType))

    #logger.debug("Axis: {0}, getBin: {1}".format(axis(hist).GetName(), getBin))
    setOfBinsContent = []
    setOfBinsError = []
    #for index in range(0, axis(hist).GetNbins() + 2):
    # Temp 1-nBins to agree with the previous function
    for index in range(1, axis(hist).GetNbins() + 1):
        setOfBinsContent.append(hist.GetBinContent(getBin(hist, binOfInterest, index)))
        setOfBinsError.append(hist.GetBinError(getBin(hist, binOfInterest, index)))

        if (scaleFactor >= 0):
            # -1 since index starts at 1
            # NOTE: If the above for loop is changed to 0 to Nbins+1, then the -1 should be removed!
            hist.SetBinContent(getBin(hist, binOfInterest, index), setOfBinsContent[index - 1] / scaleFactor)
            hist.SetBinError(getBin(hist, binOfInterest, index), setOfBinsError[index - 1] / scaleFactor)

    return setOfBinsContent

def normalizeResponseMatrix(hist, rmNormalizationType):
    """ Normalize response matrix

    For each given x bin (detector level pt), we take all associated y bins (truth leve), and normalize them to 1."""
    projectionFunction = None
    if rmNormalizationType == RMNormalizationType.kNormalizeEachDetectorBin:
        projectionFunction = ROOT.TH2.ProjectionY
        maxBins = hist.GetXaxis().GetNbins() + 1
    elif rmNormalizationType == RMNormalizationType.kNormalizeEachTruthBin:
        projectionFunction = ROOT.TH2.ProjectionX
        maxBins = hist.GetYaxis().GetNbins() + 1

    # NOTE: Range is selected to not include overflow bins. This could be changed if desired.
    for index in range(1, maxBins):
        binsContent = []
        norm = 0

        # Access bins
        binsContent = accessSetOfValuesAssociatedWithABin(hist, index, rmNormalizationType)

        norm = sum(binsContent)
        # NOTE: The upper bound on integrals is inclusive!
        ptYProjection = projectionFunction(hist, "{0}_projection{1}".format(hist.GetName(), index), index, index)
        #ptYProjection = hist.ProjectionY("{0}_projection{1}".format(hist.GetName(), index), index, index+1)

        # Sanity checks
        # NOTE: The upper bound on integrals is inclusive!
        # NOTE: Integral() == Integral(1, proj.GetXaxis().GetNbins())
        if not np.isclose(norm, ptYProjection.Integral(1, ptYProjection.GetXaxis().GetNbins())):
            logger.error("Mismatch between sum and integral! norm: {0}, integral: {1}".format(norm, ptYProjection.Integral(1, ptYProjection.GetXaxis().GetNbins())))
        if not np.isclose(ptYProjection.Integral(), ptYProjection.Integral(1, ptYProjection.GetXaxis().GetNbins())):
            logger.error("Integral mismatch! Full: {0} 1-nBins: {1}".format(ptYProjection.Integral(), ptYProjection.Integral(1, ptYProjection.GetXaxis().GetNbins())))

        # Avoid scaling by 0
        if not norm > 0.0:
            continue

        # normalization by sum
        accessSetOfValuesAssociatedWithABin(hist, index, rmNormalizationType, scaleFactor = norm)

def checkNormalization(hist, rmNormalizationType, useFloatHists):
    """ Check to ensure that the normalization was successful. """
    for index in range(1, hist.GetXaxis().GetNbins() + 1):
        binsContent = []
        norm = 0

        # Access bins
        binsContent = accessSetOfValuesAssociatedWithABin(hist, index, rmNormalizationType)
        # Get norm
        norm = sum(binsContent)

        if useFloatHists:
            # Otherwise, it will freqeuntly fail
            comparisonLimit = 1e-7
        else:
            # Preferred value
            comparisonLimit = 1e-9
        if not np.isclose(norm, 0, atol = comparisonLimit) and not np.isclose(norm, 1, atol = comparisonLimit):
            logger.error("Normalization not successful for bin {0}. Norm: {1}".format(index, norm))

def removeOutliers(hist, limit, outputPath):
    if isinstance(hist, ROOT.TH3):
        raise TypeError(type(hist), "Outlier removal does not work on TH3")

    # Project a TH2 to a TH1 to simplfy the algorithm
    projection = False
    if isinstance(hist, ROOT.TH2):
        # Project
        projection = hist.ProjectionX("{histName}_temp".format(histName = hist.GetName()))

    histToCheck = projection if projection else hist

    # Check with moving average
    foundAboveLimit = False
    cutLimitReached = False
    # The cut index is where we decided cut on that row
    cutIndex = -1
    nBinsBelowLimitAfterLimit = 0
    # n bins that are below threshold before all bins are cut
    nBinsThreshold = 4

    (preMean, preMedian) = GetHistMeanAndMedian(histToCheck)

    for index in range(0, histToCheck.GetNcells()):
        #logger.debug("---------")
        avg = MovingAverage(histToCheck, index = index, numberOfCountsBelowIndex = 2, numberOfCountsAboveIndex = 2)
        #logger.debug("Index: {0}, Avg: {1}, BinContent: {5}, foundAboveLimit: {2}, cutIndex: {3}, cutLimitReached: {4}".format(index, avg, foundAboveLimit, cutIndex, cutLimitReached, histToCheck.GetBinContent(index)))
        if avg > limit:
            foundAboveLimit = True

        if not cutLimitReached:
            if foundAboveLimit and avg <= limit:
                if cutIndex == -1:
                    cutIndex = index
                nBinsBelowLimitAfterLimit += 1

            if nBinsBelowLimitAfterLimit != 0 and avg > limit:
                # Reset
                cutIndex = -1
                nBinsBelowLimitAfterLimit = 0

            if nBinsBelowLimitAfterLimit > nBinsThreshold:
                cutLimitReached = True
        # Do not perform removal here because then we miss values between the avg going below
        # the limit and crossing the nBinsThreshold

    logger.debug("Hist checked: {0}, cut index: {1}".format(histToCheck.GetName(), cutIndex))

    """
    Format of the file is as follows:
    histName: limit

    Note that the values should be quite similar for different hists from the same pt hard bin, but
    they may not be exactly the same.
    """
    outlierLimitsFilename = os.path.join(os.path.join(outputPath, "outlierRemovalLimits.yaml"))
    try:
        with open(outlierLimitsFilename, "rb") as f:
            #logger.info("Loading existing outlier limits reference values")
            outlierLimits = yaml.safe_load(f)
    except IOError:
        # The file didn't exist, so just define the limits as an empty dict.
        outlierLimits = {}

    #logger.debug("outlierLimits read from file: {}".format(outlierLimits))
    if not outlierLimits:
        # The file doesn't yet have a dict
        logger.warning("Creating empty outlier limits dict")
        outlierLimits = {}
    reference = outlierLimits.get(hist.GetName(), None)

    # Get reference cut index
    if reference:
        if cutIndex > reference:
            # Use the larger cut index value
            logger.info("Setting reference cut value {} to larger extracted cut index {}".format(reference, cutIndex))
            reference = cutIndex
        else:
            # Use the larger reference value
            logger.info("Using cut index reference value of {} instead of extracted cut index {}".format(reference, cutIndex))
            cutIndex = reference
    else:
        # Define a new reference
        logger.info("Creating a new reference cut index of {} for hist {}".format(cutIndex, hist.GetName()))
        reference = cutIndex

    outlierLimits[hist.GetName()] = reference
    #logger.debug("outlierLimits right before write: {}".format(outlierLimits))
    with open(outlierLimitsFilename, "wb") as f:
        yaml.safe_dump(outlierLimits, f, default_flow_style = False)

    # Use on both TH1 and TH2 since we don't start removing immediately, but instead only after the limit
    if cutLimitReached:
        #logger.debug("Removing outliers")
        # Check for values above which they should be removed by translating the global index
        x = ctypes.c_int(0)
        y = ctypes.c_int(0)
        z = ctypes.c_int(0)
        for index in range(0, hist.GetNcells()):
            # Get the bin x, y, z from the global bin
            hist.GetBinXYZ(index, x, y, z)
            # Watch out for any problems
            if hist.GetBinContent(index) < hist.GetBinError(index):
                logger.warning("Bin content < error. Name: {}, Bin content: {}, Bin error: {}, index: {}, ({}, {})".format(hist.GetName(), hist.GetBinContent(index), hist.GetBinError(index), index, x.value, y.value))
            if x.value >= cutIndex:
                #logger.debug("Cutting for index {}. x bin {}. Cut index: {}".format(index, x, cutIndex))
                hist.SetBinContent(index, 0)
                hist.SetBinError(index, 0)
    else:
        logger.info("Hist {} did not have any outliers to cut".format(hist.GetName()))

    # Check the mean and median
    # Use another temporary hist in the case of a TH2 to simply extracting values
    if isinstance(hist, ROOT.TH2):
        # Project
        projection = hist.ProjectionX("{histName}_temp2".format(histName = hist.GetName()))
    histToCheck = projection if projection else hist

    (postMean, postMedian) = GetHistMeanAndMedian(histToCheck)
    logger.info("Pre  outliers removal mean: {}, median: {}".format(preMean, preMedian))
    logger.info("Post outliers removal mean: {}, median: {}".format(postMean, postMedian))

def GetHistMeanAndMedian(hist):
    # Median
    # See: https://root-forum.cern.ch/t/median-of-histogram/7626/5
    x = ctypes.c_double(0)
    q = ctypes.c_double(0.5)
    # Apparently needed to be safe(?)
    hist.ComputeIntegral()
    hist.GetQuantiles(1, x, q)

    mean = hist.GetMean()

    return (mean, x.value)

def MovingAverage(hist, index, numberOfCountsBelowIndex = 0, numberOfCountsAboveIndex = 2):
    """
    # [-2, 2] includes -2, -1, 0, 1, 2
    """
    # Check inputs
    if numberOfCountsBelowIndex < 0 or numberOfCountsAboveIndex < 0:
        logger.critical("Moving average number of counts above or below must be >= 0. Please check the values!")
        sys.exit(1)

    count = 0.
    average = 0.
    for i in range(index - numberOfCountsBelowIndex, index + numberOfCountsAboveIndex + 1):
        # Avoid going over histogram limits
        if i < 0 or i >= hist.GetNcells():
            continue
        # TEMP - Hack due to entries in pt hard value = 1 for large pt hard bins for no apparent reason
        if i == 1:
            continue
        #logger.debug("Adding {}".format(hist.GetBinContent(i)))
        average += hist.GetBinContent(i)
        count += 1

    #if count != (numberOfCountsBelowIndex + numberOfCountsAboveIndex + 1):
        #logger.debug("Count: {}, summed: {}".format(count, (numberOfCountsBelowIndex + numberOfCountsAboveIndex + 1)))
        #exit(0)

    return average / count

####################
# Plotting Utilities
####################
def plot1DPtHardHists(full, ptHardList, canvas, outputPath, ptHardBinning = []):
    """ Basic plotting macro to show the contributions of pt hard bins on 1D hists """
    # Plot jet spectra
    # Ensure that it is possible to see the points
    full.SetMarkerStyle(ROOT.kFullCircle)
    full.SetMarkerSize(1)
    full.GetXaxis().SetRangeUser(0, 500)
    full.SetLineColor(ROOT.kBlack)
    full.Draw()

    if ptHardBinning:
        legend = ROOT.TLegend(0.55, 0.55, 0.95, 0.9)
        # ROOT 5 sux
        if ROOT.gROOT.GetVersionInt() > 60000:
            legend.SetHeader("#matit{p}_{#mathrm{T}} bins", "C")
        else:
            legend.SetHeader("#mathit{p}_{#mathrm{T}} bins")
        legend.SetNColumns(2)
        legend.SetBorderSize(0)

    for (ptHardBin, spectra), color in zip(ptHardList.iteritems(), sns.husl_palette(n_colors=len(ptHardList))):
        color = ROOT.TColor.GetColor(*color)
        #logger.debug("Color: {0}".format(color))
        spectra.SetMarkerStyle(ROOT.kFullCircle + int(ptHardBin))
        spectra.SetMarkerColor(color)
        spectra.SetLineColor(color)
        spectra.SetMarkerSize(1)
        #spectra.SetDirectory(0)

        logger.debug("ptHardBin: {0}, color: {1}".format(ptHardBin, color))

        if ptHardBinning:
            #legend.AddEntry(spectra, "p_{{T}}^{{Hard}} Bin: {0}".format(ptHardBin))
            legend.AddEntry(spectra, "{0}<=p_{{T}}<{1}".format(ptHardBinning[int(ptHardBin) - 1], ptHardBinning[int(ptHardBin)]))

        drawOptions = "same"
        spectra.Draw(drawOptions)

    if ptHardBinning:
        legend.Draw()

    canvas.SaveAs(os.path.join(outputPath, "{0}.pdf".format(full.GetName())))

def plotParticleSpectraProjection(JetHResponseEP):
    """

    """
    includeAllAngles = True
    normalizeByNJets = True

    colors = [ROOT.kBlack, ROOT.kBlue - 7, 8, ROOT.kRed - 4]
    markers = [ROOT.kFullDiamond, ROOT.kFullSquare, ROOT.kFullTriangleUp, ROOT.kFullCircle]
    # See: https://github.com/rootpy/rootpy/blob/master/rootpy/plotting/base.py#L838
    fillTypes = [1001, 3003, 3345, 3013]
    maxViewableRange = 100

    canvas = ROOT.TCanvas("canvas", "canvas")
    canvas.SetTopMargin(0.04)
    canvas.SetLeftMargin(0.12)
    canvas.SetRightMargin(0.04)
    canvas.SetBottomMargin(0.15)

    latexLabels = []
    latexLabels.append(ROOT.TLatex(0.605, 0.90, "ALICE #sqrt{s_{NN}} = 2.76 TeV"))
    # TODO: Grab the centrality from the config.
    latexLabels.append(ROOT.TLatex(0.475, 0.84, "30-50% Pb-Pb Embedded PYTHIA"))
    latexLabels.append(ROOT.TLatex(0.525, 0.78, "20 GeV/#it{c} < #it{p}_{T,jet}^{det} < 40 GeV/#it{c}"))
    latexLabels.append(ROOT.TLatex(0.565, 0.69, "#it{p}_{T}^{ch,det}#it{c}, E_{T}^{clus,det} > 3.0 GeV"))
    latexLabels.append(ROOT.TLatex(0.635, 0.61, "E_{T}^{lead clus, det} > 6.0 GeV"))
    latexLabels.append(ROOT.TLatex(0.72, 0.545, "anti-#it{k}_{T}  R = 0.2"))

    legend = ROOT.TLegend(0.14, 0.17, 0.42, 0.47)
    # Remove border
    legend.SetBorderSize(0)
    # Increase text size
    legend.SetTextSize(0.06)
    # Make the legend transparent
    legend.SetFillStyle(0)
    # ROOT 5 sux
    #if ROOT.gROOT.GetVersionInt() > 60000:
    #    legend.SetHeader("Event Plane Angle", "C")
    #else:
    #    legend.SetHeader("Event Plane Angle")

    # Store the cloned histograms so they stay in scope
    hists = collections.OrderedDict()

    for i, ((epAngle, jetH), color, marker, fillType) in enumerate(zip(JetHResponseEP.iteritems(), colors, markers, fillTypes)):
        histTemp = jetH.hists["partSpectraProjection"]
        hist = histTemp.Clone("{histName}_clone".format(histName = histTemp.GetName()))
        hists[epAngle] = hist
        # Enlarge axis title size
        hist.GetXaxis().SetTitleSize(0.055)
        hist.GetYaxis().SetTitleSize(0.055)
        # Ensure there is enough space
        hist.GetXaxis().SetTitleOffset(1.15)
        hist.GetYaxis().SetTitleOffset(1.05)
        # Enlarge axis label size
        hist.GetXaxis().SetLabelSize(0.06)
        hist.GetYaxis().SetLabelSize(0.06)
        # Center axis title
        hist.GetXaxis().CenterTitle(True)
        hist.GetYaxis().CenterTitle(True)
        # Increase marker size slightly
        hist.SetMarkerSize(1.1)

        # Skip all angles
        if epAngle == ReactionPlaneOrientation.kAll:
            if not includeAllAngles:
                continue

        # Rebin to 5 GeV bin width
        if True:
            newBinWidth = 5
            hist.Rebin(newBinWidth)
            hist.Scale(1.0 / newBinWidth)

            # Cut below 5 GeV
            #for i in range(0, hist.GetNcells()):
            #    if not hist.IsBinUnderflow(i) and not hist.IsBinOverflow(i) and i <= hist.FindBin(5):
            #        hist.SetBinContent(i, 0)
            #        hist.SetBinError(i, 0)
            # Cut below 5 GeV
            # Note that this will modify the overall number of entries
            hist.SetBinContent(1, 0)
            hist.SetBinError(1, 0)
            # Potentially cut 5-10 GeV bin
            # TEMP
            #hist.SetBinContent(2, 0)
            #hist.SetBinError(2, 0)
            # ENDTEMP

        # View the interesting range
        # Note that this must be set after removing any bins that we might want to remove
        hist.GetXaxis().SetRangeUser(0, maxViewableRange)

        # Scale by N_{jets}
        # The number of entries should be equal to the number of jets. However, it's not a straightfoward
        # number to extract because of all of the scalings related to pt hard bins
        if normalizeByNJets:
            # 1e-5 is to ensure we do the integral from [0, 100) (ie not inclusive of the bin beyond 100)
            entries = hist.Integral(hist.FindBin(0), hist.FindBin(maxViewableRange - utils.epsilon))
            logger.debug("entries: {}, integral: {}".format(hist.GetEntries(), entries))
            hist.Scale(1.0 / entries)
            # Update the y axis title to represent the change
            hist.GetYaxis().SetTitle("(1/N_{jets})dN/d#mathit{p}_{T}")

        logger.debug("angle: {}, hist: {}, hist entries: {}, integral: {} histTemp: {}, histTemp entries: {} (decrease due to cutting out the 0-5 bin)".format(epAngle, hist, hist.GetEntries(), hist.Integral(), histTemp, histTemp.GetEntries()))
        #logger.debug("color: {}".format(color))
        hist.SetLineColor(color)
        hist.SetMarkerColor(color)
        hist.SetMarkerStyle(marker)
        #hist.SetMarkerSize(1.2)
        if False:
            hist.SetFillColorAlpha(color, 0.5)
            hist.SetFillStyle(fillType)

        # Offset points
        # See: https://root.cern.ch/root/roottalk/roottalk03/2765.html
        if False:
            shift = i * 0.1 * hist.GetBinWidth(1)
            xAxis = hist.GetXaxis()
            xAxis.SetLimits(xAxis.GetXmin() + shift, xAxis.GetXmax() + shift)

        # Label
        legend.AddEntry(hist, "{}".format(epAngle.displayStr()))

        hist.Draw("same")

        #if includeAllAngles:
        if False:
            # Need to adjust the range in this case, which requires redrawing the hist because ROOT sux...
            if normalizeByNJets:
                minVal = 1e-6
                maxVal = 4e-3
            else:
                minVal = 1e-4
                maxVal = 3

            logger.info("Setting y range to ({}, {})".format(minVal, maxVal))
            hist.GetYaxis().SetRangeUser(minVal, maxVal)
            hist.Draw("same")

    for tex in latexLabels:
        tex.SetNDC(True)
        tex.Draw()
    legend.Draw()

    canvas.SetLogy()

    outputPath = next(JetHResponseEP.itervalues()).OutputPathWithoutAngleInName()
    logger.info("Particle level projection outputPath: {0}".format(outputPath))
    filename = os.path.join(outputPath, "partSpectraProjection.{}")
    canvas.SaveAs(filename.format("pdf"))
    canvas.SaveAs(filename.format("C"))

def checkAgreementOfAllAnglesVsSumOfAngles(JetHResponseEP):
    outputPath = next(JetHResponseEP.itervalues()).OutputPathWithoutAngleInName()
    outputPath = os.path.join(outputPath, "{filenameLabel}.pdf")

    hist = JetHResponseEP[ReactionPlaneOrientation.kInPlane].hists["partSpectraProjection"]
    histSumOfAngles = hist.Clone("{histName}_sum.pdf".format(histName = hist.GetName()))
    logger.info("Immediately after clone, get entries: Original: {0}, clone: {1}".format(hist.GetEntries(), histSumOfAngles.GetEntries()))
    histSumOfAngles.Reset()
    histSumOfAngles.Sumw2()
    logger.info("Reset hist entries: {0} (0 expected)".format(histSumOfAngles.GetEntries()))

    for epAngle in ReactionPlaneOrientation:
        if epAngle == ReactionPlaneOrientation.kAll:
            continue

        jetH = JetHResponseEP[epAngle]
        h = jetH.hists["partSpectraProjection"]
        logger.info("{0}: {1}, integral: {2}".format(str(epAngle), h.GetEntries(), h.Integral()))
        ROOT.TH1.Add(histSumOfAngles, h)
        logger.info("Summing hists: Get Entries: {0}, Integral(): {1}".format(histSumOfAngles.GetEntries(), histSumOfAngles.Integral()))

    c = ROOT.TCanvas("canvas", "canvas")
    histSumOfAngles.Draw()
    logger.info("histSumOfAngles Entries: {0}, integral: {1}".format(histSumOfAngles.GetEntries(), histSumOfAngles.Integral()))
    #filenameLabel = "histSumOfAllAngles"
    #c.SaveAs(outputPath.format(filenameLabel = filenameLabel))

    hAll = JetHResponseEP[ReactionPlaneOrientation.kAll].hists["partSpectraProjection"]
    logger.info("hAll entries: {0}, integral: {1}".format(hAll.GetEntries(), hAll.Integral()))

    # Subtract the two to compare
    #histSumOfAngles.Add(hAll, -1)
    ROOT.TH1.Add(histSumOfAngles, hAll, -1)

    histSumOfAngles.SetTitle("(in + mid + out) - All projection")

    histSumOfAngles.Draw()
    filenameLabel = "allAnglesVsSumComparison"
    c.SaveAs(outputPath.format(filenameLabel = filenameLabel))

    # Take abs value so we can look at the log
    for i in range(0, histSumOfAngles.GetNcells()):
        histSumOfAngles.SetBinContent(i, abs(histSumOfAngles.GetBinContent(i)))

    histSumOfAngles.SetTitle("{0} (absolute value)".format(histSumOfAngles.GetTitle()))

    c.SetLogy()
    histSumOfAngles.Draw()
    c.SaveAs(outputPath.format(filenameLabel = "{}Log".format(filenameLabel)))

def packageMatricesIntoOneFile(JetHResponseEP):
    jetH = next(JetHResponseEP.itervalues())
    outputPath = jetH.OutputPathWithoutAngleInName()
    responseMatrixName = jetH.GetResponseMatrixName()

    hists = collections.OrderedDict()
    for epAngle, jetH in JetHResponseEP.iteritems():
        hist = jetH.hists["responseMatrix"]
        hists[epAngle] = hist.Clone("{histName}_{epAngle}".format(histName = responseMatrixName, epAngle = str(epAngle)))

    outputFilename = os.path.join(outputPath, "JESCorrection.root")
    logger.info("Writing merged output to {0}".format(outputFilename))
    with histogram.RootOpen(outputFilename, "RECREATE") as f:  # noqa: F841
        for h in hists.itervalues():
            h.Write()

def parseArguments():
    """ Parse arugments to this module. """
    # Setup command line parser
    parser = argparse.ArgumentParser(description = "Creating response matrix for JetH analysis")
    # Define arguments
    group = parser.add_mutually_exclusive_group(required = False)
    # Possible operation modes
    group.add_argument("-e", "--embeddingExamplePlots",
                       action="store_true",
                       help="Plot example plots used to explain the usage of the response matrix for embedding.")
    group.add_argument("-t", "--testMatrix",
                       action="store_true",
                       help="Create a testing response matrix.")

    # General options
    parser.add_argument("-c", "--configFile", metavar="configFile",
                        type=str, default=None,
                        help="Path to config filename")
    parser.add_argument("-b", "--clusterBias", metavar="bias",
                        type=float, default=None,
                        help="Value of the cluster bias")
    parser.add_argument("-p", "--productionRootFile",
                        action="store_true",
                        help="Create a production root file with only the response matrix")
    parser.add_argument("-f", "--useFloatHists",
                        action="store_true", default=None,
                        help="Use TH2F instead of TH2D.")
    parser.add_argument("-s", "--collisionSystem", metavar="system",
                        type=str, default="",
                        help="Set the collision system. Possible values include: \"pp\", \"PbPb\"")
    parser.add_argument("-n", "--nonInteractive", default=None,
                        action="store_true",
                        help="Run non-interactively by skipping the IPython embed.")
    parser.add_argument("-r", "--runAllAngles", default=None,
                        action="store_true",
                        help="Run each EP angle, as well as all angles.")
    parser.add_argument("-i", "--initFromRootFile", default=None,
                        action="store_true",
                        help="Initialize from file instead of running analysis.")

    # Test matrix sepcific options
    parser.add_argument("--testMatrixWeight", metavar="weight",
                        type=float, default=1,
                        help="Weight to use when filling the testing response matrix. Default: 1 = Identity matrix.")

    # Parse arguments
    args = parser.parse_args()

    return {"embeddingExamplePlots": args.embeddingExamplePlots,
            "testMatrix": args.testMatrix,
            "configFile": args.configFile,
            "clusterBias": args.clusterBias,
            "productionRootFile": args.productionRootFile,
            "useFloatHists": args.useFloatHists,
            "collisionSystem": args.collisionSystem,
            "nonInteractive": args.nonInteractive,
            "runAllAngles": args.runAllAngles,
            "initFromRootFile": args.initFromRootFile
            }

def validateArguments(args):
    """ Validate passed arguments. """
    # Check for test matrix options
    if args.get("testMatrix", False):
        logger.info("Creating test matrix, so the production root file must be created, as the other hists are not meaningful!")
        args["productionRootFile"] = True
    # Convert collision systems to enumeration
    if args.get("collisionSystem", False):
        try:
            # Needs the additional "k" to find the enumeration
            args["collisionSystem"] = analysis_objects.CollisionSystem["k" + args["collisionSystem"]]
        except KeyError:
            logger.critical("Could not find collision system \"{0}\"".format(args["collisionSystem"]))
            sys.exit(1)

    return args

if __name__ == "__main__":
    args = parseArguments()
    JetHResponseMatrix.run(args)

