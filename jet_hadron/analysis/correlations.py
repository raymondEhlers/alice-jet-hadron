#!/usr/bin/env python

""" Main jet-hadron correlations analysis module

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

# Py2/3 compatibility
from future.utils import iteritems
from future.utils import itervalues

import collections
import copy
import ctypes
import enum
#import IPython
import logging
import os
import pprint
import math
#import ruamel.yaml as yaml
import sys
import warnings

import scipy
import scipy.signal
import scipy.interpolate
import numpy as np

from pachyderm import generic_config
from pachyderm import histogram
from pachyderm import projectors
from pachyderm.projectors import HistAxisRange
from pachyderm import utils
from pachyderm.utils import epsilon

from jet_hadron.base import analysis_config
from jet_hadron.base import analysis_objects
from jet_hadron.base import params
from jet_hadron.plot import general as plot_general
from jet_hadron.plot import correlations as plot_correlations
from jet_hadron.plot import fit as plot_fit
from jet_hadron.plot import extracted as plot_extracted
from jet_hadron.analysis import fit as fitting

import rootpy.ROOT as ROOT
from rootpy.io import root_open
# Tell ROOT to ignore command line options so args are passed to python
# NOTE: Must be immediately after import ROOT!
ROOT.PyConfig.IgnoreCommandLineOptions = True

# Handle rootpy warning
warnings.filterwarnings(action='ignore', category=RuntimeWarning, message=r'creating converter for unknown type "_Atomic\(bool\)"')

# Setup logger
logger = logging.getLogger(__name__)
# Quiet down the matplotlib logging
logging.getLogger("matplotlib").setLevel(logging.INFO)

# Run in batch mode
ROOT.gROOT.SetBatch(True)

# TODO: Update these enums to lower case names and remove the "k"!!
class JetHCorrelationSparse(enum.Enum):
    """ Defines the axes in the Jet-Hadron THn Sparses. """
    kCentrality = 0
    kJetPt = 1
    kTrackPt = 2
    kDeltaEta = 3
    kDeltaPhi = 4
    kLeadingJet = 5
    kJetHadronDeltaR = 6
    kReactionPlaneOrientation = 7

class JetHTriggerSparse(enum.Enum):
    """ Define the axes in the Jet-Hadron Trigger Sparse. """
    kCentrality = 0
    kJetPt = 1
    kReactionPlaneOrientation = 2

class JetHCorrelationAxis(enum.Enum):
    """ Define the axes of Jet-H 2D correlation hists. """
    kDeltaPhi = projectors.TH1AxisType.x_axis.value
    kDeltaEta = projectors.TH1AxisType.y_axis.value

    def __str__(self):
        """ Turns kDeltaPhi into "deltaPhi" """
        tempStr = self.name.replace("k", "", 1)
        tempStr = tempStr[:1].lower() + tempStr[1:]
        return tempStr

class JetHObservableSparseProjector(projectors.HistProjector):
    """ Projector for THnSparse into analysis_objects.Observable objects. """
    def __init__(self, observable_dict, observables_to_project_from, projectionNameFormat, projectionInformation):
        super().__init__(observable_dict, observables_to_project_from, projectionNameFormat, projectionInformation)

    def OutputHist(self, outputHist, projectionName, *args, **kwargs):
        """ Creates a HistContainer in a Observable to store the output. """
        # In principle, we could pass `**kwargs`, but this could get dangerous if names changes later and
        # initialize something unexpected in the constructor, so instead we'll be explicit
        outputObservable = analysis_objects.Observable(hist = analysis_objects.HistContainer(outputHist))
        return outputObservable

class JetHCorrelationSparseProjector(projectors.HistProjector):
    """ Projector for THnSparse into analysis_objects.CorrelationObservable objects. """
    def __init__(self, observable_dict, observables_to_project_from, projectionNameFormat, projectionInformation):
        super().__init__(observable_dict, observables_to_project_from, projectionNameFormat, projectionInformation)

    def OutputHist(self, outputHist, projectionName, *args, **kwargs):
        """ Creates a HistContainer in a CorrelationObservable to store the output. """
        # In principle, we could pass `**kwargs`, but this could get dangerous if names changes later and
        # initialize something unexpected in the constructor, so instead we'll be explicit
        outputObservable = analysis_objects.CorrelationObservable(
            jetPtBin = kwargs["jetPtBin"],
            trackPtBin = kwargs["trackPtBin"],
            hist = analysis_objects.HistContainer(outputHist)
        )
        return outputObservable

class JetHCorrelationProjector(projectors.HistProjector):
    """ Projector for the Jet-h 2D correlation hists to 1D correlation hists. """
    def __init__(self, observable_dict, observables_to_project_from, projectionNameFormat, projectionInformation):
        super().__init__(observable_dict, observables_to_project_from, projectionNameFormat, projectionInformation)

    def ProjectionName(self, **kwargs):
        """ Define the projection name for the JetH RM projector """
        observable = kwargs["inputObservable"]
        trackPtBin = observable.trackPtBin
        jetPtBin = observable.jetPtBin
        logger.info("Projecting hist name: {}".format(self.projectionNameFormat.format(trackPtBin = trackPtBin, jetPtBin = jetPtBin, **kwargs)))
        return self.projectionNameFormat.format(trackPtBin = trackPtBin, jetPtBin = jetPtBin, **kwargs)

    def GetHist(self, observable, *args, **kwargs):
        """ Return the histogram which is inside of an HistContainer object which is stored in an CorrelationObservable object."""
        return observable.hist.hist

    def OutputHist(self, outputHist, projectionName, *args, **kwargs):
        """ Creates a HistContainer in a CorrelationObservable to store the output. """
        # In principle, we could pass `**kwargs`, but this could get dangerous if names changes later and initialize something
        # unexpected in the constructor, so instead we'll be explicit
        inputObservable = kwargs["inputObservable"]
        outputObservable = analysis_objects.CorrelationObservable1D(
            jetPtBin = inputObservable.jetPtBin,
            trackPtBin = inputObservable.trackPtBin,
            axis = kwargs["axis"],
            correlationType = kwargs["correlationType"],
            hist = analysis_objects.HistContainer(outputHist)
        )
        return outputObservable

class JetHAnalysis(analysis_objects.JetHBase):
    """ Main jet-hadron analysis task. """

    # Properties
    # Define as static variables since they don't depend on any particular instance
    histNameFormat = "jetH%(label)s_jetPt{jetPtBin}_trackPt{trackPtBin}_{tag}"
    histNameFormat2D = histNameFormat % {"label": "DEtaDPhi"}
    # Standard 1D hists
    histNameFormatDPhi = histNameFormat % {"label": "DPhi"}
    histNameFormatDPhiArray = histNameFormat % {"label": "DPhi"} + "Array"
    histNameFormatDEta = histNameFormat % {"label": "DEta"}
    histNameFormatDEtaArray = histNameFormat % {"label": "DEta"} + "Array"
    # Subtracted 1D hists
    histNameFormatDPhiSubtracted = histNameFormatDPhi + "_subtracted"
    histNameFormatDPhiSubtractedArray = histNameFormatDPhiArray + "_subtracted"
    histNameFormatDEtaSubtracted = histNameFormatDEta + "_subtracted"
    histNameFormatDEtaSubtractedArray = histNameFormatDEtaArray + "_subtracted"

    # These is nothing here to format - it's just the jet spectra
    # However, the variable name will stay the same for clarity
    histNameFormatTrigger = "jetHTriggerPt"
    fitNameFormat = histNameFormat % {"label": "Fit"}

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
        self.generalHists1D = collections.OrderedDict()
        self.generalHists2D = collections.OrderedDict()

        self.generalHists = [self.generalHists1D, self.generalHists2D]

        # Trigger jet pt. Used for calculating N_trig
        self.triggerJetPt = collections.OrderedDict()

        # 2D correlations
        self.rawSignal2D = collections.OrderedDict()
        self.mixedEvents2D = collections.OrderedDict()
        self.signal2D = collections.OrderedDict()

        # All 2D hists
        # NOTE: We can't use an iterator here because we rely on them being separate for initialzing from ROOT files.
        #       Would look something like:
        #self.hists2D = itertools.chain(self.rawSignal2D, self.mixedEvents2D, self.signal2D)
        self.hists2D = [self.rawSignal2D, self.mixedEvents2D, self.signal2D]

        # 1D correlations
        self.dPhi = collections.OrderedDict()
        self.dPhiArray = collections.OrderedDict()
        self.dPhiSubtracted = collections.OrderedDict()
        self.dPhiSubtractedArray = collections.OrderedDict()
        self.dPhiSideBand = collections.OrderedDict()
        self.dPhiSideBandArray = collections.OrderedDict()
        self.dEtaNS = collections.OrderedDict()
        self.dEtaNSArray = collections.OrderedDict()
        self.dEtaNSSubtracted = collections.OrderedDict()
        self.dEtaNSSubtractedArray = collections.OrderedDict()

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
        self.dPhiFit = collections.OrderedDict()
        self.dPhiSideBandFit = collections.OrderedDict()
        self.dPhiSubtractedFit = collections.OrderedDict()
        self.dEtaNSFit = collections.OrderedDict()
        self.dEtaNSSubtractedFit = collections.OrderedDict()

        # All 1D fits
        self.fits1D = [self.dPhiFit, self.dPhiSideBandFit, self.dPhiSubtractedFit, self.dEtaNSFit, self.dEtaNSSubtractedFit]
        # Standard
        self.fits1DStandard = [self.dPhiFit, self.dPhiSideBandFit, self.dEtaNSFit]
        # Subtracted
        self.fits1DSubtracted = [self.dPhiSubtractedFit, self.dEtaNSSubtractedFit]

        # Yields
        self.yieldsNS = collections.OrderedDict()
        self.yieldsAS = collections.OrderedDict()
        self.yieldsDEtaNS = collections.OrderedDict()

        # All yields
        self.yields = [self.yieldsNS, self.yieldsAS, self.yieldsDEtaNS]

        # Widths
        self.widthsNS = collections.OrderedDict()
        self.widthsAS = collections.OrderedDict()
        self.widthsDEtaNS = collections.OrderedDict()

        # All widths
        self.widths = [self.widthsNS, self.widthsAS, self.widthsDEtaNS]

    def retrieveInputHists(self):
        """ Run general setup tasks. """
        # Retrieve all histograms
        self.inputHists = utils.getHistogramsInList(self.inputFilename, self.inputListName)

    def assignGeneralHistsFromDict(self, histDict, outputDict):
        """ Simple helper to assign hists named in a dict to an output dict. """
        for name, histName in iteritems(histDict):
            # NOTE: The hist may not always exist, so we return None if it doesn't!
            outputDict[name] = self.inputHists.get(histName, None)

    def generalHistograms(self):
        """ Process some general histograms such as centralty, Z vertex, very basic QA spectra, etc. """
        # Get configuration
        processingOptions = self.taskConfig["processingOptions"]

        if processingOptions["generalHistograms"]:
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

        # Project the histograms
        # Includes the trigger, raw signal 2D, and mixed event 2D hists
        for projector in self.sparseProjectors:
            projector.Project()

        # Determine nTrig for the jet pt bins
        nTrig = {}
        triggerObservable = self.triggerJetPt[self.histNameFormatTrigger]
        for iJetPtBin in params.iterateOverJetPtBins(self.config):
            """
            When retrieving the number of triggers, carefully noting the information below.
            >>> hist = ROOT.TH1D("test", "test", 10, 0, 10)
            >>> x = 2, y = 5
            >>> hist.FindBin(x)
            2
            >>> hist.FindBin(x+epsilon)
            2
            >>> hist.FindBin(y)
            6
            >>> hist.FindBin(y-epsilon)
            5

            NOTE: The bin + epsilon on the lower bin is not strictly necessary, but it is used for consistency.
            """
            triggerHist = triggerObservable.hist
            logger.debug("Find bin({}+epsilon): {} to Find bin({}-epsilon): {}".format(
                params.jetPtBins[iJetPtBin],
                triggerHist.FindBin(params.jetPtBins[iJetPtBin] + epsilon),
                params.jetPtBins[iJetPtBin + 1],
                triggerHist.FindBin(params.jetPtBins[iJetPtBin + 1] - epsilon))
            )
            nTrigInJetPtBin = triggerHist.Integral(
                triggerHist.FindBin(params.jetPtBins[iJetPtBin] + epsilon),
                triggerHist.FindBin(params.jetPtBins[iJetPtBin + 1] - epsilon)
            )
            logger.info("nTrig for [{}, {}): {}".format(params.jetPtBins[iJetPtBin], params.jetPtBins[iJetPtBin + 1], nTrigInJetPtBin))
            nTrig[iJetPtBin] = nTrigInJetPtBin

        # TODO: Use a broader range of pt for mixed events like Joel?
        for (rawObservable, mixedEventObservable) in zip(itervalues(self.rawSignal2D), itervalues(self.mixedEvents2D)):
            # Check to ensure that we've zipped properly
            if rawObservable.jetPtBin != mixedEventObservable.jetPtBin or rawObservable.trackPtBin != mixedEventObservable.trackPtBin:
                raise ValueError("Mismatch in jet or track pt bins. raw: (jet: {}, track: {}), mixed: (jet: {}, track: {})!!".format(
                    rawObservable.jetPtBin,
                    rawObservable.trackPtBin,
                    mixedEventObservable.jetPtBin,
                    mixedEventObservable.trackPtBin)
                )

            logger.debug("Processing correlations raw: {} (hist: {}), and mixed: {} (hist: {})".format(rawObservable, rawObservable.hist.hist, mixedEventObservable, mixedEventObservable.hist.hist))

            # Helper dict
            binningDict = {"trackPtBin": rawObservable.trackPtBin, "jetPtBin": rawObservable.jetPtBin}

            # Normalize and post process the raw observable
            postProcessingArgs = {"observable": rawObservable,
                                  "normalizationFactor": nTrig[binningDict["jetPtBin"]],
                                  "titleLabel": "Raw signal"}
            postProcessingArgs.update(binningDict)
            JetHAnalysis.postProjectionProcessing2DCorrelation(**postProcessingArgs)

            # Normalize and post process the mixed event observable
            normalizationFactor = self.measureMixedEventNormalization(mixedEvent = mixedEventObservable.hist, **binningDict)
            postProcessingArgs.update({"observable": mixedEventObservable,
                                       "normalizationFactor": normalizationFactor,
                                       "titleLabel": "Mixed Event"})
            JetHAnalysis.postProjectionProcessing2DCorrelation(**postProcessingArgs)

            # Rebin
            # TODO: Work on rebin quality...

    def generate2DSignalCorrelation(self):
        """ Generate 2D signal correlation.

        Intentionally decoupled for creating the raw and mixed event hists so that the THnSparse can be swapped out when desired.
        """
        for (rawObservable, mixedEventObservable) in zip(itervalues(self.rawSignal2D), itervalues(self.mixedEvents2D)):
            # Check to ensure that we've zipped properly
            if rawObservable.jetPtBin != mixedEventObservable.jetPtBin or rawObservable.trackPtBin != mixedEventObservable.trackPtBin:
                raise ValueError("Mismatch in jet or track pt bins. raw: (jet: {}, track: {}), mixed: (jet: {}, track: {})!!".format(
                    rawObservable.jetPtBin,
                    rawObservable.trackPtBin,
                    mixedEventObservable.jetPtBin,
                    mixedEventObservable.trackPtBin)
                )

            # Define for convenience
            jetPtBin = rawObservable.jetPtBin
            trackPtBin = rawObservable.trackPtBin
            binningDict = {"trackPtBin": trackPtBin, "jetPtBin": jetPtBin}

            # Correlation - Divide signal by mixed events
            correlation = rawObservable.hist.Clone(self.histNameFormat2D.format(jetPtBin = jetPtBin, trackPtBin = trackPtBin, tag = "corr"))
            correlation.Divide(mixedEventObservable.hist.hist)

            # Create the observable
            signal2DObservable = analysis_objects.CorrelationObservable(
                jetPtBin = jetPtBin,
                trackPtBin = trackPtBin,
                hist = analysis_objects.HistContainer(correlation)
            )

            # Post process the signal 2D hist
            postProcessingArgs = {}
            postProcessingArgs.update({"observable": signal2DObservable,
                                       "normalizationFactor": 1.0,
                                       "titleLabel": "Correlation"})
            postProcessingArgs.update(binningDict)
            JetHAnalysis.postProjectionProcessing2DCorrelation(**postProcessingArgs)

            # Save the observable
            self.signal2D[correlation.GetName()] = signal2DObservable

    def generate1DCorrelations(self):
        """ Generate 1D Correlation by projecting 2D correlations. """

        # Project the histograms
        # Includes the dPhi signal dominated, dPhi background dominated, and dEta near side
        for projector in self.correlationProjectors:
            projector.Project()

        # Post process and scale
        for hists in self.hists1DStandard:
            for name, observable in iteritems(hists):
                logger.info("Post projection processing of 1D correlation {}".format(name))

                # Define for convenience
                jetPtBin = observable.jetPtBin
                trackPtBin = observable.trackPtBin
                binningDict = {"trackPtBin": trackPtBin, "jetPtBin": jetPtBin}

                # Determine normalization factor
                # We only apply this so we don't unnecessarily scale the signal region.
                # However, it is then important that we report the eta range in which we measure!
                normalizationFactor = 1
                # TODO: IMPORTANT: Remove hard code here and restore proper scaling!
                # Scale is dependent on the signal and background range
                # Since this is hard-coded, it is calculated very explicitly so it will
                # be caught if the values are modified.
                # Ranges are multiplied by 2 because the ranges are symmetric
                signalMinVal = params.etaBins[params.etaBins.index(0.0)]
                signalMaxVal = params.etaBins[params.etaBins.index(0.6)]
                signalRange = (signalMaxVal - signalMinVal) * 2
                backgroundMinVal = params.etaBins[params.etaBins.index(0.8)]
                backgroundMaxVal = params.etaBins[params.etaBins.index(1.2)]
                backgroundRange = (backgroundMaxVal - backgroundMinVal) * 2

                ################
                # If we wanted to plug into the projectors (it would take some work), we could do something like:
                ## Determine the min and max values
                #axis = projector.axis(observable.hist.hist)
                #minVal = rangeSet.minVal(axis)
                #maxVal = rangeSet.maxVal(axis)
                ## Determine the projection range for proper scaling.
                #projectionRange += (axis.GetBinUpEdge(maxVal) - axis.GetBinLowEdge(minVal))
                ################
                # Could also consider trying to get the projector directly and apply it to a hist
                ################

                if observable.correlationType == analysis_objects.JetHCorrelationType.background_dominated:
                    # Scale by (signal region)/(background region)
                    # NOTE: Will be applied as `1/normalizationFactor`, so the value is the inverse
                    #normalizationFactor = backgroundRange/signalRange
                    normalizationFactor = backgroundRange
                    logger.debug("Scaling background by normalizationFactor {}".format(normalizationFactor))
                else:
                    normalizationFactor = signalRange
                    logger.debug("Scaling signal by normalizationFactor {}".format(normalizationFactor))

                # Post process and scale
                postProcessingArgs = {}
                postProcessingArgs.update({"observable": observable,
                                           "normalizationFactor": normalizationFactor,
                                           "rebinFactor": 2,
                                           "titleLabel": "{}, {}".format(observable.correlationType.name, str(observable.axis))})
                postProcessingArgs.update(binningDict)
                JetHAnalysis.postProjectionProcessing1DCorrelation(**postProcessingArgs)

    def post1DProjectionScaling(self):
        """ Perform post projection scalings.

        In particular, scale the 1D hists by their bin widths so no further scaling is necessary.
        """
        for hists in self.hists1DStandard:
            #logger.debug("len(hists): {}, hists.keys(): {}".format(len(hists), hists.keys()))
            for name, observable in iteritems(hists):
                scaleFactor = observable.hist.calculateFinalScaleFactor()
                #logger.info("Post projection scaling of hist {} with scale factor 1/{}".format(name, 1/scaleFactor))
                observable.hist.Scale(scaleFactor)

    def generateSparseProjectors(self):
        """ Generate sparse projectors """
        # Helper which defines the full axis range
        fullAxisRange = {"minVal": HistAxisRange.ApplyFuncToFindBin(None, 1),
                         "maxVal": HistAxisRange.ApplyFuncToFindBin(ROOT.TAxis.GetNbins)}

        # Define common axes
        # Centrality axis
        centralityCutAxis = HistAxisRange(
            axisType = JetHCorrelationSparse.kCentrality,
            axisRangeName = "centrality",
            minVal = HistAxisRange.ApplyFuncToFindBin(ROOT.TAxis.FindBin, 0 + epsilon),
            maxVal = HistAxisRange.ApplyFuncToFindBin(ROOT.TAxis.FindBin, 10 - epsilon)
        )
        # Event plane selection
        reaction_plane_orientation_cut_axis = None
        if self.reaction_plane_orientation:
            if self.reaction_plane_orientation == params.ReactionPlaneOrientation.all:
                eventPlaneAxisRange = fullAxisRange
                logger.info("Using full EP angle range")
            else:
                # TODO: Update enum to have both axis number and range of the selection.
                eventPlaneAxisRange = {
                    "minVal": HistAxisRange.ApplyFuncToFindBin(None, self.reaction_plane_orientation.value),
                    "maxVal": HistAxisRange.ApplyFuncToFindBin(None, self.reaction_plane_orientation.value)
                }
                logger.info("Using selected EP angle range {self.reaction_plane_orientation.name}")

            reaction_plane_orientation_cut_axis = HistAxisRange(
                axisType = JetHCorrelationSparse.kReactionPlaneOrientation,
                axisRangeName = "eventPlane", **eventPlaneAxisRange
            )
        # dPhi full axis
        dPhiAxis = HistAxisRange(axisType = JetHCorrelationSparse.kDeltaPhi,
                                 axisRangeName = "deltaPhi", **fullAxisRange)
        # dEta full axis
        dEtaAxis = HistAxisRange(axisType = JetHCorrelationSparse.kDeltaEta,
                                 axisRangeName = "deltaEta", **fullAxisRange)

        ###########################
        # Trigger projector
        #
        # Note that it has no jet pt or trigger pt dependence.
        # We will select jet pt ranges later when determining nTrig
        ###########################
        projectionInformation = {}
        # Attempt to format for consistency, although it doesn't do anything for the trigger projection
        triggerInputDict = {self.histNameFormatTrigger.format(**projectionInformation): self.inputHists["fhnTrigger"]}
        triggerProjector = JetHObservableSparseProjector(
            observable_dict = self.triggerJetPt,
            observables_to_project_from = triggerInputDict,
            projectionNameFormat = self.histNameFormatTrigger,
            projectionInformation = projectionInformation
        )
        # Take advantage of existing centrality and event plane object, but need to copy and modify the axis type
        if self.collisionSystem != params.collisionSystem.pp:
            triggerCentralityCutAxis = copy.deepcopy(centralityCutAxis)
            triggerCentralityCutAxis.axisType = JetHTriggerSparse.kCentrality
            triggerProjector.additionalAxisCuts.append(triggerCentralityCutAxis)
        if reaction_plane_orientation_cut_axis:
            triggerReactionPlaneOrientationCutAxis = copy.deepcopy(reaction_plane_orientation_cut_axis)
            triggerReactionPlaneOrientationCutAxis.axisType = JetHTriggerSparse.kReactionPlaneOrientation
            triggerProjector.additionalAxisCuts.append(triggerReactionPlaneOrientationCutAxis)
        # No projection dependent cut axes
        triggerProjector.projectionDependentCutAxes.append([])
        # Projection axis
        triggerProjector.projectionAxes.append(
            HistAxisRange(
                axisType = JetHTriggerSparse.kJetPt,
                axisRangeName = "jetPt",
                **fullAxisRange
            )
        )
        self.sparseProjectors.append(triggerProjector)

        # Jet and track pt bin dependent cuts
        for (iJetPtBin, iTrackPtBin) in params.iterateOverJetAndTrackPtBins(self.config):
            jetAxisRange = {
                "minVal": HistAxisRange.ApplyFuncToFindBin(ROOT.TAxis.FindBin, params.jetPtBins[iJetPtBin] + epsilon),
                "maxVal": HistAxisRange.ApplyFuncToFindBin(ROOT.TAxis.FindBin, params.jetPtBins[iJetPtBin + 1] - epsilon)
            }
            trackAxisRange = {
                "minVal": HistAxisRange.ApplyFuncToFindBin(ROOT.TAxis.FindBin, params.trackPtBins[iTrackPtBin] + epsilon),
                "maxVal": HistAxisRange.ApplyFuncToFindBin(ROOT.TAxis.FindBin, params.trackPtBins[iTrackPtBin + 1] - epsilon)
            }
            projectionInformation = {"jetPtBin": iJetPtBin, "trackPtBin": iTrackPtBin}

            ###########################
            # Raw signal projector
            ###########################
            projectionInformation["tag"] = "raw"
            rawSignalInputDict = {self.histNameFormat2D.format(**projectionInformation): self.inputHists["fhnJH"]}
            rawSignalProjector = JetHCorrelationSparseProjector(
                observable_dict = self.rawSignal2D,
                observables_to_project_from = rawSignalInputDict,
                projectionNameFormat = self.histNameFormat2D,
                projectionInformation = projectionInformation
            )
            if self.collisionSystem != params.collisionSystem.pp:
                rawSignalProjector.additionalAxisCuts.append(centralityCutAxis)
            if reaction_plane_orientation_cut_axis:
                rawSignalProjector.additionalAxisCuts.append(reaction_plane_orientation_cut_axis)
            # TODO: Do these projectors really need projection dependent cut axes?
            #       It seems like additionalAxisCuts would be sufficient.
            projectionDependentCutAxes = []
            projectionDependentCutAxes.append(
                HistAxisRange(
                    axisType = JetHCorrelationSparse.kJetPt,
                    axisRangeName = "jetPt{}".format(iJetPtBin),
                    **jetAxisRange
                )
            )
            projectionDependentCutAxes.append(
                HistAxisRange(
                    axisType = JetHCorrelationSparse.kTrackPt,
                    axisRangeName = "trackPt{}".format(iTrackPtBin),
                    **trackAxisRange
                )
            )
            # NOTE: We are passing a list to the list of cuts. Therefore, the two cuts defined above will be applied on the same projection!
            rawSignalProjector.projectionDependentCutAxes.append(projectionDependentCutAxes)
            # Projection Axes
            rawSignalProjector.projectionAxes.append(dPhiAxis)
            rawSignalProjector.projectionAxes.append(dEtaAxis)
            self.sparseProjectors.append(rawSignalProjector)

            ###########################
            # Mixed Event projector
            ###########################
            projectionInformation["tag"] = "mixed"
            mixedEventInputDict = {self.histNameFormat2D.format(**projectionInformation): self.inputHists["fhnMixedEvents"]}
            mixedEventProjector = JetHCorrelationSparseProjector(
                observable_dict = self.mixedEvents2D,
                observables_to_project_from = mixedEventInputDict,
                projectionNameFormat = self.histNameFormat2D,
                projectionInformation = projectionInformation
            )
            if self.collisionSystem != params.collisionSystem.pp:
                mixedEventProjector.additionalAxisCuts.append(centralityCutAxis)
            if reaction_plane_orientation_cut_axis:
                mixedEventProjector.additionalAxisCuts.append(reaction_plane_orientation_cut_axis)
            projectionDependentCutAxes = []
            projectionDependentCutAxes.append(
                HistAxisRange(
                    axisType = JetHCorrelationSparse.kJetPt,
                    axisRangeName = "jetPt{}".format(iJetPtBin),
                    **jetAxisRange
                )
            )
            projectionDependentCutAxes.append(
                HistAxisRange(
                    axisType = JetHCorrelationSparse.kTrackPt,
                    axisRangeName = "trackPt{}".format(iTrackPtBin),
                    **trackAxisRange
                )
            )
            # NOTE: We are passing a list to the list of cuts. Therefore, the two cuts defined above will be applied on the same projection!
            mixedEventProjector.projectionDependentCutAxes.append(projectionDependentCutAxes)
            # Projection Axes
            mixedEventProjector.projectionAxes.append(dPhiAxis)
            mixedEventProjector.projectionAxes.append(dEtaAxis)
            self.sparseProjectors.append(mixedEventProjector)

    def generateCorrelationProjectors(self):
        """ Generate correlation projectors (2D -> 1D) """
        # Helper which defines the full axis range
        fullAxisRange = {"minVal": HistAxisRange.ApplyFuncToFindBin(None, 1),
                         "maxVal": HistAxisRange.ApplyFuncToFindBin(ROOT.TAxis.GetNbins)}

        ###########################
        # dPhi Signal
        ###########################
        projectionInformation = {"correlationType": analysis_objects.JetHCorrelationType.signal_dominated,
                                 "axis": JetHCorrelationAxis.kDeltaPhi}
        projectionInformation["tag"] = projectionInformation["correlationType"].str()
        dPhiSignalProjector = JetHCorrelationProjector(
            observable_dict = self.dPhi,
            observables_to_project_from = self.signal2D,
            projectionNameFormat = self.histNameFormatDPhi,
            projectionInformation = projectionInformation
        )
        # Select signal dominated region in eta
        # Could be a single range, but this is conceptually clearer when compared to the background
        # dominated region. Need to do this as projection dependent cuts because it is selecting different
        # ranges on the same axis
        dPhiSignalProjector.projectionDependentCutAxes.append([
            HistAxisRange(
                axisType = JetHCorrelationAxis.kDeltaEta,
                axisRangeName = "NegativeEtaSignalDominated",
                minVal = HistAxisRange.ApplyFuncToFindBin(ROOT.TAxis.FindBin, -1 * params.etaBins[params.etaBins.index(0.6)] + epsilon),
                maxVal = HistAxisRange.ApplyFuncToFindBin(ROOT.TAxis.FindBin, -1 * params.etaBins[params.etaBins.index(0)] - epsilon)
            )
        ])
        dPhiSignalProjector.projectionDependentCutAxes.append([
            HistAxisRange(
                axisType = JetHCorrelationAxis.kDeltaEta,
                axisRangeName = "PositiveEtaSignalDominated",
                minVal = HistAxisRange.ApplyFuncToFindBin(ROOT.TAxis.FindBin, params.etaBins[params.etaBins.index(0)] + epsilon),
                maxVal = HistAxisRange.ApplyFuncToFindBin(ROOT.TAxis.FindBin, params.etaBins[params.etaBins.index(0.6)] - epsilon)
            )
        ])
        dPhiSignalProjector.projectionAxes.append(
            HistAxisRange(
                axisType = JetHCorrelationAxis.kDeltaPhi,
                axisRangeName = "deltaPhi",
                **fullAxisRange
            )
        )
        self.correlationProjectors.append(dPhiSignalProjector)

        ###########################
        # dPhi Background dominated
        ###########################
        projectionInformation = {"correlationType": analysis_objects.JetHCorrelationType.background_dominated,
                                 "axis": JetHCorrelationAxis.kDeltaPhi}
        projectionInformation["tag"] = projectionInformation["correlationType"].str()
        dPhiBackgroundProjector = JetHCorrelationProjector(
            observable_dict = self.dPhiSideBand,
            observables_to_project_from = self.signal2D,
            projectionNameFormat = self.histNameFormatDPhi,
            projectionInformation = projectionInformation
        )
        # Select background dominated region in eta
        # Redundant to find the index, but it helps check that it is actually in the list!
        # Need to do this as projection dependent cuts because it is selecting different ranges
        # on the same axis
        dPhiBackgroundProjector.projectionDependentCutAxes.append([
            HistAxisRange(
                axisType = JetHCorrelationAxis.kDeltaEta,
                axisRangeName = "NegativeEtaBackgroundDominated",
                minVal = HistAxisRange.ApplyFuncToFindBin(ROOT.TAxis.FindBin, -1 * params.etaBins[params.etaBins.index(1.2)] + epsilon),
                maxVal = HistAxisRange.ApplyFuncToFindBin(ROOT.TAxis.FindBin, -1 * params.etaBins[params.etaBins.index(0.8)] - epsilon)
            )
        ])
        dPhiBackgroundProjector.projectionDependentCutAxes.append([
            HistAxisRange(
                axisType = JetHCorrelationAxis.kDeltaEta,
                axisRangeName = "PositiveEtaBackgroundDominated",
                minVal = HistAxisRange.ApplyFuncToFindBin(ROOT.TAxis.FindBin, params.etaBins[params.etaBins.index(0.8)] + epsilon),
                maxVal = HistAxisRange.ApplyFuncToFindBin(ROOT.TAxis.FindBin, params.etaBins[params.etaBins.index(1.2)] - epsilon)
            )
        ])
        dPhiBackgroundProjector.projectionAxes.append(
            HistAxisRange(
                axisType = JetHCorrelationAxis.kDeltaPhi,
                axisRangeName = "deltaPhi",
                **fullAxisRange
            )
        )
        self.correlationProjectors.append(dPhiBackgroundProjector)

        ###########################
        # dEta NS
        ###########################
        projectionInformation = {"correlationType": analysis_objects.JetHCorrelationType.near_side,
                                 "axis": JetHCorrelationAxis.kDeltaEta}
        projectionInformation["tag"] = projectionInformation["correlationType"].str()
        dEtaNSProjector = JetHCorrelationProjector(
            observable_dict = self.dEtaNS,
            observables_to_project_from = self.signal2D,
            projectionNameFormat = self.histNameFormatDEta,
            projectionInformation = projectionInformation
        )
        # Select near side in delta phi
        dEtaNSProjector.additionalAxisCuts.append(
            HistAxisRange(
                axisType = JetHCorrelationAxis.kDeltaPhi,
                axisRangeName = "deltaPhiNearSide",
                minVal = HistAxisRange.ApplyFuncToFindBin(ROOT.TAxis.FindBin, params.phiBins[params.phiBins.index(-1. * math.pi / 2.)] + epsilon),
                maxVal = HistAxisRange.ApplyFuncToFindBin(ROOT.TAxis.FindBin, params.phiBins[params.phiBins.index(1. * math.pi / 2.)] - epsilon)
            )
        )
        # No projection dependent cut axes
        dEtaNSProjector.projectionDependentCutAxes.append([])
        dEtaNSProjector.projectionAxes.append(
            HistAxisRange(
                axisType = JetHCorrelationAxis.kDeltaEta,
                axisRangeName = "deltaEta",
                **fullAxisRange
            )
        )
        self.correlationProjectors.append(dEtaNSProjector)

    def convert1DRootHistsToArray(self, inputHists, outputHists):
        """ Convert requested 1D hists to hist array format. """
        for observable in itervalues(inputHists):
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

        for name, observable in iteritems(hists):
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
        for ((name, observable), signalFit, bgFit) in zip(iteritems(hists), itervalues(signalFits), itervalues(backgroundFits)):
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

        for location, (centralValue, yields, hists, fits) in iteritems(parameters):
            for (name, observable), fit in zip(iteritems(hists), itervalues(fits)):
                # Extract yield
                minVal = observable.hist.GetXaxis().FindBin(centralValue - yieldLimit + utils.epsilon)
                maxVal = observable.hist.GetXaxis().FindBin(centralValue + yieldLimit - utils.epsilon)
                yieldError = ctypes.c_double(0)
                yieldValue = observable.hist.IntegralAndError(minVal, maxVal, yieldError, "width")

                # Convert ctype back to python type for convenience
                yieldError = yieldError.value

                # Scale by track pt bin width
                trackPtBinWidth = params.trackPtBins[observable.trackPtBin + 1] - params.trackPtBins[observable.trackPtBin]
                yieldValue /= trackPtBinWidth
                yieldError /= trackPtBinWidth

                # Store yield
                yields["{}_yield".format(name)] = analysis_objects.ExtractedObservable(
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

        for location, (parameterNumber, widths, hists, fits) in iteritems(parameters):
            for (name, observable), fit in zip(iteritems(hists), itervalues(fits)):
                widths["{}_width".format(name)] = analysis_objects.ExtractedObservable(
                    jetPtBin = observable.jetPtBin,
                    trackPtBin = observable.trackPtBin,
                    value = fit.GetParameter(parameterNumber),
                    error = fit.GetParError(parameterNumber)
                )

    def writeToRootFile(self, observable_dict, mode = "UPDATE"):
        """ Write output list to a file """
        filename = os.path.join(self.outputPrefix, self.outputFilename)

        logger.info("Saving correlations to {}".format(filename))

        with root_open(filename, mode) as fOut:  # noqa: 854
            for histCollection in observable_dict:
                for name, observable in iteritems(histCollection):
                    if isinstance(observable, analysis_objects.Observable):
                        hist = observable.hist
                    else:
                        hist = observable

                    hist.Write()

    def writeHistsToYAML(self, observable_dict, mode = "wb"):
        """ Write hist to YAML file. """

        logger.info("Saving hist arrays!")

        for histCollection in observable_dict:
            for name, observable in iteritems(histCollection):
                if isinstance(observable, analysis_objects.Observable):
                    hist = observable.hist
                else:
                    hist = observable

                hist.saveToYAML(prefix = self.outputPrefix,
                                objType = observable.correlationType,
                                jetPtBin = observable.jetPtBin,
                                trackPtBin = observable.trackPtBin,
                                fileAccessMode = mode)

    def writeGeneralHistograms(self):
        """ Write general histograms to file. """
        self.writeToRootFile(self.generalHists)

    def writeTriggerJetSpectra(self):
        """ Write trigger jet spectra to file. """
        self.writeToRootFile([self.triggerJetPt])

    def write2DCorrelations(self):
        """ Write the 2D Correlations to file. """
        self.writeToRootFile(self.hists2D)

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

        # TODO: Depending on this ROOT file when opening a YAML file is not really right.
        #       However, it's fine for now because the ROOT file should almost always exist

        # Open the ROOT file
        with root_open(os.path.join(self.outputPrefix, self.outputFilename), "READ") as fIn:
            if GeneralHists:
                logger.critical("General hists are not yet implemented!")
                sys.exit(1)

            if TriggerJetSpectra:
                # We only have one trigger jet hist at the moment, but by using the same approach
                # as for the correlations, it makes it straightforward to generalize later if needed
                histName = self.histNameFormatTrigger
                hist = fIn.Get(histName)
                if hist:
                    self.triggerJetPt[histName] = analysis_objects.Observable(hist = analysis_objects.HistContainer(hist))
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
                        hist = fIn.Get(histName)
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
                        hists1D.append([self.dPhi,         self.histNameFormatDPhi, analysis_objects.JetHCorrelationType.signal_dominated,     JetHCorrelationAxis.kDeltaPhi])  # noqa: E241
                        hists1D.append([self.dPhiSideBand, self.histNameFormatDPhi, analysis_objects.JetHCorrelationType.background_dominated, JetHCorrelationAxis.kDeltaPhi])  # noqa: E241
                        hists1D.append([self.dEtaNS,       self.histNameFormatDEta, analysis_objects.JetHCorrelationType.near_side,            JetHCorrelationAxis.kDeltaEta])  # noqa: E241
                    if Correlations1DSubtracted:
                        hists1D.append([self.dPhiSubtracted,   self.histNameFormatDPhiSubtracted, analysis_objects.JetHCorrelationType.signal_dominated,     JetHCorrelationAxis.kDeltaPhi])  # noqa: E241
                        hists1D.append([self.dEtaNSSubtracted, self.histNameFormatDEtaSubtracted, analysis_objects.JetHCorrelationType.near_side,            JetHCorrelationAxis.kDeltaEta])  # noqa: E241
                    if Correlations1DArray:
                        logger.debug("Correlations1DArray hists")
                        retrieveArray = True
                        hists1D.append([self.dPhiArray,         self.histNameFormatDPhiArray, analysis_objects.JetHCorrelationType.signal_dominated,     JetHCorrelationAxis.kDeltaPhi])  # noqa: E241
                        hists1D.append([self.dPhiSideBandArray, self.histNameFormatDPhiArray, analysis_objects.JetHCorrelationType.background_dominated, JetHCorrelationAxis.kDeltaPhi])  # noqa: E241
                        hists1D.append([self.dEtaNS,            self.histNameFormatDEtaArray, analysis_objects.JetHCorrelationType.near_side,            JetHCorrelationAxis.kDeltaEta])  # noqa: E241
                    if Correlations1DSubtractedArray:
                        retrieveArray = True
                        hists1D.append([self.dPhiSubtractedArray,   self.histNameFormatDPhiSubtractedArray, analysis_objects.JetHCorrelationType.signal_dominated, JetHCorrelationAxis.kDeltaPhi])  # noqa: E241
                        hists1D.append([self.dEtaNSSubtractedArray, self.histNameFormatDEtaSubtractedArray, analysis_objects.JetHCorrelationType.near_side,        JetHCorrelationAxis.kDeltaEta])  # noqa: E241

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
                            hist = fIn.Get(histName)
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
        """ Determine normalization of the mixed event.
        """
        # Project to 1D dPhi so it can be used with the signal finder
        # We need to project over a range of constant eta to be able to use the extracted max in the 2D
        # mixed event. Joel uses [-0.4, 0.4], but it really seems to drop in the 0.4 bin, so instead I'll
        # use 0.3 This value also depends on the max track eta. For 0.9, it should be 0.4 (0.9-0.5), but
        # for 0.8, it should be 0.3 (0.8-0.5)
        etaLimits = self.taskConfig["mixedEventNormalizationOptions"].get("etaLimits", [-0.3, 0.3])
        # Scale the 1D norm by the eta range.
        etaLimitBins = [mixedEvent.GetYaxis().FindBin(etaLimits[0] + epsilon), mixedEvent.GetYaxis().FindBin(etaLimits[1] - epsilon)]
        # This is basically just a sanity check that the selected values align with the binning
        projectionLength = mixedEvent.GetYaxis().GetBinUpEdge(etaLimitBins[1]) - mixedEvent.GetYaxis().GetBinLowEdge(etaLimitBins[0])
        logger.info("Scale factor from 1D to 2D: {}".format(mixedEvent.GetYaxis().GetBinWidth(1) / projectionLength))
        peakFindingHist = mixedEvent.ProjectionX("{}_peakFindingHist".format(mixedEvent.GetName()), etaLimitBins[0], etaLimitBins[1])
        peakFindingHist.Scale(mixedEvent.GetYaxis().GetBinWidth(1) / projectionLength)
        peakFindingArray = histogram.Histogram1D(peakFindingHist).y
        #logger.debug("peakFindingArray: {}".format(peakFindingArray))

        # Using moving average
        movingAvg = utils.movingAverage(peakFindingArray, n = 36)
        maxMovingAvg = max(movingAvg)

        compareNormalizationOptions = self.taskConfig["mixedEventNormalizationOptions"].get("compareOptions", False)
        if compareNormalizationOptions:
            logger.info("Comparing mixed event normalization options!")
            self.compareMixedEventNormalizationOptions(
                mixedEvent = mixedEvent, jetPtBin = jetPtBin, trackPtBin = trackPtBin,
                etaLimits = etaLimits,
                peakFindingHist = peakFindingHist,
                peakFindingArray = peakFindingArray,
                maxMovingAvg = maxMovingAvg
            )

        mixedEventNormalization = maxMovingAvg
        if not mixedEventNormalization != 0:
            logger.warning("Could not normalize the mixed event hist \"{0}\" due to no data at (0,0)!".format(mixedEvent.GetName()))
            mixedEventNormalization = 1

        return mixedEventNormalization

    def compareMixedEventNormalizationOptions(self, mixedEvent, jetPtBin, trackPtBin, etaLimits, peakFindingHist, peakFindingArray, maxMovingAvg):
        """ Compare mixed event normalization options.

        The large window over which the normalization is extracted seems to be important to avoid fluctatuions.

        Also allows for comparison of:
            - Continuous wave transform with width ~ pi
            - Smoothing data assuming the points are distributed as a gaussian with options of:
                - Max of smoothed function
                - Moving average over pi of smoothed function
            - Moving average over pi
            - Linear 1D fit
            - Linear 2D fit

        All of the above were also performed over a 2 bin rebin except for the gaussian smoothed function.
        """
        # Create rebinned hist
        # The rebinned hist may be less susceptible to noise, so it should be compared.
        # Only rebin the 2D in dPhi because otherwise the dEta bins will not align with the limits
        mixedEventRebin = mixedEvent.Rebin2D(2, 1, mixedEvent.GetName() + "Rebin")
        mixedEventRebin.Scale(1. / (2. * 1.))
        peakFindingHistRebin = peakFindingHist.Rebin(2, peakFindingHist.GetName() + "Rebin")
        peakFindingHistRebin.Scale(1. / 2.)
        # Note that peak finding will only be performed on the 1D hist
        peakFindingArrayRebin = histogram.Histogram1D(peakFindingHistRebin).y

        # Define points where the plots and functions can be evaluted
        linSpace = np.linspace(-0.5 * np.pi, 3. / 2 * np.pi, len(peakFindingArray))
        linSpaceRebin = np.linspace(-0.5 * np.pi, 3. / 2 * np.pi, len(peakFindingArrayRebin))

        # Using CWT
        # See: https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.find_peaks_cwt.html
        # and: https://stackoverflow.com/a/42285002
        peakLocations = scipy.signal.find_peaks_cwt(peakFindingArray, widths = np.arange(20, 50, .1))
        peakLocationsRebin = scipy.signal.find_peaks_cwt(peakFindingArrayRebin, widths = np.arange(10, 25, .05))
        logger.info("peakLocations: {}, values: {}".format(peakLocations, peakFindingArray[peakLocations]))

        # Using gaussian smoothing
        # See: https://stackoverflow.com/a/22291860
        f = scipy.interpolate.interp1d(linSpace, peakFindingArray)
        # Resample for higher resolution
        linSpaceResample = np.linspace(-0.5 * np.pi, 3. / 2 * np.pi, 7200)
        fResample = f(linSpaceResample)
        # Gaussian
        # std deviation is in x!
        window = scipy.signal.gaussian(1000, 300)
        smoothedArray = scipy.signal.convolve(fResample, window / window.sum(), mode="same")
        #maxSmoothed = np.amax(smoothedArray)
        #logger.debug("maxSmoothed: {}".format(maxSmoothed))
        # Moving average on smoothed curve
        smoothedMovingAvg = utils.movingAverage(smoothedArray, n = int(len(smoothedArray) // 2))
        maxSmoothedMovingAvg = max(smoothedMovingAvg)

        # Moving average with rebin
        movingAvgRebin = utils.movingAverage(peakFindingArrayRebin, n = 18)
        maxMovingAvgRebin = max(movingAvgRebin)

        # Fit using TF1 over some range
        # Fit the deltaPhi away side
        fit1D = fitting.fit1DMixedEventNormalization(peakFindingHist, [1. / 2. * np.pi, 3. / 2. * np.pi])
        maxLinearFit1D = fit1D.GetParameter(0)
        fit1DRebin = fitting.fit1DMixedEventNormalization(peakFindingHistRebin, [1. / 2. * np.pi, 3. / 2. * np.pi])
        maxLinearFit1DRebin = fit1DRebin.GetParameter(0)
        fit2D = fitting.fit2DMixedEventNormalization(mixedEvent, [1. / 2. * np.pi, 3. / 2. * np.pi], etaLimits)
        maxLinearFit2D = fit2D.GetParameter(0)
        fit2DRebin = fitting.fit2DMixedEventNormalization(mixedEventRebin, [1. / 2. * np.pi, 3. / 2. * np.pi], etaLimits)
        maxLinearFit2DRebin = fit2DRebin.GetParameter(0)

        logger.debug("linear1D: {}, linear1DRebin: {}".format(maxLinearFit1D, maxLinearFit1DRebin))
        logger.debug("linear2D: {}, linear2DRebin: {}".format(maxLinearFit2D, maxLinearFit2DRebin))

        plot_correlations.mixedEventNormalization(
            self,
            # For labeling purposes
            histName = peakFindingHist.GetName(),
            etaLimits = etaLimits,
            jetPtTitle = params.generateJetPtRangeString(jetPtBin),
            trackPtTitle = params.generateTrackPtRangeString(trackPtBin),
            # Basic data
            linSpace = linSpace,
            peakFindingArray = peakFindingArray,
            linSpaceRebin = linSpaceRebin,
            peakFindingArrayRebin = peakFindingArrayRebin,
            # CWT
            peakLocations = peakLocations,
            peakLocationsRebin = peakLocationsRebin,
            # Moving Average
            maxMovingAvg = maxMovingAvg,
            maxMovingAvgRebin = maxMovingAvgRebin,
            # Smoothed gaussian
            linSpaceResample = linSpaceResample,
            smoothedArray = smoothedArray,
            maxSmoothedMovingAvg = maxSmoothedMovingAvg,
            # Linear fits
            maxLinearFit1D = maxLinearFit1D,
            maxLinearFit1DRebin = maxLinearFit1DRebin,
            maxLinearFit2D = maxLinearFit2D,
            maxLinearFit2DRebin = maxLinearFit2DRebin,
        )

    #################################
    # Utility functions for the class
    #################################
    def generateTeXToIncludeFiguresInAN(self):
        """ Generate latex to put into AN. """
        outputText = ""

        # Define the histograms that should be included
        # Of the form (name, tag, caption)
        histsToWriteOut = collections.OrderedDict()
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
            for (name, tag, description) in itervalues(histsToWriteOut):
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
    def postProjectionProcessing2DCorrelation(observable, normalizationFactor, titleLabel, jetPtBin, trackPtBin):
        """ Basic post processing tasks for a new 2D correlation observable. """
        hist = observable.hist

        # Scale
        hist.Scale(1.0 / normalizationFactor)

        # Set title, labels
        jetPtBinsTitle = params.generateJetPtRangeString(jetPtBin)
        trackPtBinsTitle = params.generateTrackPtRangeString(trackPtBin)
        hist.SetTitle("{} with {}, {}".format(titleLabel, jetPtBinsTitle, trackPtBinsTitle))
        hist.GetXaxis().SetTitle("#Delta#varphi")
        hist.GetYaxis().SetTitle("#Delta#eta")

    @staticmethod
    def postProjectionProcessing1DCorrelation(observable, normalizationFactor, rebinFactor, titleLabel, jetPtBin, trackPtBin):
        """ Basic post processing tasks for a new 1D correlation observable. """
        hist = observable.hist

        # Rebin to decrease the fluctuations in the correlations
        hist.Rebin(rebinFactor)
        hist.Scale(1.0 / rebinFactor)

        # Scale
        hist.Scale(1.0 / normalizationFactor)

        # Set title, labels
        jetPtBinsTitle = params.generateJetPtRangeString(jetPtBin)
        trackPtBinsTitle = params.generateTrackPtRangeString(trackPtBin)
        hist.SetTitle("{} with {}, {}".format(titleLabel, jetPtBinsTitle, trackPtBinsTitle))
        hist.GetXaxis().SetTitle("#Delta#varphi")
        hist.GetYaxis().SetTitle("#Delta#eta")

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
        processingOptions = self.taskConfig["processingOptions"]

        # Only need to check if file exists for the first if statement because we cannot get past there without somehow having some hists
        fileExists = os.path.isfile(os.path.join(self.outputPrefix, self.outputFilename))

        # NOTE: Only normalize hists when plotting, and then only do so to a copy!
        #       The exceptions are the 2D correlations, which are normalized by nTrig for the raw correlation and the maximum efficiency
        #       for the mixed events. They are excepted because we don't have a purpose for such unnormalized hists.
        if processingOptions["generate2DCorrelations"] or not fileExists:
            # Generate and process the 2D correlations
            # First generate the projectors
            logger.info("Generating 2D projectors")
            self.generateSparseProjectors()
            # Then generate the correlations by utilizing the projectors
            logger.info("Projecting 2D correlations")
            self.generate2DCorrelationsTHnSparse()
            # Create the signal correlation
            self.generate2DSignalCorrelation()

            # Write the correlations
            self.write2DCorrelations()
            # Write triggers
            self.writeTriggerJetSpectra()

            # Ensure we execute the next step
            processingOptions["generate1DCorrelations"] = True
        else:
            # Initialize the 2D correlations from the file
            logger.info("Loading 2D correlations and trigger jet spectra from file")
            self.InitFromRootFile(Correlations2D = True)
            self.InitFromRootFile(TriggerJetSpectra = True)

        if processingOptions["plot2DCorrelations"]:
            logger.info("Plotting 2D correlations")
            plot_correlations.plot2DCorrelations(self)
            logger.info("Plotting RPF example region")
            plot_correlations.plotRPFFitRegions(self)

        if processingOptions["generate1DCorrelations"]:
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

            if processingOptions["plot1DCorrelations"]:
                logger.info("Plotting 1D correlations")
                plot_correlations.plot1DCorrelations(self)

            # Ensure that the next step in the chain is run
            processingOptions["fit1DCorrelations"] = True
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
        processingOptions = firstAnalysis.taskConfig["processingOptions"]

        # Run the fitting code
        if processingOptions["fit1DCorrelations"]:
            if firstAnalysis.collisionSystem == params.collisionSystem.PbPb:
                # Run the combined fit over the analyses
                epFit = JetHAnalysis.fitCombinedSignalAndBackgroundRegion(analyses)

                if processingOptions["plot1DCorrelationsWithFits"]:
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
                processingOptions["subtract1DCorrelations"] = True
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
        processingOptions = firstAnalysis.taskConfig["processingOptions"]

        # Subtracted fit functions from the correlations
        if processingOptions["subtract1DCorrelations"]:
            logger.info("Subtracting EP dPhi hists")
            epFit.SubtractEPHists()

            if processingOptions["plotSubtracted1DCorrelations"]:
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
        processingOptions = self.taskConfig["processingOptions"]

        # Subtracted fit functions from the correlations
        if processingOptions["subtract1DCorrelations"]:
            # Subtract fit functions
            logger.info("Subtracting fit functions.")
            logger.info("Subtracting side-band fit from signal region.")
            self.subtractSignalRegion()
            logger.info("Subtracting fit from near-side dEta.")
            self.subtractDEtaNS()

            # Write output
            self.writeSubtracted1DCorrelations()
            self.writeSubtracted1DFits()

            if processingOptions["plot1DCorrelationsWithFits"]:
                logger.info("Plotting 1D correlations with fits")
                plot_correlations.plot1DCorrelationsWithFits(self)

            # Ensure that the next step in the chain is run
            processingOptions["extractWidths"] = True
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
        processingOptions = self.taskConfig["processingOptions"]

        # Extract yields
        if processingOptions["extractYields"]:
            # Setup yield limits
            # 1.0 is the value from the semi-central analysis.
            yieldLimit = self.config.get("yieldLimit", 1.0)

            logger.info("Extracting yields with yield limit: {}".format(yieldLimit))
            self.extractYields(yieldLimit = yieldLimit)
            #logger.info("jetH AS yields: {}".format(self.yieldsAS))

            # Plot
            if processingOptions["plotYields"]:
                plot_extracted.plotYields(self)

            processingOptions["extractWidths"] = True
        else:
            # Initialize yields from the file
            pass

        # Extract widths
        if processingOptions["extractWidths"]:
            # Extract widths
            logger.info("Extracting widths from the fits.")
            self.extractWidths()
            #logger.info("jetH AS widths: {}".format(self.widthsAS))

            # Plot
            if processingOptions["plotWidths"]:
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

def runFromTerminal():
    (configFilename, terminalArgs, additionalArgs) = analysis_config.determineSelectedOptionsFromKwargs(taskName = "correlations analysis")
    analyses = JetHAnalysis.run(
        configFilename = configFilename,
        selectedAnalysisOptions = terminalArgs
    )

    return analyses
