#!/usr/bin/env python

# EMCal corrections and embedding plotting code
#
# Author: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
# Date: 24 Mar 2018

# This must be at the start
from __future__ import print_function
from builtins import super
from future.utils import iteritems

import os
import sys
import aenum
import collections
import logging
# Setup logger
logger = logging.getLogger(__name__)
import warnings
# Handle rootpy warning
warnings.filterwarnings(action='ignore', category=RuntimeWarning, message=r'creating converter for unknown type "_Atomic\(bool\)"')
thisModule = sys.modules[__name__]

import IPython

import jetH.base.params as params
import jetH.base.analysisConfig as analysisConfig
import jetH.analysis.genericHists as genereicHists

import rootpy.ROOT as ROOT
# Tell ROOT to ignore command line options so args are passed to python
# NOTE: Must be immediately after import ROOT and sometimes must be the first ROOT related import!
ROOT.PyConfig.IgnoreCommandLineOptions = True
import rootpy
import rootpy.io

class EMCalCorrectionsLabels(aenum.Enum):
    """

    """
    standard = ""
    embed = "Embed"
    combined = "Data + Embed"
    data = "Data"

    def __str__(self):
        """ Return the label """
        return self.value

    def str(self):
        """ Helper for __str__ to allow it to be accessed the same as the other str functions. """
        return self.__str__()

    def filenameStr(self):
        """ Return the name to be used for the output path. """
        return self.name

class PlotEMCalCorrections(genericHists.PlotTaskHists):
    """

    """
    def __init__(self, taskLabel, *args, **kwargs):
        self.taskLabel = taskLabel
        # Add the task label to the output prefix
        kwargs["config"]["outputPrefix"] = os.path.join(kwargs["config"]["outputPrefix"], self.taskLabel.filenameStr())
        # Need to add it as "_label" so it ends up as "name_label_histos"
        # If it is the standard correction task, then we just put in an emptry string
        correlationsLabel = "_{}".format(taskLabel.filenameStr()) if self.taskLabel != EMCalCorrectionsLabels.standard else EMCalCorrectionsLabels.standard.value
        kwargs["config"]["inputListName"] = kwargs["config"]["inputListName"].format(correctionsLabel = correlationsLabel)

        # Afterwards, we can initialize the base class
        super().__init__(*args, **kwargs)

    def histSpecificProcessing(self):
        """ Perform histogram specific processing. """
        # Loop over available components in the hists
        for componentName in self.hists:
            # Clusterizer
            if "Clusterizer" in componentName:
                # Only perform the scaling if the hist actually exists.
                if "hCPUTime" in self.hists[componentName]:
                    scaleCPUTime(self.hists[componentName]["hCPUTime"])

            if "ClusterExotics" in componentName:
                histName = "hExoticsEtaPhiRemoved"
                beforeHistName = "hEtaPhiDistBefore"
                afterHistName = "hEtaPhiDistAfter"
                if beforeHistName in self.hists[componentName] and afterHistName in self.hists[componentName]:
                    self.hists[componentName][histName] = etaPhiRemoved(histName = histName,
                            beforeHist = self.hists[componentName][beforeHistName],
                            afterHist = self.hists[componentName][afterHistName])

    def histOptionsSpecificProcessing(self, histOptionsName, options):
        """ Run a particular processing functions for some set of hist options.

        It looks for a function named in the configuration, so a bit of care is
        required to this safely.
        """
        if "processing" in options:
            funcName = options["processing"]["funcName"]
            func = getattr(thisModule, funcName)
            if func:
                logger.debug("Calling funcName {} (func {}) for hist options {}".format(funcName, func, histOptionsName))
                options = func(histOptionsName, options)
                logger.debug("Options after return: {}".format(options))
            else:
                logger.critical("Requested funcName {} for hist options {}, but it doesn't exist!".format(funcName, histOptionsName))
                sys.exit(1)

        return options

    @staticmethod
    def constructFromConfigurationFile(configFilename, selectedAnalysisOptions):
        """ Helper function to construct EMCal corrections plotting objects. """
        return analysisConfig.constructFromConfigurationFile(taskName = "EMCalCorrections",
                configFilename = configFilename,
                selectedAnalysisOptions = selectedAnalysisOptions,
                obj = PlotEMCalCorrections,
                additionalIterators = {"taskLabel" : EMCalCorrectionsLabels})

def etaPhiMatchHistNames(histOptionsName, options):
    """ Generate hist names based on the available options.

    This approach allows generating of hist config options using for loops
    while still being defined in YAML.
    """
    # Pop this value so it won't cause issues when creating the hist plotter later.
    processingOptions = options.pop("processing")
    # Get the hist name template
    # We don't care about the hist title
    histName = next(iter(next(iter(options["histNames"]))))
    # Angle name
    angles = processingOptions["angles"]
    # {Number: label}
    centBins = processingOptions["centBins"]
    # {Number: label}
    etaDirections = processingOptions["etaDirections"]
    # List of pt bins
    ptBins = processingOptions["ptBins"]
    # We don't load these from YAML to avoid having to frequently copy them
    ptBinRanges = [0.15, 0.5, 1, 1.5, 2, 3, 4, 5, 8, 200]

    histNames = []
    for angle in angles:
        for centDict in centBins:
            centBin, centLabel = next(iter(iteritems(centDict)))
            for ptBin in ptBins:
                for etaDict in etaDirections:
                    etaDirection, etaDirectionLabel = next(iter(iteritems(etaDict)))
                    # Determine hist name
                    name = histName.format(angle = angle, cent = centBin, ptBin = ptBin, etaDirection = etaDirection)
                    # Determine label
                    # NOTE: Can't use generateTrackPtRangeString because it includes "assoc" in
                    # the pt label. Instead, handle more directly.
                    ptBinLabel = params.generatePtRangeString(arr = ptBinRanges,
                            binVal = trackPtBin,
                            lowerLabel = r"\mathrm{T}",
                            upperLabel = r"")

                    angleLabel = determineAngleLabel(angle)
                    # Ex: "$\\Delta\\varphi$, Pos. tracks, $\\eta < 0$, $4 < p_{\\mathrm{T}}< 5$"
                    label = "{}, {}, {}, {}".format(angleLabel, centLabel, etaDirectionLabel, ptBinLabel)
                    # Save in the expected format
                    histNames.append({name : label})
                    #logger.debug("name: \"{}\", label: \"{}\"".format(name, label))

    # Assign the newly created names
    options["histNames"] = histNames
    logger.debug("Assigning histNames {}".format(histNames))

    return options

def determineAngleLabel(angle):
    """ Determine the angle label and return the corresponding latex. """
    returnValue = r"$\Delta"
    # Need to lower because the label in the hist name is upper
    angle = angle.lower()
    if angle == "phi":
        # "phi" -> "varphi"
        angle = "var" + angle
    returnValue += r"\%s$" % (angle)
    return returnValue

#def determinePtBinLabel(ptBin, ptBins):
#    """ Determine the pt bin label based on the selected bin.
#
#    TODO: Can we update this?
#    Could try to use the params functions to create the pt bin string, but it
#    would require modification, so we will just handle it here for simplicity.
#    """
#    #logger.debug("ptBin: {}, ptBins: {}".format(ptBin, ptBins))
#    ptBinLabel = r"$%(lowerPtLabel)s p_{\mathrm{T}}%(upperPtLabel)s$"
#    lowerPtLabel = "{} <".format(ptBins[ptBin])
#    upperPtLabel = "< {}".format(ptBins[ptBin + 1])
#    # Only show the lower value for the last pt bin
#    if ptBin == (len(ptBins) - 2):
#        upperPtLabel = ""
#    return ptBinLabel % {"lowerPtLabel" : lowerPtLabel, "upperPtLabel" : upperPtLabel}

def scaleCPUTime(hist):
    """ Time is only reported in increments of 10 ms.

    So we rebin by those 10 bins (since each bin is 1 ms) and then
    scale them down to be on the same scale as the real time hist.
    We can perform this scaling in place.

    NOTE: This scaling appears to be the same as one would usually do
          for a rebin, but it is slightly more sublte, because it is
          as if the data was already binned. That being said, the end
          result is effectively the same.

    Args:
        hist (ROOT.TH1): CPU time histogram to be scaled.
    """
    logger.debug("Performing CPU time hist scaling.")
    timeIncrement = 10
    hist.rebin(timeIncrement)
    hist.Scale(1.0/timeIncrement)

def etaPhiRemoved(histName, beforeHist, afterHist):
    """ Show the eta phi locations of clusters removed by the exotics cut.

    Args:
        histName (str): Name of the new hist showing the removed clusters
        beforeHist (ROOT.TH2): Eta-Phi histogram before exotic clusters removal
        afterHist (ROOT.TH2): Eta-Phi histogram after exotic cluster removal
    """
    # Create a new hist and remove the after hist
    hist = beforeHist.Clone(histName)
    hist.Add(afterHist, -1)

    return hist

def runEMCalCorrectionsHistsFromTerminal():
    """

    """
    (configFilename, terminalArgs, additionalArgs) = analysisConfig.determineSelectedOptionsFromKwargs(taskName = taskName)
    analyses = PlotEMCalCorrections.run(configFilename = configFilename,
            selectedAnalysisOptions = terminalArgs)

    return analyses

class PlotEMCalEmbedding(genericHists.PlotTaskHists):
    """

    """
    def __init__(self, *args, **kwargs):
        # Afterwards, we can initialize the base class
        super().__init__(*args, **kwargs)

    @staticmethod
    def constructFromConfigurationFile(configFilename, selectedAnalysisOptions):
        """ Helper function to construct EMCal embedding plotting objects. """
        return analysisConfig.constructFromConfigurationFile(taskName = "EMCalEmbedding",
                configFilename = configFilename,
                selectedAnalysisOptions = selectedAnalysisOptions,
                obj = PlotEMCalEmbedding,
                additionalIterators = {})

def runEMCalEmbeddingHistsFromTerminal():
    """

    """
    (configFilename, terminalArgs, additionalArgs) = analysisConfig.determineSelectedOptionsFromKwargs(taskName = taskName)
    analyses = PlotEMCalEmbedding.run(configFilename = configFilename,
            selectedAnalysisOptions = terminalArgs)

    return analyses

