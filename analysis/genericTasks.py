#!/usr/bin/env python

# Generic task hist plotting code
#
# Author: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
# Date: 01 June 2018

# Py2/3 compatibility
from __future__ import print_function
from builtins import super
from future.utils import iteritems
from future.utils import itervalues

import os
import copy
import collections
import logging
# Setup logger
logger = logging.getLogger(__name__)

import IPython
import pprint

import PlotGenericHist

import JetHConfig
import JetHUtils

class PlotTaskHists(JetHConfig.JetHBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Both replaced by task label
        #self.taskName = taskName
        #self.taskDescription = taskDescription
        # Unneeded
        #self.componentBaseName = componentBaseName
        #self.label = label
        # These are the objects for each component stored in the YAML config
        self.componentsFromYAML = self.taskConfig.get("componentsToPlot")
        if self.componentsFromYAML is None:
            raise KeyError("Were \"componentsToPlot\" defined in the task configuration?")
        # Contain the actual components, which consist of lists of hist configs
        self.components = collections.OrderedDict()
        # Store the input histograms
        self.hists = {}

    def getHistsFromInputFile(self):
        """ Retrieve hists corresponding to the task name.

        We already depend on JetHUtils, so we may as well take advantage of them here."""
        self.hists = JetHUtils.getHistogramsInList(self.inputFilename, self.inputListName)

        # Don't process this line unless we are debugging because pprint may be slow
        # see: https://stackoverflow.com/a/11093247
        #if logger.isEnabledFor(logging.DEBUG):
        #    logger.debug("Hists:")
        #    logger.debug(pprint.pformat(self.hists))

    def histSpecificProcessing(self):
        """
        
        """
        pass

    def histOptionsSpecificProcessing(self, histOptionsName, options):
        """

        """
        return options

    def definePlotObjectsForComponent(self, componentName, componentHistsOptions):
        """ Define Hist Plotter options based on the provided YAML config. """
        # Make the output directory
        componentOutputPath = os.path.join(self.outputPrefix, componentName)
        if not os.path.exists(componentOutputPath):
            os.makedirs(componentOutputPath)

        # Pop the value so all other names can be assumed to be histogram names
        plotAdditional = componentHistsOptions.pop("plotAdditional", False)
        if plotAdditional:
            logger.info("Plotting additional histograms in component \"{}\"".format(componentName))

        # Create the defined component hists
        histsConfigurationOptions = {}
        for histOptionsName, options in iteritems(componentHistsOptions):
            # Copy the options dict so we can add to it
            histOptions = {}
            histOptions.update(options)
            logger.debug("hist options: {}".format(histOptions))
            histNames = histOptions.get("histNames", None)
            if not histNames:
                # Use the hist options name as a proxy for the hist names if not specified
                histOptions["histNames"] = [{histOptionsName: ""}]
                logger.debug("Using hist names from config {}. histNames: {}".format(histOptionsName, histOptions["histNames"]))

            #logger.debug("Pre processing:  hist options name: {}, histOptions: {}".format(histOptionsName, histOptions))
            histOptions = self.histOptionsSpecificProcessing(histOptionsName, histOptions)
            logger.debug("Post processing: hist options name: {}, histOptions: {}".format(histOptionsName, histOptions))

            histsConfigurationOptions[histOptionsName] = PlotGenericHist.HistPlotter(**histOptions)

        return (histsConfigurationOptions, plotAdditional)

    def assignHistsToPlotObjects(self, componentHistsInFile, histsConfigurationOptions, plotAdditional):
        """ Assign hists retreived from a file to the defined Hist Plotter configs. """
        componentHists = {}
        # First iterate over the available hists
        for hist in itervalues(componentHistsInFile):
            logger.debug("Looking for match to hist {}".format(hist.GetName()))
            foundMatch = False
            # Then iterate over the Hist Plotter configurations in the particular component.
            # We are looking for the hists which belong in a particular config
            for histObjectName, histObject in iteritems(histsConfigurationOptions):
                if logger.isEnabledFor(logging.DEBUG):
                    debugHistNames = [next(iter(histLabel)) for histLabel in histObject.histNames]
                    # Only print the first five items so we aren't overwhelmed with information
                    # "..." indicates that there are more than 5
                    debugHistNamesLen = len(debugHistNames)
                    debugHistNames = ", ".join(debugHistNames[:5]) if len(debugHistNames) < 5 else ", ".join(debugHistNames[:5] + ["..."])
                    logger.debug("Considering histObject \"{}\" for exactNameMatch: {}, histNames (len {}): {}".format(histObjectName, histObject.exactNameMatch, debugHistNamesLen, debugHistNames))

                # Iterate over the hist names stored in the config object
                # CommentedMapItemsView apparently returns items when accessed
                # directly iterated over, so we ignore the value.
                for histLabel in histObject.histNames:
                    # There should only be one key in the hist label
                    if len(histLabel) > 1:
                        logger.critical("Too many entries in the histLabel object! Should only be 1! Object: {}".format(histLabel))
                    histName, histTitle = next(iter(iteritems(histLabel)))

                    #logger.debug("Considering histName from histObject: {}, hist.GetName(): {} with exactNameMatch: {}".format(histName, hist.GetName(), histObject.exactNameMatch))

                    # Check for match between the config object and the current hist
                    # The name can be required to be exact, but by default, it doesn't need to be
                    if (histObject.exactNameMatch and histName == hist.GetName()) or (not histObject.exactNameMatch and histName in hist.GetName()):
                        logger.debug("Found match of hist name: {} and options config name {}".format(hist.GetName(), histObjectName))
                        foundMatch = True
                        # Keep the title as the hist name if we haven't specified it so we don't lose information
                        histTitle = histTitle if histTitle != "" else hist.GetTitle() if hist.GetTitle() != "" else hist.GetName()

                        # Check if object is already in the component hists
                        obj = componentHists.get(histObjectName, None)
                        if obj:
                            # If the object already exists, we should add the hist to the list if there are multiple
                            # hists requested for the object. If not, just ignore it
                            if len(histObject.histNames) > 1:
                                logger.debug("Adding hist to existing config")
                                # Set the hist title (for legend, plotting, etc)
                                hist.SetTitle(histTitle)

                                obj.hists.append(hist)
                                logger.debug("Hists after adding: {}".format(obj.hists))
                            else:
                                logger.debug("Skipping hist because this object is only supposed to have one hist")
                                continue
                        else:
                            # Create a copy to store this particular histogram if we haven't already created
                            # an object with this type of histogram because we can have multiple hists covered
                            # by the same configurations options set
                            obj = copy.deepcopy(histObject)

                            # Name will be from outputName if it is defined. If not, use histObjectName if there
                            # are multiple hists, or just the hist name if it only relates to one hist.
                            name = obj.outputName if obj.outputName != "" else hist.GetName() if len(obj.histNames) == 1 else histObjectName
                            logger.debug("Creating config for hist stored under {}".format(name))
                            componentHists[name] = obj

                            # Set the hist title (for legend, plotting, etc)
                            hist.SetTitle(histTitle)
                            # Store the hist in the object
                            obj.hists.append(hist)

            # Create a default hist object
            if not foundMatch:
                if plotAdditional:
                    logger.debug("No hist config options match found. Using default options...")
                    componentHists[hist.GetName()] = PlotGenericHist.HistPlotter(hist = hist)
                else:
                    logger.debug("No match found and not plotting additional, so hist {} will not be plotted.".format(hist.GetName()))

        return componentHists

    def matchHistsToHistConfigurations(self):
        """ Match retrieved histograms to components and their plotting options. """
        for componentNameInFile, componentHistsInFile in iteritems(self.hists):
            # componentNameInFile is the name of the componentHistsInFile to which we want to compare
            for componentName, componentHistsOptions in iteritems(self.componentsFromYAML):
                # componentHistsOptions are the config options for a component
                if componentName in componentNameInFile:
                    # We've now matched the component name and and can move on to dealing with
                    # the individual hists
                    (histsConfigurationOptions, plotAdditional) = self.definePlotObjectsForComponent(componentName = componentName,
                            componentHistsOptions = componentHistsOptions)
                    logger.debug("Component name: {}, histsConfigurationOptions: {}".format(componentName, histsConfigurationOptions))

                    # Assign hists from the component in the input file to a hist object
                    componentHists = self.assignHistsToPlotObjects(componentHistsInFile = componentHistsInFile,
                            histsConfigurationOptions = histsConfigurationOptions,
                            plotAdditional = plotAdditional)

                    self.components[componentName] = componentHists

                    logger.debug("componentHists: {}".format(pprint.pformat(componentHists)))
                    for componentHist in itervalues(componentHists):
                        # Even though the hist names could be defined in order in the configuration,
                        # the hists will not necessarily show up in alphabetical order when they are assigned to
                        # the plot objects. So we sort them alphabetically here
                        componentHist.hists = sorted(componentHist.hists, key = lambda hist : hist.GetName())
                        logger.debug("componentHist: {}, hists: {}, (first) hist name: {}".format(componentHist, componentHist.hists, componentHist.getFirstHist().GetName()))

    def plotHistograms(self):
        """ """
        for componentName, componentHists in iteritems(self.components):
            logger.info("Plotting hists for component {}".format(componentName))

            # Apply the options, draw the plot and save it
            for histName, histObj in iteritems(componentHists):
                logger.debug("Processing hist obj {}".format(histName))
                histObj.plot(self, outputName = os.path.join(componentName, histName))

    @classmethod
    def run(cls, configFilename, selectedAnalysisOptions, runPlotting = True):
        """

        """
        # Construct tasks
        (selectedOptionNames, tasks) = cls.constructFromConfigurationFile(configFilename = configFilename,
                selectedAnalysisOptions = selectedAnalysisOptions)

        # Run the analysis
        logger.info("About to process")
        for keys, task in JetHConfig.unrollNestedDict(tasks):
            opts = ["{name}: {value}".format(name = name, value = value.str()) for name, value in zip(selectedOptionNames, keys)]
            logger.info("Processing plotting task {} with options: {}".format(task.taskName, ", ".join(opts)))

            task.getHistsFromInputFile()
            task.histSpecificProcessing()
            task.matchHistsToHistConfigurations()
            # Plot
            if runPlotting:
                task.plotHistograms()

        return tasks

    @staticmethod
    def constructFromConfigurationFile(configFilename, selectedAnalysisOptions):
        """ Helper function to construct plotting objects. """
        raise NotImplemented("Need to implement the constructFromConfigurationFile.")


