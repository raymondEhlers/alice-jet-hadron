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
logger = logging.getLogger(__name__)
# Quiet down the matplotlib logging
logging.getLogger("matplotlib").setLevel(logging.INFO)

import pprint

import jetH.base.utils as utils
import jetH.base.analysisConfig as analysisConfig
import jetH.plot.genericHist as plotGenericHist

class PlotTaskHists(analysisConfig.JetHBase):
    """ Generic class to plot hists in analysis task.

    Hists are selected and configured by a configuration file.

    Args:
        taskLabel (aenum.Enum): Enum which labels the task and can be converted into a string.
        args (list): Additional arguments to pass along to the base config class.
        kwargs (dict): Additional arguments to pass along to the base config class.
    """
    def __init__(self, taskLabel, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.taskLabel = taskLabel
        # These are the objects for each component stored in the YAML config
        self.componentsFromYAML = self.taskConfig.get("componentsToPlot")
        if self.componentsFromYAML is None:
            raise KeyError("Were \"componentsToPlot\" defined in the task configuration?")
        # Contain the actual components, which consist of lists of hist configs
        self.components = collections.OrderedDict()
        # Store the input histograms
        self.hists = {}

    def getHistsFromInputFile(self):
        """ Retrieve hists corresponding to the task name. They are stored in the object. """
        self.hists = utils.getHistogramsInList(self.inputFilename, self.inputListName)

        # Don't process this line unless we are debugging because pprint may be slow
        # see: https://stackoverflow.com/a/11093247
        #if logger.isEnabledFor(logging.DEBUG):
        #    logger.debug("Hists:")
        #    logger.debug(pprint.pformat(self.hists))

    def histSpecificProcessing(self):
        """ Perform processing on specific histograms in the input hists.

        Each component and histogram in the input hists are searched for particular histograms.
        When they are found, particular functions are applied to those hists, which are then
        stored in the input hists (depending on the function, it is sometimes saved as a
        replacement of the existing hist and sometimes as an additional hist).
        """
        pass

    def histOptionsSpecificProcessing(self, histOptionsName, options):
        """ Run a particular processing functions for some set of hist options.

        It looks for a function name specified in the configuration, so a bit of care is
        required to this safely.

        Args:
            histOptionsName (str): Name of the hist options.
            options (dict): Associated set of hist options.
        Returns:
            dict: Updated set of hist options.
        """
        return options

    def definePlotObjectsForComponent(self, componentName, componentHistsOptions):
        """ Define Hist Plotter options for a particular component based on the provided YAML config.

        Args:
            componentName (str): Name of the component.
            componentHistsOptions (dict): Hist options for a particular component.
        Returns:
            (dict, bool): (Component hists configuration options contained in HistPlotter objects, whether
                other hists in the component that are not configured by HistPlotter objects should also be plotted).
        """
        # Make the output directory for the component.
        componentOutputPath = os.path.join(self.outputPrefix, componentName)
        if not os.path.exists(componentOutputPath):
            os.makedirs(componentOutputPath)

        # Pop the value so all other names can be assumed to be histogram names
        plotAdditional = componentHistsOptions.pop("plotAdditional", False)
        if plotAdditional:
            logger.info("Plotting additional histograms in component \"{}\"".format(componentName))

        # Create the defined component hists
        # Each histogram in a component has a corresponding set of histOptions.
        componentHistsConfigurationOptions = {}
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

            componentHistsConfigurationOptions[histOptionsName] = plotGenericHist.HistPlotter(**histOptions)

        return (componentHistsConfigurationOptions, plotAdditional)

    def assignHistsToPlotObjects(self, componentHistsInFile, histsConfigurationOptions, plotAdditional):
        """ Assign input hists retrieved from a file to the defined Hist Plotters.

        Args:
            componentHistsInFile (dict): Hists that are in a particular component. Keys are hist names
            histsConfigurationOptions (dict): HistPlotter hist configuration objects for a particular component.
            plotAdditional (bool): If true, plot additional histograms in the component that are not specified in the config.
        Returns:
            dict: HistPlotter configuration objects with input hists assigned to each object according to the configuration.
        """
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
                                logger.critical("Skipping hist because this object is only supposed to have one hist")
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
                    componentHists[hist.GetName()] = plotGenericHist.HistPlotter(hist = hist)
                else:
                    logger.debug("No match found and not plotting additional, so hist {} will not be plotted.".format(hist.GetName()))

        return componentHists

    def matchHistsToHistConfigurations(self):
        """ Match retrieved histograms to components and their plotting options.

        This method iterates over the available hists from the file and then over the
        hist options defined in YAML to try to find a match. Once a match is found,
        the options are iterated over to create HistPlotter objects. Then the hists are assigned
        to those newly created objects.

        The results are stored in the components dict of the class.
        """
        #if isinstance(next(iter(self.hists)), ROOT.TH1):

        # componentNameInFile is the name of the componentHistsInFile to which we want to compare
        for componentNameInFile, componentHistsInFile in iteritems(self.hists):
            # componentHistsOptions are the config options for a component
            for componentName, componentHistsOptions in iteritems(self.componentsFromYAML):
                if componentName in componentNameInFile:
                    # We've now matched the component name and and can move on to dealing with
                    # the individual hists
                    (histsConfigurationOptions, plotAdditional) = self.definePlotObjectsForComponent(
                        componentName = componentName,
                        componentHistsOptions = componentHistsOptions
                    )
                    logger.debug("Component name: {}, histsConfigurationOptions: {}".format(componentName, histsConfigurationOptions))

                    # Assign hists from the component in the input file to a hist object
                    componentHists = self.assignHistsToPlotObjects(
                        componentHistsInFile = componentHistsInFile,
                        histsConfigurationOptions = histsConfigurationOptions,
                        plotAdditional = plotAdditional
                    )

                    logger.debug("componentHists: {}".format(pprint.pformat(componentHists)))
                    for componentHist in itervalues(componentHists):
                        # Even though the hist names could be defined in order in the configuration,
                        # the hists will not necessarily show up in alphabetical order when they are assigned to
                        # the plot objects. So we sort them alphabetically here
                        componentHist.hists = sorted(componentHist.hists, key = lambda hist: hist.GetName())
                        logger.debug("componentHist: {}, hists: {}, (first) hist name: {}".format(componentHist, componentHist.hists, componentHist.getFirstHist().GetName()))

                    self.components[componentName] = componentHists

    def plotHistograms(self):
        """ Driver function to plotting the histograms contained in the object. """
        for componentName, componentHists in iteritems(self.components):
            logger.info("Plotting hists for component {}".format(componentName))

            # Apply the options, draw the plot and save it
            for histName, histObj in iteritems(componentHists):
                logger.debug("Processing hist obj {}".format(histName))
                histObj.plot(self, outputName = os.path.join(componentName, histName))

    @classmethod
    def run(cls, configFilename, selectedAnalysisOptions, runPlotting = True):
        """ Main driver function to create, process, and plot task hists.

        Args:
            configFilename (str): Filename of the yaml config.
            selectedAnalysisOptions (params.selectedAnalysisOptions): Selected analysis options.
            runPlotting (bool): If true, run plotting after the processing.
        Returns:
            nested tuple: Tuple of nested analysis objects as described in analysisConfig.constructFromConfigurationFile(...).
        """
        # Create logger
        logging.basicConfig(level=logging.DEBUG)

        # Construct tasks
        (selectedOptionNames, tasks) = cls.constructFromConfigurationFile(
            configFilename = configFilename,
            selectedAnalysisOptions = selectedAnalysisOptions
        )

        # Run the analysis
        logger.info("About to process")
        for keys, task in analysisConfig.unrollNestedDict(tasks):
            # Print the task selected analysis options
            opts = ["{name}: \"{value}\"".format(name = name, value = value.str()) for name, value in zip(selectedOptionNames, keys)]
            logger.info("Processing plotting task {} with options:\n\t{}".format(task.taskName, "\n\t".join(opts)))

            # Setup and run the processing
            task.getHistsFromInputFile()
            task.histSpecificProcessing()
            task.matchHistsToHistConfigurations()
            # Plot
            if runPlotting:
                task.plotHistograms()

        return tasks

    @staticmethod
    def constructFromConfigurationFile(configFilename, selectedAnalysisOptions):
        """ Helper function to construct plotting objects.

        Must be implemented by the derived class. Usually, this is a simple wrapper around
        analysisConfig.constructFromConfigurationFile(...) that is filled in with options
        specific to the particular task.

        Args:
            configFilename (str): Filename of the yaml config.
            selectedAnalysisOptions (params.selectedAnalysisOptions): Selected analysis options.
        Returns:
            nested tuple: Tuple of nested analysis objects as described in analysisConfig.constructFromConfigurationFile(...).
        """
        raise NotImplementedError("Need to implement the constructFromConfigurationFile.")

