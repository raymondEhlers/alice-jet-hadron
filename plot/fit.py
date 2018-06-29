#!/usr/bin/env python

#######################
# Plot fits
#
# Predominately related to RPF plots
#######################

# Py2/3
from future.utils import iteritems

import os
import collections
import itertools
# Setup logger
import logging
logger = logging.getLogger(__name__)

import numpy as np

import jetH.base.params as params
import jetH.base.utils as utils
import jetH.base.analysisConfig as analysisConfig
import jetH.base.analysisObjects as analysisObjects
import jetH.plot.base as plotBase

# Import plotting packages
# Use matplotlib in some cases
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import seaborn as sns
# And use ROOT in others
import rootpy.ROOT as ROOT

def plotMinuitQA(epFitObj, fitObj, fitsDict, minuit, jetPtBin, trackPtBin):
    """ 
    This is really just a debug plot, but it is nice to have available

    NOTE: We can't really modify the plot layout much because this uses a predefined function
          in the probfit package that doesn't seem to take well to modification.

    Args:
        epFitObj (JetHEPFit): The fit object for this 
    """
    # NOTE: Turned off parameter printing to make it easier to see
    # NOTE: Can enable parts = true to draw each part of an added PDF separately, but it
    #       is often difficult to get a good view of it, so it's not super useful
    fitObj.draw(minuit, print_par=False)

    # Retreive fig, axes to attempt to customize the plots some
    fig = plt.gcf()
    axes = fig.get_axes()

    # Label the plots for clarity
    #logger.debug("Drawing legend with axes: {}, fits.iterkeys(): {}".format(axes, fits.keys()))
    for ax, label in zip(axes, fitsDict.keys()):
        logger.debug("label: {}".format(label))
        ax.set_title(label)

    # Reduce overlap
    fig.tight_layout()

    # Save plot
    plotBase.savePlot(epFitObj, fig, epFitObj.fitNameFormat.format(jetPtBin = jetPtBin, trackPtBin = trackPtBin, tag = epFitObj.overallFitLabel.str() + "Minuit"))
    # Cleanup
    plt.close(fig)

def PlotRPF(epFitObj):
    # Get current plotting settings to access values consistently
    plottingSettings = sns.plotting_context()
    colors = sns.color_palette()

    colorIter = iter(colors)
    colorsMap = { (analysisObjects.jetHCorrelationType.signalDominated, "Fit") : next(colorIter),
        (analysisObjects.jetHCorrelationType.signalDominated, "Data") : next(colorIter),
        (analysisObjects.jetHCorrelationType.backgroundDominated, "Fit") : next(colorIter),
        (analysisObjects.jetHCorrelationType.backgroundDominated, "Data") : next(colorIter)}
    zOrder = { (analysisObjects.jetHCorrelationType.signalDominated, "Fit") : 10,
        (analysisObjects.jetHCorrelationType.signalDominated, "FitErrorBars") : 9,
        (analysisObjects.jetHCorrelationType.signalDominated, "Data") : 6,
        (analysisObjects.jetHCorrelationType.backgroundDominated, "Fit") : 8,
        (analysisObjects.jetHCorrelationType.backgroundDominated, "FitErrorBars") : 7,
        (analysisObjects.jetHCorrelationType.backgroundDominated, "Data") : 5 }

    for (jetPtBin, trackPtBin), fitCont in iteritems(epFitObj.fitContainers):
        # Define axes for plot
        fig, axes = plt.subplots(1, 4, sharey = True, sharex = True, figsize = (12,6))
        # TODO: Residual = data-fit/fit, not just data-fit
        figResidual, axesResidual = plt.subplots(1, 4, sharey = True, sharex = True)

        # Store legend information
        handles = []
        labels = []

        handlesResidual = []
        labelsResidual = []

        # Store the all angles data generated from the other angles
        allAnglesSummedFromFit = {analysisObjects.jetHCorrelationType.backgroundDominated : None,
                                  analysisObjects.jetHCorrelationType.signalDominated : None}

        # Put the all angles at the end for consistnecy
        epAngles = [angle for angle in params.eventPlaneAngle]
        epAngles.append(epAngles.pop(epAngles.index(params.eventPlaneAngle.all)))

        for i, (epAngle, ax, axResidual) in enumerate(zip(epAngles, axes, axesResidual)):
            # Main analysis object
            _, jetH = next(analysisConfig.unrollNestedDict(epFitObj.analyses[epAngle]))
            assert jetH.eventPlaneAngle == epAngle

            # Set labels in individual panels
            # NOTE: If text is attached to the figure (fig.text()), we can just plot it whenever
            # Set title
            ax.set_title(epAngle.displayStr(), fontsize = 17)
            # Axis labels
            ax.set_xlabel(r"$\Delta\varphi$", fontsize = 17)

            # Set y label
            (jetFinding, constituentCuts, leadingHadron, jetPt) = params.jetPropertiesLabel(jetPtBin)
            if i == 0:
                ax.set_ylabel(r"1/$\mathrm{N}_{\mathrm{trig}}$dN/d$\Delta\varphi$", fontsize = 17)
                text = ""
                text += params.generateTrackPtRangeString(trackPtBin) + "\n"
                text += constituentCuts + "\n"
                text += leadingHadron
                ax.text(0.5, 0.9, text, horizontalalignment='center',
                        verticalalignment='center',
                        multialignment="left",
                        transform = ax.transAxes)
                #anchorText = AnchoredText(text, loc=9, frameon=False)
                #ax.add_artist(anchorText)

            if i == 1:
                text = ""
                text += params.systemLabel(energy = jetH.collisionEnergy, system = jetH.collisionSystem, activity = jetH.eventActivity) + "\n"
                text += jetPt + "\n"
                text += jetFinding
                ax.text(0.5, 0.9, text, horizontalalignment='center',
                        verticalalignment='center',
                        multialignment="left",
                        transform = ax.transAxes)

                #anchorText = AnchoredText(text, loc=9, frameon=False)
                #ax.add_artist(anchorText)
                #anchorText = """Background: $0.8<|\Delta\eta|<1.2$\nSignal + Background: $|\Delta\eta|<0.6$\n"""
                #anchorText += """{jetPtString}\n{trackPtString}""".format(jetPtString = params.generateJetPtRangeString(jetPtBin),
                #        trackPtString = params.generateTrackPtRangeString(trackPtBin))
                #fig.text(0.34, 0.78, anchorText, fontsize = 10, transform=ax.transAxes)
                #ax.text(0.34, 0.78, anchorText, transform=ax.transAxes)

            if i == 2:
                text = ""
                text += jetH.aliceLabel.str() + "\n"
                text += "Background: $0.8<|\Delta\eta|<1.2$\n"
                text += "Signal + Background: $|\Delta\eta|<0.6$"
                ax.text(0.5, 0.9, text, horizontalalignment='center',
                        verticalalignment='center',
                        multialignment="left",
                        transform = ax.transAxes)

                #anchorText = AnchoredText(text, loc=9, frameon=False)
                #ax.add_artist(anchorText)

            for correlationType, correlationDict in [(analysisObjects.jetHCorrelationType.signalDominated, jetH.dPhiArray),
                                                     (analysisObjects.jetHCorrelationType.backgroundDominated, jetH.dPhiSideBandArray)]:
                # Observable name
                observableName = jetH.histNameFormatDPhiArray.format(jetPtBin = jetPtBin, trackPtBin = trackPtBin, tag = correlationType)
                observable = correlationDict[observableName]

                # Plot data
                # Plot S+B, B for all angles, but only B for EP angles
                if correlationType == analysisObjects.jetHCorrelationType.backgroundDominated or (correlationType == analysisObjects.jetHCorrelationType.signalDominated and epAngle == params.eventPlaneAngle.all):
                    x = observable.hist.x
                    y = observable.hist.array
                    errors = observable.hist.errors
                    ax.errorbar(x, y, yerr = errors, marker = "o", zorder = zOrder[(correlationType, "Data")], color = colorsMap[correlationType, "Data"], label = correlationType.displayStr())

                # Check if the fit was perofmred and therefore should be plotted
                retVal = epFitObj.CheckIfFitIsEnabled(epAngle, correlationType)
                if retVal == False:
                    # Also plot the fit in the case of background dominated in all angles
                    # Although need to clarify that we didn't actually fit - this is just showing that component
                    if not (correlationType == analysisObjects.jetHCorrelationType.backgroundDominated and epAngle == params.eventPlaneAngle.all):
                        continue
                    else:
                        plotLabel = "Background (Simultaneous Fit)"
                else:
                    plotLabel = correlationType.displayStr() + " Fit"

                logger.info("Plotting {}, {}".format(epAngle.str(), correlationType.str()))

                # x values for the fit to be evaluated at
                # Plot fit at same points as data
                # Defined seperately because we may want to change this in the future
                #xForFitFunc = np.linspace(-0.5*np.pi, 1.5*np.pi, 36)
                xForFitFunc = observable.hist.x

                # Evaluate fit
                fit = epFitObj.EvaluateFit(epAngle = epAngle, fitType = correlationType, xValue = xForFitFunc, fitContainer = fitCont)
                #logger.debug("fit: {}".format(fit))

                # Retrieve errors and plot
                errors = fitCont.errors[(epAngle.str(), correlationType.str())]
                #plot = ax.errorbar(xForFitFunc, fit, yerr = errors, zorder = 10, label = correlationType.displayStr() + " Fit")
                logger.debug("Label: {}".format(correlationType.displayStr() + " Fit"))
                plot = ax.plot(xForFitFunc, fit, zorder = zOrder[(correlationType, "Fit")], color = colorsMap[(correlationType, "Fit")], label = plotLabel)
                # Fill in the error band
                # See: https://stackoverflow.com/a/12958534
                ax.fill_between(xForFitFunc, fit - errors, fit + errors, facecolor = plot[0].get_color(), zorder = zOrder[(correlationType, "FitErrorBars")], alpha = 0.8)

                # Plot residual on separate axes
                residual = observable.hist.array - fit
                residualPlot = axResidual.plot(xForFitFunc, residual)
                axResidual.fill_between(xForFitFunc, residual - errors, residual + errors, facecolor = residualPlot[0].get_color(), label = correlationType.displayStr() + " fit residual")

                h, l = axResidual.get_legend_handles_labels()
                logger.debug("handlesResidual: {}, labelsResidual: {}".format(handlesResidual, labelsResidual))
                handlesResidual += h
                labelsResidual += l

                # Build up event plane fit to get all angles as a cross check
                # TODO: This should probably be refactored back to JetHFitting
                if epAngle != params.eventPlaneAngle.all:
                    if allAnglesSummedFromFit[correlationType] is None:
                        allAnglesSummedFromFit[correlationType] = np.zeros(len(fit))
                    #loger.debug("fit: {}, len(fit): {}, allAnglesSummedFromFit[correlationType]: {}, len(allAnglesSummedFromFit[correlationType]): {}".format(fit, len(fit), allAnglesSummedFromFit[correlationType], len(allAnglesSummedFromFit[correlationType])))
                    # Store fit for all angles
                    allAnglesSummedFromFit[correlationType] = np.add(fit, allAnglesSummedFromFit[correlationType])

                # Store legend label
                h, l = ax.get_legend_handles_labels()
                logger.debug("handles: {}, labels: {}".format(handles, labels))
                handles += h
                labels += l

            # Need to perform after plotting all angles to ensure that we get a good
            # estimate for y max
            if i == 3:
                # Make room for the legend
                # Only needed for higher track pt bins
                if trackPtBin > 4:
                    yMin, yMax = ax.get_ylim()
                    # Scale in a data dependent manner
                    yMax = yMax + 0.3*(yMax-yMin)
                    ax.set_ylim(yMin, yMax)

        # Plot a possible cross check
        if epFitObj.plotSummedFitCrosscheck:
            for correlationType, fit in iteritems(allAnglesSummedFromFit):
                logger.debug("Fit: {}".format(fit))
                if fit is not None:
                    # Fit can be None if, for example, we fit the all angles signal, such that the EP signal is not fit
                    # TODO: Is the trivial factor of 3 here correct?
                    logger.info("Plotting summed all angles for correlation type {}".format(correlationType.str()))
                    ax.plot(xForFitFunc, fit/3., zorder = 10, label = correlationType.displayStr() + " fit cross-check")
                    #ax.errorbar(xForFitFunc, fit/3., yerr = fitCont.errors["{}_{}".format(epAngle.str(), correlationType.str())], zorder = 10, label = "Signal fit")
                else:
                    logger.debug("Skipping plot of all angles summed up from each EP angle since it was empty")

        # Tight the plotting up
        fig.tight_layout()
        # Then adjust spacing between subplots
        # Must go second so it isn't reset by tight_layout()
        fig.subplots_adjust(hspace = 0, wspace = 0.05, bottom = 0.12, left = 0.05)

        # Show legend
        logger.debug("handles: {}, labels: {}".format(handles, labels))
        # Remove duplicates
        noDuplicates = collections.OrderedDict(zip(labels, handles))
        axes[3].legend(handles = noDuplicates.values(), labels = noDuplicates.keys(), loc="best", fontsize = plottingSettings["legend.fontsize"])

        # Save plot
        plotBase.savePlot(epFitObj, fig, epFitObj.fitNameFormat.format(jetPtBin = jetPtBin, trackPtBin = trackPtBin, tag = epFitObj.overallFitLabel.str()))

        # Cleanup
        plt.close(fig)

        # Tight the plotting up
        figResidual.tight_layout()
        # Then adjust spacing between subplots
        # Must go second so it isn't reset by tight_layout()
        figResidual.subplots_adjust(hspace = 0, wspace = 0.05, bottom = 0.12, left = 0.1)

        # Show legend
        logger.debug("handles: {}, labels: {}".format(handlesResidual, labelsResidual))
        # Remove duplicates
        noDuplicates = collections.OrderedDict(zip(labelsResidual, handlesResidual))
        axesResidual[3].legend(handles = noDuplicates.values(), labels = noDuplicates.keys(), loc="best", fontsize = plottingSettings["legend.fontsize"])

        # Save plot
        plotBase.savePlot(epFitObj, figResidual, epFitObj.fitNameFormat.format(jetPtBin = jetPtBin, trackPtBin = trackPtBin, tag = epFitObj.overallFitLabel.str() + "Residual"))

        # Cleanup
        plt.close(figResidual)

def PlotSubtractedEPHists(epFitObj):
    # Get current plotting settings to access values consistently
    plottingSettings = sns.plotting_context()
    colors = sns.color_palette()

    # TODO: Use the color map defined in PlotRPF (it's the same, just copied here)
    colorIter = iter(colors)
    colorsMap = { (analysisObjects.jetHCorrelationType.signalDominated, "Fit") : next(colorIter),
        (analysisObjects.jetHCorrelationType.signalDominated, "Data") : next(colorIter),
        (analysisObjects.jetHCorrelationType.backgroundDominated, "Fit") : next(colorIter),
        (analysisObjects.jetHCorrelationType.backgroundDominated, "Data") : next(colorIter)}

    # Iterate over the data and subtract the hists
    for (jetPtBin, trackPtBin), fitCont in iteritems(epFitObj.fitContainers):

        # Define axes for plot
        fig, axes = plt.subplots(1, 4, sharey = True, sharex = True)

        # Just for the all angles subtracted
        figAll, axisAll = plt.subplots(figsize=(5,7.5))

        # Store legend information
        handles = []
        labels = []

        # Put the all angles at the end for consistnecy
        epAngles = [angle for angle in params.eventPlaneAngle]
        epAngles.append(epAngles.pop(epAngles.index(params.eventPlaneAngle.all)))

        for i, (epAngle, ax) in enumerate(zip(epAngles, axes)):
            # Set labels in individual panels
            # Set title
            ax.set_title(epAngle.displayStr(), fontsize = 17)
            # Axis labels
            ax.set_xlabel(r"$\Delta\varphi$", fontsize = 17)
            # Set y label
            if i == 0:
                ax.set_ylabel(r"1/$\mathrm{N}_{\mathrm{trig}}$d(N-B)/d$\Delta\varphi$", fontsize = 17)

            # Main analysis object
            _, jetH = next(analysisConfig.unrollNestedDict(epFitObj.analyses[epAngle]))

            observableName = jetH.histNameFormatDPhiSubtractedArray.format(jetPtBin = jetPtBin, trackPtBin = trackPtBin, tag = analysisObjects.jetHCorrelationType.signalDominated)
            logger.debug("Processing observable {}".format(observableName))
            logger.debug("Subtracted hist arrays: {}".format(jetH.dPhiSubtractedArray))
            observable = jetH.dPhiSubtractedArray[observableName]

            # x values for the fit to be evaluated at
            # Plot fit at same points as data
            # Defined seperately because we may want to change this in the future
            xForFitFunc = observable.hist.binCenters

            # Retrieve fit data
            #fit = epFitObj.EvaluateFit(epAngle = epAngle, fitType = analysisObjects.jetHCorrelationType.backgroundDominated, xValue = xForFitFunc, fitContainer = fitCont)
            # We want to subtract the background function for each EP angle, so we need the errors from the background dominated fit params
            #logger.debug("fitCont.errors: {}".format(fitCont.errors))
            # TODO: Is it right to be retrieving the background errors here?? I'm not so certain for the all angles case that this is right...
            fitErrors = fitCont.errors[(epAngle.str(), analysisObjects.jetHCorrelationType.backgroundDominated.str())]

            plot = ax.errorbar(xForFitFunc, observable.hist.array, yerr = observable.hist.errors, zorder = 5, color = colorsMap[(observable.correlationType, "Data")], label = observable.correlationType.displayStr() + " Subtracted")
            # Following Joel's example, plot the fit error on the same points as the correlation error
            # Fill in the error band
            # See: https://stackoverflow.com/a/12958534
            ax.fill_between(xForFitFunc, observable.hist.array - fitErrors, observable.hist.array + fitErrors, label = "Fit error",facecolor = colorsMap[(observable.correlationType, "Fit")], zorder = 10, alpha = 0.8)

            # Store legend label
            h, l = ax.get_legend_handles_labels()
            logger.debug("handles: {}, labels: {}".format(handles, labels))
            handles += h
            labels += l

            if epAngle == params.eventPlaneAngle.all:
                # TODO: Include all angles in label
                axisAll.set_title(epAngle.displayStr(), fontsize = 17)
                # Axis labels
                axisAll.set_xlabel(r"$\Delta\varphi$", fontsize = 17)
                # Set y label
                axisAll.set_ylabel(r"1/$\mathrm{N}_{\mathrm{trig}}$d(N-B)/d$\Delta\varphi$", fontsize = 17)

                # Add labels
                # NOTE: Cannot end in "\n". It will cause an crash.
                # TODO: Add to other subtracted plots
                (jetFinding, constituentCuts, leadingHadron, jetPt) = params.jetPropertiesLabel(jetPtBin)
                text = ""
                text += params.systemLabel(energy = jetH.collisionEnergy, system = jetH.collisionSystem, activity = jetH.eventActivity)
                text += "\n" + jetFinding + constituentCuts
                text += "\n" + jetPt + ", " + leadingHadron
                text += "\n" + params.generateTrackPtRangeString(trackPtBin) + ", " + "$|\Delta\eta|<0.6$"
                text += "\nScale uncertainty: 6\%"
                text += "\n" + jetH.aliceLabel.str()
                #logger.debug("text: {}".format(text))
                axisAll.text(0.5, 0.82, text, horizontalalignment='center',
                        verticalalignment='center',
                        multialignment="center",
                        fontsize=12.5,
                        transform = axisAll.transAxes)

                plot = axisAll.errorbar(xForFitFunc, observable.hist.array, yerr = observable.hist.errors, zorder = 5, color = colorsMap[(observable.correlationType, "Data")], label = observable.correlationType.displayStr() + " Subtracted")
                axisAll.fill_between(xForFitFunc, observable.hist.array - fitErrors, observable.hist.array + fitErrors, label = "Fit error",facecolor = colorsMap[(observable.correlationType, "Fit")], zorder = 10, alpha = 0.8)

                # Adjust after we know the range of the data
                yMin, yMax = axisAll.get_ylim()
                # Scale in a data dependent manner
                yMax = yMax + 0.35*(yMax-yMin)
                axisAll.set_ylim(yMin, yMax)

        # Tight the plotting up
        fig.tight_layout()
        # Then adjust spacing between subplots
        # Must go second so it isn't reset by tight_layout()
        fig.subplots_adjust(hspace = 0, wspace = 0.05, bottom = 0.12, left = 0.1)

        # Show legend
        logger.debug("handles: {}, labels: {}".format(handles, labels))
        # Remove duplicates
        noDuplicates = collections.OrderedDict(zip(labels, handles))
        axes[3].legend(handles = noDuplicates.values(), labels = noDuplicates.keys(), loc="best", fontsize = plottingSettings["legend.fontsize"])

        # Save plot
        plotBase.savePlot(epFitObj, fig, epFitObj.fitNameFormat.format(jetPtBin = jetPtBin, trackPtBin = trackPtBin, tag = epFitObj.overallFitLabel.str() + "_subtracted"))

        # Cleanup
        plt.close(fig)

        # Tight the plotting up
        figAll.tight_layout()
        # Then adjust spacing between subplots
        # Must go second so it isn't reset by tight_layout()
        #figAll.subplots_adjust(hspace = 0, wspace = 0.05, bottom = 0.12, left = 0.1)

        axisAll.legend(loc="best", fontsize = plottingSettings["legend.fontsize"])

        # Save plot
        plotBase.savePlot(epFitObj, figAll, epFitObj.fitNameFormat.format(jetPtBin = jetPtBin, trackPtBin = trackPtBin, tag = epFitObj.overallFitLabel.str() + "AllAngles_subtracted"))

        # Cleanup
        plt.close(figAll)

def CompareToJoel(epFitObj):
    # Open ROOT file
    filename = "RPF_sysScaleCorrelations{trackPtBin}rebinX2bg.root"

    joelAllAnglesName = "allReconstructedSignalwithErrorsNOM"
    joelAllAnglesErrorMinName = "allReconstructedSignalwithErrorsMIN"
    joelAllAnglesErrorMaxName = "allReconstructedSignalwithErrorsMAX"

    # Iterate over the data and subtract the hists
    for (jetPtBin, trackPtBin), fitCont in iteritems(epFitObj.fitContainers):
        logger.info("Comparing with Joel's code for trackPtBin {}".format(trackPtBin))

        # TODO: Remove hard code
        # TODO: Move this out of here if possible (but perhaps it's fine)
        fIn = ROOT.TFile.Open(os.path.join("output", "plotting", "PbPb", "joelCentral", filename.format(trackPtBin = trackPtBin)), "READ")
        joelAllAngles = fIn.Get(joelAllAnglesName)
        joelAllAnglesErrorMin = fIn.Get(joelAllAnglesErrorMinName)
        joelAllAnglesErrorMax = fIn.Get(joelAllAnglesErrorMaxName)

        # Define axes for plot
        fig, ax = plt.subplots()

        jetPtString = params.generateJetPtRangeString(jetPtBin)
        trackPtString = params.generateTrackPtRangeString(trackPtBin)
        formatStr = """{jetPtString}\n{trackPtString}""".format(jetPtString = jetPtString, trackPtString = trackPtString)
        ax.set_title(r"$\Delta\varphi$ subtracted correlations " + " for {}".format(formatStr))
        # Axis labels
        ax.set_xlabel(r"$\Delta\varphi$")
        ax.set_ylabel(r"dN/d$\Delta\varphi$")

        epAngle = params.eventPlaneAngle.all
        #data = epFitObj.subtractedHistData[(jetPtBin, trackPtBin)][epAngle][analysisObjects.jetHCorrelationType.signalDominated]
        _, jetH = next(analysisConfig.unrollNestedDict(epFitObj.analyses[epAngle]))
        observableName = jetH.histNameFormatDPhiSubtractedArray.format(jetPtBin = jetPtBin, trackPtBin = trackPtBin, tag = analysisObjects.jetHCorrelationType.signalDominated)
        observable = jetH.dPhiSubtractedArray[observableName]

        # Plot my dada
        myDataPlot = ax.errorbar(observable.hist.binCenters, observable.hist.array, yerr = observable.hist.errors, label = "This analysis")
        myDataPlotColor = myDataPlot[0].get_color()
        fitErrors = fitCont.errors[(epAngle.str(), analysisObjects.jetHCorrelationType.signalDominated.str())]
        ax.fill_between(observable.hist.binCenters, observable.hist.array - fitErrors, observable.hist.array + fitErrors, facecolor = myDataPlotColor, zorder = 10, alpha = 0.8)

        # Plot joel data
        joelData = utils.getArrayFromHist(joelAllAngles)
        joelDataPlot = ax.errorbar(joelData["binCenters"], joelData["y"], yerr = joelData["errors"], label = "Joel")
        joelDataPlotColor = joelDataPlot[0].get_color()
        joelErrorMin = utils.getArrayFromHist(joelAllAnglesErrorMin)
        ax.fill_between(joelData["binCenters"], joelData["y"], joelErrorMin["y"], facecolor = joelDataPlotColor)
        joelErrorMax = utils.getArrayFromHist(joelAllAnglesErrorMax)
        ax.fill_between(joelData["binCenters"], joelData["y"], joelErrorMax["y"], facecolor = joelDataPlotColor)

        # Tight the plotting up
        # TODO: Shorten up title (convert the information to a text box), and re-enable this option
        #       For explanation of the error, see: https://github.com/mwaskom/seaborn/issues/954
        #fig.tight_layout()
        # Then adjust spacing between subplots
        # Must go second so it isn't reset by tight_layout()
        # Tuned for "paper" context
        fig.subplots_adjust(hspace = 0, wspace = 0.05, bottom = 0.12, left = 0.1)

        # Show legend
        plt.legend(loc="best")

        # Save plot
        # TODO: Define this name in the class!
        plotBase.savePlot(epFitObj, fig, "joelComparison_jetPt{jetPtBin}_trackPt{trackPtBin}_{tag}_subtracted".format(jetPtBin = jetPtBin, trackPtBin = trackPtBin, tag = epFitObj.overallFitLabel.str()))

        # Cleanup
        plt.close(fig)
