#!/usr/bin/env python

import math
# Import for convenience when defining the RP functions
from math import sin, cos
import sys
import time
import os
import copy
import collections
import itertools
import logging
logger = logging.getLogger(__name__)

import IPython

import numpy as np

# Modules for fits
import iminuit
import probfit
# For gradients
import numdifftools as nd

import rootpy.ROOT as ROOT

import JetHParams
import JetHUtils

import JetHPlotFit

def fitDeltaPhiBackground(hist, trackPtBin, zyam = True, disableVN = True, setFixedVN = False):
    """ Fit the delta phi background. """
    deltaPhi = hist

    fitFunc = ROOT.TF1("deltaPhiBackground", "[0]*1 + (2*[1]*cos(2*x) + 2*[2]*cos(3*x))",
                        -0.5*ROOT.TMath.Pi(), 1.5*ROOT.TMath.Pi())

    # Backgound
    # Pedestal
    fitFunc.SetParLimits(0, 0.,100)
    if zyam:
        # Seed with ZYAM value
        fitFunc.SetParameter(0, deltaPhi.GetBinContent(deltaPhi.GetMinimumBin()))
    fitFunc.SetParName(0, "Pedestal")
    # v2
    fitFunc.SetParLimits(1, -1, 1)
    fitFunc.SetParName(1, "v_{2}")
    # v3
    fitFunc.SetParLimits(2, -1, 1)
    fitFunc.SetParName(2, "v_{3}")
    if disableVN:
        fitFunc.FixParameter(1, 0)
        fitFunc.FixParameter(2, 0)
    if setFixedVN:
        #v2Assoc = math.sqrt((v2Cent00_05Values[trackPtBin] + v2Cent05_10Values[trackPtBin])/2.)
        v2Assoc = (v2Cent00_05Values[trackPtBin] + v2Cent05_10Values[trackPtBin])/2.
        # From https://arxiv.org/pdf/1509.07334v2.pdf
        v2Jet = 0.03
        v3Assoc = 0
        v3Jet = v3Assoc
        fitFunc.FixParameter(7, v2Assoc*v2Jet)
        fitFunc.FixParameter(8, v3Assoc*v3Jet)

    # Set styling
    fitFunc.SetLineColor(ROOT.kBlue+2)
    fitFunc.SetLineStyle(1)

    # Fit to the given histogram
    # R uses the range defined in the fit function
    # 0 ensures that the fit isn't drawn
    # Q ensures minimum printing
    # + adds the fit to function list to ensure that it is not deleted on the creation of a new fit
    deltaPhi.Fit(fitFunc, "RIB0")

    return fitFunc

def fitDeltaPhi(hist, trackPtBin, zyam = True, disableVN = True, setFixedVN = False):
    """ Define 1D gaussian fit function with one gaussian each for the near and away sides, along with a gaussian offset by +/-2Pi"""
    deltaPhi = hist

    #fitFunc = ROOT.TF1("symmetricGaussian","[0]*exp(-0.5*((x-[1])/[2])**2)+[3]+[4]*exp(-0.5*((x-[5])/[6])**2)+[0]*exp(-0.5*((x-[1]+2.*TMath::Pi())/[2])**2)+[4]*exp(-0.5*((x-[5]-2.*TMath::Pi())/[6])**2)", -0.5*ROOT.TMath.Pi(), 1.5*ROOT.TMath.Pi())
    # NOTE: This is not symmetric! Instead, the extra fits are because of how it wraps around. Even if our data doesn't go there, it is still relevant
    fitFunc = ROOT.TF1("dPhiWithGaussians",
                        "[6]*1 + (2*[7]*cos(2*x) + 2*[8]*cos(3*x)) + "
                        "[0]*(TMath::Gaus(x, [1], [2]) + TMath::Gaus(x, [1]-2.*TMath::Pi(), [2]) + TMath::Gaus(x, [1]+2.*TMath::Pi(), [2])) + "
                        "[3]*(TMath::Gaus(x, [4], [5]) + TMath::Gaus(x, [4]-2.*TMath::Pi(), [5]) + TMath::Gaus(x, [4]+2.*TMath::Pi(), [5]))",
                        -0.5*ROOT.TMath.Pi(), 1.5*ROOT.TMath.Pi())

    # Setup parameters
    amplitudeLimits = [0.0, 100.0]
    sigmaLimits = [0.05, 2.0]
    # Near side
    # Amplitude
    fitFunc.SetParLimits(0, amplitudeLimits[0], amplitudeLimits[1])
    fitFunc.SetParName(0, "NS Amplitude")
    # Offset
    fitFunc.FixParameter(1, 0)
    fitFunc.SetParName(1, "NS Offset")
    # Sigma
    fitFunc.SetParLimits(2, sigmaLimits[0], sigmaLimits[1])
    fitFunc.SetParName(2, "NS #sigma")
    # Seed for sigma
    fitFunc.SetParameter(2, sigmaLimits[0])

    # Away side
    # Amplitude
    fitFunc.SetParLimits(3, amplitudeLimits[0], amplitudeLimits[1])
    fitFunc.SetParName(3, "AS Amplitude")
    # Offset
    fitFunc.FixParameter(4, ROOT.TMath.Pi())
    fitFunc.SetParName(4, "AS Offset")
    # Sigma
    fitFunc.SetParLimits(5, sigmaLimits[0], sigmaLimits[1])
    fitFunc.SetParName(5, "AS #sigma")
    # Seed for sigma
    fitFunc.SetParameter(5, sigmaLimits[0])

    # Backgound
    # Pedestal
    fitFunc.SetParLimits(6, 0.,100)
    if zyam:
        # Seed with ZYAM value
        fitFunc.SetParameter(6, deltaPhi.GetBinContent(deltaPhi.GetMinimumBin()))
    fitFunc.SetParName(6, "Pedestal")
    # v2
    fitFunc.SetParLimits(7, -1, 1)
    fitFunc.SetParName(7, "v_{2}")
    # v3
    fitFunc.SetParLimits(8, -1, 1)
    fitFunc.SetParName(8, "v_{3}")
    if disableVN:
        fitFunc.FixParameter(7, 0)
        fitFunc.FixParameter(8, 0)
    if setFixedVN:
        #v2Assoc = math.sqrt((v2Cent00_05Values[trackPtBin] + v2Cent05_10Values[trackPtBin])/2.)
        v2Assoc = (v2Cent00_05Values[trackPtBin] + v2Cent05_10Values[trackPtBin])/2.
        # From https://arxiv.org/pdf/1509.07334v2.pdf
        v2Jet = 0.03
        v3Assoc = 0
        v3Jet = v3Assoc
        #fitFunc.SetParameter(7, v2Assoc*v2Jet)
        fitFunc.FixParameter(7, v2Assoc*v2Jet)
        fitFunc.FixParameter(8, v3Assoc*v3Jet)

    # Set styling
    fitFunc.SetLineColor(ROOT.kRed + 2)
    fitFunc.SetLineStyle(1)

    # Fit to the given histogram
    # R uses the range defined in the fit function
    # 0 ensures that the fit isn't drawn
    # Q ensures minimum printing
    # + adds the fit to function list to ensure that it is not deleted on the creation of a new fit
    deltaPhi.Fit(fitFunc, "RIB0")

    # And return the fit
    return fitFunc

def fitDeltaEta(hist, trackPtBin, zyam = True, disableVN = True, setFixedVN = False):
    """ dEta near-side fit implementation. """
    fitFunc = ROOT.TF1("deltaEtaNS", "[6] + [0]*TMath::Gaus(x, [1], [2])", -1, 1)

    # Setup parameters
    amplitudeLimits = [0.0, 100.0]
    sigmaLimits = [0.05, 2.0]
    # Near side
    # Amplitude
    fitFunc.SetParLimits(0, amplitudeLimits[0], amplitudeLimits[1])
    fitFunc.SetParName(0, "NS Amplitude")
    # Offset
    fitFunc.FixParameter(1, 0)
    fitFunc.SetParName(1, "NS Offset")
    # Sigma
    fitFunc.SetParLimits(2, sigmaLimits[0], sigmaLimits[1])
    fitFunc.SetParName(2, "NS #sigma")
    # Seed for sigma
    fitFunc.SetParameter(2, sigmaLimits[0])

    # Backgound
    fitFunc.SetParLimits(6, 0.,100)
    if zyam:
        # Seed with ZYAM value
        fitFunc.SetParameter(6, hist.GetBinContent(hist.GetMinimumBin()))
    fitFunc.SetParName(6, "Pedestal")

    # Set styling
    fitFunc.SetLineColor(ROOT.kGreen + 2)
    fitFunc.SetLineStyle(1)

    # Fit to the given histogram
    # R uses the range defined in the fit function
    # 0 ensures that the fit isn't drawn
    # Q ensures minimum printing
    # + adds the fit to function list to ensure that it is not deleted on the creation of a new fit
    hist.Fit(fitFunc, "RIB0")

    # And return the fit
    return fitFunc

def fit1DMixedEventNormalization(hist, dPhiLimits):
    """ Alternative to determine the mixed event normalization.

    A lienar function is fit to the dPhi mixed event normalization for some predefined range. """
    fitFunc = ROOT.TF1("mixedEventNormalization1D", "[0] + 0.0*x", dPhiLimits[0], dPhiLimits[1])

    # Fit to the given histogram
    # R uses the range defined in the fit function
    # 0 ensures that the fit isn't drawn
    # Q ensures minimum printing
    # + adds the fit to function list to ensure that it is not deleted on the creation of a new fit
    hist.Fit(fitFunc, "RIB0")

    # And return the fit
    return fitFunc

def fit2DMixedEventNormalization(hist, dPhiLimits, dEtaLimits):
    """ Alternative to determine the mixed event normalization.

    A lienar function is fit to the dPhi-dEta mixed event normalization for some predefined range. """
    fitFunc = ROOT.TF2("mixedEventNormalization2D", "[0] + 0.0*x + 0.0*y", dPhiLimits[0], dPhiLimits[1],
                                                                           dEtaLimits[0], dEtaLimits[1])

    # Fit to the given histogram
    # R uses the range defined in the fit function
    # 0 ensures that the fit isn't drawn
    # Q ensures minimum printing
    # + adds the fit to function list to ensure that it is not deleted on the creation of a new fit
    hist.Fit(fitFunc, "RIB0")

    # And return the fit
    return fitFunc

#####################
# Reaction Plane Fit
#
# Actual fit functions based on https://github.com/miguelignacio/BackgroundFit from Miguel
#####################
class JetHEPFit(object):
    def __init__(self, jetHAnalyses):
        # Jet H analysis objects
        self.analyses = jetHAnalyses
        # Store the necessary information per EP angle
        # Dicts are of the form [(jetPtBin, trackPtBin)][epAngle][correlationType]
        self.fits = collections.OrderedDict()
        self.fitContainers = collections.OrderedDict()

        # Determine configuration
        # The configurations should all be the same, except for the EP (which isn't in the config anyway)
        jetHAllAngles = self.analyses[JetHUtils.EventPlaneAngle.kAll]
        analysisConfig = next(self.analyses.itervalues()).config

        # Store to simplify plotting
        self.outputPrefix = jetHAllAngles.outputPrefix
        self.printingExtensions = jetHAllAngles.printingExtensions
        self.fitNameFormat = jetHAllAngles.fitNameFormat

        # Fit options
        # Load fit configuration from YAML
        self.fitConfig = analysisConfig["fitOptions"]
        self.performFit = self.fitConfig["performFit"]
        self.includeSignalInFit = self.fitConfig["includeSignalInFit"]
        self.allAnglesSignal = self.fitConfig["allAnglesSignal"]
        self.allAnglesBackground = self.fitConfig["allAnglesBackground"]
        self.logLikelihoodTrackPtBins = self.fitConfig["logLikelihoodTrackPtBins"]
        self.includeFitError = self.fitConfig["includeFitError"]
        self.drawMinuitQAPlots = self.fitConfig["drawMinuitQAPlots"]
        self.plotSummedFitCrosscheck = self.fitConfig["plotSummedFitCrosscheck"]
        # This is generally slow (due to needed to calculate the derivatives?)
        self.calculateFitError = self.fitConfig["calculateFitError"]

        # Useful when labeling fit conatiner objects
        self.overallFitLabel = JetHUtils.JetHCorrelationType.backgroundDominated
        if self.includeSignalInFit:
            self.overallFitLabel = JetHUtils.JetHCorrelationType.signalDominated

    def GetFitFunction(self, fitType, epAngle):
        """ Simple wrapper to get the fit function corresponding to the selected fitType. """
        fitFunctionMap = {JetHUtils.JetHCorrelationType.signalDominated : GetSignalDominatedFitFunction,
                          JetHUtils.JetHCorrelationType.backgroundDominated : GetBackgroundDominatedFitFunction}

        # The fit container stores the fit type as a string. We need it as a JetHCorrelationType
        if isinstance(fitType, str):
            fitType = JetHUtils.JetHCorrelationType[fitType]
        if isinstance(epAngle, str):
            epAngle = JetHUtils.EventPlaneAngle[epAngle]
        # Retrieve the function
        uncalledFitFunc = fitFunctionMap[fitType]
        # Call the function on return
        return uncalledFitFunc(epAngle = epAngle, fitConfig = self.fitConfig)

    def EvaluateFit(self, epAngle, fitType, xValue, fitContainer):
        """  """
        func = self.GetFitFunction(epAngle = epAngle, fitType = fitType)
        argsForFuncCall = self.GetArgsForFunc(func = func, xValue = xValue, fitContainer = fitContainer)
        # Don't need "x" here, since AddPdf can't handle an np.array of x args...
        argsForFuncCall.pop("x")
        #logger.debug("describe func: {}".format(probfit.describe(func)))

        # Apply each value to the fit function
        fit = probfit.nputil.vector_apply(func, xValue, *list(argsForFuncCall.itervalues()))

        return fit

    def CheckIfFitIsEnabled(self, epAngle, correlationType):
        """

        """
        retVal = True
        if correlationType == JetHUtils.JetHCorrelationType.signalDominated:
            # Skip signal fit
            if not self.includeSignalInFit:
                retVal = False

            # Skip EP angles if we fit all angles signal. Otherwise, it would be double counting
            if epAngle != JetHUtils.EventPlaneAngle.kAll and self.allAnglesSignal:
                retVal = False
        else:
            # Skip all angles background unless it's explicitly enabled
            if epAngle == JetHUtils.EventPlaneAngle.kAll and not self.allAnglesBackground:
                retVal = False

            # Skip EP angles if we fit all angles background. Otherwise, it would be double counting
            if epAngle != JetHUtils.EventPlaneAngle.kAll and self.allAnglesBackground:
                retVal = False

        return retVal

    def DefineFits(self):
        # Setup fit and cost functions
        # Define the fits
        for epAngle, jetH in self.analyses.iteritems():
            #for signalDominated, backgroundDominated in zip(jetH.dPhiArray.itervalues(), jetH.dPhiSideBandArray.itervalues()):
            for observable in itertools.chain(jetH.dPhiArray.itervalues(), jetH.dPhiSideBandArray.itervalues()):
                retVal = self.CheckIfFitIsEnabled(epAngle, observable.correlationType)
                if retVal == False:
                    continue

                # Create the dict if it doesn't already exist
                if (observable.jetPtBin, observable.trackPtBin) not in self.fits:
                    self.fits[(observable.jetPtBin, observable.trackPtBin)] = collections.OrderedDict()

                # Retrieve data
                x = observable.hist.x
                y = observable.hist.array
                errors = observable.hist.errors

                # Define fit function
                fitFunc = self.GetFitFunction(fitType = observable.correlationType, epAngle = epAngle)

                # Restricted the background fit range
                if observable.correlationType == JetHUtils.JetHCorrelationType.backgroundDominated:
                    # Use only near-side data (ie dPhi < pi/2)
                    NSrange = int(len(x)/2.)
                    x = x[:NSrange]
                    y = y[:NSrange]
                    errors = errors[:NSrange]

                # Define cost function
                #
                # NOTE: We don't want the binned cost function versions - they will bin the data that is given,
                #       which is definitely not what we want
                # For lower pt assoc bin, use Chi2
                # For higher pt assoc bin, use log likelihood because the statistics are not as good
                if observable.trackPtBin in self.logLikelihoodTrackPtBins.get(observable.jetPtBin, []):
                    logger.debug("Using log likelihood for {}, {}".format(epAngle.str(), observable.correlationType.str()))
                    # Generally will use for higher pt bins where the statistics are not as good
                    # Errors are extracted by assuming a poisson distribution, so we don't need to pass them explcitily(?)
                    # TODO: I think this is actually supposed to be binned!
                    costFunction = probfit.UnbinnedLH(f = fitFunc, data = x)
                    #costFunction = probfit.BinnedLH(f = fitFunc, data = x, bins = len(x), bound=(x[0], x[-1]))
                else:
                    logger.debug("Using Chi2 for {}, {}".format(epAngle.str(), observable.correlationType.str()))
                    costFunction = probfit.Chi2Regression(f = fitFunc,
                            x = x, y = y, error = errors)

                self.fits[(observable.jetPtBin, observable.trackPtBin)][(epAngle.str(), observable.correlationType.str())] =  costFunction

        #logger.debug("self.fits: {}".format(self.fits))

    def PerformFit(self):
        if not self.performFit:
            logger.info("Loading stored fit parameters")
            # Load the fit containers from file instead
            for (jetPtBin, trackPtBin), fitsDict in self.fits.iteritems():
                fitCont = JetHUtils.FitContainer.initFromYAML(inputPrefix = self.outputPrefix,
                        fitType = self.overallFitLabel,
                        jetPtBin = jetPtBin,
                        trackPtBin = trackPtBin)
                # May return as None if they file doesn't exist
                if fitCont:
                    self.fitContainers[(jetPtBin, trackPtBin)] = fitCont

            # We have loaded all of the fit information, so we don't need to do anything else
            return

        # Create overall cost function and perform the fit
        for (jetPtBin, trackPtBin), fitsDict in self.fits.iteritems():
            logger.info("Processing jetPtBin {}, trackPtBin: {}".format(jetPtBin, trackPtBin))

            logger.debug("fitsDict: {}".format(fitsDict))
            fitObj = probfit.SimultaneousFit(*list(fitsDict.itervalues()))

            # Definition variable initiation through a dictionary!
            minuitArgs = {}

            # Signal args
            if self.includeSignalInFit:
                # Signal default parameters
                nsSigmaInit = 0.07
                asSigmaInit = 0.2
                sigmaUpperLimit = 0.35
                sigmaLowerLimit = 0.025
                signalLimits = dict(
                        #nsAmplitude = 0.55, limit_nsAmplitude = (0.1,1.0), error_nsAmplitude=0.01,
                        #asAmplitude = 0.55, limit_asAmplitude = (0.1,1.0), error_asAmplitude=0.01,
                        nsSigma = nsSigmaInit, limit_nsSigma = (sigmaLowerLimit, sigmaUpperLimit), error_nsSigma = 0.001,
                        asSigma = asSigmaInit, limit_asSigma = (sigmaLowerLimit, sigmaUpperLimit), error_asSigma = 0.001,
                        signalPedestal = 0.0, fix_signalPedestal = True
                    )

                if not self.allAnglesSignal:
                    # Add separate parameters for each event plane
                    # Only needed if we are fitting separate event planes
                    for epAngle in JetHUtils.EventPlaneAngle:
                        if epAngle == JetHUtils.EventPlaneAngle.kAll:
                            # We don't want to use all angles here as it would be double counting
                            continue

                        # Copy for a particular EP
                        signalLimitsEP = dict(signalLimits)
                        # Add the event plane prefix
                        signalLimitsEP = iminuit.util.fitarg_rename(signalLimitsEP, lambda pname: epAngle.str() + "_" + pname)
                        # Add to the arguments
                        minuitArgs.update(signalLimitsEP)
                else:
                    # Add in the signal limits for all angles
                    minuitArgs.update(signalLimits)

            # Background arguments
            backgroundLimits = dict(
                    v2_t=0.02, limit_v2_t =(0,0.50), error_v2_t=0.001,
                    v2_a=0.02, limit_v2_a =(0,0.50), error_v2_a=0.001,
                    v4_t=0.01, limit_v4_t =(0,0.50), error_v4_t=0.001,
                    v4_a=0.01, limit_v4_a =(0,0.50), error_v4_a=0.001,
                    v3=0 , limit_v3 = (-0.1, 0.5), error_v3 =0.001,
                    v1=0.0, fix_v1=True
                )
            minuitArgs.update(backgroundLimits)

            logger.debug("Minuit args: {}".format(minuitArgs))
            minuit = iminuit.Minuit(fitObj,
                        **minuitArgs)

            # Perform the fit
            minuit.migrad()
            # More sophisticated error estimation
            #minuit.minos()
            # Plot the correlation matrix
            minuit.print_matrix()

            # Check if fit is considered valid
            if not minuit.migrad_ok():
                logger.critical("Fit was not valid for jetPtBin: {}, trackPtBin: {}. Skipping this bin!".format(jetPtBin, trackPtBin))
                continue

            # Draw result
            if self.drawMinuitQAPlots:
                # This is really just a debug plot, but it is nice to have available
                JetHPlotFit.plotMinuitQA(epFitObj = self,
                        fitObj = fitObj, fitsDict = fitsDict,
                        minuit = minuit,
                        jetPtBin = jetPtBin, trackPtBin = trackPtBin)

            # Save out the fit paramaters
            fitCont = JetHUtils.FitContainer(jetPtBin = jetPtBin, trackPtBin = trackPtBin, fitType = self.overallFitLabel,
                    values = minuit.values, params = minuit.fitarg, covarianceMatrix = minuit.covariance)

            # They are the same for each EP angle
            self.fitContainers[(jetPtBin, trackPtBin)] = fitCont

        # Since we performed the fit, we should save out the information in the containers
        for fitCont in self.fitContainers.itervalues():
            fitCont.saveToYAML(self.outputPrefix)

        # We need to reclculate the errors if they will be shown, since we've modified the fit
        self.calculateFitError = True

    def DetermineFitErrors(self):
        # Perform error calculate
        for epAngle, jetH in self.analyses.iteritems():
            for observable in itertools.chain(jetH.dPhiArray.itervalues(), jetH.dPhiSideBandArray.itervalues()):
                retVal = self.CheckIfFitIsEnabled(epAngle, observable.correlationType)
                # Always calculate for all angles because we will subtract and we want the errors from the fit on that plot
                # TODO: Is this okay / right???
                #       Compare error bars from S+B to B only in all angles
                if epAngle != JetHUtils.EventPlaneAngle.kAll and retVal == False:
                    continue

                # Retrieve fit container
                # We know that it exists because we just checked.
                # TODO: It actually may not exist if a fit failed. Address this more carefully.
                fitCont = self.fitContainers[(observable.jetPtBin, observable.trackPtBin)]

                identifier = (epAngle.str(), observable.correlationType.str())
                if self.includeFitError:
                    if self.calculateFitError:
                        # Calculate errors
                        logger.debug("Determine fit errors for {}, {}".format(observable.correlationType.str(), epAngle.str()))
                        errors = self.CalculateRPFError(fitFunc = self.GetFitFunction(fitType = observable.correlationType, epAngle = epAngle),
                                    histArray = observable.hist,
                                    fitContainer = fitCont,
                                    epAngle = epAngle)

                        # Store errors in the fit container
                        fitCont.errors[identifier] = errors
                    else:
                        # Load stored errors
                        if not self.performFit:
                            #logger.debug("Checking error fit data for {}: {}".format(identifier, fitCont.errors))
                            for identifier, data in fitCont.errors.iteritems():
                                #logger.warning("len(data): {}, data: {}, identifier: {}".format(len(data), data, identifier))
                                if len(data) == 0:
                                    logger.warning("Errors for fit container ({},{}), identifier: {} should already be loaded, but don't appear to be. Please check the error object!".format(jetPtBin, trackPtBin, identifier))
                                    logger.warning("data: {}".format(data))
                        else:
                            logger.critical("Attempting to load errors, but the fit has been modified! Please enable error recalculation! Exiting")
                            sys.exit(1)
                else:
                    logger.debug("Requested to not include errors for dataType {}. Will use 0".format(dataType))
                    fitCont.errors[identifier] = np.zeros(len(data["binCenters"]))

            # Need to do after the above loops are completed because the errors depend on epAngle, dataType
            # Could write above, but it would be wasteful (freuqently overwritting the same yaml file)
            if self.includeFitError and self.calculateFitError:
                for fitCont in self.fitContainers.itervalues():
                    # Rewrite the fit container with the new errors
                    logger.debug("Writing errors for fitCont ({}, {})".format(fitCont.jetPtBin, fitCont.trackPtBin))
                    fitCont.saveToYAML(self.outputPrefix)

            logger.debug("fitCont ({},{}) errors: {}".format(fitCont.jetPtBin, fitCont.trackPtBin, fitCont.errors))

    def CalculateRPFError(self, fitFunc, histArray, fitContainer, epAngle):
        # Wrapper needed to call the function because numdifftools requires that multiple arguments
        # are in a single list. The wrapper expands that list for us
        def funcWrap(x):
            # Need to expand the arguments
            return fitFunc(*x)

        # Determine the arguments for the fit function
        argsForFuncCall = JetHEPFit.GetArgsForFunc(func = fitFunc, xValue = None, fitContainer = fitContainer)
        logger.debug("argsForFuncCall: {}".format(argsForFuncCall))

        # Retrieve the parameters to use in calculating the fit errors
        funcArgs = probfit.describe(fitFunc)
        # Remove "x" as an arg, because we don't want to evaluate the error on it
        funcArgs.pop(funcArgs.index("x"))
        # Remove free parameters, as they won't contribute to the error and will cause problems for the gradient
        for param in fitContainer.params:
            if "fix_" in param and fitContainer.params[param] == True:
                # This parameter is fixed. We need to remove it from the funcArgs!
                funcArgParamName = param.replace("fix_", "")
                # Remove it from funcArgs if it exists
                if funcArgParamName in funcArgs:
                    funcArgs.pop(funcArgs.index(funcArgParamName))
        logger.debug("funcArgs: {}".format(funcArgs))

        # Compute the derivative
        partialDerivatives = nd.Gradient(funcWrap)

        # To store the errors for each point
        # Just using "binCenters" as a proxy
        errorVals = np.zeros(len(histArray.binCenters))
        #logger.debug("len(histArray.binCneters]): {}, histArray.binCenters: {}".format(len(histArray.binCenters), histArray.binCenters))

        for i, val in enumerate(histArray.binCenters):
            # Add in x for func the function call
            argsForFuncCall["x"] = val

            #logger.debug("Actual list of args: {}".format(list(argsForFuncCall.itervalues())))

            # We need to calculate the derivative once per x value
            start = time.time()
            logger.debug("Calculating the gradient for point {}.".format(i))
            partialDerivative = partialDerivatives(list(argsForFuncCall.itervalues()))
            end = time.time()
            logger.debug("Finished calculating the graident in {} seconds.".format(end-start))

            # Calculate error
            errorVal = 0
            for iName in funcArgs:
                for jName in funcArgs:
                    # Evaluate the partial derivative at a point
                    # Must be called as a list!
                    listOfArgsForFuncCall = list(argsForFuncCall.itervalues())
                    iNameIndex = listOfArgsForFuncCall.index(argsForFuncCall[iName])
                    jNameIndex = listOfArgsForFuncCall.index(argsForFuncCall[jName])
                    #logger.debug("Calculating error for iName: {}, iNameIndex: {} jName: {}, jNameIndex: {}".format(iName, iNameIndex, jName, jNameIndex))
                    #logger.debug("Calling partial derivative for args {}".format(argsForFuncCall))

                    # Add error to overall error value
                    errorVal += partialDerivative[iNameIndex] * partialDerivative[jNameIndex] * fitContainer.covarianceMatrix[(iName, jName)]

            # Modify from error squared to error
            errorVal = np.sqrt(errorVal)

            # Store
            #logger.debug("i: {}, errorVal: {}".format(i, errorVal))
            errorVals[i] = errorVal

        return errorVals

    def SubtractEPHists(self):
        for epAngle, jetH in self.analyses.iteritems():
            for observable in jetH.dPhiArray.itervalues():
                #retVal = self.CheckIfFitIsEnabled(epAngle, observable.correlationType)
                #if retVal == False:
                #    continue

                # x values for the fit to be evaluated at
                # Plot fit at same points as data
                # Defined seperately because we may want to change this in the future
                xForFitFunc = observable.hist.binCenters

                # Retrieve fit container
                # We know that it exists because we just checked.
                fitCont = self.fitContainers[(observable.jetPtBin, observable.trackPtBin)]

                # Retrieve fit data
                fit = self.EvaluateFit(epAngle = epAngle, fitType = JetHUtils.JetHCorrelationType.backgroundDominated, xValue = xForFitFunc, fitContainer = fitCont)

                # Subtract the fit from the hist
                subtracted = observable.hist.array - fit
                # TODO: Error prop?
                subtractedErrors = observable.hist.errors

                histArray = JetHUtils.HistArray(_binCenters = observable.hist.x,
                        _array = subtracted,
                        _errors = subtractedErrors)

                subtractedHistName = jetH.histNameFormatDPhiSubtractedArray.format(jetPtBin = observable.jetPtBin,
                        trackPtBin = observable.trackPtBin,
                        tag = observable.correlationType)

                jetH.dPhiSubtractedArray[subtractedHistName] = JetHUtils.CorrelationObservable1D(jetPtBin = observable.jetPtBin,
                        trackPtBin = observable.trackPtBin,
                        axis = observable.axis,
                        correlationType = observable.correlationType,
                        hist = histArray)

                logger.debug("Storing subtracted hist under name {}".format(subtractedHistName))

                # Save out the newly subtracted hist
                # TODO: This will overrite the unsubtracted hist. Determine how to make these names unique
                #histArray.saveToYAML(self.outputPrefix, histType = observable.correlationType, jetPtBin = observable.jetPtBin, trackPtBin = observable.trackPtBin)

    def RetrieveWidths(self):
        """ Extract widths from the fits. """
        widths = collections.OrderedDict()
        # Retrieve the widths parameter and it's error
        for location in ["ns", "as"]:
            widths[location] = collections.OrderedDict()
            for (jetPtBin, trackPtBin), fitCont in self.fitContainers.iteritems():
                value = fitCont.params["{}Sigma".format(location)]
                error = fitCont.params["error_{}Sigma".format(location)]
                widths[location][(jetPtBin, trackPtBin)] = JetHUtils.ExtractedObservable(jetPtBin = jetPtBin,
                        trackPtBin = trackPtBin,
                        value = value,
                        error = error)

        return widths

    @staticmethod
    def GetArgsForFunc(func, xValue, fitContainer):
        """

        Args:
            func (Callable): Function for which the arguments should be determined
            xValue (int, float, or np.array): Whatever the x value (or values for an np.array) that should be called
            fitContainer (JetHUtils.FitContainer): Fit container which holds the values that will be used when calling the function
        """
        # Get description of the arguments of the function
        funcDescription = probfit.describe(func)
        # Remove "x" because we need to assign it manually (or want to remove it)
        funcDescription.pop(funcDescription.index("x"))

        # Define the arguments we will call
        argsForFuncCall = collections.OrderedDict()
        # Store the argument for x first
        argsForFuncCall["x"] = xValue
        # Store the rest of the arguments in order
        for name in funcDescription:
            argsForFuncCall[name] = fitContainer.values[name]

        return argsForFuncCall

def GetSignalDominatedFitFunction(epAngle, fitConfig):
    """
    All angles signal dominated is Fourier + Gauss
    EP signal dominated is RPF + Gauss

    """
    # Signal function
    signalFunc = SignalWrapper
    # Background function
    backgroundFunc = GetBackgroundDominatedFitFunction(epAngle, fitConfig)

    if epAngle == JetHUtils.EventPlaneAngle.kAll:
        backgroundFunc = Fourier

        # We don't need to rename the all angles function because we can only use
        # the signal fit on all angles alone. If we fit the other event plane angles
        # at the same time, it will double count
        signalDominatedFunc = probfit.functor.AddPdf(signalFunc, backgroundFunc)
    else:
        # Rename the variables so each signal related variable is independent for each EP
        # We do this by renaming all parameters that are _not_ used in the background
        # NOTE: prefixSkipParameters includes the variable "x", but that is fine, as it
        #       would be automatically excluded (and we don't want to prefix it anyway)!
        prefixSkipParameters = probfit.describe(backgroundFunc)

        # Sum the functions together
        # NOTE: "BG" shouldn't ever need to be used, but it is included so that it fails clearly in the case
        #       that a mistake is made and the prefix is actually matched to and applied to some paramter
        signalDominatedFunc = probfit.functor.AddPdf(signalFunc, backgroundFunc, prefix = [epAngle.str() + "_", "BG"], skip_prefix = prefixSkipParameters)

    logger.debug("epAngle: {}, signalDominatedFunc: {}".format(epAngle.str() , probfit.describe(signalDominatedFunc)))

    return signalDominatedFunc

def GetBackgroundDominatedFitFunction(epAngle, fitConfig):
    """
    All angles background is Fourier
    EP angles background is RPF
    """
    if epAngle == JetHUtils.EventPlaneAngle.kAll:
        backgroundFunc = Fourier
    else:
        # Define constraints
        # Define here for convenience
        # Center of event plane bins
        phiS = {}
        phiS[JetHUtils.EventPlaneAngle.kInPlane] = 0
        phiS[JetHUtils.EventPlaneAngle.kMidPlane] = np.pi/4.0
        phiS[JetHUtils.EventPlaneAngle.kOutOfPlane] = np.pi/2.0
        # EP bin widths
        c = {}
        c[JetHUtils.EventPlaneAngle.kInPlane] = np.pi/6.0
        # NOTE: This value is doubled in the fit to account for the non-continuous regions
        c[JetHUtils.EventPlaneAngle.kMidPlane] = np.pi/12.0
        c[JetHUtils.EventPlaneAngle.kOutOfPlane] = np.pi/6.0
        # Resolution parameters
        resolutionParameters = fitConfig["epResolutionParameters"]

        # Finally define the function
        backgroundFunc = BackgroundWrapperEP(phi = phiS[epAngle],
                                    c = c[epAngle],
                                    resolutionParameters = resolutionParameters)

    logger.debug("epAngle: {}, backgroundFunc: {}".format(epAngle.str() , probfit.describe(backgroundFunc)))

    return backgroundFunc

def SignalWrapper(x, nsAmplitude, asAmpltiude, nsSigma, asSigma, signalPedestal, **kwargs):
    """ Wrapper that basically reassigning parameter names from more descriptive, longer names
    to shorter names for ease in defining the functions.

    kwargs just absorbs the possible extra parameters from minuit.
    """
    signalParameters = {}
    signalParameters["A1"] = nsAmplitude
    signalParameters["A2"] = asAmpltiude
    signalParameters["s1"] = nsSigma
    signalParameters["s2"] = asSigma
    signalParameters["pedestal"] = signalPedestal

    return Signal(x, **signalParameters)

def Signal(x, A1, A2, s1, s2, pedestal):
    """ Signal function has two gaussian peaks (one at the NS (0.0) and one at the AS (np.pi)), along with a pedestal.

    Gaussians are of the form:

    ```
    1/(\sigma*\sqrt(2*\pi) * e^{-(x-0.0)^{2}/(2*\sigma^{2})}
    ```
    """
    return A1*1/(s1*np.sqrt(2*np.pi))*np.exp(-(x-0.0)**2/(2*s1**2) ) + A2*1/(s2*np.sqrt(2*np.pi))*np.exp(-(x-np.pi)**2/(2*s2**2) ) + pedestal

def BackgroundWrapperEP(phi, c, resolutionParameters):
    """ Wrapper around the background function to pass in the relevant parameters

    Args:
        phi (double): Center of tbe event plane bin. Matches up to phi_s in the RPF paper
        c (double): Width of the event plane bin. Matches up to c in the RPF paper
        resolutionParameters (dict): Map from resolution paramaeters of the form "R22" (for the R_{2,2} parameter) to the value

    Returns:
        Wrapper around the actual background function with the parameters set above.
    """
    def BackgroundWrapper(x,
            B, v2_t, v2_a, v4_t, v4_a, v1, v3, **kwargs):
        """ Defines the background function as it will be passed to a particular cost function (and eventually, minuit)

        The arguments must be specified explicitly here because minuit uses it to deteremine the arguments for the function
        via the `iminuit.util.describe()` function.

        kwargs just absorbs the possible extra parameters from minuit.
        """
        backgroundParameters = {}

        backgroundParameters["B"] = B
        backgroundParameters["v2_t"] = v2_t
        backgroundParameters["v2_a"] = v2_a
        backgroundParameters["v4_t"] = v4_t
        backgroundParameters["v4_a"] = v4_a
        backgroundParameters["v1"] = v1
        backgroundParameters["v3"] = v3

        # The resolution parameters are passed directly instead of via the backgroundParameters because
        # they are not something that should vary in our fit
        return Background(x, phi = phi, c = c, resolutionParameters = resolutionParameters, **backgroundParameters)

    return BackgroundWrapper

def Background(x, phi, c, resolutionParameters, B, v2_t, v2_a, v4_t, v4_a, v1, v3, **kwargs):
    """ Background is of the form specified in the RPF paper.

    Resolution parameters implemented include R{2,2} through R{8,2}, which should be the only meaningful values up to v_4^{eff}.

    kwargs just absorbs the possible extra parameters from minuit.
    """
    R22  = resolutionParameters["R22"]
    R42  = resolutionParameters["R42"]
    R62  = resolutionParameters["R62"]
    R82  = resolutionParameters["R82"]

    num = v2_t + cos(2*phi)*sin(2*c)/(2*c)*R22 + v4_t*cos(2*phi)*sin(2*c)/(2*c)*R22 + v2_t*cos(4*phi)*sin(4*c)/(4*c)*R42 + v4_t*cos(6*phi)*sin(6*c)/(6*c)*R62
    den = 1 + 2*v2_t*cos(2*phi)*sin(2*c)/(2*c)*R22 + 2*v4_t*cos(4*phi)*sin(4*c)/(4*c)*R42
    v2R = num/den
    num2 = v4_t + cos(4*phi)*sin(4*c)/(4*c)*R42 + v2_t*cos(2*phi)*sin(2*c)/(2*c)*R22 + v2_t*cos(6*phi)*sin(6*c)/(6*c)*R62 + v4_t*cos(8*phi)*sin(8*c)/(8*c)*R82
    v4R = num2/den
    BR = B*den*c*2/np.pi
    factor = 1.0
    if(c==np.pi/12.0): factor=2.0 # In the case of mid-plane, it has 4 regions instead of 2
    BR = BR*factor
    return Fourier(x, BR, v2R, v2_a, v4R, v4_a, v1, v3)

def Fourier(x, BG, v2_t, v2_a, v4_t, v4_a, v1, v3, **kwargs):
    """ Defines a Fourier series.

    Note that B was renamed to BG so the value would be decoupled from the background of the RPF!

    kwargs just absorbs the possible extra parameters from minuit.
    """
    return BG*(1 + 2*v1*np.cos(x) + 2*v2_t*v2_a*np.cos(2*x) + 2*v3*np.cos(3*x) + 2*v4_t*v4_a*np.cos(4*x))

