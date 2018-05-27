#!/usr/bin/env python

# Tests for the JetHParams. Developed to work with pytest.
#
# author: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
# date: 8 May 2018

import pytest
import logging
# Setup logger
logger = logging.getLogger(__name__)

import JetHParams

# Set logging level as a global variable to simplify configuration.
# This is not ideal, but fine for simple tests.
loggingLevel = logging.DEBUG

def getRangeFromBinArray(array):
    """ Helper function to return bin indices from an array.
    Args:
        array (list): Array from which the bin indcies will be extracted.
    Returns:
        list: The bin indices
    """
    return range(len(array) - 1)

def testIterateOverTrackPtBins(caplog):
    """ Test the track pt bins generator.

    Note that we wrap the function in list so we get all of the values from the generator.
    """
    caplog.set_level(loggingLevel)
    assert len(JetHParams.trackPtBins) == 10
    assert list(JetHParams.iterateOverTrackPtBins()) == list(getRangeFromBinArray(JetHParams.trackPtBins))

def testIterateOverTrackPtBinsWithConfig(caplog):
    """ Test the track pt bins generator with some bins skipped.

    The values to skip were not selected with any paticular critera except to be non-continuous.
    """
    caplog.set_level(loggingLevel)

    skipBins = [2, 6]
    comparisonBins = [x for x in getRangeFromBinArray(JetHParams.trackPtBins) if not x in skipBins]
    config = {"skipPtBins" : {"track" : skipBins}}
    assert list(JetHParams.iterateOverTrackPtBins(config = config)) == comparisonBins

def testIterateOverJetPtBins(caplog):
    """ Test the jet pt bins generator.

    Note that we wrap the function in list so we get all of the values from the generator.
    """
    caplog.set_level(loggingLevel)

    # Ensure that we have the expected number of jet pt bins
    assert len(JetHParams.jetPtBins) == 5
    # Then test the actual iterable.
    assert list(JetHParams.iterateOverJetPtBins()) == list(getRangeFromBinArray(JetHParams.jetPtBins))

def testIterateOverJetPtBinsWithConfig(caplog):
    """ Test the jet pt bins generator with some bins skipped.

    The values to skip were not selected with any paticular critera except to be non-continuous.
    """
    caplog.set_level(loggingLevel)

    skipBins = [0, 2]
    comparisonBins = [x for x in getRangeFromBinArray(JetHParams.jetPtBins) if not x in skipBins]
    config = {"skipPtBins" : {"jet" : skipBins}}
    assert list(JetHParams.iterateOverJetPtBins(config = config)) == comparisonBins

def testIterateOverJetAndTrackPtBins(caplog):
    """ Test the jet and track pt bins generator.

    Note that we wrap the function in list so we get all of the values from the generator.
    """
    caplog.set_level(loggingLevel)

    comparisonBins = [(x, y) for x in getRangeFromBinArray(JetHParams.jetPtBins) for y in getRangeFromBinArray(JetHParams.trackPtBins)]
    assert list(JetHParams.iterateOverJetAndTrackPtBins()) == comparisonBins

def testIterateOverJetAndTrackPtBinsWithConfig(caplog):
    """ Test the jet and track pt bins generator with some bins skipped.

    The values to skip were not selected with any paticular critera except to be non-continuous.
    """
    caplog.set_level(loggingLevel)

    skipJetPtBins = [0, 3]
    skipTrackPtBins = [2, 6]
    comparisonBins = [(x,y) for x in getRangeFromBinArray(JetHParams.jetPtBins) for y in getRangeFromBinArray(JetHParams.trackPtBins) if not x in skipJetPtBins and not y in skipTrackPtBins]
    config = {"skipPtBins": {"jet" : skipJetPtBins, "track" : skipTrackPtBins}}
    assert list(JetHParams.iterateOverJetAndTrackPtBins(config = config)) == comparisonBins
    assert comparisonBins == [(1, 0), (1, 1), (1, 3), (1, 4), (1, 5), (1, 7), (1, 8), (2, 0), (2, 1), (2, 3), (2, 4), (2, 5), (2, 7), (2, 8)]

def testOutOfRangeSkipBin(caplog):
    """ Test that an except is generated if a skip bin is out of range.

    The test is performed both with a in range and out of range bin to ensure
    the exception is thrown on the right value.
    """
    caplog.set_level(loggingLevel)

    skipBins = [2, 38]
    config = {"skipPtBins" : {"track" : skipBins}}
    caughtExpectedException = False
    exceptionValue = None
    try:
        list(JetHParams.iterateOverTrackPtBins(config = config))
    except ValueError as e:
        caughtExpectedException = True
        # The first arg is the value which caused the ValueError.
        exceptionValue = e.args[0]

    # An exception should be thrown for the scond value, which is out of range.
    assert caughtExpectedException == True
    # The exception should have returned the value it failed on.
    assert exceptionValue == skipBins[1]

#############
# Latex tests
#############
def testTrackPtStrings(caplog):
    """ Test the track pt string generation functions. Each bin is tested.  """
    caplog.set_level(loggingLevel)

    for ptBin in JetHParams.iterateOverTrackPtBins():
        print(ptBin)
        assert JetHParams.generateTrackPtRangeString(ptBin) == r"$%(lower)s < p_{\mathrm{T}}^{\mathrm{assoc}} < %(upper)s\:\mathrm{GeV/\mathit{c}}$" % {"lower" : JetHParams.trackPtBins[ptBin], "upper" : JetHParams.trackPtBins[ptBin+1]}

def testJetPtString(caplog):
    """ Test the jet pt string generation functions. Each bin (except for the last) is tested.
    The last pt bin is left for a separate test because it is printed differently. (See testJetPtStringForLastPtBin())
    """
    caplog.set_level(loggingLevel)

    # We retrieve the generator as a list and cut off the last value because we need
    # to handle it separately in testJetPtStringForLastPtBin()
    for ptBin in list(JetHParams.iterateOverJetPtBins())[:-1]:
        assert JetHParams.generateJetPtRangeString(ptBin) == r"$%(lower)s < p_{\mathrm{T \,unc,jet}}^{\mathrm{ch+ne}} < %(upper)s\:\mathrm{GeV/\mathit{c}}$" % {"lower" : JetHParams.jetPtBins[ptBin], "upper" : JetHParams.jetPtBins[ptBin+1]}

def testJetPtStringForLastPtBin(caplog):
    """ Test the jet pt string generation function for the last jet pt bin.

    In the case of the last pt bin, we only want to show the lower range.
    """
    caplog.set_level(loggingLevel)

    ptBin = len(JetHParams.jetPtBins) - 2
    assert JetHParams.generateJetPtRangeString(ptBin) == r"$%(lower)s < p_{\mathrm{T \,unc,jet}}^{\mathrm{ch+ne}}\:\mathrm{GeV/\mathit{c}}$" % {"lower" : JetHParams.jetPtBins[ptBin]}

def testPPSystemLabel(caplog):
    """ Test the pp system label. """
    caplog.set_level(loggingLevel)

    assert JetHParams.systemLabel(collisionSystem = "pp", eventActivity = "inclusive", energy = 2.76) == r"$\mathrm{pp}\:\sqrt{s_{\mathrm{NN}}} = 2.76\:\mathrm{TeV}$"

def testPbPbCentralSystemLabel(caplog):
    """ Test the PbPb Central system label"""
    caplog.set_level(loggingLevel)

    assert JetHParams.systemLabel(collisionSystem = "PbPb", eventActivity = "central", energy = 2.76) == r"$\mathrm{Pb\mbox{-}Pb}\:\sqrt{s_{\mathrm{NN}}} = 2.76\:\mathrm{TeV},\:0\mbox{-}10\mbox{\%}$"

def testPbPbSemiCentralSystemLabel(caplog):
    """ Test the PbPb semi-central system label"""
    caplog.set_level(loggingLevel)

    assert JetHParams.systemLabel(collisionSystem = "PbPb", eventActivity = "semiCentral", energy = 2.76) == r"$\mathrm{Pb\mbox{-}Pb}\:\sqrt{s_{\mathrm{NN}}} = 2.76\:\mathrm{TeV},\:30\mbox{-}50\mbox{\%}$"

def testWithoutEventActivityForBackwardsCompatability(caplog):
    """ Test the backwards compatiable functionality where the event activity is not specified.
    In that case, it depends on the collision system.
    """
    caplog.set_level(loggingLevel)

    assert JetHParams.systemLabel(collisionSystem = "pp", energy = 2.76) == r"$\mathrm{pp}\:\sqrt{s_{\mathrm{NN}}} = 2.76\:\mathrm{TeV}$"
    assert JetHParams.systemLabel(collisionSystem = "PbPb", energy = 2.76) == r"$\mathrm{Pb\mbox{-}Pb}\:\sqrt{s_{\mathrm{NN}}} = 2.76\:\mathrm{TeV},\:0\mbox{-}10\mbox{\%}$"

def testDifferentEnergySystemLabel(caplog):
    """ Test the system label for a different energy. """
    caplog.set_level(loggingLevel)

    assert JetHParams.systemLabel(collisionSystem = "PbPb", energy = 5.02) == r"$\mathrm{Pb\mbox{-}Pb}\:\sqrt{s_{\mathrm{NN}}} = 5.02\:\mathrm{TeV},\:0\mbox{-}10\mbox{\%}$"

def testJetPropertiesLabels(caplog):
    """ Test the jet properties labels. """
    caplog.set_level(loggingLevel)

    jetPtBin = 1
    (jetFindingExpected, constituentCutsExpected, leadingHadronExpected, jetPtExpected) = (r"$\mathrm{anti\mbox{-}k}_{\mathrm{T}}\;R=0.2$",
            r"$p_{\mathrm{T}}^{\mathrm{ch}}\:\mathrm{\mathit{c},}\:\mathrm{E}_{\mathrm{T}}^{\mathrm{clus}} > 3\:\mathrm{GeV}$",
            r"$p_{\mathrm{T}}^{\mathrm{lead,ch}} > 5\:\mathrm{GeV/\mathit{c}}$",
            r"$20.0 < p_{\mathrm{T \,unc,jet}}^{\mathrm{ch+ne}} < 40.0\:\mathrm{GeV/\mathit{c}}$")

    (jetFinding, constituentCuts, leadingHadron, jetPt) = JetHParams.jetPropertiesLabel(jetPtBin)

    assert jetFinding == jetFindingExpected
    assert constituentCuts == constituentCutsExpected
    assert leadingHadron == leadingHadronExpected
    assert jetPt == jetPtExpected

