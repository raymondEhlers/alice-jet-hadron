#!/usr/bin/env python

# Tests for the params. Developed to work with pytest.
#
# author: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
# date: 8 May 2018

import pytest
import logging
# Setup logger
logger = logging.getLogger(__name__)

import jetH.base.params as params

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
    assert len(params.trackPtBins) == 10
    assert list(params.iterateOverTrackPtBins()) == list(getRangeFromBinArray(params.trackPtBins))

def testIterateOverTrackPtBinsWithConfig(caplog):
    """ Test the track pt bins generator with some bins skipped.

    The values to skip were not selected with any paticular critera except to be non-continuous.
    """
    caplog.set_level(loggingLevel)

    skipBins = [2, 6]
    comparisonBins = [x for x in getRangeFromBinArray(params.trackPtBins) if not x in skipBins]
    config = {"skipPtBins" : {"track" : skipBins}}
    assert list(params.iterateOverTrackPtBins(config = config)) == comparisonBins

def testIterateOverJetPtBins(caplog):
    """ Test the jet pt bins generator.

    Note that we wrap the function in list so we get all of the values from the generator.
    """
    caplog.set_level(loggingLevel)

    # Ensure that we have the expected number of jet pt bins
    assert len(params.jetPtBins) == 5
    # Then test the actual iterable.
    assert list(params.iterateOverJetPtBins()) == list(getRangeFromBinArray(params.jetPtBins))

def testIterateOverJetPtBinsWithConfig(caplog):
    """ Test the jet pt bins generator with some bins skipped.

    The values to skip were not selected with any paticular critera except to be non-continuous.
    """
    caplog.set_level(loggingLevel)

    skipBins = [0, 2]
    comparisonBins = [x for x in getRangeFromBinArray(params.jetPtBins) if not x in skipBins]
    config = {"skipPtBins" : {"jet" : skipBins}}
    assert list(params.iterateOverJetPtBins(config = config)) == comparisonBins

def testIterateOverJetAndTrackPtBins(caplog):
    """ Test the jet and track pt bins generator.

    Note that we wrap the function in list so we get all of the values from the generator.
    """
    caplog.set_level(loggingLevel)

    comparisonBins = [(x, y) for x in getRangeFromBinArray(params.jetPtBins) for y in getRangeFromBinArray(params.trackPtBins)]
    assert list(params.iterateOverJetAndTrackPtBins()) == comparisonBins

def testIterateOverJetAndTrackPtBinsWithConfig(caplog):
    """ Test the jet and track pt bins generator with some bins skipped.

    The values to skip were not selected with any paticular critera except to be non-continuous.
    """
    caplog.set_level(loggingLevel)

    skipJetPtBins = [0, 3]
    skipTrackPtBins = [2, 6]
    comparisonBins = [(x,y) for x in getRangeFromBinArray(params.jetPtBins) for y in getRangeFromBinArray(params.trackPtBins) if not x in skipJetPtBins and not y in skipTrackPtBins]
    config = {"skipPtBins": {"jet" : skipJetPtBins, "track" : skipTrackPtBins}}
    assert list(params.iterateOverJetAndTrackPtBins(config = config)) == comparisonBins
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
        list(params.iterateOverTrackPtBins(config = config))
    except ValueError as e:
        caughtExpectedException = True
        # The first arg is the value which caused the ValueError.
        exceptionValue = e.args[0]

    # An exception should be thrown for the scond value, which is out of range.
    assert caughtExpectedException == True
    # The exception should have returned the value it failed on.
    assert exceptionValue == skipBins[1]

#############
# Label tests
#############
def testRootLatexConversion(caplog):
    """ Test converting latex to ROOT compatiable latex. """
    caplog.set_level(loggingLevel)

    assert params.useLabelWithRoot(r"\textbf{test}") == r"#textbf{test}"
    assert params.useLabelWithRoot(r"$\mathrm{test}$") == r"#mathrm{test}"

def testAliceLabel(caplog):
    """ Tests for ALICE labeling. """
    caplog.set_level(loggingLevel)

    testParams = [
            ("workInProgress", {"str" : "ALICE Work in Progress"}),
            ("preliminary", {"str" : "ALICE Preliminary"}),
            ("final", {"str" : "ALICE"}),
            ("thesis", {"str" : "This thesis"})
        ]

    for label, expected in testParams:
        aliceLabel = params.aliceLabel[label]
        assert aliceLabel.str() == expected["str"]

def testTrackPtStrings(caplog):
    """ Test the track pt string generation functions. Each bin is tested.  """
    caplog.set_level(loggingLevel)

    for ptBin in params.iterateOverTrackPtBins():
        print(ptBin)
        assert params.generateTrackPtRangeString(ptBin) == r"$%(lower)s < p_{\mathrm{T}}^{\mathrm{assoc}} < %(upper)s\:\mathrm{GeV/\mathit{c}}$" % {"lower" : params.trackPtBins[ptBin], "upper" : params.trackPtBins[ptBin+1]}

def testJetPtString(caplog):
    """ Test the jet pt string generation functions. Each bin (except for the last) is tested.
    The last pt bin is left for a separate test because it is printed differently. (See testJetPtStringForLastPtBin())
    """
    caplog.set_level(loggingLevel)

    # We retrieve the generator as a list and cut off the last value because we need
    # to handle it separately in testJetPtStringForLastPtBin()
    for ptBin in list(params.iterateOverJetPtBins())[:-1]:
        assert params.generateJetPtRangeString(ptBin) == r"$%(lower)s < p_{\mathrm{T \,unc,jet}}^{\mathrm{ch+ne}} < %(upper)s\:\mathrm{GeV/\mathit{c}}$" % {"lower" : params.jetPtBins[ptBin], "upper" : params.jetPtBins[ptBin+1]}

def testJetPtStringForLastPtBin(caplog):
    """ Test the jet pt string generation function for the last jet pt bin.

    In the case of the last pt bin, we only want to show the lower range.
    """
    caplog.set_level(loggingLevel)

    ptBin = len(params.jetPtBins) - 2
    assert params.generateJetPtRangeString(ptBin) == r"$%(lower)s < p_{\mathrm{T \,unc,jet}}^{\mathrm{ch+ne}}\:\mathrm{GeV/\mathit{c}}$" % {"lower" : params.jetPtBins[ptBin]}

def testPPSystemLabel(caplog):
    """ Test the pp system label. """
    caplog.set_level(loggingLevel)

    assert params.systemLabel(collisionSystem = "pp", eventActivity = "inclusive", energy = 2.76) == r"$\mathrm{pp}\:\sqrt{s_{\mathrm{NN}}} = 2.76\:\mathrm{TeV}$"

def testPbPbCentralSystemLabel(caplog):
    """ Test the PbPb Central system label"""
    caplog.set_level(loggingLevel)

    assert params.systemLabel(collisionSystem = "PbPb", eventActivity = "central", energy = 2.76) == r"$\mathrm{Pb\mbox{-}Pb}\:\sqrt{s_{\mathrm{NN}}} = 2.76\:\mathrm{TeV},\:0\mbox{-}10\mbox{\%}$"

def testPbPbSemiCentralSystemLabel(caplog):
    """ Test the PbPb semi-central system label"""
    caplog.set_level(loggingLevel)

    assert params.systemLabel(collisionSystem = "PbPb", eventActivity = "semiCentral", energy = 2.76) == r"$\mathrm{Pb\mbox{-}Pb}\:\sqrt{s_{\mathrm{NN}}} = 2.76\:\mathrm{TeV},\:30\mbox{-}50\mbox{\%}$"

def testWithoutEventActivityForBackwardsCompatability(caplog):
    """ Test the backwards compatiable functionality where the event activity is not specified.
    In that case, it depends on the collision system.
    """
    caplog.set_level(loggingLevel)

    assert params.systemLabel(collisionSystem = "pp", energy = 2.76) == r"$\mathrm{pp}\:\sqrt{s_{\mathrm{NN}}} = 2.76\:\mathrm{TeV}$"
    assert params.systemLabel(collisionSystem = "PbPb", energy = 2.76) == r"$\mathrm{Pb\mbox{-}Pb}\:\sqrt{s_{\mathrm{NN}}} = 2.76\:\mathrm{TeV},\:0\mbox{-}10\mbox{\%}$"

def testDifferentEnergySystemLabel(caplog):
    """ Test the system label for a different energy. """
    caplog.set_level(loggingLevel)

    assert params.systemLabel(collisionSystem = "PbPb", energy = 5.02) == r"$\mathrm{Pb\mbox{-}Pb}\:\sqrt{s_{\mathrm{NN}}} = 5.02\:\mathrm{TeV},\:0\mbox{-}10\mbox{\%}$"

def testJetPropertiesLabels(caplog):
    """ Test the jet properties labels. """
    caplog.set_level(loggingLevel)

    jetPtBin = 1
    (jetFindingExpected, constituentCutsExpected, leadingHadronExpected, jetPtExpected) = (r"$\mathrm{anti\mbox{-}k}_{\mathrm{T}}\;R=0.2$",
            r"$p_{\mathrm{T}}^{\mathrm{ch}}\:\mathrm{\mathit{c},}\:\mathrm{E}_{\mathrm{T}}^{\mathrm{clus}} > 3\:\mathrm{GeV}$",
            r"$p_{\mathrm{T}}^{\mathrm{lead,ch}} > 5\:\mathrm{GeV/\mathit{c}}$",
            r"$20.0 < p_{\mathrm{T \,unc,jet}}^{\mathrm{ch+ne}} < 40.0\:\mathrm{GeV/\mathit{c}}$")

    (jetFinding, constituentCuts, leadingHadron, jetPt) = params.jetPropertiesLabel(jetPtBin)

    assert jetFinding == jetFindingExpected
    assert constituentCuts == constituentCutsExpected
    assert leadingHadron == leadingHadronExpected
    assert jetPt == jetPtExpected

def testCollisionEnergy(caplog):
    """ Test collision energy values. """
    caplog.set_level(loggingLevel)

    output276 = {"str" : "2.76", "value" : 2.76}
    output502 = {"str" : "5.02", "value" : 5.02}
    testParams = [
            # Default test
            (params.collisionEnergy(2.76), output276),
            # Test alternative initialization
            (params.collisionEnergy["twoSevenSix"], output276),
            # Test different energy
            (params.collisionEnergy(5.02), output502)
            ]

    for energy, expected in testParams:
        assert str(energy) == expected["str"]
        assert energy.str() == expected["str"]
        assert energy.value == expected["value"]

def testCollisionSystem(caplog):
    """ Test collision system values. """
    caplog.set_level(loggingLevel)

    testParams = [
            # Default tests
            (params.collisionSystem["pp"], {"str" : "pp", "filenameStr" : "pp", "value" : 0}),
            (params.collisionSystem["PbPb"], {"str" : "PbPb", "filenameStr" : "PbPb", "value" : 2}),
            # Alias to pp
            (params.collisionSystem["embedPP"], {"str" : "embedPP", "filenameStr" : "embedPP", "value" : params.collisionSystem.pp.value})
            ]
    for system, expected in testParams:
        assert str(system) == expected["str"]
        assert system.str() == expected["str"]
        assert system.filenameStr() == expected["filenameStr"]
        assert system.value == expected["value"]

def testEventActivity(caplog):
    """ Test event activity values. """
    caplog.set_level(loggingLevel)

    assert False

def testLeadingHadron(caplog):
    """ Test determining the leading hadron bias. """
    caplog.set_level(loggingLevel)

    assert False

def testEventPlaneAngleStrings(caplog):
    """ Test event plane angle strings. """
    caplog.set_level(loggingLevel)

    # Also test out of plane, with args something like
    tests = [
        (params.eventPlaneAngle.all,
            {"str" : "all",
             "filenameStr" : "eventPlaneAll",
             "displayStr" : "All"}),
        (params.eventPlaneAngle.outOfPlane,
            {"str" : "outOfPlane",
             "filenameStr" : "eventPlaneOutOfPlane",
             "displayStr" : "Out-of-plane"})
        ]

    for angle, testValues in tests:
        assert str(angle) == testValues["str"]
        assert angle.str() == testValues["str"]
        assert angle.filenameStr() == testValues["filenameStr"]
        assert angle.displayStr() == testValues["displayStr"]

def testQVectorStrings(caplog):
    """ Test q vector strings. """
    caplog.set_level(loggingLevel)

    # Also test out of plane, with args something like
    tests = [
        (params.qVector.all,
            {"str" : "all",
             "filenameStr" : "qVectorAll",
             "displayStr" : "All"}),
        (params.qVector.bottom10,
            {"str" : "bottom10",
             "filenameStr" : "qVectorBottom10",
             "displayStr" : "Bottom 10%"})
        ]

    for angle, testValues in tests:
        assert str(angle) == testValues["str"]
        assert angle.str() == testValues["str"]
        assert angle.filenameStr() == testValues["filenameStr"]
        assert angle.displayStr() == testValues["displayStr"]

