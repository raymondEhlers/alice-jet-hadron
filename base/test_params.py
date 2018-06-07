#!/usr/bin/env python

# Tests for analysis params.
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
    # Check that the comparison bins are as expected.
    assert comparisonBins == [(1, 0), (1, 1), (1, 3), (1, 4), (1, 5), (1, 7), (1, 8), (2, 0), (2, 1), (2, 3), (2, 4), (2, 5), (2, 7), (2, 8)]
    # Then check the actual output.
    assert list(params.iterateOverJetAndTrackPtBins(config = config)) == comparisonBins

def testOutOfRangeSkipBin(caplog):
    """ Test that an except is generated if a skip bin is out of range.

    The test is performed both with a in range and out of range bin to ensure
    the exception is thrown on the right value.
    """
    caplog.set_level(loggingLevel)

    skipBins = [2, 38]
    config = {"skipPtBins" : {"track" : skipBins}}
    with pytest.raises(ValueError) as exceptionInfo:
        list(params.iterateOverTrackPtBins(config = config))
    # NOTE: ExecptionInfo is a wrapper around the exception. `.value` is the actual exectpion
    #       and then we want to check the value of the first arg, which contains the value
    #       that causes the exception.
    assert exceptionInfo.value.args[0] == skipBins[1]

#############
# Label tests
#############
@pytest.mark.parametrize("value, expected", [
        (r"\textbf{test}", r"#textbf{test}"),
        (r"$\mathrm{test}$", r"#mathrm{test}")
    ], ids = ["just latex", "latex in math mode"])
def testRootLatexConversion(value, expected, caplog):
    """ Test converting latex to ROOT compatiable latex. """
    caplog.set_level(loggingLevel)

    assert params.useLabelWithRoot(value) == expected

@pytest.mark.parametrize("label, expected", [
        ("workInProgress", {"str" : "ALICE Work in Progress"}),
        ("preliminary", {"str" : "ALICE Preliminary"}),
        ("final", {"str" : "ALICE"}),
        ("thesis", {"str" : "This thesis"})
    ], ids = ["workInProgress", "preliminary", "final", "thesis"])
def testAliceLabel(label, expected, caplog):
    """ Tests ALICE labeling. """
    caplog.set_level(loggingLevel)

    aliceLabel = params.aliceLabel[label]
    assert aliceLabel.str() == expected["str"]

@pytest.mark.parametrize("ptBin", params.iterateOverTrackPtBins())
def testTrackPtStrings(ptBin, caplog):
    """ Test the track pt string generation functions. Each bin is tested.  """
    caplog.set_level(loggingLevel)

    assert params.generateTrackPtRangeString(ptBin) == r"$%(lower)s < p_{\mathrm{T}}^{\mathrm{assoc}} < %(upper)s\:\mathrm{GeV/\mathit{c}}$" % {"lower" : params.trackPtBins[ptBin], "upper" : params.trackPtBins[ptBin+1]}

# We retrieve the generator as a list and cut off the last value because we need
# to handle it separately in testJetPtStringForLastPtBin()
@pytest.mark.parametrize("ptBin", list(params.iterateOverJetPtBins())[:-1])
def testJetPtString(ptBin, caplog):
    """ Test the jet pt string generation functions. Each bin (except for the last) is tested.
    The last pt bin is left for a separate test because it is printed differently. (See testJetPtStringForLastPtBin())
    """
    caplog.set_level(loggingLevel)

    assert params.generateJetPtRangeString(ptBin) == r"$%(lower)s < p_{\mathrm{T \,unc,jet}}^{\mathrm{ch+ne}} < %(upper)s\:\mathrm{GeV/\mathit{c}}$" % {"lower" : params.jetPtBins[ptBin], "upper" : params.jetPtBins[ptBin+1]}

def testJetPtStringForLastPtBin(caplog):
    """ Test the jet pt string generation function for the last jet pt bin.

    In the case of the last pt bin, we only want to show the lower range.
    """
    caplog.set_level(loggingLevel)

    ptBin = len(params.jetPtBins) - 2
    assert params.generateJetPtRangeString(ptBin) == r"$%(lower)s < p_{\mathrm{T \,unc,jet}}^{\mathrm{ch+ne}}\:\mathrm{GeV/\mathit{c}}$" % {"lower" : params.jetPtBins[ptBin]}

@pytest.mark.parametrize("energy, system, activity, expected", [
        (2.76, "pp", "inclusive", r"$\mathrm{pp}\:\sqrt{s_{\mathrm{NN}}} = 2.76\:\mathrm{TeV}$"),
        (2.76, "PbPb", "central", r"$\mathrm{Pb\mbox{-}Pb}\:\sqrt{s_{\mathrm{NN}}} = 2.76\:\mathrm{TeV},\:0\mbox{-}10\mbox{\%}$"),
        (2.76, "PbPb", "semiCentral", r"$\mathrm{Pb\mbox{-}Pb}\:\sqrt{s_{\mathrm{NN}}} = 2.76\:\mathrm{TeV},\:30\mbox{-}50\mbox{\%}$"),
        (5.02, "PbPb", "central", r"$\mathrm{Pb\mbox{-}Pb}\:\sqrt{s_{\mathrm{NN}}} = 5.02\:\mathrm{TeV},\:0\mbox{-}10\mbox{\%}$"),
        ("fiveZeroTwo", "PbPb", "central", r"$\mathrm{Pb\mbox{-}Pb}\:\sqrt{s_{\mathrm{NN}}} = 5.02\:\mathrm{TeV},\:0\mbox{-}10\mbox{\%}$"),
        ("5.02", "PbPb", "central", r"$\mathrm{Pb\mbox{-}Pb}\:\sqrt{s_{\mathrm{NN}}} = 5.02\:\mathrm{TeV},\:0\mbox{-}10\mbox{\%}$"),
        (params.collisionEnergy.fiveZeroTwo, params.collisionSystem.PbPb, params.eventActivity.central, r"$\mathrm{Pb\mbox{-}Pb}\:\sqrt{s_{\mathrm{NN}}} = 5.02\:\mathrm{TeV},\:0\mbox{-}10\mbox{\%}$")
    ], ids = ["Inclusive pp", "Central PbPb", "Semi-central PbPb", "Central PbPb at 5.02", "Energy as string fiveZeroTwo", "Energy as string \"5.02\"", "Using enums directly"])
def testSystemLabel(energy, system, activity, expected, caplog):
    """ Test system labels. """
    caplog.set_level(loggingLevel)

    assert params.systemLabel(energy = energy, system = system, activity = activity) == expected

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

@pytest.mark.parametrize("energy, expected", [
        (params.collisionEnergy(2.76),
            {"str" : "2.76",
                "displayStr" : "\sqrt{s_{\mathrm{NN}}} = 2.76\:\mathrm{TeV}",
                "value" : 2.76}),
        (params.collisionEnergy["twoSevenSix"],
            {"str" : "2.76",
                "displayStr" : "\sqrt{s_{\mathrm{NN}}} = 2.76\:\mathrm{TeV}",
                "value" : 2.76}),
        (params.collisionEnergy(5.02),
            {"str" : "5.02",
                "displayStr" : "\sqrt{s_{\mathrm{NN}}} = 5.02\:\mathrm{TeV}",
                "value" : 5.02})
    ], ids = ["2.76 standard", "twoSevenSix alternative intialization", "5.02 standard"])
def testCollisionEnergy(energy, expected, caplog):
    """ Test collision energy values. """
    caplog.set_level(loggingLevel)

    assert str(energy) == expected["str"]
    assert energy.str() == expected["str"]
    assert energy.displayStr() == expected["displayStr"]
    assert energy.value == expected["value"]

@pytest.mark.parametrize("system, expected", [
        (params.collisionSystem["pp"],
            {"str" : "pp",
                "displayStr" : "pp"}),
        (params.collisionSystem["pythia"],
            {"str" : "pythia",
                "displayStr" : "PYTHIA"}),
        (params.collisionSystem["PbPb"],
            {"str" : "PbPb",
                "displayStr" : params.PbPbLatexLabel}),
        (params.collisionSystem["embedPP"],
            {"str" : "embedPP",
                "displayStr" : r"pp \bigotimes %(PbPb)s" % {"PbPb" : params.PbPbLatexLabel}})
    ], ids = ["pp", "pythia", "PbPb", "embedded pp"])
def testCollisionSystem(system, expected, caplog):
    """ Test collision system values. """
    caplog.set_level(loggingLevel)

    assert str(system) == expected["str"]
    assert system.str() == expected["str"]
    assert system.displayStr() == expected["displayStr"]

@pytest.mark.parametrize("activity, expected", [
        (params.eventActivity["inclusive"],
            { "str" : "inclusive",
                "displayStr" : "",
                "range" : params.selectedRange(min = -1, max = -1)}),
        (params.eventActivity["central"],
            { "str" : "central",
                "displayStr" : r",\:0\mbox{-}10\mbox{\%}",
                "range" : params.selectedRange(min = 0, max = 10)}),
        (params.eventActivity["semiCentral"],
            { "str" : "semiCentral",
                "displayStr" : r",\:30\mbox{-}50\mbox{\%}",
                "range" : params.selectedRange(min = 30, max = 50)})
    ], ids = ["inclusive", "central", "semiCentral"])
def testEventActivity(activity, expected, caplog):
    """ Test event activity values. """
    caplog.set_level(loggingLevel)

    assert str(activity) == expected["str"]
    assert activity.str() == expected["str"]
    assert activity.displayStr() == expected["displayStr"]
    assert activity.range() == expected["range"]

@pytest.mark.parametrize("bias, expected", [
        ("NA", {"str" : "NA"}),
        ("track", {"str" : "track"}),
        ("cluster", {"str" : "cluster"}),
        ("both", {"str" : "both"})
    ], ids = ["NA", "track", "cluster", "both"])
def testLeadingHadronBiasType(bias, expected, caplog):
    """ Test the leading hadron bias enum. """
    caplog.set_level(loggingLevel)

    bias = params.leadingHadronBiasType[bias]
    assert str(bias) == expected["str"]
    assert bias.str() == expected["str"]

@pytest.mark.parametrize("type, value, expected", [
        ("NA", 0, {"filenameStr" : "NA0"}),
        ("NA", 5, {"value" : 0, "filenameStr" : "NA0"}),
        ("track", 5, {"filenameStr" : "track5"}),
        ("cluster", 6, {"filenameStr" : "cluster6"}),
        ("both", 10, {"filenameStr" : "both10"})
    ], ids = ["NA", "NAPassedWrongValue", "track", "cluster", "both"])
def testLeadingHadronBias(type, value, expected, caplog):
    """ Test the leading hadron bias class. """
    caplog.set_level(loggingLevel)

    type = params.leadingHadronBiasType[type]
    bias = params.leadingHadronBias(type = type, value = value)
    # Handle value with a bit of care in the case of "NAPassedWrongValue"
    value = expected["value"] if "value" in expected else value
    assert bias.type == type
    assert bias.value == value
    assert bias.filenameStr() == expected["filenameStr"]

@pytest.mark.parametrize("epAngle, expected", [
        ("all",
            {"str" : "all",
                "filenameStr" : "eventPlaneAll",
                "displayStr" : "All"}),
        ("outOfPlane",
            {"str" : "outOfPlane",
                "filenameStr" : "eventPlaneOutOfPlane",
                "displayStr" : "Out-of-plane"})
    ], ids = ["epAngleAll", "epAngleOutOfPlane"])
def testEventPlaneAngleStrings(epAngle, expected, caplog):
    """ Test event plane angle strings. """
    caplog.set_level(loggingLevel)

    epAngle = params.eventPlaneAngle[epAngle]
    assert str(epAngle) == expected["str"]
    assert epAngle.str() == expected["str"]
    assert epAngle.filenameStr() == expected["filenameStr"]
    assert epAngle.displayStr() == expected["displayStr"]

@pytest.mark.parametrize("qVector, expected", [
        ("all",
            {"str" : "all",
                "filenameStr" : "qVectorAll",
                "displayStr" : "All",
                "range" : params.selectedRange(min = 0, max = 100)}),
        ("bottom10",
            {"str" : "bottom10",
                "filenameStr" : "qVectorBottom10",
                "displayStr" : "Bottom 10%",
                "range" : params.selectedRange(min = 0, max = 10)})
    ], ids = ["qVectorAll", "qVectorBottom10"])
def testQVectorStrings(qVector, expected, caplog):
    """ Test q vector strings. """
    caplog.set_level(loggingLevel)

    qVector = params.qVector[qVector]
    assert str(qVector) == expected["str"]
    assert qVector.str() == expected["str"]
    assert qVector.filenameStr() == expected["filenameStr"]
    assert qVector.displayStr() == expected["displayStr"]
    assert qVector.range() == expected["range"]

