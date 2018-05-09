#!/usr/bin/env python

# Tests for the JetHUtils. Developed to work with pytest.
#
# author: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
# date: 8 May 2018

import pytest
import logging
# Setup logger
logger = logging.getLogger(__name__)

import JetHParams

def testTrackPtStrings():
    """ Test the track pt string generation functions. Each bin is tested.  """
    for ptBin in JetHParams.iterateOverTrackPtBins():
        print(ptBin)
        assert JetHParams.generateTrackPtRangeString(ptBin) == "$%(lower)s < p_{\\mathrm{T}}^{\\mathrm{assoc}} < %(upper)s\\:\\mathrm{GeV/\\mathit{c}}$" % {"lower" : JetHParams.trackPtBins[ptBin], "upper" : JetHParams.trackPtBins[ptBin+1]}

def testJetPtString():
    """ Test the jet pt string generation functions. Each bin (except for the last) is tested.
    The last pt bin is left for a separate test because it is printed differently. (See testJetPtStringForLastPtBin())
    """
    # We retrieve the generator as a list and cut off the last value because we need
    # to handle it separately in testJetPtStringForLastPtBin()
    for ptBin in list(JetHParams.iterateOverJetPtBins())[:-1]:
        assert JetHParams.generateJetPtRangeString(ptBin) == "$%(lower)s < p_{\\mathrm{T \\,unc,jet}}^{\\mathrm{ch+ne}} < %(upper)s\\:\\mathrm{GeV/\\mathit{c}}$" % {"lower" : JetHParams.jetPtBins[ptBin], "upper" : JetHParams.jetPtBins[ptBin+1]}

def testJetPtStringForLastPtBin():
    """ Test the jet pt string generation function for the last jet pt bin.

    In the case of the last pt bin, we only want to show the lower range.
    """
    ptBin = len(JetHParams.jetPtBins) - 2
    assert JetHParams.generateJetPtRangeString(ptBin) == "$%(lower)s < p_{\\mathrm{T \\,unc,jet}}^{\\mathrm{ch+ne}}\\:\\mathrm{GeV/\\mathit{c}}$" % {"lower" : JetHParams.jetPtBins[ptBin]}

def getRangeFromBinArray(array):
    """ Helper function to return bin indices from an array.
    Args:
        array (list): Array from which the bin indcies will be extracted.
    Returns:
        list: The bin indices
    """
    return range(len(array) - 1)

def testIterateOverTrackPtBins():
    """ Test the track pt bins generator.

    Note that we wrap the function in list so we get all of the values from the generator.
    """
    assert len(JetHParams.trackPtBins) == 10
    assert list(JetHParams.iterateOverTrackPtBins()) == getRangeFromBinArray(JetHParams.trackPtBins)

def testIterateOverTrackPtBinsWithConfig():
    """ Test the track pt bins generator with some bins skipped.

    The values to skip were not selected with any paticular critera except to be non-continuous.
    """
    skipBins = [2, 6]
    comparisonBins = [x for x in getRangeFromBinArray(JetHParams.trackPtBins) if not x in skipBins]
    config = {"skipPtBins" : {"track" : skipBins}}
    assert list(JetHParams.iterateOverTrackPtBins(config = config)) == comparisonBins

def testIterateOverJetPtBins():
    """ Test the jet pt bins generator.

    Note that we wrap the function in list so we get all of the values from the generator.
    """
    # Ensure that we have the expected number of jet pt bins
    assert len(JetHParams.jetPtBins) == 5
    # Then test the actual iterable.
    assert list(JetHParams.iterateOverJetPtBins()) == getRangeFromBinArray(JetHParams.jetPtBins)

def testIterateOverJetPtBinsWithConfig():
    """ Test the jet pt bins generator with some bins skipped.

    The values to skip were not selected with any paticular critera except to be non-continuous.
    """
    skipBins = [0, 2]
    comparisonBins = [x for x in getRangeFromBinArray(JetHParams.jetPtBins) if not x in skipBins]
    config = {"skipPtBins" : {"jet" : skipBins}}
    assert list(JetHParams.iterateOverJetPtBins(config = config)) == comparisonBins

def testIterateOverJetAndTrackPtBins():
    """ Test the jet and track pt bins generator.

    Note that we wrap the function in list so we get all of the values from the generator.
    """
    comparisonBins = [(x, y) for x in getRangeFromBinArray(JetHParams.jetPtBins) for y in getRangeFromBinArray(JetHParams.trackPtBins)]
    assert list(JetHParams.iterateOverJetAndTrackPtBins()) == comparisonBins

def testIterateOverJetAndTrackPtBinsWithConfig():
    """ Test the jet and track pt bins generator with some bins skipped.

    The values to skip were not selected with any paticular critera except to be non-continuous.
    """
    skipJetPtBins = [0, 3]
    skipTrackPtBins = [2, 6]
    comparisonBins = [(x,y) for x in getRangeFromBinArray(JetHParams.jetPtBins) for y in getRangeFromBinArray(JetHParams.trackPtBins) if not x in skipJetPtBins and not y in skipTrackPtBins]
    config = {"skipPtBins": {"jet" : skipJetPtBins, "track" : skipTrackPtBins}}
    assert list(JetHParams.iterateOverJetAndTrackPtBins(config = config)) == comparisonBins
    assert comparisonBins == [(1, 0), (1, 1), (1, 3), (1, 4), (1, 5), (1, 7), (1, 8), (2, 0), (2, 1), (2, 3), (2, 4), (2, 5), (2, 7), (2, 8)]

def testOutOfRangeSkipBin():
    """ Test that an except is generated if a skip bin is out of range.

    The test is performed both with a in range and out of range bin to ensure
    the exception is thrown on the right value.
    """
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
