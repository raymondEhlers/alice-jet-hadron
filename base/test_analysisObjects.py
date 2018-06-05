#!/usr/bin/env python

# Tests for the analysisObjects module. Developed to work with pytest.
#
# author: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
# date: 8 May 2018

import pytest
import logging
# Setup logger
logger = logging.getLogger(__name__)

import jetH.base.analysisObjects as analysisObjects

@pytest.fixture
def testHist():
    """ Create an empty TH1F for testing.

    Args:
        None
    Returns:
        TH1F: Empty hist to use for testing.
    """
    import warnings
    # Handle rootpy warning
    warnings.filterwarnings(action='ignore', category=RuntimeWarning, message=r'creating converter for unknown type "_Atomic\(bool\)"')
    import rootpy.ROOT as ROOT

    # Define the hist to test with 
    hist = ROOT.TH1F("test", "test", 10, 0, 1)

    return hist

def testHistContainer(testHist):
    """ Test the hist container class """

    # Create the container
    cont = analysisObjects.HistContainer(testHist)
    # Test the basic properties
    assert "test" == cont.GetName()
    assert "testFunc" == cont.testFunc()


