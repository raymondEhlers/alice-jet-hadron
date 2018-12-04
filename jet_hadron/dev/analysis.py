#!/usr/bin/env python

# Run the jet-hadron analysis based on a tree extracted from ALICE data

from __future__ import print_function

import rootpy
import rootpy.ROOT as ROOT
import JetHUtils

class JetHadronAnalysis(object):
    """

    """
    def __init__(self):
        """ Initialize the analysis """
        pass

    def Initialize(self):
        pass

    def RunAnalysis(self):
        pass

def runJetHAnalysis():
    """ Driver function for running the analysis"""
    jetH = JetHadronAnalysis()

    jetH.Initialize()

    jetH.RunAnalysis()

if __name__ == "__main__":
    testHistContainer()

    runJetHAnalysis()

