#!/usr/bin/env python

""" Test writing and reading a ROOT object...

"""

#from pachyderm import histogram
#from pachyderm import yaml

import ROOT

class RootHist(ROOT.TH1):

    @classmethod
    def to_yaml(cls):
        pass
