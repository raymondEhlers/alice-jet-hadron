#!/usr/bin/env pythoon

import pytest

@pytest.fixture
def testRootHists():
    """ Create minimal TH*F hists in 1D, 2D, and 3D. Each has been filled once.

    Args:
        None
    Returns:
        tuple: (TH1F, TH2F, TH3F) for testing
    """
    import warnings
    # Handle rootpy warning
    warnings.filterwarnings(action='ignore', category=RuntimeWarning, message=r'creating converter for unknown type "_Atomic\(bool\)"')
    import rootpy.ROOT as ROOT
    import collections

    rootHists = collections.namedtuple("rootHists", ["hist1D", "hist2D", "hist3D"])

    # Define the hist to test with 
    hist = ROOT.TH1F("test", "test", 10, 0, 1)
    hist.Fill(.1)
    hist2D = ROOT.TH2F("test2", "test2", 10, 0, 1, 10, 0, 10)
    hist2D.Fill(.1, 1)
    hist3D = ROOT.TH3F("test3", "test3", 10, 0, 1, 10, 0, 10, 10, 0, 100)
    hist3D.Fill(.1, 1, 10)

    return rootHists(hist1D = hist, hist2D = hist2D, hist3D = hist3D)

