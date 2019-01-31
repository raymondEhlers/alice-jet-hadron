#!/usr/bin/env python

""" Typing helpers

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

from typing import Any, Union

# Typing helper
try:
    import ROOT
    Axis = ROOT.TAxis
    Hist = Union[ROOT.TH1, ROOT.THnBase]
    Canvas = ROOT.TCanvas
except ImportError:
    # It doesn't like the possibility of redefining this, so we need to tell ``mypy`` to ignore it.
    Axis = Any  # type: ignore
    Hist = Any  # type: ignore
    Canvas = Any  # type: ignore

# Tell ROOT to ignore command line options so args are passed to python
# NOTE: Must be immediately after import ROOT and sometimes must be the first ROOT related import!
#       We do this here (even though it is unrelated to typing helpers) because it is the most common
#       first import which requires ROOT. So by putting it here, it should cover all executables.
ROOT.PyConfig.IgnoreCommandLineOptions = True
