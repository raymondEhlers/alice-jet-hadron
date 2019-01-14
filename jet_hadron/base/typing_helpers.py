#!/usr/bin/env python

""" Typing helpers

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

from typing import Any, Union

# Typing helper
try:
    import ROOT
    Hist = Union[ROOT.TH1, ROOT.THnBase]
except ImportError:
    # It doesn't like the possibility of redefining this, so we need to tell ``mypy`` to ignore it.
    Hist = Any  # type: ignore

