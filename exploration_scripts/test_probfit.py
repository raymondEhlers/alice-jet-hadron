#!/usr/bin/env python3

""" Test some capabilities of probfit. """

import numpy as np
import probfit
from typing import Callable

from pachyderm import histogram

class MyBinnedLH(probfit.BinnedLH):  # type: ignore
    def __init__(self, f: Callable[..., float], data: histogram.Histogram1D,
                 use_w2: bool = True, extended: bool = False):
        self.f = f
        self.func_code = ...
        self.use_w2 = use_w2
        self.extedned = extended

        self.mymin, self.mymax = np.min(data.y), np.max(data.y)
        self.edges = data.bin_edges
        self.h = data.y
        self.N = np.sum(data.y)
        # NOTE: This was a nice idea, but sadly it doesn't work because the
        #       attributes are declared read only in the cython, so it appears that
        #       it can only be modified in hthe base class init.

if __name__ == "__main__":
    pass
