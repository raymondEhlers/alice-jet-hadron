#!/usr/bin/env python3

""" Trying to get statistics out of a ROOT hist...

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import numpy as np
import ROOT

def get_stats() -> None:
    # Setup
    h = ROOT.TH1F("tests", "tests", 10, 0, 10)
    h.Sumw2()
    h.Fill(2)
    h.Fill(3)

    #stats = np.ctypeslib.ndpointer(dtype = ctypes.c_double, shape = (4,))
    #stats = ctypes.POINTER(ctypes.c_double * 4)
    # Works!!
    #stats = (ctypes.c_double * 4)()
    # This also works!
    stats = np.array([0., 0., 0., 0.], dtype = np.float64)
    ptr_stats = np.ctypeslib.as_ctypes(stats)
    print(f"ptr_stats: {ptr_stats}")
    print(f"Pre stats : {stats}")
    h.GetStats(np.ctypeslib.as_ctypes(stats))
    print(f"Post stats: {stats}")

if __name__ == "__main__":
    get_stats()

