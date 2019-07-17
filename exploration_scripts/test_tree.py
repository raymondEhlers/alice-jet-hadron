#!/usr/bin/env python

import IPython
import numpy as np
#import ROOT

DTYPE_BASE = np.dtype([("pT", np.float64), ("eta", np.float64), ("phi", np.float64), ("m", np.float64)])
print(f"DTYPE_BASE: {DTYPE_BASE}")

#status_dtype = DTYPE_EP.descr + [("status_code", np.int32)]

DTYPE_JETS = [(f"{label}_{name}", dtype) for label in ["part", "det"] for name, dtype in DTYPE_BASE.descr]
print(f"DTYPE_JETS: {DTYPE_JETS}")

output_array = np.zeros(1, dtype = DTYPE_JETS)

part_jet = np.array((1, 0.5, 0.5, 0), dtype = DTYPE_BASE)
det_jet = np.array((2, 0.5, 0.75, 1), dtype = DTYPE_BASE)

IPython.embed()

# None of this works...
#temp = np.concatenate(part_jet[:], det_jet[:], axis = 1)
#output_array[:4] = part_jet
#output_array[0] = temp

print(f"output_array: {output_array}")
