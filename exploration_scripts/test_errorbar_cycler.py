#!/usr/bin/env python

""" Tets for bug in errorbar with fillstyle cycler.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

from cycler import cycler
import matplotlib.pyplot as plt

fillstyle_cycler = cycler("fillstyle", ["none", "full"])
fig, ax = plt.subplots()
#ax.set_prop_cycle(fillstyle_cycler)
# Will work if `yerr` is not specified.
ax.errorbar([1, 2, 3], [4, 5, 6], yerr = [1, 1, 1], marker = "o", fillstyle = "none")
#ax.errorbar([1, 2, 3], [4, 5, 6], marker = "o", fillstyle = "full")

fig.savefig("test.pdf")

