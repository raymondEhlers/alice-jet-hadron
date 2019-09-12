#!/usr/bin/env python3

""" Test log scale formatting in MPL.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import pachyderm.plot
pachyderm.plot.configure()

def log_minor_tick_formatter(y: int, pos: float) -> str:
    """ Provide reasonable minor tick formatting for a log y axis.

    Provides ticks on the 2, 3, and 5 for every decade.

    Args:
        y: Tick value.
        pos: Tick position.
    Returns:
        Formatted label.
    """
    ret_val = ""
    # The positions of major ticks appear to be skipped, so the numbering starts at 2
    # Thus, to labe the 2, 3, and 5 ticks, we need to retun the label for the 0th, 1st, and
    # 3rd labels.
    values_to_plot = [0, 1, 3]
    # The values 2 - 9 are availble for the minor ticks, so we take the position mod 8 to
    # ensure that we are repeating the same labels over multiple decades.
    if (pos % 8) in values_to_plot:
        # "g" auto formats to a reasonable presentation for most numbers.
        ret_val = f"{y:g}"
    return ret_val

def log_plot() -> None:
    fig, ax = plt.subplots(figsize = (8, 6))

    x = np.array([1, 2, 3, 4, 5])
    ax.plot(x, np.exp(-x))

    # Set log on the y axis
    ax.set_yscale("log")

    # Increase the number of ticks and labels
    #ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base = 0.2))
    #ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.2))
    #ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.01))
    #ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    #ax.yaxis.set_minor_formatter(matplotlib.ticker.LogFormatter(minor_thresholds = (2, 0.4)))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(log_minor_tick_formatter))
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    #ax.yaxis.set_minor_formatter(matplotlib.ticker.LogFormatterSciNotation(minor_thresholds = (2, 0.3)))
    #ax.ticklabel_format(style='plain', axis='y', useOffset = False)
    #ax.set_ylim(0.9e-2, 5)

    fig.tight_layout()

    fig.savefig("log.pdf")

if __name__ == "__main__":
    log_plot()
