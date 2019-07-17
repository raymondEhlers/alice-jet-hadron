#!/usr/bin/env python3

""" Tests for combining Gaussians together.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple

def create_gaussian_inputs() -> Tuple[np.ndarray, np.ndarray]:
    g1 = np.random.normal(0, 1, size = 1000000)
    g2 = np.random.normal(0, 2, size = 1000000)

    return (g1, g2)

#def calculate_var(a_1, sigma_1, a_2, sigma_2):
#    return 1 / (f_1.GetParameter(0) * np.sqrt(2 * np.pi * f_1.GetParameter(2) ** 2) + f_2.GetParameter(0) * np.sqrt(2 * np.pi * f_2.GetParameter(2) ** 2)) * \
#        np.sqrt(2 * np.pi) * (f_1.GetParameter(0) * f_1.GetParameter(2) ** 3 + f_2.GetParameter(0) * f_2.GetParameter(2) ** 3)

def combine_gaussians() -> None:
    hist_args = {
        "bins": 200,
        "range": (-10, 10),
        "alpha": 0.5,
    }
    g1, g2 = create_gaussian_inputs()

    g_combined = np.append(g1, g2)

    fig, ax = plt.subplots(figsize = (8, 6))

    g1_counts, g1_bin_edges, _ = ax.hist(g1, label = "Gen. mu = 1", **hist_args)
    g2_counts, g2_bin_edges, _ = ax.hist(g2, label = "Gen. mu = 2", **hist_args)
    ax.hist(g_combined, label = "Appended arrays", **hist_args)
    ax.plot(g1_bin_edges[:-1] + (g1_bin_edges[1:] - g1_bin_edges[:-1]) / 2.0, g1_counts + g2_counts, label = "Added hist")
    x = np.linspace(-10, 10, 101)
    ax.plot(x, 50000 * 1 / (np.sqrt(2 * np.pi) * 1.0) * np.exp(- x ** 2 / (2 * 1.0)), label = "Gaussian with sigma = 1.0")
    ax.plot(x, 200000 * 1 / (np.sqrt(2 * np.pi) * 4.0) * np.exp(- x ** 2 / (2 * 4.0)), label = "Gaussian with sigma = 2.0")
    #ax.plot(x, 375000 * 1 / (np.sqrt(2 * np.pi) * 2.5) * np.exp(- x ** 2 / (2 * 2.5)), label = "Gaussian with mu = 2.5")
    ax.plot(x, 300000 * 1 / (np.sqrt(2 * np.pi) * 3.0) * np.exp(- x ** 2 / (2 * 3)), label = "Gaussian with sigma = sqrt(3)")

    std_dev = np.std(g_combined)
    print(f"std_dev: {std_dev}, squared: {std_dev * std_dev}, var: {np.var(g_combined)}")
    print(f"Sanity check on std dev: mu = 1: {np.std(g1)}, mu = 2: {np.std(g2)}")
    print(f"Sanity check on variance: mu = 1: {np.var(g1)}, mu = 2: {np.var(g2)}")

    ax.legend()
    fig.tight_layout()
    fig.savefig("gaussian.pdf")

if __name__ == "__main__":
    combine_gaussians()
