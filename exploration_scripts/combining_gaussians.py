#!/usr/bin/env python3

""" Tests for combining Gaussians together.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

from typing import Dict, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

from pachyderm import fit, histogram

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

def unnormalized_gaussian(x: Union[np.ndarray, float], mean: float, sigma: float, amplitude: float) -> Union[np.ndarray, float]:
    r""" Unnormalized gaussian.

    .. math::

        f = A * \exp{-\frac{(x - \mu)^{2}}{(2 * \sigma^{2}}}

    The width in the amplitude is implicitly excluded.

    Args:
        x: Value(s) where the gaussian should be evaluated.
        mean: Mean of the gaussian distribution.
        sigma: Width of the gaussian distribution.
        amplitude: Amplitude of the gaussian.
    Returns:
        Calculated gaussian value(s).
    """
    return amplitude * np.exp(-1.0 / 2.0 * np.square((x - mean) / sigma))

def new_combine_gaussians(label: str, widths: Sequence[float]) -> None:
    """ Use a new approach devised in July 2019. """
    # Validation
    if len(widths) != 3:
        raise ValueError(f"Must pass only three widths. Passed: {widths}")

    # Imagine with 3 EP angles + inclusive
    # First define the 3 EP angles
    gaussians = []
    #for width in range(1, 4):
    for width in widths:
        gaussians.append(np.random.normal(0, width, size = 1000000))
    n_trigs = [375, 300, 325]
    # Add the inclusive to the start of the number of trigs
    n_trigs.insert(0, np.sum(n_trigs))

    # Create the inclusive, and then histogram it
    inclusive = np.array([(i, val) for i, gaussian in enumerate(gaussians, 1) for val in gaussian])

    # Recall that [:, 0] contains 1-3, while [:, 1] contains the x values.
    h_inclusive, x_bin_edges, y_bin_edges = np.histogram2d(
        inclusive[:, 1], inclusive[:, 0], bins = [200, np.linspace(0.5, 3.5, 3 + 1)], range = [[-10, 10], [0.5, 3.5]]
    )

    print(f"h_inclusive.shape: {h_inclusive.shape}")
    #print(f"x_bin_edges: {x_bin_edges}")
    #print(f"y_bin_edges: {y_bin_edges}")
    print(f"inclusive[:, 0]: {inclusive[:, 0]}")

    fig, ax = plt.subplots(figsize = (8, 6))

    X, Y = np.meshgrid(x_bin_edges, y_bin_edges)
    ax.pcolormesh(X, Y, h_inclusive.T)
    ax.set_xlabel("x")
    ax.set_ylabel("EP orientation proxy")

    fig.tight_layout()
    fig.savefig("gaussian_2d_histogram.pdf")

    # Basically, the first two arguments are h.x and h.y
    binned_mean, _, _ = scipy.stats.binned_statistic(
        inclusive[:, 0], inclusive[:, 1], "std", bins = np.linspace(0.5, 3.5, 3 + 1)
    )
    print(f"Binned mean: {binned_mean}")
    inclusive_binned_mean, _, _ = scipy.stats.binned_statistic(inclusive[:, 0], inclusive[:, 1], "std", bins = 1)
    print(f"Inclusive binned mean: {inclusive_binned_mean}")

    hists = []
    # Inclusive
    hists.append(histogram.Histogram1D(
        bin_edges = x_bin_edges, y = np.sum(h_inclusive, axis = 1), errors_squared = np.sum(h_inclusive[:, 0]),
    ))
    # Width = 1
    hists.append(histogram.Histogram1D(
        bin_edges = x_bin_edges, y = h_inclusive[:, 0], errors_squared = np.copy(h_inclusive[:, 0]),
    ))
    # Width = 2
    hists.append(histogram.Histogram1D(
        bin_edges = x_bin_edges, y = h_inclusive[:, 1], errors_squared = np.copy(h_inclusive[:, 1]),
    ))
    # Width = 3
    hists.append(histogram.Histogram1D(
        bin_edges = x_bin_edges, y = h_inclusive[:, 2], errors_squared = np.copy(h_inclusive[:, 2]),
    ))

    # Scale by number of triggers.
    #for h, n_trig in zip(hists, n_trigs):
    #for i, n_trig in enumerate(n_trigs):
    #    print(f" pre scale {i}: {np.max(hists[i].y)}")
    #    print(f"scale by {1 / n_trig}")
    #    hists[i] *= 1.0 / n_trig
    #    #h = h * 1.0 / n_trig
    #    print(f"post scale {i}: {np.max(hists[i].y)}")

    # Quickly plot hists
    fig, ax = plt.subplots(figsize = (8, 6))
    for i, h in enumerate(hists):
        ax.errorbar(h.x, h.y, yerr = h.errors, marker = "o", linestyle = "", label = f"Data {i}")
    # Final adjustments
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"gaussian_data.pdf")

    # Fit to gaussians
    def scaled_gaussian(x: float, mean: float, sigma: float, amplitude: float) -> float:
        return amplitude * fit.gaussian(x = x, mean = mean, sigma = sigma)

    fig, ax = plt.subplots(figsize = (8, 6))
    gaussian_fit_results = []
    for i, h in enumerate(hists):
        cost_func = fit.BinnedLogLikelihood(f = scaled_gaussian, data = h)
        minuit_args: Dict[str, Union[float, Tuple[float, float]]] = {
            "mean": 0, "fix_mean": True,
            "sigma": 1.0, "error_sigma": 0.1, "limit_sigma": (0, 100),
            "amplitude": 100.0, "error_amplitude": 0.1,
        }
        fit_result, _ = fit.fit_with_minuit(
            cost_func = cost_func, minuit_args = minuit_args, log_likelihood = True, x = h.x
        )
        gaussian_fit_results.append(fit_result)

        # Plot for a sanity check
        plot_label: str
        if i > 0:
            plot_label = fr"$\sigma = {widths[i-1]:0.2f}$"
        else:
            plot_label = "inclusive"
        ax.errorbar(h.x, h.y, yerr = h.errors, marker = "o", linestyle = "", label = f"Data {plot_label}")
        ax.plot(h.x, scaled_gaussian(h.x, *list(fit_result.values_at_minimum.values())), label = f"Fit {plot_label}", zorder = 5)

    values_at_zero_from_hist = []
    for h in hists:
        values_at_zero_from_hist.append(h.y[h.find_bin(0.0)])
    values_at_zero_from_fits = []
    for fit_result in gaussian_fit_results:
        values_at_zero_from_fits.append(scaled_gaussian(0, *list(fit_result.values_at_minimum.values())))
    sum_of_last_3_from_hist = np.sum(values_at_zero_from_hist[1:])
    sum_of_last_3_from_fit = np.sum(values_at_zero_from_fits[1:])
    print(f"Values at 0 from hist: {values_at_zero_from_hist}, Sum of last 3: {sum_of_last_3_from_hist}, Diff: {_percent_diff(values_at_zero_from_hist[0], sum_of_last_3_from_hist):.3f}%")
    print(f"Values at 0 from fit: {values_at_zero_from_fits}, Sum of last 3: {sum_of_last_3_from_fit}, Diff: {_percent_diff(values_at_zero_from_fits[0], sum_of_last_3_from_fit):.3f}%")

    # TODO: Predict gaussian fit and plot

    # Final adjustments
    ax.legend()
    ax.set_title(f"{label} widths")
    fig.tight_layout()
    fig.savefig(f"gaussian_fit_{label}.pdf")

def _percent_diff(expected: float, predicted: float) -> float:
    """ Calculate the percent difference.

    Args:
        expected: The value that the predicted should reproduce.
        predicted: The value that attempts to match the expected.
    Returns:
        The percent difference between the two values.
    """
    return (predicted - expected) / expected * 100.

if __name__ == "__main__":
    print("---- New explorations ----")
    new_combine_gaussians(label = "broad", widths = [1, 2, 3])
    new_combine_gaussians(label = "narrow", widths = [0.95, 1.0, 1.05])
    print("---- Older explorations -----")
    combine_gaussians()

