#!/usr/bin/env python3

""" Tests for combining Gaussians together.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import coloredlogs
import logging
from functools import reduce
from typing import Dict, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

from pachyderm import fit, histogram
from pachyderm.utils import epsilon

logger = logging.getLogger(__name__)

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
    logger.debug(f"std_dev: {std_dev}, squared: {std_dev * std_dev}, var: {np.var(g_combined)}")
    logger.debug(f"Sanity check on std dev: mu = 1: {np.std(g1)}, mu = 2: {np.std(g2)}")
    logger.debug(f"Sanity check on variance: mu = 1: {np.var(g1)}, mu = 2: {np.var(g2)}")

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

def restrict_hist_range(hist: histogram.Histogram1D, min_x: float, max_x: float) -> histogram.Histogram1D:
    """ Restrict the histogram to only be within a provided x range.

    Args:
        hist: Histogram to be restricted.
        min_x: Minimum x value.
        max_x: Maximum x value.
    Returns:
        Restricted histogram.
    """
    selected_range = slice(hist.find_bin(min_x + epsilon), hist.find_bin(max_x - epsilon) + 1)
    bin_edges_selected_range = ((hist.bin_edges >= min_x) & (hist.bin_edges <= max_x))
    return histogram.Histogram1D(
        bin_edges = hist.bin_edges[bin_edges_selected_range], y = hist.y[selected_range],
        errors_squared = hist.errors_squared[selected_range]
    )

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

    logger.debug(f"h_inclusive.shape: {h_inclusive.shape}")
    #logger.debug(f"x_bin_edges: {x_bin_edges}")
    #logger.debug(f"y_bin_edges: {y_bin_edges}")
    logger.debug(f"inclusive[:, 0]: {inclusive[:, 0]}")

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
    logger.debug(f"Binned mean: {binned_mean}")
    inclusive_binned_mean, _, _ = scipy.stats.binned_statistic(inclusive[:, 0], inclusive[:, 1], "std", bins = 1)
    logger.debug(f"Inclusive binned mean: {inclusive_binned_mean}")

    hists = []
    # Inclusive
    hists.append(histogram.Histogram1D(
        bin_edges = x_bin_edges, y = np.sum(h_inclusive, axis = 1), errors_squared = np.sum(h_inclusive, axis = 1),
    ))
    # Width = 1
    hists.append(histogram.Histogram1D(
        bin_edges = x_bin_edges, y = h_inclusive[:, 0], errors_squared = h_inclusive[:, 0],
    ))
    # Width = 2
    hists.append(histogram.Histogram1D(
        bin_edges = x_bin_edges, y = h_inclusive[:, 1], errors_squared = h_inclusive[:, 1],
    ))
    # Width = 3
    hists.append(histogram.Histogram1D(
        bin_edges = x_bin_edges, y = h_inclusive[:, 2], errors_squared = h_inclusive[:, 2],
    ))

    # Scale by number of triggers.
    #for h, n_trig in zip(hists, n_trigs):
    #for i, n_trig in enumerate(n_trigs):
    #    logger.debug(f" pre scale {i}: {np.max(hists[i].y)}")
    #    logger.debug(f"scale by {1 / n_trig}")
    #    hists[i] *= 1.0 / n_trig
    #    #h = h * 1.0 / n_trig
    #    logger.debug(f"post scale {i}: {np.max(hists[i].y)}")

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
        # First try just -3 to 3
        #cost_func = fit.BinnedLogLikelihood(f = scaled_gaussian, data = restrict_hist_range(h, -3, 3))
        cost_func = fit.BinnedLogLikelihood(f = unnormalized_gaussian, data = restrict_hist_range(h, -3, 3))
        minuit_args: Dict[str, Union[float, Tuple[float, float]]] = {
            "mean": 0, "fix_mean": True,
            "sigma": 1.0, "error_sigma": 0.1, "limit_sigma": (0, 10),
            "amplitude": 100.0, "error_amplitude": 0.1,
        }
        fit_result, _ = fit.fit_with_minuit(
            cost_func = cost_func, minuit_args = minuit_args, x = h.x
        )
        gaussian_fit_results.append(fit_result)

        # Plot for a sanity check
        plot_label: str
        if i > 0:
            plot_label = fr"$\sigma = {widths[i-1]:0.2f}$"
        else:
            plot_label = "inclusive"
        ax.errorbar(h.x, h.y, yerr = h.errors, marker = "o", linestyle = "", label = f"Data {plot_label}")
        #ax.plot(h.x, scaled_gaussian(h.x, *list(fit_result.values_at_minimum.values())), label = f"Fit {plot_label}", zorder = 5)
        ax.plot(h.x, unnormalized_gaussian(h.x, *list(fit_result.values_at_minimum.values())), label = f"Fit {plot_label}", zorder = 5)

    values_at_zero_from_hist = []
    for h in hists:
        values_at_zero_from_hist.append(h.y[h.find_bin(0.0)])
    values_at_zero_from_fits = []
    for fit_result in gaussian_fit_results:
        #values_at_zero_from_fits.append(scaled_gaussian(0, *list(fit_result.values_at_minimum.values())))
        values_at_zero_from_fits.append(unnormalized_gaussian(0, *list(fit_result.values_at_minimum.values())))
    sum_of_last_3_from_hist = np.sum(values_at_zero_from_hist[1:])
    sum_of_last_3_from_fit = np.sum(values_at_zero_from_fits[1:])
    logger.debug(f"Values at 0 from hist: {values_at_zero_from_hist}, Sum of last 3: {sum_of_last_3_from_hist}, Diff: {_percent_diff(values_at_zero_from_hist[0], sum_of_last_3_from_hist):.3f}%")
    logger.debug(f"Values at 0 from fit: {values_at_zero_from_fits}, Sum of last 3: {sum_of_last_3_from_fit}, Diff: {_percent_diff(values_at_zero_from_fits[0], sum_of_last_3_from_fit):.3f}%")

    # Predict gaussian fit and plot based on previous fit
    calculate_variances(hists, gaussian_fit_results)

    import IPython
    IPython.embed()

    # Final adjustments
    ax.legend()
    ax.set_title(f"{label} widths")
    fig.tight_layout()
    fig.savefig(f"gaussian_fit_{label}.pdf")

def calculate_variances(hists: Sequence[histogram.Histogram1D], gaussian_fit_results: Sequence[fit.FitResult]) -> None:
    """ Calculate variances in a variety of routes. """
    def f(x: np.ndarray, amplitudes: Sequence[float], sigmas: Sequence[float]) -> np.ndarray:
        return reduce(
            lambda x, y: x + y,
            [A / sigma * np.exp(-x ** 2 / (2 * sigma ** 2)) for A, sigma in zip(amplitudes, sigmas)]
        )

    amplitudes_list = []
    sigmas_list = []
    x = hists[0].x
    for fit_result in gaussian_fit_results[1:]:
        #amplitudes_list.append(scaled_gaussian(0, *list(fit_result.values_at_minimum.values())))
        amplitudes_list.append(unnormalized_gaussian(0, *list(fit_result.values_at_minimum.values())))
        sigmas_list.append(fit_result.values_at_minimum["sigma"])
    amplitudes = np.array(amplitudes_list)
    sigmas = np.array(sigmas_list)
    numerator = np.sum(amplitudes / sigmas)
    denominator = f(x, amplitudes, sigmas)
    sigma_inclusive = np.sqrt(x ** 2 / (2 * np.log1p(numerator / denominator)))

    A_inclusive_at_0 = np.sum(amplitudes)
    A_inclusive = np.sum(amplitudes / sigmas) * sigma_inclusive
    logger.debug(f"A_inclusive_at_0: {A_inclusive_at_0}, A_inclusive: {A_inclusive}")
    logger.debug(f"sigma_inclusive: {sigma_inclusive}")

    # Check variance of the x weighted by f(x) = A_i / sqrt(2 * np.pi * sigma_i ** 2) * np.exp(-x ** 2 / (2 * sigma_i **2))
    mean = np.average(x, weights = f(x, amplitudes, sigmas))
    variance = np.average((x - mean) ** 2, weights = f(x, amplitudes, sigmas))
    # Check out varaince for comparison
    stats = histogram.calculate_binned_stats(hists[0].bin_edges, hists[0].y, hists[0].errors_squared)
    variance_hist = histogram.binned_variance(stats)
    logger.debug(f"Calculated mean: {mean}, variance: {variance}, sqrt: {np.sqrt(variance)}")
    logger.debug(f"Calculated variance: {variance}, from histogram: {variance_hist}, percent difference: {_percent_diff(variance_hist, variance):.3f}%")
    logger.debug("Conclusion: The variance difference it too large when the widths are too different :-(")

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
    # Basic setup
    coloredlogs.install(
        level = logging.DEBUG,
        fmt = "%(asctime)s %(name)s:%(lineno)d %(levelname)s %(message)s"
    )
    # Quiet down the matplotlib logging
    logging.getLogger("matplotlib").setLevel(logging.INFO)

    logger.debug("---- New explorations ----")
    new_combine_gaussians(label = "narrow", widths = [0.95, 1.0, 1.05])
    new_combine_gaussians(label = "broad", widths = [1, 2, 3])
    logger.debug("---- Older explorations -----")
    combine_gaussians()

