#!/usr/bin/env python3

""" Plotting a specified tracking efficiency.

Tracking efficiencies are specified via ``AliAnalysisTaskEmcalJetHUtils``, so there
is a hard AliPhysics dependency.
"""

import coloredlogs
import IPython
import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from typing import Any, cast, Callable, Dict, List, Tuple

from pachyderm import histogram
from pachyderm.utils import epsilon

from jet_hadron.base import analysis_objects
from jet_hadron.base import labels
from jet_hadron.base import params
from jet_hadron.plot import base as plot_base

logger = logging.getLogger(__name__)

# Types
T_PublicUtils = Any

def generate_parameters(system: params.CollisionSystem) -> Tuple[np.ndarray, np.ndarray, int, Dict[int, params.SelectedRange]]:
    """ Generate the analysis parameters.

    This can be called multiple times if necessary to retrieve the parameters easily in any function.

    Args:
        system: Collision system.
    Returns:
        (pt_values, eta_values, n_cent_bins, centrality_ranges): Pt values where the efficiency should be evaluated,
            eta values where the efficiency should be evaluated, number of centrality bins, map from centrality bin
            number to centrality bin ranges.
    """
    pt_values = np.linspace(0.05, 9.95, 100)
    eta_values = np.linspace(-0.85, 0.85, 35)
    n_cent_bins = 4 if system != params.CollisionSystem.pp else 1
    centrality_ranges = {
        0: params.SelectedRange(0, 10),
        1: params.SelectedRange(10, 30),
        2: params.SelectedRange(30, 50),
        3: params.SelectedRange(50, 90),
    }

    return pt_values, eta_values, n_cent_bins, centrality_ranges

def setup_AliPhysics(period: str) -> Tuple[Callable[..., float], Any, T_PublicUtils]:
    """ Retrieve the efficiency function, period and internal efficiency functions and params from AliPhysics.

    Re-factored to a separate function so we can isolate the dependency.

    Args:
        period: Name of the data period.
    Returns:
        (efficiency_function, efficiency_period, T_PublicUtils): The general efficiency function, and
        the period enum value used in calling that function, and a derived class to make the internal
        efficiency functions and parameters public for further testing.
    """
    import ROOT
    # The tasks are in the PWGJE::EMCALJetTasks namespace, so we first retrieve that namespace for convenience.
    user_namespace = ROOT.PWGJE.EMCALJetTasks
    # First retrieve the function
    efficiency_function = user_namespace.AliAnalysisTaskEmcalJetHUtils.DetermineTrackingEfficiency
    # Then the period by string
    efficiency_period = getattr(user_namespace.AliAnalysisTaskEmcalJetHUtils, "k" + period)

    # Provide full access to the rest of the attributes by inheriting and making the members public.
    # The winding route to get there is due to namespace issues in PyROOT.
    code = """
    // Tell ROOT where to find AliRoot headers
    R__ADD_INCLUDE_PATH($ALICE_ROOT)
    // Tell ROOT where to find AliPhysics headers
    R__ADD_INCLUDE_PATH($ALICE_PHYSICS)
    #include <AliAnalysisTaskEmcalJetHUtils.h>
    class JetHUtilsPublic : public PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalJetHUtils {
      public:
        // Can't just use (appears to be because of a PyROOT limitation...):
        //using PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalJetHUtils::LHC11aTrackingEfficiency;
        // So we define the function by hand.
        static double LHC11aTrackingEfficiency(const double trackPt, const double trackEta, const int centralityBin,
                                               const std::string& taskName)
        {
            return PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalJetHUtils::LHC11aTrackingEfficiency(trackPt, trackEta, centralityBin, taskName);
        }
        // LHC11h
        static double LHC11hTrackingEfficiency(const double trackPt, const double trackEta, const int centralityBin,
                                               const std::string& taskName)
        {
            return PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalJetHUtils::LHC11hTrackingEfficiency(trackPt, trackEta, centralityBin, taskName);
        }
        // LHC15o
        static double LHC15oTrackingEfficiency(const double trackPt, const double trackEta, const int centralityBin,
                                               const std::string& taskName)
        {
            return PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalJetHUtils::LHC15oTrackingEfficiency(trackPt, trackEta, centralityBin, taskName);
        }
        static double LHC15oPtEfficiency(const double trackPt, const int centBin)
        {
            std::map<int, const double*> centMap = {
                { 0, PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalJetHUtils::LHC15oParam_0_10_pt },
                { 1, PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalJetHUtils::LHC15oParam_10_30_pt },
                { 2, PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalJetHUtils::LHC15oParam_30_50_pt },
                { 3, PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalJetHUtils::LHC15oParam_50_90_pt },
            };
            return (trackPt <= 3.5) * PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalJetHUtils::LHC15oLowPtEfficiency(trackPt, centMap[centBin], 0) +
                   (trackPt > 3.5) * PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalJetHUtils::LHC15oHighPtEfficiency(trackPt, centMap[centBin], 5);
        }
        static double LHC15oEtaEfficiency(const double trackEta, const int centBin)
        {
            std::map<int, const double*> centMap = {
                { 0, PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalJetHUtils::LHC15oParam_0_10_eta },
                { 1, PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalJetHUtils::LHC15oParam_10_30_eta },
                { 2, PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalJetHUtils::LHC15oParam_30_50_eta },
                { 3, PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalJetHUtils::LHC15oParam_50_90_eta },
            };
            return  (trackEta <= -0.04) * PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalJetHUtils::LHC15oEtaEfficiency(trackEta, centMap[centBin], 0) +
                    (trackEta > -0.04) * PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalJetHUtils::LHC15oEtaEfficiency(trackEta, centMap[centBin], 6);
        }
    };
    """
    # Create the class
    ROOT.gInterpreter.ProcessLine(code)
    PublicUtils = ROOT.JetHUtilsPublic

    return cast(Callable[..., float], efficiency_function), efficiency_period, PublicUtils

def get_15o_eta_max_efficiency(PublicUtils: T_PublicUtils, n_cent_bins: int) -> List[List[float]]:
    """ Get the max of the eta efficiencies for properly normalizing the functions.

    Takes advantage of the public interface to the ``AliAnalysisTaskEmcalJetHUtils`` defiend in the setup.

    Args:
        PublicUtils: Jet-H public utils class.
        n_cent_bins: Number of centrality bins
    Returns:
        Max value in the eta efficiency.
    """
    efficiency_function = PublicUtils.LHC15oEtaEfficiency
    max_values = []
    for centrality_bin in range(n_cent_bins):
        # Then determine the max eta values.
        values_left = []
        values_right = []
        # Record the values with high granularity
        for eta in np.linspace(-0.9, 0, 1000):
            values_left.append(efficiency_function(eta, centrality_bin))
        for eta in np.linspace(-0.06, 0.9, 1000):
            values_right.append(efficiency_function(eta, centrality_bin))

        # Store the result
        max_values.append([np.max(values_left), np.max(values_right)])

    return max_values

def jetH_task_efficiency_for_comparison(period: str, system: params.CollisionSystem,
                                        pt_values: np.ndarray, eta_values: np.ndarray) -> np.ndarray:
    """ Provide the efficiencies from the Jet-hadron correlations task for comparison.

    Requires a hack of the jet-hadron correlations task to make ``fCentBin`` and ``fCurrentRunNumber``.
    One way to do so is:

    .. code-block:: cpp

        void SetCentralityBin(int centBin) { fCentBin = centBin; }
        void SetCurrentRun(int run) { fCurrentRunNumber = run; }

    The comparison only works for LHC11h or LHC11a.

    Args:
        period: Period for calculating the efficiencies.
        system: Collision system.
        pt_values: pT values where the efficiency should be evaluated.
        eta_values: eta values where the efficiency should be evaluated.
    Returns:
        Calculated efficiencies for the same range as the JetHUtils implementation.
    """
    # Setup
    n_cent_bins = 4 if system != params.CollisionSystem.pp else 1
    import ROOT
    jetHTask = ROOT.PWGJE.EMCALJetTasks.AliAnalysisTaskEmcalJetHCorrelations()

    # Corresponds to good runs
    jetHTask.SetCurrentRun(167902)
    # Enable efficiency correction
    efficiency = jetHTask.kEffPP if period == "LHC11a" else jetHTask.kEffAutomaticConfiguration
    system_enum = ROOT.AliAnalysisTaskEmcal.kpp if system == params.CollisionSystem.pp else ROOT.AliAnalysisTaskEmcal.kAA
    jetHTask.SetSingleTrackEfficiencyType(efficiency)

    # Loop over values
    efficiencies = np.zeros(shape = (n_cent_bins, len(pt_values), len(eta_values)))
    for centrality_bin in range(n_cent_bins):
        jetHTask.SetCentralityBin(centrality_bin)
        for pt_index, pt in enumerate(pt_values):
            for eta_index, eta in enumerate(eta_values):
                efficiencies[centrality_bin, pt_index, eta_index] = jetHTask.EffCorrection(
                    eta, pt, system_enum
                )

    return efficiencies

def check_for_accidentally_repeated_parameters(n_cent_bins: int, efficiencies: np.ndarray) -> bool:
    """ Check for accidnetally repeated parameters.

    Namely, it checks whether the efficiencies for one centrality bin is the same as another.
    If they are the same, it indicates that parameters are accidentally repeated.

    Args:
        n_cent_bins: Number of centrality bins.
        efficiencies: Calculated efficiencies.
    Returns:
        True if there are duplicated parameters
    """
    for centrality_bin in range(n_cent_bins):
        for other_centrality_bin in range(n_cent_bins):
            # True if the indices are the same, false if not
            comparison_value = (centrality_bin == other_centrality_bin)
            assert np.allclose(efficiencies[centrality_bin], efficiencies[other_centrality_bin]) is comparison_value

    return False

def calculate_efficiencies(n_cent_bins: int, pt_values: np.ndarray, eta_values: np.ndarray, efficiency_period: Any, efficiency_function: Callable[..., float]) -> np.ndarray:
    """ Caluclate the efficiency given the parameters.

    Args:
        n_cent_bins: Number of centrality bins.
        pt_values: Pt values to evaluate.
        eta_values: Eta values to evaluate.
        efficiency_period: Enum value correspoonding to the period.
        efficiency_function: The actual efficiency function.
    Returns:
        Efficiency evaluated at all centrality bins, pt values, and eta values, indexed as
            [centrality_bin, pt_value, eta_value].
    """
    # Centrality, pt, eta
    efficiencies = np.zeros(shape = (n_cent_bins, len(pt_values), len(eta_values)))
    for centrality_bin in range(n_cent_bins):
        for pt_index, pt in enumerate(pt_values):
            for eta_index, eta in enumerate(eta_values):
                efficiencies[centrality_bin, pt_index, eta_index] = efficiency_function(
                    pt, eta, centrality_bin, efficiency_period, "task_name"
                )

    return efficiencies

def efficiency_properties(n_cent_bins: int, efficiencies: np.ndarray, PublicUtils: T_PublicUtils) -> bool:
    """ Determine and check a variety of efficiency properties.

    Args:
        n_cent_bins: Number of centrality bins.
        efficiencies: Calculated efficiencies.
        PublicUtils: Jet-H public utils class.
    Returns:
        True if the properties were calculated and checked.
    """
    # Determine maximum
    for centrality_bin in range(n_cent_bins):
        m = np.max(efficiencies[centrality_bin])
        logger.debug(f"centrality bin: {centrality_bin}, max: {m}")

    # Find maxima for eta normalization
    if period == "LHC15o":
        logger.info("Checking max eta efficiency normalization")
        max_eta_efficiencies = get_15o_eta_max_efficiency(
            PublicUtils = PublicUtils, n_cent_bins = n_cent_bins,
        )

        for centrality_bin, (left, right) in enumerate(max_eta_efficiencies):
            if np.isclose(left, 1.0, atol = 1e-4) or np.isclose(right, 1.0, atol = 1e-4):
                logger.info(f"Eta efficiency for LHC15o centrality bin {centrality_bin} is properly normalized.")
            else:
                logger.warning(
                    f"Eta efficiency for LHC15o centrality bin {centrality_bin} appears not to be normalized."
                    "Check if this is expected!"
                )
        logger.info(f"Max eta efficiencies: {max_eta_efficiencies}")

    # Check that the parameters are all set correctly.
    logger.info("Checking for accidentally repeated parameters")
    duplicated = check_for_accidentally_repeated_parameters(n_cent_bins = n_cent_bins, efficiencies = efficiencies)
    if duplicated:
        raise ValueError("Some parameters appear to be accidentally repeated.")
    logger.info("No repeated parameters!")

    return True

def plot_tracking_efficiency_parametrization(efficiencies: np.ndarray, n_cent_bins: int, period: str,
                                             system: params.CollisionSystem, output_info: analysis_objects.PlottingOutputWrapper) -> None:
    """ Plot the given tracking efficiencies parametrization.

    Args:
        efficiencies: Calculated tracking efficiencies.
        n_cent_bins: Number of centrality bins.
        period: Data taking period.
        system: Collision system.
        output_info: Output info for saving figures.
    Returns:
        None.
    """
    # Get the parameters
    pt_values, eta_values, n_cent_bins, centrality_ranges = generate_parameters(system)

    logger.debug("Plotting efficiencies")
    for centrality_bin in range(n_cent_bins):
        fig, ax = plt.subplots(figsize = (8, 6))
        im = ax.imshow(
            efficiencies[centrality_bin].T,
            extent = [np.min(pt_values), np.max(pt_values), np.min(eta_values), np.max(eta_values)],
            interpolation = "nearest", aspect = "auto", origin = "lower",
            norm = matplotlib.colors.Normalize(vmin = 0.5, vmax = 1), cmap = "viridis",
        )

        # Add the colorbar
        fig.colorbar(im, ax = ax)

        # Labels
        ax.set_xlabel(fr"${labels.pt_display_label()}\:({labels.momentum_units_label_gev()})$")
        ax.set_ylabel(r"$\varphi$")
        title = f"{period} tracking efficiency parametrization"
        if system != params.CollisionSystem.pp:
            centrality_range = centrality_ranges[centrality_bin]
            title += rf", ${centrality_range.min} \textendash {centrality_range.max}\%$"
        ax.set_title(title, size = 16)

        # Final adjustments
        fig.tight_layout()
        name = f"efficiency_{period}"
        if system != params.CollisionSystem.pp:
            centrality_range = centrality_ranges[centrality_bin]
            name += f"_centrality_parametrization_{centrality_range.min}_{centrality_range.max}"
        plot_base.save_plot(output_info, fig, name)

        # Cleanup
        plt.close(fig)

def comparison_to_data() -> None:
    ...

def characterize_tracking_efficiency(period: str, system: params.CollisionSystem) -> None:
    """ Characterize the tracking efficiency.

    Args:
        period: Period for calculating the efficiencies.
        system: Collision system.
    Returns:
        Calculated efficiencies for the same range as the JetHUtils implementation.
    """
    # Setup
    logger.warning(f"Plotting efficiencies for {period}, system {system}")
    efficiency_function, efficiency_period, PublicUtils = setup_AliPhysics(period)
    # Setting up evaluation ranges.
    pt_values, eta_values, n_cent_bins, centrality_ranges = generate_parameters(system)
    # Plotting output location
    output_info = analysis_objects.PlottingOutputWrapper(
        output_prefix = f"output/{system}/{period}/trackingEfficiency",
        printing_extensions = ["pdf"],
    )

    # Caluclate the efficiency.
    efficiencies = calculate_efficiencies(
        n_cent_bins = n_cent_bins, pt_values = pt_values, eta_values = eta_values,
        efficiency_period = efficiency_period, efficiency_function = efficiency_function,
    )

    # Calculate and check efficiency properties.
    result = efficiency_properties(
        n_cent_bins = n_cent_bins, efficiencies = efficiencies, PublicUtils = PublicUtils
    )
    if not result:
        raise RuntimeError("Failed to calculate all efficiency properties. Check the logs!")

    # Retrieve efficiency data
    #efficiencies_data = retrieve_efficiencies_data()

    # Comparison to previous task
    if period in ["LHC11a", "LHC11h"]:
        try:
            comparison_efficiencies = jetH_task_efficiency_for_comparison(period, system, pt_values, eta_values)
            np.testing.assert_allclose(efficiencies, comparison_efficiencies)
            logger.info("Efficiencies agree!")
        except AttributeError as e:
            logger.warning(f"{e.args[0]}. Skipping!")
    else:
        logger.info("Skipping efficiencies comparison because no other implementation exists.")

    # Now, plot the result.
    plot_tracking_efficiency_parametrization(efficiencies, n_cent_bins, period, system, output_info)

    # Plot residuals if available.
    if period == "LHC15o":
        logger.info("Calculating and plotting residuals")
        # Setup
        residuals: np.ndarray = None
        #residuals = np.zeros(shape = efficiencies.shape)
        hists = histogram.get_histograms_in_list(
            filename = "trains/PbPbMC/55/AnalysisResults.root",
            list_name = "AliAnalysisTaskPWGJEQA_tracks_caloClusters_emcalCells_histos"
        )
        matched_sparse = hists["tracks_Matched"]
        generator_sparse = hists["tracks_PhysPrim"]

        # Project
        for centrality_bin, centrality_range in centrality_ranges.items():
            # Select in centrality
            matched_sparse.GetAxis(0).SetRangeUser(centrality_range.min + epsilon, centrality_range.max - epsilon)
            generator_sparse.GetAxis(0).SetRangeUser(centrality_range.min + epsilon, centrality_range.max - epsilon)
            # Restrict pt range to < 10 GeV.
            matched_sparse.GetAxis(1).SetRangeUser(0, 10)
            # (pt_gen, eta_gen) - order is reversed because 2D API is backwards...
            hPtGenMatched = matched_sparse.Projection(2, 1)
            hPtGenMatched.SetName(f"hPtGenMatched_cent_{centrality_bin}")
            # (pt_gen, eta_gen, findable)
            hPtGen2D = generator_sparse.Projection(1, 2, 4)
            hPtGen2D.SetName(f"hPtGen2D_cent_{centrality_bin}")
            # Restrict pt range to < 10 GeV.
            hPtGen2D.GetXaxis().SetRangeUser(0, 10)
            # Select only findable particles and use that efficiency.
            hPtGen2D.GetZaxis().SetRange(2, 2)
            hPtGenFindable = hPtGen2D.Project3D("yx")
            logger.debug(f"hPtGenMatched: {hPtGenMatched}, hPtGenFindable: {hPtGenFindable}")

            efficiency_hist = hPtGenFindable.Clone()
            efficiency_hist.Divide(hPtGenMatched, hPtGenFindable, 1.0, 1.0, "B")

            # Get 1D efficiency for a test.
            # NOTE: We can't just project from the 2D efficiency. Integrating over eta will get the wrong
            #       wrong values. Here, we project before we divide to get the right answer.
            hPtGenMatched1D = matched_sparse.Projection(1)
            hPtGenMatched1D.SetName(f"hPtGenMatched_cent_{centrality_bin}")
            hPtGen1D = generator_sparse.Projection(4, 1)
            hPtGen1D.SetName(f"hPtGetn1D_cent_{centrality_bin}")
            # Restrict pt range to < 10 GeV.
            hPtGen1D.GetXaxis().SetRangeUser(0, 10)
            # Select only findable particles and use that efficiency.
            hPtGen1D.GetYaxis().SetRange(2, 2)
            hPtGenFindable1D = hPtGen1D.ProjectionX()
            logger.debug(f"hPtGenMatched1D: {hPtGenMatched1D}, hPtGenFindable1D: {hPtGenFindable1D}")

            efficiency_1D = hPtGenFindable1D.Clone()
            efficiency_1D.Divide(hPtGenMatched1D, hPtGenFindable1D, 1.0, 1.0, "B")

            if residuals is None:
                residuals = np.zeros(
                    shape = (n_cent_bins, efficiency_hist.GetXaxis().GetNbins(), efficiency_hist.GetYaxis().GetNbins())
                )
            logger.debug(f"residuals shape: {residuals.shape}")

            # Calculate the residuals.
            pts = []
            etas = []
            for pt_index, x in enumerate(range(1, efficiency_hist.GetXaxis().GetNbins() + 1)):
                pt = efficiency_hist.GetXaxis().GetBinCenter(x)
                pts.append(pt)
                for eta_index, y in enumerate(range(1, efficiency_hist.GetYaxis().GetNbins() + 1)):
                    eta = efficiency_hist.GetYaxis().GetBinCenter(y)
                    etas.append(eta)
                    efficiency_at_value = efficiency_function(pt, eta, centrality_bin, efficiency_period, "task_name")
                    if np.abs(efficiency_hist.GetBinContent(x, y)) < epsilon:
                        value = np.nan
                    else:
                        value = (efficiency_hist.GetBinContent(x, y) - efficiency_at_value) / efficiency_at_value * 100.
                    residuals[centrality_bin, pt_index, eta_index] = value

            # Check max values
            logger.debug(f"min efficiency_hist: {efficiency_hist.GetMinimum()}, max efficiency_hist: {efficiency_hist.GetMaximum()}")
            logger.debug(f"min residual: {np.nanmin(residuals[centrality_bin])}, max residual: {np.nanmax(residuals[centrality_bin])}")
            logger.debug(f"mean: {np.nanmean(residuals[centrality_bin])}")

            # Plot them.
            fig, ax = plt.subplots(figsize = (8, 6))
            im = ax.imshow(
                residuals[centrality_bin].T, extent = [np.nanmin(pts), np.nanmax(pts), np.nanmin(etas), np.nanmax(etas)],
                interpolation = "nearest", aspect = "auto", origin = "lower",
                norm = matplotlib.colors.Normalize(
                    #vmin = np.nanmin(residuals[centrality_bin]), vmax = np.nanmax(residuals[centrality_bin])
                    vmin = -40, vmax = 40
                ),
                cmap = "RdBu",
            )

            # Add the colorbar
            color_bar = fig.colorbar(im, ax = ax)
            color_bar.set_label(r"(data - fit)/fit (\%)")

            # Labels
            ax.set_xlabel(fr"${labels.pt_display_label()}\:({labels.momentum_units_label_gev()})$")
            ax.set_ylabel(r"$\varphi$")
            title = f"{period} tracking efficiency residuals"
            if system != params.CollisionSystem.pp:
                centrality_range = centrality_ranges[centrality_bin]
                title += rf", ${centrality_range.min} \textendash {centrality_range.max}\%$"
            ax.set_title(title, size = 16)

            # Final adjustments
            fig.tight_layout()
            name = f"efficiency_residuals_{period}"
            if system != params.CollisionSystem.pp:
                centrality_range = centrality_ranges[centrality_bin]
                name += f"_centrality_{centrality_range.min}_{centrality_range.max}"
            fig.savefig(f"{name}.pdf")

            # Cleanup
            plt.close(fig)

            # Plot the efficiency itself
            X, Y, efficiency_data = histogram.get_array_from_hist2D(hist = efficiency_hist)
            logger.debug(f"efficiency data min: {np.nanmin(efficiency_data)}, max: {np.nanmax(efficiency_data)}")
            fig, ax = plt.subplots(figsize = (8, 6))
            im = ax.imshow(
                efficiency_data.T, extent = [np.nanmin(X), np.nanmax(X), np.nanmin(Y), np.nanmax(Y)],
                interpolation = "nearest", aspect = "auto", origin = "lower",
                norm = matplotlib.colors.Normalize(
                    vmin = np.nanmin(efficiency_data), vmax = np.nanmax(efficiency_data)
                    #vmin = 0.5, vmax = 1,
                ),
                cmap = "viridis",
            )

            # Add the colorbar
            color_bar = fig.colorbar(im, ax = ax)
            color_bar.set_label("Efficiency")

            # Labels
            ax.set_xlabel(fr"${labels.pt_display_label()}\:({labels.momentum_units_label_gev()})$")
            ax.set_ylabel(r"$\varphi$")
            title = f"{period} tracking efficiency data"
            if system != params.CollisionSystem.pp:
                centrality_range = centrality_ranges[centrality_bin]
                title += rf", ${centrality_range.min} \textendash {centrality_range.max}\%$"
            ax.set_title(title, size = 16)

            # Final adjustments
            fig.tight_layout()
            name = f"efficiency_{period}"
            if system != params.CollisionSystem.pp:
                centrality_range = centrality_ranges[centrality_bin]
                name += f"_centrality_{centrality_range.min}_{centrality_range.max}"
            fig.savefig(f"{name}.pdf")

            # Cleanup
            plt.close(fig)

            # 1D efficiency as a function of pt
            logger.debug(f"max efficiency_1D: {efficiency_1D.GetMaximum()}")
            h = histogram.Histogram1D.from_existing_hist(efficiency_1D)
            fig, ax = plt.subplots(figsize = (8, 6))
            ax.errorbar(
                h.x, h.y, yerr = h.errors,
                label = "${labels.pt_display_label()}$",
                color = "black", marker = ".", linestyle = "",
            )

            # Ensure that it's on a consistent axis
            ax.set_ylim(0, 1)

            # Labels
            ax.set_xlabel(fr"${labels.pt_display_label()}\:({labels.momentum_units_label_gev()})$")
            ax.set_ylabel(r"Efficiency")
            title = f"{period} ${labels.pt_display_label()}$ tracking efficiency"
            if system != params.CollisionSystem.pp:
                centrality_range = centrality_ranges[centrality_bin]
                title += rf", ${centrality_range.min} \textendash {centrality_range.max}\%$"
            ax.set_title(title, size = 16)

            # Final adjustments
            fig.tight_layout()
            name = f"efficiency_pt_{period}"
            if system != params.CollisionSystem.pp:
                centrality_range = centrality_ranges[centrality_bin]
                name += f"_centrality_{centrality_range.min}_{centrality_range.max}"
            fig.savefig(f"{name}.pdf")

if __name__ == "__main__":
    # Basic setup
    coloredlogs.install(
        level = logging.DEBUG,
        fmt = "%(asctime)s %(name)s:%(lineno)d %(levelname)s %(message)s"
    )
    # Quiet down the matplotlib logging
    logging.getLogger("matplotlib").setLevel(logging.INFO)

    # Run for each period
    #for period, system in [("LHC11a", params.CollisionSystem.pp),
    #                       ("LHC11h", params.CollisionSystem.PbPb),
    #                       ("LHC15o", params.CollisionSystem.PbPb)]:
    for period, system in [("LHC15o", params.CollisionSystem.PbPb)]:
        characterize_tracking_efficiency(period, system)

