#!/usr/bin/env python3

""" Plotting a specified tracking efficiency.

Tracking efficiencies are specified via ``AliAnalysisTaskEmcalJetHUtils``, so there
is a hard AliPhysics dependency.
"""

import coloredlogs
import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from typing import Any, cast, Callable, Dict, List, Tuple

# NOTE: This is out of the expected order, but it must be here to prevent ROOT from stealing the command
#       line options
from jet_hadron.base.typing_helpers import Hist

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
    pt_values = np.linspace(0.15, 9.95, 100 - 1)
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
            return PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalJetHUtils::LHC15oPtEfficiency(trackPt, centMap[centBin]);
        }
        static double LHC15oEtaEfficiency(const double trackEta, const int centBin)
        {
            std::map<int, const double*> centMap = {
                { 0, PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalJetHUtils::LHC15oParam_0_10_eta },
                { 1, PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalJetHUtils::LHC15oParam_10_30_eta },
                { 2, PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalJetHUtils::LHC15oParam_30_50_eta },
                { 3, PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalJetHUtils::LHC15oParam_50_90_eta },
            };
            return PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalJetHUtils::LHC15oEtaEfficiency(trackEta, centMap[centBin]);
        }
        static double LHC15oEtaEfficiencyNormalization(const double centralityBin)
        {
            std::map<int, const double*> centMap = {
                { 0, PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalJetHUtils::LHC15oParam_0_10_eta },
                { 1, PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalJetHUtils::LHC15oParam_10_30_eta },
                { 2, PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalJetHUtils::LHC15oParam_30_50_eta },
                { 3, PWGJE::EMCALJetTasks::AliAnalysisTaskEmcalJetHUtils::LHC15oParam_50_90_eta },
            };
            return centMap[centralityBin][12];
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
                    " Check if this is expected!"
                )
        logger.info(f"Max eta efficiencies: {max_eta_efficiencies}")

    # Check that the parameters are all set correctly.
    logger.info("Checking for accidentally repeated parameters")
    duplicated = check_for_accidentally_repeated_parameters(n_cent_bins = n_cent_bins, efficiencies = efficiencies)
    if duplicated:
        raise ValueError("Some parameters appear to be accidentally repeated.")
    logger.info("No repeated parameters!")

    return True

def plot_tracking_efficiency_parametrization(efficiency: np.ndarray, centrality_range: params.SelectedRange,
                                             period: str, system: params.CollisionSystem,
                                             output_info: analysis_objects.PlottingOutputWrapper) -> None:
    """ Plot the given tracking efficiencies parametrization.

    Args:
        efficiency: Calculated tracking efficiencies.
        centrality_range: Associated centrality range.
        period: Data taking period.
        system: Collision system.
        output_info: Output info for saving figures.
    Returns:
        None.
    """
    # Get the parameters
    pt_values, eta_values, n_cent_bins, centrality_ranges = generate_parameters(system)

    logger.debug(r"Plotting efficiencies for {centrality_range.min}--{centrality_range.max}%")
    fig, ax = plt.subplots(figsize = (8, 6))
    im = ax.imshow(
        efficiency.T,
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
        title += rf", ${centrality_range.min} \textendash {centrality_range.max}\%$"
    ax.set_title(title, size = 16)

    # Final adjustments
    fig.tight_layout()
    name = f"efficiency_{period}"
    if system != params.CollisionSystem.pp:
        name += f"_centrality_parametrization_{centrality_range.min}_{centrality_range.max}"
    plot_base.save_plot(output_info, fig, name)

    # Cleanup
    plt.close(fig)

def retrieve_efficiency_data(n_cent_bins: int, centrality_ranges: Dict[int, params.SelectedRange]) -> Tuple[List[Hist], List[Hist], List[Hist]]:
    """ Retrieve efficiency data.

    Args:
        n_cent_bins: Number of centrality bins.
        centrality_ranges: Map from centrality bin numbers to centrality ranges.
    Returns:
        (2D efficiency data, 1D pt efficiency data, 1D eta efficiency data)
    """
    # Retrieve histograms
    hists = histogram.get_histograms_in_list(
        filename = "trains/PbPbMC/55/AnalysisResults.root",
        list_name = "AliAnalysisTaskPWGJEQA_tracks_caloClusters_emcalCells_histos"
    )
    matched_sparse = hists["tracks_Matched"]
    generator_sparse = hists["tracks_PhysPrim"]

    # Retrieve the centrality dependent data
    efficiency_data_1D_pt = []
    efficiency_data_1D_eta = []
    efficiency_data_2D = []
    for dimension in ["1D", "2D"]:
        for centrality_bin, centrality_range in centrality_ranges.items():
            # Select in centrality
            matched_sparse.GetAxis(0).SetRangeUser(centrality_range.min + epsilon, centrality_range.max - epsilon)
            generator_sparse.GetAxis(0).SetRangeUser(centrality_range.min + epsilon, centrality_range.max - epsilon)
            # Restrict pt range to < 10 GeV
            matched_sparse.GetAxis(1).SetRangeUser(0.15, 10)
            generator_sparse.GetAxis(1).SetRangeUser(0.15, 10)

            if dimension == "2D":
                # (pt_gen, eta_gen) - order is reversed because 2D API is backwards...
                pt_gen_matched = matched_sparse.Projection(2, 1)
                pt_gen_matched.SetName(f"pt_gen_matched_cent_{centrality_bin}")
                # (pt_gen, eta_gen, findable)
                pt_gen_2d = generator_sparse.Projection(1, 2, 4)
                pt_gen_2d.SetName(f"pt_gen_2D_cent_{centrality_bin}")
                # Select only findable particles and use that efficiency.
                pt_gen_2d.GetZaxis().SetRange(2, 2)
                pt_gen_findable = pt_gen_2d.Project3D("yx")
                logger.debug(f"pt_gen_matched: {pt_gen_matched}, pt_gen_findable: {pt_gen_findable}")

                efficiency_hist = pt_gen_findable.Clone()
                efficiency_hist.Divide(pt_gen_matched, pt_gen_findable, 1.0, 1.0, "B")
                efficiency_data_2D.append(efficiency_hist)

            elif dimension == "1D":
                # pT 1D efficiency
                # NOTE: We can't just project from the 2D efficiency. Integrating over eta will get the wrong
                #       wrong values. Here, we project before we divide to get the right answer.
                pt_gen_matched1D = matched_sparse.Projection(1)
                pt_gen_matched1D.SetName(f"pt_gen_matched_1D_cent_{centrality_bin}")
                pt_gen_1d = generator_sparse.Projection(4, 1)
                pt_gen_1d.SetName(f"pt_gen_1D_cent_{centrality_bin}")
                # Select only findable particles and use that efficiency.
                pt_gen_1d.GetYaxis().SetRange(2, 2)
                pt_gen_findable = pt_gen_1d.ProjectionX()
                logger.debug(f"pt_gen_matched1D: {pt_gen_matched1D}, pt_gen_findable: {pt_gen_findable}")

                efficiency_1D = pt_gen_findable.Clone()
                efficiency_1D.Divide(pt_gen_matched1D, pt_gen_findable, 1.0, 1.0, "B")
                efficiency_data_1D_pt.append(efficiency_1D)

                # Eta 1D
                eta_gen_matched_1D = matched_sparse.Projection(2)
                eta_gen_matched_1D.SetName(f"eta_gen_matched_1D_cent_{centrality_bin}")
                eta_gen_1D = generator_sparse.Projection(4, 2)
                eta_gen_1D.SetName(f"eta_gen_1D_cent_{centrality_bin}")
                # Select only findable particles and use that efficiency.
                eta_gen_1D.GetYaxis().SetRange(2, 2)
                eta_gen_findable = eta_gen_1D.ProjectionX()
                logger.debug(f"eta_gen_matched_1D: {eta_gen_matched_1D}, eta_gen_findable: {eta_gen_findable}")

                efficiency_1D = eta_gen_findable.Clone()
                efficiency_1D.Divide(eta_gen_matched_1D, eta_gen_findable, 1.0, 1.0, "B")
                efficiency_data_1D_eta.append(efficiency_1D)
            else:
                # Shouldn't ever really happen, but just for sanity.
                raise RuntimeError(f"Invalid dimension {dimension}")

    return efficiency_data_2D, efficiency_data_1D_pt, efficiency_data_1D_eta

def calculate_residual_2D(efficiency_data: Hist, efficiency_function: Callable[..., float],
                          efficiency_period: Any, centrality_bin: int) -> Tuple[np.ndarray, List[float], List[float]]:
    """ Calculate residual for 2D tracking efficiency.

    There is a separate 1D and 2D function for convenience. If there is no entries for a particular
    bin, we set the value to NaN so that it can be ignored later when plotting.

    Args:
        efficiency_data: 2D efficiency data.
        efficiency_function: Efficiency function.
        efficiency_period: Efficiency period.
        centrality_bin: Centrality bin.
    Returns:
        Calculated residual, pt values where it was evaluated, eta values where it was evaluated.
    """
    pts = [efficiency_data.GetXaxis().GetBinCenter(x) for x in range(1, efficiency_data.GetXaxis().GetNbins() + 1)]
    etas = [efficiency_data.GetYaxis().GetBinCenter(y) for y in range(1, efficiency_data.GetYaxis().GetNbins() + 1)]
    residual = np.zeros(shape = (efficiency_data.GetXaxis().GetNbins(),
                                 efficiency_data.GetYaxis().GetNbins()))
    # Loop over all of the bins in the data histogram.
    for pt_index, pt in enumerate(pts):
        for eta_index, eta in enumerate(etas):
            x = pt_index + 1
            y = eta_index + 1
            # Calculate the efficiency. It's calculated again here to ensure that it's evaluated at exactly
            # the same location as in the data histogram.
            efficiency_at_value = efficiency_function(pt, eta, centrality_bin, efficiency_period, "task_name")

            # Determine the histogram value, setting it to NaN if there's no entries.
            if np.abs(efficiency_data.GetBinContent(x, y)) < epsilon:
                value = np.nan
            else:
                value = (efficiency_data.GetBinContent(x, y) - efficiency_at_value) / efficiency_at_value * 100.

            residual[pt_index, eta_index] = value

    # Check max values
    logger.debug(f"min efficiency_data: {efficiency_data.GetMinimum()}, "
                 f"max efficiency_data: {efficiency_data.GetMaximum()}")
    logger.debug(f"min residual: {np.nanmin(residual)}, max residual: {np.nanmax(residual)}")
    logger.debug(f"standard mean: {np.nanmean(residual)}")
    logger.debug(f"restricted mean: {np.nanmean(residual[:,np.abs(etas) < 0.8])}")
    logger.debug(f"len(pts): {len(pts)}, len(etas): {len(etas)}")

    return residual, pts, etas

def plot_residual(residual: np.ndarray, pts: List[float], etas: List[float],
                  period: str, centrality_bin: int, centrality_ranges: Dict[int, params.SelectedRange],
                  output_info: analysis_objects.PlottingOutputWrapper) -> None:
    """ Plot the residual between the data and the parametrization.

    Args:
        residual: Calculated residual.
        pts: Pt values where the residual was evaluated.
        etas: Eta values where the residual was evaluated.
        period: Name of the data taking period.
        centrality_bin: Centrality bin.
        centraliy_ranges: Map of centrality bins to ranges.
        output_info: Output info for saving figures.
    Returns:
        None.
    """
    fig, ax = plt.subplots(figsize = (8, 6))
    im = ax.imshow(
        residual.T, extent = [np.nanmin(pts), np.nanmax(pts), np.nanmin(etas), np.nanmax(etas)],
        interpolation = "nearest", aspect = "auto", origin = "lower",
        # An even normalization is better for the colorscheme.
        # NOTE: This causes clipping at the lowest pt values, but I don't think this is a big problem.
        norm = matplotlib.colors.Normalize(
            #vmin = np.nanmin(residuals[centrality_bin]), vmax = np.nanmax(residuals[centrality_bin])
            vmin = -40, vmax = 40
        ),
        # This is a good diverging color scheme when it's centered at 0.
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
    plot_base.save_plot(output_info, fig, name)

    # Cleanup
    plt.close(fig)

def plot_2D_efficiency_data(efficiency_hist: Hist, centrality_range: params.SelectedRange, output_info: analysis_objects.PlottingOutputWrapper) -> None:
    """ Plot the 2D efficiency data

    Args:
        efficiency_hist: Efficiecny histogram.
        centrality_range: Centrality range.
        output_info: Output info for saving figures.
    Returns:
        None.
    """
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
        title += rf", ${centrality_range.min} \textendash {centrality_range.max}\%$"
    ax.set_title(title, size = 16)

    # Final adjustments
    fig.tight_layout()
    name = f"efficiency_{period}"
    if system != params.CollisionSystem.pp:
        name += f"_centrality_{centrality_range.min}_{centrality_range.max}"
    plot_base.save_plot(output_info, fig, name)

    # Cleanup
    plt.close(fig)

def plot_1D_pt_efficiency(efficiency: Hist, PublicUtils: T_PublicUtils, efficiency_period: Any,
                          centrality_bin: int, centrality_range: params.SelectedRange,
                          output_info: analysis_objects.PlottingOutputWrapper) -> None:
    """ Plot 1D pt efficiency.

    Args:
        efficiency: Pt efficiency hist.
        PublicUtils: Jet-H public utils class.
        efficiency_period: Data taking period in the efficiency enum.
        centrality_bin: int
        centrality_range: Centrality range.
        output_info: Output info for saving figures.
    Returns:
        None.
    """
    # 1D efficiency as a function of pt
    logger.debug(f"max efficiency_1D: {efficiency.GetMaximum()}")
    h = histogram.Histogram1D.from_existing_hist(efficiency)
    fig, ax = plt.subplots(figsize = (8, 6))
    ax.errorbar(
        h.x, h.y, yerr = h.errors,
        label = "${labels.pt_display_label()}$",
        color = "black", marker = ".", linestyle = "",
    )

    # Efficiency function
    parametrization = []
    for x in h.x:
        parametrization.append(PublicUtils.LHC15oPtEfficiency(x, centrality_bin))
    ax.plot(
        h.x, parametrization,
        label = "${labels.pt_display_label()}$ param.",
        color = "red",
    )

    # Ensure that it's on a consistent axis
    ax.set_ylim(0.6, 1)

    # Labels
    ax.set_xlabel(fr"${labels.pt_display_label()}\:({labels.momentum_units_label_gev()})$")
    ax.set_ylabel(r"Efficiency")
    title = f"{period} ${labels.pt_display_label()}$ tracking efficiency"
    if system != params.CollisionSystem.pp:
        title += rf", ${centrality_range.min} \textendash {centrality_range.max}\%$"
    ax.set_title(title, size = 16)

    # Final adjustments
    fig.tight_layout()
    name = f"efficiency_pt_{period}"
    if system != params.CollisionSystem.pp:
        name += f"_centrality_{centrality_range.min}_{centrality_range.max}"
    plot_base.save_plot(output_info, fig, name)

def plot_1D_eta_efficiency(efficiency: Hist, PublicUtils: T_PublicUtils, efficiency_period: Any,
                           centrality_bin: int, centrality_range: params.SelectedRange,
                           output_info: analysis_objects.PlottingOutputWrapper) -> None:
    """ Plot 1D eta efficiency.

    Args:
        efficiency: Eta efficiency hist.
        PublicUtils: Jet-H public utils class.
        efficiency_period: Data taking period in the efficiency enum.
        centrality_bin: int
        centrality_range: Centrality range.
        output_info: Output info for saving figures.
    Returns:
        None.
    """
    # 1D efficiency as a function of eta
    logger.debug(f"max efficiency_1D: {efficiency.GetMaximum()}")
    h = histogram.Histogram1D.from_existing_hist(efficiency)
    fig, ax = plt.subplots(figsize = (8, 6))
    ax.errorbar(
        h.x, h.y, yerr = h.errors,
        label = r"$\eta$",
        color = "black", marker = ".", linestyle = "",
    )

    # Efficiency function
    parametrization = []
    for x in h.x:
        # Only evaluate for eta < 0.9 - otherwise the function goes back up again.
        if np.abs(x) < 0.9:
            # Need to undo the scale factor here to get something that's reasonably close..
            value = PublicUtils.LHC15oEtaEfficiency(x, centrality_bin) * \
                PublicUtils.LHC15oEtaEfficiencyNormalization(centrality_bin)
        else:
            # Just set to 0 since it's not meaningful
            value = 0
        parametrization.append(value)

    # And plot it
    ax.plot(
        h.x, parametrization,
        label = r"$\eta$ param.",
        color = "red",
    )

    # Ensure that it's on a consistent axis
    ax.set_ylim(0.6, 1)

    # Labels
    ax.set_xlabel(fr"$\eta$")
    ax.set_ylabel(r"Efficiency")
    title = fr"{period} $\eta$ tracking efficiency"
    if system != params.CollisionSystem.pp:
        title += rf", ${centrality_range.min} \textendash {centrality_range.max}\%$"
    ax.set_title(title, size = 16)

    # Final adjustments
    fig.tight_layout()
    name = f"efficiency_eta_{period}"
    if system != params.CollisionSystem.pp:
        name += f"_centrality_{centrality_range.min}_{centrality_range.max}"
    plot_base.save_plot(output_info, fig, name)

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

    # Calculate the efficiency.
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

    # Done checking the efficiency parametrization, so now plot it.
    for centrality_bin in range(n_cent_bins):
        plot_tracking_efficiency_parametrization(
            efficiencies[centrality_bin], centrality_ranges[centrality_bin],
            period, system, output_info
        )

    # Next, compare to the actual efficiency data.
    # We only have the LHC15o data easily available.
    if period == "LHC15o":
        # First, retrieve efficiency data
        efficiency_data_2D, efficiency_data_1D_pt, efficiency_data_1D_eta = \
            retrieve_efficiency_data(n_cent_bins, centrality_ranges)

        # Calculate residuals.
        logger.info("Calculating and plotting residuals")
        # Setup
        residuals: np.ndarray = None
        for centrality_bin, centrality_range in centrality_ranges.items():
            # Finish the setup
            if residuals is None:
                residuals = np.zeros(
                    shape = (n_cent_bins, efficiency_data_2D[centrality_bin].GetXaxis().GetNbins(),
                             efficiency_data_2D[centrality_bin].GetYaxis().GetNbins())
                )
            logger.debug(f"residuals shape: {residuals.shape}")

            # Calculate the residuals.
            residuals[centrality_bin], pts, etas, = calculate_residual_2D(
                efficiency_data_2D[centrality_bin], efficiency_function, efficiency_period, centrality_bin
            )

            # Plot the residuals.
            plot_residual(residuals[centrality_bin], pts, etas, period, centrality_bin, centrality_ranges, output_info)

            # Plot the 2D efficiency data
            plot_2D_efficiency_data(efficiency_data_2D[centrality_bin],
                                    centrality_ranges[centrality_bin], output_info)

            # 1D efficiency comparison
            plot_1D_pt_efficiency(efficiency_data_1D_pt[centrality_bin],
                                  PublicUtils, efficiency_period,
                                  centrality_bin, centrality_ranges[centrality_bin],
                                  output_info)
            plot_1D_eta_efficiency(efficiency_data_1D_eta[centrality_bin],
                                   PublicUtils, efficiency_period,
                                   centrality_bin, centrality_ranges[centrality_bin],
                                   output_info)

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

