#!/usr/bin/env python3

""" Use TGlauberMC to calculate expected path lengths with respect to event plane orientations.

"""

from dataclasses import dataclass
import enlighten
import numpy as np
import os
import logging
from pathlib import Path
import requests
import tarfile
from typing import Any, cast, List

# NOTE: This is out of the expected order, but it must be here to prevent ROOT from stealing the command
#       line options
from jet_hadron.base.typing_helpers import Hist  # noqa: F401

from pachyderm import histogram

from jet_hadron.base import analysis_manager
from jet_hadron.base import analysis_objects
from jet_hadron.base import params
from jet_hadron.plot import base as plot_base

import ROOT

logger = logging.getLogger(__name__)

def _setup_TGlauberMC(version: str, path: Path) -> bool:
    """ Setup TGlauberMC by downloading and extracting it.

    Args:
        version: TGlauberMC version.
        path: Path to the directory where TGlauberMC will be stored.
    Returns:
        True if it was successfully setup.
    Raises:
        RuntimeError: If the package was not downloaded succefully.
    """
    # Setup
    url = f"https://tglaubermc.hepforge.org/downloads/?f=TGlauberMC-{version}.tar.gz"
    filename = path / url[url.find("=") + 1:]
    filename.parent.mkdir(parents = True, exist_ok = True)

    # Request the file
    logger.debug(f"Downloading TGlauberMC v. {version}")
    r = requests.get(url)

    if not r.status_code == requests.codes.ok:
        raise RuntimeError(f"Unable to download package at {url}")

    # Save it.
    with open(filename, "wb") as f:
        f.write(r.content)

    # Untar
    logger.debug("Extracting...")
    with tarfile.open(filename, "r:gz") as f_tar:
        # Help out mypy...
        f_tar.extractall(path = str(filename.parent))

    # Remove the tgz file - it's not needed anymore.
    #filename.unlink()

    return True

def _configure_TGlauberMC(version: str, path: Path) -> bool:
    """ Configure and compile TGlauberMC so that it's accessible through ROOT.

    Args:
        version: TGlauberMC version.
        path: Path to the directory where TGlauberMC is stored.
    Returns:
        True if it was successfully configured.
    """
    logger.info("Configuring TGlauberMC")

    # Pre-requisite for running the code
    ROOT.gSystem.Load("libMathMore")

    # Compile the Glauber code.
    logger.debug("Compiling TGlauberMC. This may take a moment...")
    ROOT.gROOT.LoadMacro(os.path.join(path, f"runglauber_v{version}.C+"))

    # The Glauber classes can now be accessed through ROOT.
    return True

@dataclass
class CrossSection:
    """ Define an input cross section. """
    value: float
    width: float

def _calculate_array_RMS(arr: np.ndarray) -> float:
    """ Calculate the RMS of the given array using the same proceudre as ROOT. """
    return cast(float, np.sqrt(1 / len(arr) * np.sum((arr - np.mean(arr)) ** 2)))

class GlauberPathLengthAnalysis:
    def __init__(self, cross_section: CrossSection, impact_parameter_range: params.SelectedRange) -> None:
        # Store base propreties.
        self.cross_section = cross_section
        self.impact_parameter_range = impact_parameter_range

        # Store output from each event
        self.max_x: np.ndarray = []
        self.max_y: np.ndarray = []
        self.eccentricity: np.ndarray = []
        # Calculated length ratio
        self.ratio: np.ndarray

        # The actual Glauber object
        self.glauber: Any

    def setup(self) -> bool:
        # Setup the glauber object
        # Arguments are (Nuclei, Nuclei, cross section, cross section width)
        self.glauber = ROOT.TGlauberMC("Pb", "Pb", self.cross_section.value, self.cross_section.width)
        # Specify the impact parameters
        self.glauber.SetBmin(self.impact_parameter_range.min)
        self.glauber.SetBmax(self.impact_parameter_range.max)
        # Recommended value.
        self.glauber.SetMinDistance(0.4)

        return True

    def event_loop(self, n_events: int, progress_manager: enlighten._manager.Manager) -> bool:
        """ Run the Glauber event loop.

        Args:
            n_events: Number of events to run.
            progress_manager: Progress manager to keep track of execution progress.
        Returns:
            True if executed successfully.
        """
        # Temporary variables to store the event-by-event results
        # We will store these in numpy arrays, but it's not convenient to expand those arrays,
        # so we will store them temporarily in easily expandable lists and then store them
        # in the analysis object after the event-by-event process is completed.
        max_x: List[float] = []
        max_y: List[float] = []
        eccentricity: List[float] = []

        #c = ROOT.TCanvas("c", "c")
        with progress_manager.counter(total = n_events,
                                      desc = "Calculating:",
                                      unit = "glauber events") as progress:
            for i in range(n_events):
                # Run one event and retrieve the nucleons.
                self.glauber.Run(1)
                nucleons = self.glauber.GetNucleons()

                x_values = []
                y_values = []
                for nucleon in nucleons:
                    # We only care about nucleons which participate.
                    if nucleon.IsWounded():
                        x_values.append(nucleon.GetX())
                        y_values.append(nucleon.GetY())

                max_x.append(np.max(x_values))
                max_y.append(np.max(y_values))
                eccentricity.append(self.glauber.GetEcc(2))

                # Uncomment if we want to save plots of  the individual events.
                #glauber.Draw()
                #c.SaveAs(f"glauber_{i}.pdf")

                progress.update()

        # Convert all of the stored values to numpy arrays for convenience.
        self.max_x = np.array(max_x)
        self.max_y = np.array(max_y)
        self.eccentricity = np.array(eccentricity)

        # Define the length ratio that we are interested in.
        # Although we want to the out / in ratio, we invert it here because the small value corresponds to
        # yielding more jets. So out / in should be < 1.
        self.ratio = self.max_x / self.max_y

        # Calculate the mean and RMS
        # We do in the same way as ROOT.
        # Calculate the mean and RMS the same way as ROOT
        logger.info(f"Mean ratio: {np.mean(self.ratio)}, rms: {_calculate_array_RMS(self.ratio)}")
        # Eccentricity for comparison.
        # It shouldn't match because the measured eccentricity is not trivially related to geometric eccentricity.
        logger.info(f"Mean eccentricity: {np.mean(self.eccentricity)}, rms: {_calculate_array_RMS(self.eccentricity)}")

        # Can also directly compare against ROOT to be certain that we've calculated the mean and RMS correctly.
        #h_ratio_root = ROOT.TH1D("test", "test", 150, 0, 15)
        #for v in self.ratio:
        #    h_ratio_root.Fill(v)
        #logger.info(f"ROOT mean: {h_ratio_root.GetMean()}, rms: {h_ratio_root.GetRMS()}")

        return True

    def process_result(self, selected_analysis_options: params.SelectedAnalysisOptions,
                       glauber_version: str, output_info: analysis_objects.PlottingOutputWrapper) -> None:
        """ Perform final processing of the event-by-event results. """
        logger.info("Performing final processing.")

        # Setup
        main_font_size = 16
        general_labels = fr"TGlauberMC v{glauber_version}, ${selected_analysis_options.event_activity.display_str()}\:{selected_analysis_options.collision_system.display_str()}$"
        general_labels += "\n" + fr"$\sigma = {self.cross_section.value} \pm {self.cross_section.width}$ mb"
        general_labels += fr", $b = {self.impact_parameter_range.min} - {self.impact_parameter_range.max}$ fm"

        # Create histograms of the two axes
        h_x, x_edges = np.histogram(
            self.max_x, bins = 100, range = (0, 10),
        )
        hist_max_x = histogram.Histogram1D(
            bin_edges = x_edges, y = h_x, errors_squared = h_x,
        )

        h_y, y_edges = np.histogram(
            self.max_y, bins = 100, range = (0, 10),
        )
        hist_max_y = histogram.Histogram1D(
            bin_edges = y_edges, y = h_y, errors_squared = h_y,
        )

        # Plot the distributions
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize = (8, 6))
        ax.errorbar(
            hist_max_x.x, hist_max_x.y, yerr = hist_max_x.errors,
            marker = "o", linestyle = "",
            label = "Semi-minor",
        )
        ax.errorbar(
            hist_max_y.x, hist_max_y.y, yerr = hist_max_y.errors,
            marker = "o", linestyle = "",
            label = "Semi-major",
        )

        # Label and final adjustments
        ax.legend(loc = "upper left", frameon = False, fontsize = main_font_size)
        ax.set_xlabel("Max length (fm)", fontsize = main_font_size)
        ax.set_ylabel("Counts / 0.1", fontsize = main_font_size)
        ax.tick_params(axis='both', which='major', labelsize = main_font_size)
        ax.text(0.97, 0.97, s = general_labels,
                horizontalalignment = "right",
                verticalalignment = "top",
                multialignment = "right",
                fontsize = main_font_size,
                transform = ax.transAxes)
        fig.tight_layout()

        # Save plot and cleanup
        plot_base.save_plot(output_info, fig, "glauber_lengths")
        plt.close(fig)

        # Plot ratio
        fig, ax = plt.subplots(figsize = (8, 6))
        h, edges = np.histogram(
            self.ratio, bins = 60, range = (-1, 2)
        )
        h = histogram.Histogram1D(
            bin_edges = edges, y = h, errors_squared = h,
        )
        ax.errorbar(
            h.x, h.y, yerr = h.errors,
            marker = "o", linestyle = "",
            label = "out-of-plane / in-plane",
        )

        # Label and final adjustments
        ax.legend(
            loc = "upper right", bbox_to_anchor = (0.99, 0.99), borderaxespad = 0,
            frameon = False, fontsize = main_font_size
        )
        ax.set_xlabel("Length ratio", fontsize = main_font_size)
        ax.set_ylabel("Counts / 0.05", fontsize = main_font_size)
        ax.tick_params(axis='both', which='major', labelsize = main_font_size)
        ratio_label = general_labels
        ratio_label += "\n" + fr"Mean: ${np.mean(self.ratio):.2f} \pm {_calculate_array_RMS(self.ratio):.2f}$ (RMS)"
        ax.text(0.03, 0.97, s = ratio_label,
                horizontalalignment = "left",
                verticalalignment = "top",
                multialignment = "left",
                fontsize = main_font_size,
                transform = ax.transAxes)
        fig.tight_layout()

        # Plot and cleanup
        plot_base.save_plot(output_info, fig, "glauber_ratio")
        plt.close(fig)

class GlauberPathLengthManager(analysis_manager.Manager):
    def __init__(self, config_filename: str, selected_analysis_options: params.SelectedAnalysisOptions, **kwargs: str):
        super().__init__(
            config_filename = config_filename, selected_analysis_options = selected_analysis_options,
            manager_task_name = "GlauberToyModelManager", **kwargs,
        )

        # Analysis task
        self.analysis: GlauberPathLengthAnalysis

        # Properties for setting up the Glauber code
        self.glauber_version = self.task_config.get("glauber_version", "3.2")
        self.glauber_directory_name: str = self.task_config.get("glauber_directory_name", "TGlauberMC")

        # Properties
        self.n_events = self.task_config["n_events"]
        # Cross sections (from Table V in 1710.07098)
        self.cross_sections = {
            params.CollisionEnergy.two_seven_six: CrossSection(value = 61.8, width = 0.9),
            params.CollisionEnergy.five_zero_two: CrossSection(value = 67.6, width = 0.6),
        }
        # Impact parameters
        # NOTE: Although ALICE doesn't provide impact parameters, we can compare values (N_coll and N_part)
        #       to ALICE calculated values: https://alice-notes.web.cern.ch/node/711)
        self.impact_parameters = {
            params.EventActivity.central: params.SelectedRange(0, 4.92),
            params.EventActivity.semi_central: params.SelectedRange(8.5, 11),
        }

    def setup_TGlauberMC(self) -> bool:
        """ Setup the TGlauberMC code.

        This will guide downloading it if necessary, and then load the code using ROOT. It is accessible
        through the ``ROOT`` module.

        Args:
            version: TGlauberMC version.
            glauber_directory_name: Name of the directory where TGlauberMC is stored. Default: "TGlauberMC".
        Returns:
            True if TGlauberMC was setup successfully.
        """
        logger.info("Setting up TGlauberMC")

        # For convenience, we're going to put it in the same directory as where the toy model
        # is executing. Note that we want the directory name, so we need to remove the filename.
        path = Path(__file__).parent.resolve() / Path(self.glauber_directory_name)

        # Setup TGlauberMC
        if not path.exists() or not (path / f"runglauber_v{self.glauber_version}.C").exists():
            _setup_TGlauberMC(version = self.glauber_version, path = path)
        _configure_TGlauberMC(version = self.glauber_version, path = path)

        return True

    def _setup(self) -> bool:
        """ Setup TGlauberMC and the analysis task. """
        # Ensure that the TGlauberMC code is available.
        success = self.setup_TGlauberMC()
        if not success:
            raise RuntimeError("Failed to setup TGlauberMC.")

        # Determine the settings
        cross_section = self.cross_sections[self.selected_analysis_options.collision_energy]
        impact_parameter_range = self.impact_parameters[self.selected_analysis_options.event_activity]

        # Create the object
        self.analysis = GlauberPathLengthAnalysis(
            cross_section = cross_section,
            impact_parameter_range = impact_parameter_range,
        )

        self.analysis.setup()

        return True

    def run(self) -> bool:
        """ Setup and run the actual analysis. """
        # Setup
        result = self._setup()
        if not result:
            raise RuntimeError("Setup failed")

        # Run the analysis
        self.analysis.event_loop(
            n_events = self.n_events,
            progress_manager = self._progress_manager
        )

        # Process the final results
        self.analysis.process_result(
            selected_analysis_options = self.selected_analysis_options,
            glauber_version = self.glauber_version,
            output_info = self.output_info,
        )

        return True

def run_from_terminal() -> GlauberPathLengthManager:
    """ Driver function for running the Glauber path length toy model analysis. """
    # Basic setup
    # Quiet down some pachyderm modules
    logging.getLogger("pachyderm.generic_config").setLevel(logging.INFO)
    logging.getLogger("pachyderm.yaml").setLevel(logging.INFO)
    logging.getLogger("pachyderm.histogram").setLevel(logging.INFO)
    # Run in batch mode
    ROOT.gROOT.SetBatch(True)
    # Turn off stats box
    ROOT.gStyle.SetOptStat(0)

    # Setup and run the analysis
    manager: GlauberPathLengthManager = analysis_manager.run_helper(
        manager_class = GlauberPathLengthManager, task_name = "Glauber path length toy model analysis manager",
    )

    # Return it for convenience.
    return manager

if __name__ == "__main__":
    run_from_terminal()
