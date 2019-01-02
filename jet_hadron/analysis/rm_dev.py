#!/usr/bin/env python

""" Response matrix dev.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@yale.edu>, Yale University
"""

from dataclasses import dataclass
import logging
import pprint
from typing import Any, Dict, Iterable, Tuple

from pachyderm import generic_class

from jet_hadron.base import analysis_config
from jet_hadron.base import analysis_objects

Hist = Any

logger = logging.getLogger(__name__)

class PtHardInformation(analysis_objects.JetHBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Basic information
        self.pt_hard_bin = kwargs["pt_hard_bin"]
        self.use_after_event_selection_information: bool = False
        self.task_name = kwargs["task_name"]

        # Histograms
        self.pt_hard_spectra: Hist
        self.cross_section: Hist
        self.n_trials: Hist
        self.n_events: Hist

    def retrieve_histograms(self, input_hists: Dict[str, Any]) -> None:
        """ Retrieve relevant histogram information.

        Args:
            input_hists: Input histograms where the inforamtion will be retrieved.
        Returns:
            None.
        """
        task_name = ""
        event_sel_tag = "AfterSel" if self.use_after_event_selection_information else ""

        # Retrieve hists
        self.cross_section = input_hists[task_name][f"fHistXsection{event_sel_tag}"]
        self.n_trials = input_hists[task_name][f"fHistTrials{event_sel_tag}"]
        self.n_events = input_hists[task_name]["fHistEventCount"]
        self.pt_hard_spectra = input_hists[task_name]["fHistPtHard"]

    def extract_scale_factor(self) -> float:
        """ Extract the scale factor from the stored information. """
        pass

@dataclass
class ResponseHistograms:
    """ The main histograms for a response matrix. """
    jet_spectra: Hist
    unmatched_jet_spectra: Hist
    sample_task_jet_spectra: Hist

class ResponseMatrix(analysis_objects.JetHBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pt_hard_bin = kwargs["pt_hard_bin"]

        # Relevant histograms
        self.response_matrix: Hist
        self.response_matrix_errirs: Hist

        self.part_level_hists: ResponseHistograms
        self.det_level_hists: ResponseHistograms

        self.pt_hard_information: PtHardInformation

    def setup_pt_hard_information(self):
        #self.train_number = self.task_config["pt_hard_map"][self.pt_hard_bin]
        pass

class ResponseManager(generic_class.EqualityMixin):
    """ Analysis manager for creating response(s).

    Attributes:

    """
    def __init__(self, config_filename, selected_analysis_options, **kwargs):
        self.config_filename = config_filename
        self.selected_analysis_options = selected_analysis_options

        # Create the actual analysis objects.
        (self.key_index, self.selected_option_names, self.analyses) = self.construct_from_configuration_file()

    def construct_from_configuration_file(self) -> Tuple[Any, Iterable[Any], Iterable[Any]]:
        """ Construct ResponseMatrix objects based on iterables in a configuration file. """
        return analysis_config.construct_from_configuration_file(
            task_name = "Response",
            config_filename = self.config_filename,
            selected_analysis_options = self.selected_analysis_options,
            additional_possible_iterables = {"pt_hard_bin": None, "jet_pt_bin": None},
            obj = ResponseMatrix,
        )

    def run(self):
        logger.debug(f"key_index: {self.key_index}, selected_option_names: {self.selected_option_names}, analyses: {pprint.pformat(self.analyses)}")

        #for a in self.analyses.values():
        #    a.setup_pt_hard_information()
        #    logger.debug(f"{a.train_number}")

        # Test
        test_object = next(iter(self.analyses.values()))
        logger.info(f"Attempting to dump obj of type: {type(test_object)}")
        import ruamel.yaml
        import sys
        yaml = ruamel.yaml.YAML(typ = "rt")
        yaml.register_class(ResponseMatrix)
        yaml.dump([test_object], sys.stdout)

def run_from_terminal():
    # Basic setup
    logging.basicConfig(level = logging.DEBUG)
    # Quiet down the matplotlib logging
    logging.getLogger("matplotlib").setLevel(logging.INFO)

    # Setup the analysis
    (config_filename, terminal_args, additional_args) = analysis_config.determine_selected_options_from_kwargs(
        task_name = "Response matrix"
    )
    analysis_manager = ResponseManager(
        config_filename = config_filename,
        selected_analysis_options = terminal_args
    )
    # Finally run the analysis.
    analysis_manager.run()

    # Return it for convenience.
    return analysis_manager

if __name__ == "__main__":
    run_from_terminal()
