#!/usr/bin/env python

""" Tests for the generic analysis task plotting.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import logging
import pytest

from jet_hadron.analysis import generic_tasks
from jet_hadron.plot import generic_hist as plot_generic_hist

logger = logging.getLogger(__name__)

@pytest.fixture
def setup_input_objects_and_structures(mocker):
    """ Setup input objects and structure for testing PlotHistTasks. """
    # Hists
    hist1 = mocker.MagicMock(spec = ["GetName", "GetTitle", "SetTitle"])
    hist1.GetName.return_value = "hist1"
    hist1.GetTitle.return_value = "hist1Title"
    hist2 = mocker.MagicMock(spec = ["GetName", "GetTitle", "SetTitle"])
    hist2.GetName.return_value = "hist2"
    hist2.GetTitle.return_value = "hist2Title"
    hist3 = mocker.MagicMock(spec = ["GetName", "GetTitle", "SetTitle"])
    hist3.GetName.return_value = "hist3"
    hist3.GetTitle.return_value = "hist3Title"
    # HistPlotters
    hist_plotter_1 = mocker.MagicMock(
        spec = plot_generic_hist.HistPlotter,
        hist_names = [{"hist1": ""}],
        exact_name_match = True,
        hists = [],
    )
    hist_plotter_2 = mocker.MagicMock(
        spec = plot_generic_hist.HistPlotter,
        hist_names = [{"hist2": ""}],
        exact_name_match = True,
        hists = [],
    )
    hist_plotter_3 = mocker.MagicMock(
        spec = plot_generic_hist.HistPlotter,
        hist_names = [{"hist3": ""}],
        exact_name_match = True,
        hists = [],
    )

    # Setup input structure
    # Hists
    input_hists = {
        "AliEmcalCorrectionCellBadChannel": {
            "AliEventCuts": {
                "hist1": hist1,
            },
            "hist2": hist2,
        },
        "AliEmcalCorrectionCellEnergy": {
            "hist3": hist3,
        },
    }
    # Configuration
    plot_configurations = {
        "CellBadChannel": {
            "EventCuts": {
                "hist1": hist_plotter_1,
            },
            "hist2": hist_plotter_2,
        },
        "CellEnergy": {
            "hist3": hist_plotter_3,
        }
    }

    return (plot_configurations, input_hists,
            hist1, hist_plotter_1,
            hist2, hist_plotter_2,
            hist3, hist_plotter_3)

def test_iterate_over_plot_configurations(logging_mixin, setup_input_objects_and_structures):
    """ Test iterating over plot configurations. """
    (plot_configurations, input_hists,
     hist1, hist_plotter_1,
     hist2, hist_plotter_2,
     hist3, hist_plotter_3) = setup_input_objects_and_structures

    iter_config = generic_tasks.iterate_over_plot_configurations(plot_configurations = plot_configurations)

    assert next(iter_config) == ("hist1", hist_plotter_1, ["CellBadChannel", "EventCuts"])
    assert next(iter_config) == ("hist2", hist_plotter_2, ["CellBadChannel"])
    assert next(iter_config) == ("hist3", hist_plotter_3, ["CellEnergy"])

    # It should be exhausted now.
    with pytest.raises(StopIteration):
        next(iter_config)

def test_determine_hists_for_plot_configurations(logging_mixin, setup_input_objects_and_structures):
    """ Test assigning hists to plot configurations. """
    (plot_configurations, input_hists,
     hist1, hist_plotter_1,
     hist2, hist_plotter_2,
     hist3, hist_plotter_3) = setup_input_objects_and_structures

    # Determine which hists correspond to which plot configurations.
    generic_tasks._determine_hists_for_plot_configurations(
        plot_configurations = plot_configurations,
        input_hists = input_hists
    )

    # Check the results.
    assert plot_configurations["CellBadChannel"]["EventCuts"]["hist1"] == hist_plotter_1
    assert plot_configurations["CellBadChannel"]["EventCuts"]["hist1"].hists == [hist1]
    assert plot_configurations["CellBadChannel"]["hist2"] == hist_plotter_2
    assert plot_configurations["CellBadChannel"]["hist2"].hists == [hist2]
    assert plot_configurations["CellEnergy"]["hist3"] == hist_plotter_3
    assert plot_configurations["CellEnergy"]["hist3"].hists == [hist3]

#def my_func(config, hists, output = None):
#
#    for config_name, config in full_config:
#        for hist_name, obj in input_hists:
#            if config_name in hist_name:
#                # We've found a config that matches the hists.
#                if isinstance(config, plot_generic_hist.HistPlotter):
#                    # Assign the hist(s).
#                    pass
#                elif isinstance(config, dict):
#                    output[config_name] = {}
#                    # Go deeper
#                    my_func(config = config, hists = obj)
#
#def my_func_2(input_config, path = None):
#    if path is None:
#        path = []
#    for config_name, config in input_config:
#        if isinstance(config, plot_generic_hist.HistPlotter):
#            config.path = path + [config_name]
#            config.options_name = config_name
#        elif isinstance(config, dict):
#            path.append(config_name)
#            my_func_2(input_config)
#        else:
#            raise ValueError(f"Unrecognized type '{type(config)}' with name {config_name}")
#
#def _recursive_get_hist(config_name, hist_name, hists):
#    for p in config.paths:
#        for h in hists:
#            if p in h:
#                if isinstance(hists[h], ROOT.TH1):
#                    config_name.hists.append(hists[h])
#                hists = hists[h]
#                break
#        else:
#            raise ValueError(f"Hist corresponding to path {path} is not available")
#
#def test():
#    pass
