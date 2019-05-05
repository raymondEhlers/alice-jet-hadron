#!/usr/bin/env python

""" Tests for analysis labels.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import logging
import pytest

from jet_hadron.base import analysis_objects
from jet_hadron.base import labels
from jet_hadron.base import params

# Setup logger
logger = logging.getLogger(__name__)

@pytest.mark.parametrize("value, expected", [
    (r"$\mathrm{hello}$", r"$\mathrm{hello}$"),
    (r"\mathrm{hello}$", r"$\mathrm{hello}$"),
    (r"$\mathrm{hello}", r"$\mathrm{hello}$"),
    (r"\mathrm{hello}", r"$\mathrm{hello}$"),
    ("", ""),
], ids = ["No change", "Wrap start", "Wrap end", "Wrap both sides", "Empty string"])
def test_make_valid_latex_string(logging_mixin, value: str, expected: str):
    """ Test for making a string into a valid latex string. """
    assert labels.make_valid_latex_string(value) == expected

@pytest.mark.parametrize("value, expected", [
    (r"\textbf{test}", r"\textbf{test}"),
    (r"Pb \textendash Pb", r"Pb \mbox{-} Pb"),
    (r"$0-10\% \mathrm{test}$", r"$0-10\mbox{%} \mathrm{test}$"),
], ids = ["Nothing changes", "Replace endash", "Replace percentage sign"])
def test_root_latex_conversion(logging_mixin, value, expected):
    """ Test converting latex to ROOT compatiable latex. """
    assert labels.use_label_with_root(value) == expected

class TestTrackPtString:
    track_pt_bins = [0.15, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0]

    def test_track_pt_strings(self, logging_mixin):
        """ Test the track pt string generation functions. Each bin is tested.  """
        pt_bins = []
        for i, (min, max) in enumerate(zip(self.track_pt_bins[:-1], self.track_pt_bins[1:])):
            pt_bins.append(
                analysis_objects.TrackPtBin(
                    bin = i,
                    range = params.SelectedRange(min, max)
                )
            )

        for pt_bin, expected_min, expected_max in zip(pt_bins, self.track_pt_bins[:-1], self.track_pt_bins[1:]):
            logger.debug(f"Checking bin {pt_bin}, {pt_bin.range}, {type(pt_bin)}")
            assert labels.track_pt_range_string(pt_bin) == r"$%(lower)s < p_{\text{T}}^{\text{assoc}} < %(upper)s\:\mathrm{GeV/\mathit{c}}$" % {"lower": expected_min, "upper": expected_max}

class TestJetPtString:
    # NOTE: The -1 is important for the final bin to be understood correctly as the last bin!
    jet_pt_bins = [15.0, 20.0, 40.0, 60.0, -1]

    def test_jet_pt_string(self, logging_mixin):
        """ Test the jet pt string generation functions. Each bin (except for the last) is tested.

        The last pt bin is left for a separate test because it is printed differently
        (see ``test_jet_pt_string_for_last_pt_bin()`` for more).
        """
        pt_bins = []
        for i, (min, max) in enumerate(zip(self.jet_pt_bins[:-2], self.jet_pt_bins[1:-1])):
            pt_bins.append(
                analysis_objects.JetPtBin(
                    bin = i,
                    range = params.SelectedRange(min, max)
                )
            )

        for pt_bin, expected_min, expected_max in zip(pt_bins, self.jet_pt_bins[:-2], self.jet_pt_bins[1:-1]):
            logger.debug(f"Checking bin {pt_bin}, {pt_bin.range}, {type(pt_bin)}")
            assert labels.jet_pt_range_string(pt_bin) == r"$%(lower)s < p_{\text{T,jet}}^{\text{ch+ne}} < %(upper)s\:\mathrm{GeV/\mathit{c}}$" % {"lower": expected_min, "upper": expected_max}

    def test_jet_pt_string_for_last_pt_bin(self, logging_mixin):
        """ Test the jet pt string generation function for the last jet pt bin.

        In the case of the last pt bin, we only want to show the lower range.
        """
        pt_bin = len(self.jet_pt_bins) - 2
        jet_pt_bin = analysis_objects.JetPtBin(
            bin = pt_bin,
            range = params.SelectedRange(
                self.jet_pt_bins[pt_bin],
                self.jet_pt_bins[pt_bin + 1]
            )
        )
        assert labels.jet_pt_range_string(jet_pt_bin) == r"$%(lower)s < p_{\text{T,jet}}^{\text{ch+ne}}\:\mathrm{GeV/\mathit{c}}$" % {"lower": self.jet_pt_bins[-2]}

@pytest.mark.parametrize("energy, system, activity, expected", [
    (2.76, "pp", "inclusive", r"$\mathrm{pp}\:\sqrt{s_{\mathrm{NN}}} = 2.76\:\mathrm{TeV}$"),
    (2.76, "PbPb", "central", r"$\mathrm{Pb \textendash Pb}\:\sqrt{s_{\mathrm{NN}}} = 2.76\:\mathrm{TeV},\:0 \textendash 10 \%$"),
    (2.76, "PbPb", "semi_central", r"$\mathrm{Pb \textendash Pb}\:\sqrt{s_{\mathrm{NN}}} = 2.76\:\mathrm{TeV},\:30 \textendash 50 \%$"),
    (5.02, "PbPb", "central", r"$\mathrm{Pb \textendash Pb}\:\sqrt{s_{\mathrm{NN}}} = 5.02\:\mathrm{TeV},\:0 \textendash 10 \%$"),
    ("five_zero_two", "PbPb", "central", r"$\mathrm{Pb \textendash Pb}\:\sqrt{s_{\mathrm{NN}}} = 5.02\:\mathrm{TeV},\:0 \textendash 10 \%$"),
    ("5.02", "PbPb", "central", r"$\mathrm{Pb \textendash Pb}\:\sqrt{s_{\mathrm{NN}}} = 5.02\:\mathrm{TeV},\:0 \textendash 10 \%$"),
    (params.CollisionEnergy.five_zero_two, params.CollisionSystem.PbPb, params.EventActivity.central, r"$\mathrm{Pb \textendash Pb}\:\sqrt{s_{\mathrm{NN}}} = 5.02\:\mathrm{TeV},\:0 \textendash 10 \%$")
], ids = ["Inclusive pp", "Central PbPb", "Semi-central PbPb", "Central PbPb at 5.02", "Energy as string five_zero_two", "Energy as string \"5.02\"", "Using enums directly"])
def test_system_label(logging_mixin, energy, system, activity, expected):
    """ Test system labels. """
    assert labels.system_label(energy = energy, system = system, activity = activity) == expected

@pytest.mark.parametrize("upper_label, expected", [
    ("", r"p_{\text{T,jet}}^{\text{}}"),
    (r"det", r"p_{\text{T,jet}}^{\text{det}}")
], ids = ["Base test", "Superscript"])
def test_jet_pt_display_string(logging_mixin, upper_label, expected):
    """ Test for generating jet pt labels. """
    # Determine args. Only call with an argument if we've specified one so we can test the default args.
    kwargs = {}
    if upper_label != "":
        kwargs["upper_label"] = upper_label

    output = labels.jet_pt_display_label(**kwargs)
    assert output == expected

def test_track_pt_display_string(logging_mixin):
    """ Test for generating the track pt label. """
    labels.track_pt_display_label() == r"p_{\text{T,jet}}^{\text{assoc}}"

def test_gev_momentum_units_label(logging_mixin):
    """ Test generating GeV/c label in latex. """
    output = labels.momentum_units_label_gev()
    expected = r"\mathrm{GeV/\mathit{c}}"
    assert output == expected

@pytest.mark.parametrize("include_normalized_by_n_trig, expected", [
    (False, r"$\mathrm{d}N/\mathrm{d}\varphi$"),
    (True, r"$1/N_{\mathrm{trig}}\mathrm{d}N/\mathrm{d}\varphi$"),
], ids = ["Do not include n_trig", "Include n_trig"])
def test_delta_phi_axis_label(logging_mixin, include_normalized_by_n_trig, expected):
    """ Test for the delta phi axis label. """
    label = labels.delta_phi_axis_label(normalized_by_n_trig = include_normalized_by_n_trig)
    assert label == expected

@pytest.mark.parametrize("include_normalized_by_n_trig, expected", [
    (False, r"$\mathrm{d}N/\mathrm{d}\eta$"),
    (True, r"$1/N_{\mathrm{trig}}\mathrm{d}N/\mathrm{d}\eta$"),
], ids = ["Do not include n_trig", "Include n_trig"])
def test_delta_eta_axis_label(logging_mixin, include_normalized_by_n_trig, expected):
    """ Test for the delta eta axis label. """
    label = labels.delta_eta_axis_label(normalized_by_n_trig = include_normalized_by_n_trig)
    assert label == expected

