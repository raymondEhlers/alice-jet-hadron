#!/usr/bin/env python

""" Tests for analysis params.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import logging
import pytest

from jet_hadron.base import params
from jet_hadron.base import analysis_objects

# Setup logger
logger = logging.getLogger(__name__)

def get_range_from_bin_array(array):
    """ Helper function to return bin indices from an array.
    Args:
        array (list): Array from which the bin indcies will be extracted.
    Returns:
        list: The bin indices
    """
    return range(len(array) - 1)

def test_iterate_over_track_pt_bins(logging_mixin):
    """ Test the track pt bins generator.

    Note that we wrap the function in list so we get all of the values from the generator.
    """
    assert len(params.track_pt_bins) == 10
    assert list(params.iterate_over_track_pt_bins()) == list(get_range_from_bin_array(params.track_pt_bins))

def test_iterate_over_track_pt_bins_with_config(logging_mixin):
    """ Test the track pt bins generator with some bins skipped.

    The values to skip were not selected with any paticular critera except to be non-continuous.
    """
    skip_bins = [2, 6]
    comparison_bins = [x for x in get_range_from_bin_array(params.track_pt_bins) if x not in skip_bins]
    config = {"skipPtBins": {"track": skip_bins}}
    assert list(params.iterate_over_track_pt_bins(config = config)) == comparison_bins

def test_iterate_over_jet_pt_bins(logging_mixin):
    """ Test the jet pt bins generator.

    Note that we wrap the function in list so we get all of the values from the generator.
    """
    # Ensure that we have the expected number of jet pt bins
    assert len(params.jet_pt_bins) == 5
    # Then test the actual iterable.
    assert list(params.iterate_over_jet_pt_bins()) == list(get_range_from_bin_array(params.jet_pt_bins))

def test_iterate_over_jet_pt_bins_with_config(logging_mixin):
    """ Test the jet pt bins generator with some bins skipped.

    The values to skip were not selected with any paticular critera except to be non-continuous.
    """
    skip_bins = [0, 2]
    comparison_bins = [x for x in get_range_from_bin_array(params.jet_pt_bins) if x not in skip_bins]
    config = {"skipPtBins": {"jet": skip_bins}}
    assert list(params.iterate_over_jet_pt_bins(config = config)) == comparison_bins

def test_iterate_over_jet_and_track_pt_bins(logging_mixin):
    """ Test the jet and track pt bins generator.

    Note that we wrap the function in list so we get all of the values from the generator.
    """
    comparison_bins = [(x, y) for x in get_range_from_bin_array(params.jet_pt_bins) for y in get_range_from_bin_array(params.track_pt_bins)]
    assert list(params.iterate_over_jet_and_track_pt_bins()) == comparison_bins

def test_iterate_over_jet_and_track_pt_bins_with_config(logging_mixin):
    """ Test the jet and track pt bins generator with some bins skipped.

    The values to skip were not selected with any paticular critera except to be non-continuous.
    """
    skip_jet_pt_bins = [0, 3]
    skip_track_pt_bins = [2, 6]
    comparison_bins = [(x, y) for x in get_range_from_bin_array(params.jet_pt_bins) for y in get_range_from_bin_array(params.track_pt_bins) if x not in skip_jet_pt_bins and y not in skip_track_pt_bins]
    config = {"skipPtBins": {"jet": skip_jet_pt_bins, "track": skip_track_pt_bins}}
    # Check that the comparison bins are as expected.
    assert comparison_bins == [(1, 0), (1, 1), (1, 3), (1, 4), (1, 5), (1, 7), (1, 8), (2, 0), (2, 1), (2, 3), (2, 4), (2, 5), (2, 7), (2, 8)]
    # Then check the actual output.
    assert list(params.iterate_over_jet_and_track_pt_bins(config = config)) == comparison_bins

def test_out_of_range_skip_bin(logging_mixin):
    """ Test that an except is generated if a skip bin is out of range.

    The test is performed both with a in range and out of range bin to ensure
    the exception is thrown on the right value.
    """
    skip_bins = [2, 38]
    config = {"skipPtBins": {"track": skip_bins}}
    with pytest.raises(ValueError) as exception_info:
        list(params.iterate_over_track_pt_bins(config = config))
    # NOTE: ExecptionInfo is a wrapper around the exception. `.value` is the actual exectpion
    #       and then we want to check the value of the first arg, which contains the value
    #       that causes the exception.
    assert exception_info.value.args[0] == skip_bins[1]

#############
# Label tests
#############
@pytest.mark.parametrize("value, expected", [
    (r"\textbf{test}", r"#textbf{test}"),
    (r"$\mathrm{test}$", r"#mathrm{test}")
], ids = ["just latex", "latex in math mode"])
def test_root_latex_conversion(logging_mixin, value, expected):
    """ Test converting latex to ROOT compatiable latex. """
    assert params.use_label_with_root(value) == expected

@pytest.mark.parametrize("label, expected", [
    ("work_in_progress", {"str": "ALICE Work in Progress"}),
    ("preliminary", {"str": "ALICE Preliminary"}),
    ("final", {"str": "ALICE"}),
    ("thesis", {"str": "This thesis"})
], ids = ["work in progress", "preliminary", "final", "thesis"])
def test_alice_label(logging_mixin, label, expected):
    """ Tests ALICE labeling. """
    alice_label = params.AliceLabel[label]
    assert str(alice_label) == expected["str"]

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
            assert params.generate_track_pt_range_string(pt_bin) == r"$%(lower)s < \mathit{p}_{\mathrm{T}}^{\mathrm{assoc}} < %(upper)s\:\mathrm{GeV/\mathit{c}}$" % {"lower": expected_min, "upper": expected_max}

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
            assert params.generate_jet_pt_range_string(pt_bin) == r"$%(lower)s < \mathit{p}_{\mathrm{T \,unc,jet}}^{\mathrm{ch+ne}} < %(upper)s\:\mathrm{GeV/\mathit{c}}$" % {"lower": expected_min, "upper": expected_max}

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
        assert params.generate_jet_pt_range_string(jet_pt_bin) == r"$%(lower)s < \mathit{p}_{\mathrm{T \,unc,jet}}^{\mathrm{ch+ne}}\:\mathrm{GeV/\mathit{c}}$" % {"lower": self.jet_pt_bins[-2]}

@pytest.mark.parametrize("energy, system, activity, expected", [
    (2.76, "pp", "inclusive", r"$\mathrm{pp}\:\sqrt{s_{\mathrm{NN}}} = 2.76\:\mathrm{TeV}$"),
    (2.76, "PbPb", "central", r"$\mathrm{Pb\mbox{-}Pb}\:\sqrt{s_{\mathrm{NN}}} = 2.76\:\mathrm{TeV},\:0\mbox{-}10\mbox{\%}$"),
    (2.76, "PbPb", "semi_central", r"$\mathrm{Pb\mbox{-}Pb}\:\sqrt{s_{\mathrm{NN}}} = 2.76\:\mathrm{TeV},\:30\mbox{-}50\mbox{\%}$"),
    (5.02, "PbPb", "central", r"$\mathrm{Pb\mbox{-}Pb}\:\sqrt{s_{\mathrm{NN}}} = 5.02\:\mathrm{TeV},\:0\mbox{-}10\mbox{\%}$"),
    ("five_zero_two", "PbPb", "central", r"$\mathrm{Pb\mbox{-}Pb}\:\sqrt{s_{\mathrm{NN}}} = 5.02\:\mathrm{TeV},\:0\mbox{-}10\mbox{\%}$"),
    ("5.02", "PbPb", "central", r"$\mathrm{Pb\mbox{-}Pb}\:\sqrt{s_{\mathrm{NN}}} = 5.02\:\mathrm{TeV},\:0\mbox{-}10\mbox{\%}$"),
    (params.CollisionEnergy.five_zero_two, params.CollisionSystem.PbPb, params.EventActivity.central, r"$\mathrm{Pb\mbox{-}Pb}\:\sqrt{s_{\mathrm{NN}}} = 5.02\:\mathrm{TeV},\:0\mbox{-}10\mbox{\%}$")
], ids = ["Inclusive pp", "Central PbPb", "Semi-central PbPb", "Central PbPb at 5.02", "Energy as string five_zero_two", "Energy as string \"5.02\"", "Using enums directly"])
def test_system_label(logging_mixin, energy, system, activity, expected):
    """ Test system labels. """
    assert params.system_label(energy = energy, system = system, activity = activity) == expected

def test_jet_properties_labels(logging_mixin):
    """ Test the jet properties labels. """
    jet_pt_bin = analysis_objects.JetPtBin(bin = 1, range = params.SelectedRange(20.0, 40.0))
    (jet_finding_expected, constituent_cuts_expected, leading_hadron_expected, jet_pt_expected) = (
        r"$\mathrm{anti\mbox{-}k}_{\mathrm{T}}\;R=0.2$",
        r"$\mathit{p}_{\mathrm{T}}^{\mathrm{ch}}\:\mathrm{\mathit{c},}\:\mathrm{E}_{\mathrm{T}}^{\mathrm{clus}} > 3\:\mathrm{GeV}$",
        r"$\mathit{p}_{\mathrm{T}}^{\mathrm{lead,ch}} > 5\:\mathrm{GeV/\mathit{c}}$",
        r"$20.0 < \mathit{p}_{\mathrm{T \,unc,jet}}^{\mathrm{ch+ne}} < 40.0\:\mathrm{GeV/\mathit{c}}$"
    )

    (jet_finding, constituent_cuts, leading_hadron, jet_pt) = params.jet_properties_label(jet_pt_bin)

    assert jet_finding == jet_finding_expected
    assert constituent_cuts == constituent_cuts_expected
    assert leading_hadron == leading_hadron_expected
    assert jet_pt == jet_pt_expected

@pytest.mark.parametrize("energy, expected", [
    (params.CollisionEnergy(2.76),
        {"str": "2.76",
            "display_str": r"\sqrt{s_{\mathrm{NN}}} = 2.76\:\mathrm{TeV}",
            "value": 2.76}),
    (params.CollisionEnergy["two_seven_six"],
        {"str": "2.76",
            "display_str": r"\sqrt{s_{\mathrm{NN}}} = 2.76\:\mathrm{TeV}",
            "value": 2.76}),
    (params.CollisionEnergy(5.02),
        {"str": "5.02",
            "display_str": r"\sqrt{s_{\mathrm{NN}}} = 5.02\:\mathrm{TeV}",
            "value": 5.02})
], ids = ["2.76 standard", "two_seven_six alternative intialization", "5.02 standard"])
def test_collision_energy(logging_mixin, energy, expected):
    """ Test collision energy values. """
    assert str(energy) == expected["str"]
    assert energy.display_str() == expected["display_str"]
    assert energy.value == expected["value"]

@pytest.mark.parametrize("system, expected", [
    (params.CollisionSystem["pp"],
        {"str": "pp",
            "display_str": "pp"}),
    (params.CollisionSystem["pythia"],
        {"str": "pythia",
            "display_str": "PYTHIA"}),
    (params.CollisionSystem["PbPb"],
        {"str": "PbPb",
            "display_str": params.PbPbLatexLabel}),
    (params.CollisionSystem["embedPP"],
        {"str": "embedPP",
            "display_str": fr"pp \bigotimes {params.PbPbLatexLabel}"})
], ids = ["pp", "pythia", "PbPb", "embedded pp"])
def test_collision_system(logging_mixin, system, expected):
    """ Test collision system values. """
    assert str(system) == expected["str"]
    assert system.display_str() == expected["display_str"]

@pytest.mark.parametrize("activity, expected", [
    (params.EventActivity["inclusive"],
        {"str": "inclusive",
            "display_str": "",
            "range": params.SelectedRange(min = -1, max = -1)}),
    (params.EventActivity["central"],
        {"str": "central",
            "display_str": r"0\mbox{-}10\mbox{\%}",
            "range": params.SelectedRange(min = 0, max = 10)}),
    (params.EventActivity["semi_central"],
        {"str": "semi_central",
            "display_str": r"30\mbox{-}50\mbox{\%}",
            "range": params.SelectedRange(min = 30, max = 50)})
], ids = ["inclusive", "central", "semi_central"])
def test_event_activity(logging_mixin, activity, expected):
    """ Test event activity values. """
    assert str(activity) == expected["str"]
    assert activity.display_str() == expected["display_str"]
    assert activity.value_range == expected["range"]

@pytest.mark.parametrize("bias, expected", [
    ("NA", {"str": "NA"}),
    ("track", {"str": "track"}),
    ("cluster", {"str": "cluster"}),
    ("both", {"str": "both"})
], ids = ["NA", "track", "cluster", "both"])
def test_leading_hadron_bias_type(logging_mixin, bias, expected):
    """ Test the leading hadron bias enum. """
    bias = params.LeadingHadronBiasType[bias]
    assert str(bias) == expected["str"]

@pytest.mark.parametrize("type, value, expected", [
    ("NA", 0, {"str": "NA"}),
    ("NA", 5, {"value": 0, "str": "NA"}),
    ("track", 5, {"str": "trackBias5"}),
    ("cluster", 6, {"str": "clusterBias6"}),
    ("both", 10, {"str": "bothBias10"})
], ids = ["NA", "NAPassedWrongValue", "track", "cluster", "both"])
def test_leading_hadron_bias(logging_mixin, type, value, expected):
    """ Test the leading hadron bias class. """
    type = params.LeadingHadronBiasType[type]
    bias = params.LeadingHadronBias(type = type, value = value)
    # Handle value with a bit of care in the case of "NAPassedWrongValue"
    value = expected["value"] if "value" in expected else value
    assert bias.type == type
    assert bias.value == value
    assert str(bias) == expected["str"]

@pytest.mark.parametrize("ep_angle, expected", [
    ("all",
        {"str": "all",
            "display_str": "All"}),
    ("out_of_plane",
        {"str": "out_of_plane",
            "display_str": "Out-of-plane"})
], ids = ["epAngleAll", "epAngleOutOfPlane"])
def test_reaction_plane_orientation_strings(logging_mixin, ep_angle, expected):
    """ Test event plane angle strings. """
    ep_angle = params.ReactionPlaneOrientation[ep_angle]
    assert str(ep_angle) == expected["str"]
    assert ep_angle.display_str() == expected["display_str"]

@pytest.mark.parametrize("qvector, expected", [
    ("all",
        {"str": "all",
            "display_str": "All",
            "range": params.SelectedRange(min = 0, max = 100)}),
    ("bottom10",
        {"str": "bottom10",
            "display_str": "Bottom 10%",
            "range": params.SelectedRange(min = 0, max = 10)})
], ids = ["All", "Bottom 10"])
def test_qvector_strings(logging_mixin, qvector, expected):
    """ Test q vector strings. """
    qvector = params.QVector[qvector]
    assert str(qvector) == expected["str"]
    assert qvector.display_str() == expected["display_str"]
    assert qvector.value_range == expected["range"]

