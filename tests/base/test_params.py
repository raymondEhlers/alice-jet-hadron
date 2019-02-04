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

class TestIteratePtBins:
    _track_pt_bins = [0.15, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0]
    track_pt_bins = [
        analysis_objects.TrackPtBin(range = params.SelectedRange(min, max), bin = i + 1)
        for i, (min, max) in enumerate(zip(_track_pt_bins[:-1], _track_pt_bins[1:]))
    ]
    _jet_pt_bins = [15.0, 20.0, 40.0, 60.0, 200.0]
    jet_pt_bins = [
        analysis_objects.JetPtBin(range = params.SelectedRange(min, max), bin = i + 1)
        for i, (min, max) in enumerate(zip(_jet_pt_bins[:-1], _jet_pt_bins[1:]))
    ]

    def test_iterate_over_track_pt_bins(self, logging_mixin):
        """ Test the track pt bins generator.

        Note that we wrap the function in list so we get all of the values from the generator.
        """
        assert len(self.track_pt_bins) == 9
        assert list(params.iterate_over_track_pt_bins(self.track_pt_bins)) == list(self.track_pt_bins)

    def test_iterate_over_track_pt_bins_with_config(self, logging_mixin):
        """ Test the track pt bins generator with some bins skipped.

        The values to skip were not selected with any paticular critera except to be non-continuous.
        """
        skip_bins = [2, 6]
        comparison_bins = [x for x in self.track_pt_bins if x.bin not in skip_bins]
        config = {"skipPtBins": {"track": skip_bins}}
        assert list(params.iterate_over_track_pt_bins(bins = self.track_pt_bins, config = config)) == comparison_bins

    def test_iterate_over_jet_pt_bins(self, logging_mixin):
        """ Test the jet pt bins generator.

        Note that we wrap the function in list so we get all of the values from the generator.
        """
        # Ensure that we have the expected number of jet pt bins
        assert len(self.jet_pt_bins) == 4
        # Then test the actual iterable.
        assert list(params.iterate_over_jet_pt_bins(self.jet_pt_bins)) == list(self.jet_pt_bins)

    def test_iterate_over_jet_pt_bins_with_config(self, logging_mixin):
        """ Test the jet pt bins generator with some bins skipped.

        The values to skip were not selected with any paticular critera except to be non-continuous.
        """
        skip_bins = [1, 2]
        comparison_bins = [x for x in self.jet_pt_bins if x.bin not in skip_bins]
        config = {"skipPtBins": {"jet": skip_bins}}
        assert list(params.iterate_over_jet_pt_bins(bins = self.jet_pt_bins, config = config)) == comparison_bins

    def test_iterate_over_jet_and_track_pt_bins(self, logging_mixin):
        """ Test the jet and track pt bins generator.

        Note that we wrap the function in list so we get all of the values from the generator.
        """
        comparison_bins = [(x, y) for x in self.jet_pt_bins for y in self.track_pt_bins]
        assert list(params.iterate_over_jet_and_track_pt_bins(jet_pt_bins = self.jet_pt_bins, track_pt_bins = self.track_pt_bins)) == comparison_bins

    def test_iterate_over_jet_and_track_pt_bins_with_config(self, logging_mixin):
        """ Test the jet and track pt bins generator with some bins skipped.

        The values to skip were not selected with any paticular critera except to be non-continuous.
        """
        skip_jet_pt_bins = [1, 4]
        skip_track_pt_bins = [2, 6]
        comparison_bins = [(x, y) for x in self.jet_pt_bins for y in self.track_pt_bins if x.bin not in skip_jet_pt_bins and y.bin not in skip_track_pt_bins]
        config = {"skipPtBins": {"jet": skip_jet_pt_bins, "track": skip_track_pt_bins}}
        # Check that the comparison bins are as expected.
        comparison_bin_bins = [(x.bin, y.bin) for (x, y) in comparison_bins]
        assert comparison_bin_bins == [(2, 1), (2, 3), (2, 4), (2, 5), (2, 7), (2, 8), (2, 9), (3, 1), (3, 3), (3, 4), (3, 5), (3, 7), (3, 8), (3, 9)]
        # Then check the actual output.
        assert list(params.iterate_over_jet_and_track_pt_bins(jet_pt_bins = self.jet_pt_bins, track_pt_bins = self.track_pt_bins, config = config)) == comparison_bins

    @pytest.mark.parametrize("bin_type_name, skip_bins", [
        ("track", [2, 38]),
        ("jet", [2, 5]),
    ], ids = ["Track", "Jet"])
    def test_out_of_range_skip_track_bin(self, logging_mixin, bin_type_name, skip_bins):
        """ Test that an except is generated if a skip bin is out of range.

        The test is performed both with a in range and out of range bin to ensure
        the exception is thrown on the right value.
        """
        if bin_type_name == "track":
            bins = self.track_pt_bins
            func = params.iterate_over_track_pt_bins
        elif bin_type_name == "jet":
            bins = self.jet_pt_bins
            func = params.iterate_over_jet_pt_bins
        else:
            # Unrecognized.
            bins = None
            func = None

        config = {"skipPtBins": {bin_type_name: skip_bins}}
        with pytest.raises(ValueError) as exception_info:
            list(func(bins = bins, config = config))
        # NOTE: ExecptionInfo is a wrapper around the exception. `.value` is the actual exectpion
        #       and then we want to check the value of the first arg, which contains the value
        #       that causes the exception.
        assert exception_info.value.args[0] == skip_bins[1]

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

# NOTE: Usually, "Pb--Pb" is used in latex, but ROOT won't render it properly...
_PbPbLatexLabel = r"Pb\mbox{-}Pb"

@pytest.mark.parametrize("system, expected", [
    (params.CollisionSystem["pp"],
        {"str": "pp",
            "display_str": "pp"}),
    (params.CollisionSystem["pythia"],
        {"str": "pythia",
            "display_str": "PYTHIA"}),
    (params.CollisionSystem["PbPb"],
        {"str": "PbPb",
            "display_str": _PbPbLatexLabel}),
    (params.CollisionSystem["embedPP"],
        {"str": "embedPP",
            "display_str": fr"pp \bigotimes {_PbPbLatexLabel}"})
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
    ("inclusive",
        {"str": "inclusive",
            "display_str": "Inclusive"}),
    ("out_of_plane",
        {"str": "out_of_plane",
            "display_str": "Out-of-plane"})
], ids = ["Inclusive", "Out of Plane"])
def test_reaction_plane_orientation_strings(logging_mixin, ep_angle, expected):
    """ Test event plane angle strings. """
    ep_angle = params.ReactionPlaneOrientation[ep_angle]
    assert str(ep_angle) == expected["str"]
    assert ep_angle.display_str() == expected["display_str"]

@pytest.mark.parametrize("qvector, expected", [
    ("inclusive",
        {"str": "inclusive",
            "display_str": "Inclusive",
            "range": params.SelectedRange(min = 0, max = 100)}),
    ("bottom10",
        {"str": "bottom10",
            "display_str": "Bottom 10%",
            "range": params.SelectedRange(min = 0, max = 10)})
], ids = ["Inclusive", "Bottom 10"])
def test_qvector_strings(logging_mixin, qvector, expected):
    """ Test q vector strings. """
    qvector = params.QVector[qvector]
    assert str(qvector) == expected["str"]
    assert qvector.display_str() == expected["display_str"]
    assert qvector.value_range == expected["range"]

