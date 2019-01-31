#!/usr/bin/env python

""" Tests for the analysis_objects module.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import logging
import numpy as np
import pytest

from jet_hadron.base import analysis_config
from jet_hadron.base import analysis_objects
from jet_hadron.base import params

# Setup logger
logger = logging.getLogger(__name__)
# For reproducibility
np.random.seed(1234)

@pytest.mark.parametrize("corr_type, expected", [
    ("full_range",
        {"str": "full_range",
            "display_str": "Full Range"}),
    ("signal_dominated",
        {"str": "signal_dominated",
            "display_str": "Signal Dominated"}),
    ("near_side",
        {"str": "near_side",
            "display_str": "Near Side"})
], ids = ["full range", "dPhi signal dominated", "dEta near side"])
def test_correlation_types(logging_mixin, corr_type, expected):
    """ Test jet-hadron correlation types. """
    obj = analysis_objects.JetHCorrelationType[corr_type]

    assert str(obj) == expected["str"]
    assert obj.display_str() == expected["display_str"]

@pytest.mark.parametrize("axis, expected", [
    ("delta_phi",
        {"str": "delta_phi",
            "display_str": r"$\Delta\varphi$"}),
    ("delta_eta",
        {"str": "delta_eta",
            "display_str": r"$\Delta\eta$"}),
], ids = ["Delta phi", "Delta eta"])
def test_correlation_axis(logging_mixin, axis, expected):
    """ Test jet-hadron correaltion axis. """
    obj = analysis_objects.JetHCorrelationAxis[axis]

    assert str(obj) == expected["str"]
    assert obj.display_str() == expected["display_str"]

@pytest.mark.parametrize("leading_hadron_bias", [
    (params.LeadingHadronBiasType.track),
    (params.LeadingHadronBias(type = params.LeadingHadronBiasType.track, value = 5))
], ids = ["leadingHadronEnum", "leadingHadronClass"])
def test_JetHBase_object_construction(logging_mixin, leading_hadron_bias, object_yaml_config, override_options_helper, check_JetHBase_object, mocker):
    """ Test construction of the JetHBase object. """
    object_config, task_name = object_yaml_config
    (config, selected_analysis_options) = override_options_helper(
        object_config,
        config_containing_override = object_config[task_name]
    )

    # Avoid os.makedirs actually making directories
    mocker.patch("os.makedirs")

    config_filename = "configFilename.yaml"
    task_config = config[task_name]
    reaction_plane_orientation = params.ReactionPlaneOrientation.inclusive
    config_base = analysis_objects.JetHBase(
        task_name = task_name,
        config_filename = config_filename,
        config = config,
        task_config = task_config,
        collision_energy = selected_analysis_options.collision_energy,
        collision_system = selected_analysis_options.collision_system,
        event_activity = selected_analysis_options.event_activity,
        leading_hadron_bias = selected_analysis_options.leading_hadron_bias,
        reaction_plane_orientation = reaction_plane_orientation,
    )

    # We need values to compare against. However, namedtuples are immutable,
    # so we have to create a new one with the proper value.
    temp_selected_options = selected_analysis_options.asdict()
    temp_selected_options["leading_hadron_bias"] = leading_hadron_bias
    selected_analysis_options = params.SelectedAnalysisOptions(**temp_selected_options)
    # Only need for the case of LeadingHadronBiasType!
    if isinstance(leading_hadron_bias, params.LeadingHadronBiasType):
        selected_analysis_options = analysis_config.determine_leading_hadron_bias(config, selected_analysis_options)

    # Assertions are performed in this function
    res = check_JetHBase_object(
        obj = config_base,
        config = config,
        selected_analysis_options = selected_analysis_options,
        reaction_plane_orientation = reaction_plane_orientation
    )
    assert res is True

    # Just to be safe
    mocker.stopall()

