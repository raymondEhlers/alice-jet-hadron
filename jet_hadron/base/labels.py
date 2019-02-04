#!/usr/bin/env python

""" Labeling for plotting, etc.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import logging
import numbers
from typing import Tuple, Union

from jet_hadron.base import analysis_objects
from jet_hadron.base import params

logger = logging.getLogger(__name__)

def use_label_with_root(label: str) -> str:
    """ Automatically convert LaTeX to something that is mostly ROOT compatiable.

    Args:
        label: Label to be converted.
    Returns:
        Converted label.
    """
    # Remove "$" and map "\" -> "#""
    return label.replace("$", "").replace("\\", "#")

def pt_display_label(lower_label: str = r"\mathrm{T}", upper_label: str = "") -> str:
    """ Generate a pt display label without the "$".

    Args:
        lower_label: Subscript label for pT. Default: "\\mathrm{T}"
        upper_label: Superscript labe for pT. Default: ""
    Returns:
        Properly formatted pt string.
    """
    return r"\mathit{p}_{%(lower_label)s}^{%(upper_label)s}" % {
        "lower_label": lower_label,
        "upper_label": upper_label,
    }

def jet_pt_display_label(upper_label: str = "") -> str:
    """ Generate a display jet pt label.

    Args:
        upper_label: Superscript labe for pT. Default: ""
    Returns:
        Properly formatted pt string.
    """
    return pt_display_label(
        lower_label = r"\mathrm{T,jet}",
        upper_label = upper_label,
    )

def momentum_units_label_gev() -> str:
    """ Generate a GeV/c label.

    Args:
        None.
    Returns:
        A properly latex formatted GeV/c label.
    """
    return r"\mathrm{GeV/\mathit{c}}"

def pt_range_string(pt_bin: analysis_objects.PtBin,
                    lower_label: str, upper_label: str,
                    only_show_lower_value_for_last_bin: bool = False) -> str:
    """ Generate string to describe pt ranges for a given list.

    Args:
        pt_bin: Pt bin object which contains information about the bin and pt range.
        lower_label: Subscript label for pT.
        upper_label: Superscript labe for pT.
        only_show_lower_value_for_last_bin: If True, skip show the upper value.
    Returns:
        The pt range label.
    """
    # Cast as string so we don't have to deal with formatting the extra digits
    lower = f"{pt_bin.range.min} < "
    upper = f" < {pt_bin.range.max}"
    if only_show_lower_value_for_last_bin and pt_bin.range.max == -1:
        upper = ""
    pt_range = r"$%(lower)s%(pt_label)s%(upper)s\:%(units_label)s$" % {
        "lower": lower,
        "pt_label": pt_display_label(lower_label = lower_label, upper_label = upper_label),
        "upper": upper,
        "units_label": momentum_units_label_gev(),
    }

    return pt_range

def jet_pt_range_string(jet_pt_bin: analysis_objects.PtBin) -> str:
    """ Generate a label for the jet pt range based on the jet pt bin.

    Args:
        jet_pt_bin: Jet pt bin object.
    Returns:
        Jet pt range label.
    """
    return pt_range_string(
        pt_bin = jet_pt_bin,
        lower_label = r"\mathrm{T \,unc,jet}",
        upper_label = r"\mathrm{ch+ne}",
        only_show_lower_value_for_last_bin = True,
    )

def track_pt_range_string(track_pt_bin: analysis_objects.PtBin) -> str:
    """ Generate a label for the track pt range based on the track pt bin.

    Args:
        track_pt_bin: Track pt bin.
    Returns:
        Track pt range label.
    """
    return pt_range_string(
        pt_bin = track_pt_bin,
        lower_label = r"\mathrm{T}",
        upper_label = r"\mathrm{assoc}",
    )

def jet_properties_label(jet_pt_bin: analysis_objects.JetPtBin) -> Tuple[str, str, str, str]:
    """ Return the jet finding properties based on the jet pt bin.

    Args:
        jet_pt_bin: Jet pt bin
    Returns:
        tuple: (jet_finding, constituent_cuts, leading_hadron, jet_pt)
    """
    jet_finding = r"$\mathrm{anti\mbox{-}k}_{\mathrm{T}}\;R=0.2$"
    constituent_cuts = r"$%(pt_label)s\:\mathrm{\mathit{c},}" \
                       r"\:\mathrm{E}_{\mathrm{T}}^{\mathrm{clus}} > 3\:\mathrm{GeV}$" % {
                           "pt_label": pt_display_label(upper_label = r"\mathrm{ch}")
                       }
    leading_hadron = r"$%(pt_label)s > 5\:%(units_label)s$" % {
        "pt_label": pt_display_label(upper_label = r"\mathrm{lead,ch}"),
        "units_label": momentum_units_label_gev(),
    }
    jet_pt = jet_pt_range_string(jet_pt_bin)
    return (jet_finding, constituent_cuts, leading_hadron, jet_pt)

def system_label(energy: Union[float, params.CollisionEnergy], system: Union[str, params.CollisionSystem], activity: Union[str, params.EventActivity]) -> str:
    """ Generates the collision system, event activity, and energy label as a latex label.

    Args:
        energy: The collision energy
        system: The collision system.
        activity: The event activity selection.
    Returns:
        Label for the entire system, combining the avaialble information.
    """
    # Handle energy
    if isinstance(energy, numbers.Number):
        energy = params.CollisionEnergy(energy)
    elif isinstance(energy, str):
        try:
            e = float(energy)
            energy = params.CollisionEnergy(e)
        except ValueError:
            energy = params.CollisionEnergy[energy]  # type: ignore
    # Ensure that we've done our conversion correctly. This also helps out mypy.
    assert isinstance(energy, params.CollisionEnergy)

    # Handle collision system
    if isinstance(system, str):
        system = params.CollisionSystem[system]  # type: ignore

    # Handle event activity
    if isinstance(activity, str):
        activity = params.EventActivity[activity]  # type: ignore
    event_activity_str = activity.display_str()
    if event_activity_str:
        event_activity_str = r",\:" + event_activity_str

    system_label = r"$\mathrm{%(system)s}\:%(energy)s%(event_activity)s$" % {
        "system": system.display_str(),
        "energy": energy.display_str(),
        "event_activity": event_activity_str,
    }

    logger.debug(f"system_label: {system_label}")

    return system_label

