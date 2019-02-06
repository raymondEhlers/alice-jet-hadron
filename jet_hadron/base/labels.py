#!/usr/bin/env python

""" Labeling for plotting, etc.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import logging
import numbers
from typing import Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from jet_hadron.base import analysis_objects
    # mypy complains about this import whatever reason (but not the above), so we ignore it.
    from jet_hadron.base import params  # noqa: F401

logger = logging.getLogger(__name__)

def use_label_with_root(label: str) -> str:
    """ Automatically convert LaTeX to something that is mostly ROOT compatible.

    Conversion consists of:

    - Replace "\\textendash" -> "\\mbox{-}"
    - Replace "\\%" -> "\\mbox{%}"

    In principle, LaTeX is supported in ``TMathText``, but the support is pretty flaky.

    Args:
        label: Label to be converted.
    Returns:
        Converted label.
    """
    return label.replace(r"\textendash", r"\mbox{-}").replace(r"\%", r"\mbox{%}")

def et_display_label(lower_label: str = "T", upper_label: str = "") -> str:
    """ Generate a E_T display label without the "$".

    Args:
        lower_label: Subscript label for E_T. Default: "T"
        upper_label: Superscript label for E_T. Default: ""
    Returns:
        Properly formatted E_T string.
    """
    return r"\mathit{E}_{\mathrm{%(lower_label)s}}^{\mathrm{%(upper_label)s}}" % {
        "lower_label": lower_label,
        "upper_label": upper_label,
    }

def pt_display_label(lower_label: str = "T", upper_label: str = "") -> str:
    """ Generate a pt display label without the "$".

    Args:
        lower_label: Subscript label for pT. Default: "T"
        upper_label: Superscript label for pT. Default: ""
    Returns:
        Properly formatted pt string.
    """
    return r"\mathit{p}_{\mathrm{%(lower_label)s}}^{\mathrm{%(upper_label)s}}" % {
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
        lower_label = r"T,jet",
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

def pt_range_string(pt_bin: "analysis_objects.PtBin",
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

def jet_pt_range_string(jet_pt_bin: "analysis_objects.PtBin") -> str:
    """ Generate a label for the jet pt range based on the jet pt bin.

    Args:
        jet_pt_bin: Jet pt bin object.
    Returns:
        Jet pt range label.
    """
    return pt_range_string(
        pt_bin = jet_pt_bin,
        lower_label = r"T \,unc,jet",
        upper_label = "ch+ne",
        only_show_lower_value_for_last_bin = True,
    )

def track_pt_range_string(track_pt_bin: "analysis_objects.PtBin") -> str:
    """ Generate a label for the track pt range based on the track pt bin.

    Args:
        track_pt_bin: Track pt bin.
    Returns:
        Track pt range label.
    """
    return pt_range_string(
        pt_bin = track_pt_bin,
        lower_label = "T",
        upper_label = "assoc",
    )

def jet_finding_label(R: float = 0.2) -> str:
    """ The jet finding label.

    Args:
        R: The jet finding radius. Default: 0.2
    Returns:
        A properly latex formatted jet finding label.
    """
    return r"$\mathrm{anti \textendash} \mathit{k}_{\mathrm{T}}\;R=%(R)s$" % {"R": R}

def constituent_cuts(min_track_pt: float = 3.0, min_cluster_pt: float = 3.0, additional_label: str = "") -> str:
    """ The jet constituent cut label.

    Args:
        min_track_pt: Minimum track pt. Default: 3.0
        min_cluster_pt: Minimum cluster pt. Default: 3.0
        additional_label: Additional label for the constituents (such as denoting particle or
            detector level). Default: "".
    Returns:
        A properly latex formatted constituent cuts label.
    """
    # Validation
    # Add a comma in between the labels if the additional label is not empty.
    if additional_label and not (additional_label.startswith(",") or additional_label.startswith(r"\mathrm{,}")):
        additional_label = r"\mathrm{,}" + additional_label

    track_label = pt_display_label(upper_label = "ch" + additional_label)
    cluster_label = et_display_label(upper_label = "clus" + additional_label)

    if min_track_pt == min_cluster_pt:
        constituent_cuts = fr"${track_label}\mathit{{c}}\mathrm{{,}}\:{cluster_label} > {min_track_pt:g}\:\mathrm{{GeV}}$"
    else:
        constituent_cuts = (
            fr"${track_label} > {min_track_pt:g}\:{momentum_units_label_gev}"
            fr"\mathrm{{,}}\:{cluster_label} > {min_cluster_pt:g}\:\mathrm{{GeV}}$"
        )

    return constituent_cuts

def jet_properties_label(jet_pt_bin: "analysis_objects.JetPtBin") -> Tuple[str, str, str, str]:
    """ Return the jet finding properties based on the jet pt bin.

    Args:
        jet_pt_bin: Jet pt bin
    Returns:
        tuple: (jet_finding, constituent_cuts, leading_hadron, jet_pt)
    """
    jet_finding = jet_finding_label()
    const_cuts = constituent_cuts()
    leading_hadron = r"$%(pt_label)s > 5\:%(units_label)s$" % {
        "pt_label": pt_display_label(upper_label = "lead,ch"),
        "units_label": momentum_units_label_gev(),
    }
    jet_pt = jet_pt_range_string(jet_pt_bin)
    return (jet_finding, const_cuts, leading_hadron, jet_pt)

def system_label(energy: Union[float, "params.CollisionEnergy"],
                 system: Union[str, "params.CollisionSystem"],
                 activity: Union[str, "params.EventActivity"]) -> str:
    """ Generates the collision system, event activity, and energy label as a latex label.

    Args:
        energy: The collision energy
        system: The collision system.
        activity: The event activity selection.
    Returns:
        Label for the entire system, combining the available information.
    """
    # We defer the import until here because we need the objects, but we don't want to explicitly
    # depend on the params module (so that we can import the labels into the params module).
    from jet_hadron.base import params  # noqa: F811

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

    system_label = r"$%(system)s\:%(energy)s%(event_activity)s$" % {
        "system": system.display_str(),
        "energy": energy.display_str(),
        "event_activity": event_activity_str,
    }

    logger.debug(f"system_label: {system_label}")

    return system_label

