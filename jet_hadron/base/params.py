#!/usr/bin/env python

""" Jet-Hadron analysis parameters.

Also contains methods to access that information.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

from dataclasses import dataclass
import enum
import logging
import numpy as np
import re
from typing import Any, cast, Dict, Iterable, Iterator, Optional, Sequence, Tuple, TYPE_CHECKING, Union

from pachyderm import generic_class
from pachyderm import yaml

from jet_hadron.base import labels

if TYPE_CHECKING:
    from jet_hadron.base import analysis_objects  # noqa: F401

logger = logging.getLogger(__name__)

# Typing helpers
PtBinIteratorConfig = Optional[Dict[str, Dict[str, Iterable[float]]]]

########
# Utility functions
########
def iterate_over_pt_bins(name: str,
                         bins: Sequence["analysis_objects.PtBin"],
                         config: PtBinIteratorConfig = None) -> Iterable["analysis_objects.PtBin"]:
    """ Create a generator of the bins in a requested list.

    Bin skipping should be specified as:

    .. code-block:: python

        >>> config = {
        ...     "skipPtBins" : {
        ...         "name" : [bin1, bin2]
        ...     }
        ... }

    Args:
        name: Name of the skip bin entries in the config.
        bins: Bin edges for determining the bin indices.
        config: Containing information regarding bins to skip, as specified above.
    Returns:
        pt bins generated according to the given arguments.
    """
    # Create a default dict if none is available
    if not config:
        config = {}
    skip_pt_bins = config.get("skipPtBins", {}).get(name, [])
    # Sanity check on skip pt bins
    for val in skip_pt_bins:
        if val == 0:
            raise ValueError(val, f"Invalid bin 0! Bin counting starts at 1.")
        if val > len(bins):
            raise ValueError(val, f"Pt bin to skip {val} is outside the range of the {name} list")

    for pt_bin in bins:
        if pt_bin.bin in skip_pt_bins:
            continue

        yield pt_bin

def iterate_over_jet_pt_bins(bins: Sequence["analysis_objects.JetPtBin"],
                             config: PtBinIteratorConfig = None) -> Iterable["analysis_objects.PtBin"]:
    """ Iterate over the available jet pt bins. """
    return iterate_over_pt_bins(config = config, name = "jet", bins = bins)

def iterate_over_track_pt_bins(bins: Sequence["analysis_objects.TrackPtBin"],
                               config: PtBinIteratorConfig = None) -> Iterable["analysis_objects.PtBin"]:
    """ Iterate over the available track pt bins. """
    return iterate_over_pt_bins(config = config, name = "track", bins = bins)

def iterate_over_jet_and_track_pt_bins(
        jet_pt_bins: Sequence["analysis_objects.JetPtBin"],
        track_pt_bins: Sequence["analysis_objects.TrackPtBin"],
        config: PtBinIteratorConfig = None) -> Iterable[Tuple["analysis_objects.PtBin", "analysis_objects.PtBin"]]:
    """ Iterate over all possible combinations of jet and track pt bins. """
    for jet_pt_bin in iterate_over_jet_pt_bins(bins = jet_pt_bins, config = config):
        for track_pt_bin in iterate_over_track_pt_bins(bins = track_pt_bins, config = config):
            yield (jet_pt_bin, track_pt_bin)

def _uppercase_first_letter(s: str) -> str:
    """ Convert the first letter to uppercase without affecting the rest of the string.

    Note:
        Cannot use `str.capitalize()` or `str.title()` because they lowercase the rest of the string.

    Args:
        s: String to be convert
    Returns:
        String with first letter converted to uppercase.
    """
    return s[:1].upper() + s[1:]

############################################
# Parameter information (access and display)
############################################
class AliceLabel(enum.Enum):
    """ ALICE label types. """
    work_in_progress = "ALICE Work in Progress"
    preliminary = "ALICE Preliminary"
    final = "ALICE"
    thesis = "This thesis"

    def __str__(self) -> str:
        """ Return the value. This is just a convenience function.

        Note:
            This is backwards of the usual convention of returning the name, but the value is
            more meaningful here. The name can always be accessed with ``.name``.
        """
        return str(self.value)

    def display_str(self) -> str:
        """ Return a formatted string for display in plots, etc. Includes latex formatting. """
        # Ensure that the spacing in the words is carried over in the LaTeX
        val = self.value.replace(" ", r"\;")
        return rf"\mathrm{{{val}}}"

    # Handle YAML serialization
    to_yaml = classmethod(yaml.enum_to_yaml)

    @classmethod
    def from_yaml(cls, constructor: yaml.Constructor, node: yaml.ruamel.yaml.nodes.ScalarNode) -> "AliceLabel":
        """ Decode YAML representer. """
        return cls(node.value)

##################
# Analysis Options
##################
# These options specify the base of what is necessary to
# define an analysis.
##################

#########################
## Helpers and containers
#########################
@dataclass(frozen = True)
class SelectedRange:
    """ Helper for selected ranges. """
    min: float
    max: float

    def __iter__(self) -> Iterator[Tuple[str, float]]:
        for k, v in vars(self).items():
            yield k, v

    @classmethod
    def from_yaml(cls, constructor: yaml.Constructor,
                  data: Union[yaml.ruamel.yaml.nodes.MappingNode, yaml.ruamel.yaml.nodes.SequenceNode]) -> "SelectedRange":
        """ Decode YAML representer.

        Expected block is of the form:

        .. code-block:: yaml

            val: !SelectedRange [1, 5]

        or alternatively (which will be used when YAML is dumping the object):

        .. code-block:: yaml

            val: !SelectedRange
                min: 1
                max: 5

        which will yield:

        .. code-block:: python

            >>> val == SelectedRange(min = 1, max = 5)
        """
        # We've just passed a list, so just reconstruct it assuming that the arguments are in order.
        # This is usually used when manually defined by a user in the configuration file.
        if isinstance(data, yaml.ruamel.yaml.nodes.SequenceNode):
            values = [constructor.construct_object(v) for v in data.value]
            return cls(*values)

        # Otherwise, we should have received a MappingNode. This is usually used when YAML has
        # written the object. In this case, we convert the ``MappingNode`` into a dictionary of
        # arguments to the object.
        # NOTE: Just calling ``dict(...)`` would not be sufficient because the nodes wouldn't be converted
        arguments = {
            constructor.construct_object(key_node): constructor.construct_object(value_node)
            for key_node, value_node in data.value
        }
        return cls(**arguments)

@dataclass(frozen = True)
class ReactionPlaneBinInformation:
    """ Helper for storing reaction plane bin information.

    Attributes:
        bin: Bin corresponding to the selected reaction plane.
        center: Center of the bin. Known as phi_s in RPF expressions.
        width: Width from the center of the bin to the edge. Known as c in RPF expressions.
    """
    bin: int
    center: float
    width: float

#########
# Classes
#########
class CollisionEnergy(enum.Enum):
    """ Define the available collision system energies.

    Defined in TeV.
    """
    two_seven_six = 2.76
    five_zero_two = 5.02
    zero_point_two = 0.2

    def __str__(self) -> str:
        """ Returns a string of the value. """
        return str(self.value)

    def display_str(self) -> str:
        """ Return a formatted string for display in plots, etc. Includes latex formatting. """
        return r"\sqrt{s_{\mathrm{NN}}} = %(energy)s\:\mathrm{TeV}" % {"energy": self.value}

    # Handle YAML serialization
    to_yaml = classmethod(yaml.enum_to_yaml)

    @classmethod
    def from_yaml(cls, constructor: yaml.Constructor, data: yaml.ruamel.yaml.nodes.SequenceNode) -> "CollisionEnergy":
        """ Decode YAML representer. """
        return cls(float(data.value))

class CollisionSystem(enum.Enum):
    """ Define the collision system """
    NA = "Invalid collision system"
    pp = "pp"
    pythia = "PYTHIA"
    embedPP = r"pp \bigotimes %(embedded_additional_label)s Pb \textendash Pb"
    embedPythia = r"PYTHIA \bigotimes %(embedded_additional_label)s Pb \textendash Pb"
    pPb = r"pPb"
    PbPb = r"Pb \textendash Pb"
    AuAu = r"Au \textendash Au"

    def __str__(self) -> str:
        """ Return a string of the name of the system. """
        return self.name

    def display_str(self, embedded_additional_label: str = "") -> str:
        """ Return a formatted string for display in plots, etc. Includes latex formatting.

        Args:
            embedded_additional_label: Additional label for the embedded systems. The label
                is placed between the cross symbol and the Pb--Pb label. Default: "".
        Returns:
            Properly formatted string.
        """
        # Add some additional spacing after the label if it's been specified.
        if embedded_additional_label and not embedded_additional_label.endswith(r"\:"):
            embedded_additional_label = embedded_additional_label + r"\:"
        return rf"\mathrm{{{self.value}}}" % {"embedded_additional_label": embedded_additional_label}

    # Handle YAML serialization
    to_yaml = classmethod(yaml.enum_to_yaml)
    from_yaml = classmethod(yaml.enum_from_yaml)

class EventActivity(enum.Enum):
    """ Define the event activity.

    Values are ranges of the centrality bin, where -1 is defined as the full range!
    """
    inclusive = SelectedRange(min = -1, max = -1)
    central = SelectedRange(min = 0, max = 10)
    semi_central = SelectedRange(min = 30, max = 50)

    @property
    def value_range(self) -> SelectedRange:
        """ Return the event activity range.

        Returns:
            The min and max of the event activity range.
        """
        # Help out mypy...
        return cast(SelectedRange, self.value)

    def __str__(self) -> str:
        """ Name of the event activity range. """
        return str(self.name)

    def display_str(self) -> str:
        """ Get the event activity range as a formatted string. Includes latex formatting. """
        ret_val = ""
        # For inclusive, we want to return an empty string.
        if self != EventActivity.inclusive:
            ret_val = r"%(min)s \textendash %(max)s \%%" % dict(self.value_range)
        return ret_val

    # Handle YAML serialization
    to_yaml = classmethod(yaml.enum_to_yaml)
    from_yaml = classmethod(yaml.enum_from_yaml)

class LeadingHadronBiasType(enum.Enum):
    """ Leading hadron bias type """
    NA = -1
    track = 0
    cluster = 1
    both = 2

    def __str__(self) -> str:
        """ Return the name of the bias. It must be just the name for the config override to work properly. """
        return self.name

    # Handle YAML serialization
    to_yaml = classmethod(yaml.enum_to_yaml)
    from_yaml = classmethod(yaml.enum_from_yaml)

########################
# Final analysis options
########################
# These classes are used for final analysis specification, building
# on the analysis specification objects specified above.
########################
class LeadingHadronBias(generic_class.EqualityMixin):
    """ Full leading hadron bias class, which specifies both the type as well as the value.

    The class exists to be specified when creating an analysis object, and then the value is
    determined by the selected analysis options (including that enum). This object then
    supersedes the ``leadingHadronBiasType`` enum, storing both the type and value.

    For determining the actual value, see anaylsisConfig.determineLeadingHadronBias(...)

    Args:
        type: Type of leading hadron bias.
        value: Value of the leading hadron bias.
    """
    def __init__(self, type: LeadingHadronBiasType, value: float):
        self.type = type
        # If the leadingHadronBias is disabled, then the value is irrelevant and should be set to 0.
        if self.type == LeadingHadronBiasType.NA:
            value = 0
        self.value = value

    def __str__(self) -> str:
        """ Return a string representation.

        Return the type and value, such as "clusterBias6" or "trackBias5". In the case of the bias
        as NA, it simply returns "NA".
        """
        if self.type != LeadingHadronBiasType.NA:
            return f"{self.type}Bias{self.value}"
        else:
            return f"{self.type}"

    def display_str(self, additional_label: str = "") -> str:
        """ A formatted string for display in plots, etc. Includes latex formatting.

        Args:
            additional_label: Optional additional superscript label.
        """
        # Validation
        # Add a comma in between the labels if the additional label is not empty.
        if additional_label and not additional_label.startswith(","):
            additional_label = "," + additional_label

        track_label = labels.pt_display_label(upper_label = r"lead\:track" + additional_label)
        cluster_label = labels.et_display_label(upper_label = r"lead\:clus" + additional_label)
        gev_value_label = r"> %(value)s\:\mathrm{GeV}"

        if self.type == LeadingHadronBiasType.NA:
            return ""
        elif self.type == LeadingHadronBiasType.track:
            # Need to return GeV/c
            return f"{track_label} {gev_value_label}" % {
                "value": self.value,
            } + r"/\mathit{c}"
        elif self.type == LeadingHadronBiasType.cluster:
            return f"{cluster_label} {gev_value_label}" % {
                "value": self.value
            }
        elif self.type == LeadingHadronBiasType.both:
            # Need to have the same units, so we multiply the track pt term by c
            return fr"{track_label}\mathit{{c}}\mathrm{{,}}\:{cluster_label} {gev_value_label}" % {
                "value": self.value
            }

        raise NotImplementedError(f"Display string for leading hadron bias {self} is not implemented.")

@dataclass
class SelectedAnalysisOptions:
    collision_energy: CollisionEnergy
    collision_system: CollisionSystem
    event_activity: EventActivity
    leading_hadron_bias: Union[LeadingHadronBias, LeadingHadronBiasType]

    def astuple(self) -> Tuple[Any, ...]:
        """ Tuple of the selected analysis option values. """
        return tuple(dict(self).values())

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        for k, v in vars(self).items():
            yield k, v

# For use with overriding configuration values
SetOfPossibleOptions = SelectedAnalysisOptions(CollisionEnergy,  # type: ignore
                                               CollisionSystem,
                                               EventActivity,
                                               LeadingHadronBiasType)

##############################
# Additional selection options
##############################
# These are distinct from the above because they do not need to be used
# to specify a configuration. Thus, they don't need to be looped over.
# Instead, they are stored in a particular analysis object and used as
# analysis options.
##############################
class ReactionPlaneOrientation(enum.Enum):
    """ Selects the event plane angle in the sparse. """
    inclusive = ReactionPlaneBinInformation(bin = 0, center = -1, width = -1)
    in_plane = ReactionPlaneBinInformation(bin = 1, center = 0, width = np.pi / 6)
    mid_plane = ReactionPlaneBinInformation(bin = 2, center = np.pi / 4, width = np.pi / 12)
    out_of_plane = ReactionPlaneBinInformation(bin = 3, center = np.pi / 2, width = np.pi / 6)

    def __str__(self) -> str:
        """ Returns the event plane angle name, as is. """
        return self.name

    def display_str(self) -> str:
        """ For example, turns out_of_plane into "Out-of-plane".

        Note:
            We want the capitalize call to lowercase all other letters.
        """
        return str(self).replace("_", "-").capitalize()

    # Handle YAML serialization
    to_yaml = classmethod(yaml.enum_to_yaml)
    from_yaml = classmethod(yaml.enum_from_yaml)

class QVector(enum.Enum):
    """ Selection based on the Q vector. """
    inclusive = SelectedRange(min = 0, max = 100)
    bottom10 = SelectedRange(min = 0, max = 10)
    top10 = SelectedRange(min = 90, max = 100)

    @property
    def value_range(self) -> SelectedRange:
        """ Return the Q vector range.

        Returns:
            The min and max of the event activity range.
        """
        # Help out mypy...
        return cast(SelectedRange, self.value)

    def __str__(self) -> str:
        """ Returns the name of the selection range. """
        return self.name

    def display_str(self) -> str:
        """ Turns "bottom10" into "Bottom 10%". """
        # This also works for "inclusive" -> "Inclusive"
        match = re.match("([a-z]*)([0-9]*)", self.name)
        if not match:
            raise ValueError("Could not extract Q Vector value \"{self.name}\" for printing.")
        temp_list = match.groups()
        ret_val = _uppercase_first_letter(" ".join(temp_list))
        if self.name != "inclusive":
            ret_val += "%"
        # rstrip() is to remove the extra space after "Inclusive". Doesn't matter for the other values.
        return ret_val.rstrip(" ")

    # Handle YAML serialization
    to_yaml = classmethod(yaml.enum_to_yaml)
    from_yaml = classmethod(yaml.enum_from_yaml)

