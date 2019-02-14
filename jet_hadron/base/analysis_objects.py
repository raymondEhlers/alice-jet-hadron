#!/usr/bin/env python

""" Analysis objects for the jet-hadron anaylsis

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

from abc import ABC
from dataclasses import dataclass
import enum
import logging
import numpy as np
import re
from typing import Any, Dict, List, Mapping, Optional, Type, Union

from pachyderm import generic_class
from pachyderm import histogram
from pachyderm import projectors
from pachyderm import yaml

from reaction_plane_fit.base import FitResult

from jet_hadron.base import params
from jet_hadron.base.typing_helpers import Hist

# Setup logger
logger = logging.getLogger(__name__)

class CorrelationType(enum.Enum):
    """ 1D correlation projection type """
    full_range = 0
    # delta phi specialized
    signal_dominated = 1
    background_dominated = 2
    # delta eta specialized
    near_side = 3
    away_side = 4

    def __str__(self) -> str:
        """ Returns the name of the correlation type. """
        return self.name

    def display_str(self) -> str:
        """ Turns "signal_dominated" into "Signal Dominated". """
        # Convert to display name by splitting on camel case
        return self.name.replace("_", " ").title()

    # Handle YAML serialization
    to_yaml = classmethod(yaml.enum_to_yaml)
    from_yaml = classmethod(yaml.enum_from_yaml)

class CorrelationAxis(enum.Enum):
    """ Define the axes of Jet-H 2D correlation hists. """
    delta_phi = projectors.TH1AxisType.x_axis.value
    delta_eta = projectors.TH1AxisType.y_axis.value

    def __str__(self) -> str:
        """ Return the name of the correlations axis. """
        return self.name

    def display_str(self) -> str:
        """ Return the name of the correlation axis for display using latex. """
        angle = self.name.split("_")[1]
        if angle == "phi":
            angle = "var" + angle

        return fr"\Delta\{angle}"

    # Handle YAML serialization
    to_yaml = classmethod(yaml.enum_to_yaml)
    from_yaml = classmethod(yaml.enum_from_yaml)

####################
# Basic data classes
####################
@dataclass
class HistogramInformation:
    """ Helper class to store information about processing an hist in an analysis object.

    This basically just stores information in a nicely formatted and clear manner.

    Attributes:
        description: Description of the histogram.
        attribute_name: Name of the attribute under which the hist is stored in the analysis object.
        hist_name: Histogram safe name derived from the attribute name.
    """
    description: str
    attribute_name: str

    @property
    def hist_name(self) -> str:
        return self.attribute_name.replace(".", "_")

@dataclass
class Observable:
    """ Base observable object.

    Basically, it wraps the histogram so that we can have a consistent interface for a variety
    of observables.
    """
    hist: Hist

    @property
    def name(self) -> str:
        return f"hist_{self.hist.GetName()}"

@dataclass
class CorrelationObservable(Observable):
    """ For correlation observables. """
    type: Union[str, enum.Enum]
    axis: Union[str, enum.Enum]
    analysis_identifier: Optional[str] = None

    @property
    def name(self) -> str:
        return f"jetH_{self.axis}_{self.analysis_identifier}_{self.type}"

@dataclass
class ExtractObservable:
    """ For extracted observable such as widths or yields. """
    value: float
    error: float

@dataclass
class Fit:
    """ For fit results. """
    type: CorrelationType
    axis: CorrelationAxis
    fit: FitResult

@dataclass
class PlottingOutputWrapper:
    """ Simple wrapper to allow use of the ``jet_hadron.plot.base.save_plot`` convenience function.

    Attributes:
        output_prefix: File path to where files should be saved.
        printing_extensions: List of file extensions under which plots should be saved.
    """
    output_prefix: str
    printing_extensions: List[str]

#######################
# Main analysis classes
#######################
class JetHBase(generic_class.EqualityMixin):
    """ Base class for shared jet-hadron configuration values.

    Args:
        task_name (str): Name of the task.
        config_filename (str): Filename of the YAML configuration.
        config (dict-like object): Contains the analysis configuration. Note that it must already be
            fully configured and overridden.
        task_config (dict-like object): Contains the task specific configuration. Note that it must already be
            fully configured and overridden. Also note that by convention it is also available at
            ``config[task_name]``.
        collision_energy (params.CollisionEnergy): Selected collision energy.
        collision_system (params.CollisionSystem): Selected collision system.
        event_activity (params.EventActivity): Selected event activity.
        leading_hadron_bias (params.LeadingHadronBias or params.LeadingHadronBiasType): Selected leading hadron
            bias. The class member will contain both the type and the value.
        args (list): Absorb extra arguments. They will be ignored.
        kwargs (dict): Absorb extra named arguments. They will be ignored.
    """
    def __init__(self,
                 task_name: str, config_filename: str,
                 config: Mapping, task_config: Mapping,
                 collision_energy: params.CollisionEnergy,
                 collision_system: params.CollisionSystem,
                 event_activity: params.EventActivity,
                 leading_hadron_bias: Union[params.LeadingHadronBias, params.LeadingHadronBiasType],
                 *args, **kwargs):
        # Store the configuration
        self.task_name = task_name
        self.config_filename = config_filename
        self.config = config
        self.task_config = task_config
        self.collision_energy = collision_energy
        self.collision_system = collision_system
        self.event_activity = event_activity

        # Handle leading hadron bias depending on the type.
        if not isinstance(leading_hadron_bias, params.LeadingHadronBiasType):
            leading_hadron_bias = leading_hadron_bias.type
        self._leading_hadron_bias_type = leading_hadron_bias
        self._leading_hadron_bias = None

        # File I/O
        # If in kwargs, use that value (which inherited class may use to override the config)
        # otherwise, use the value from the value from the config
        self.input_filename = config["inputFilename"]
        self.input_list_name = config["inputListName"]
        self.output_info = PlottingOutputWrapper(
            output_prefix = config["outputPrefix"],
            printing_extensions = config["printingExtensions"],
        )
        self.output_prefix = config["outputPrefix"]
        self.output_filename = config["outputFilename"]

        # Convert the ALICE label if necessary
        alice_label = config["aliceLabel"]
        self.alice_label = params.AliceLabel[alice_label]
        self.train_number = config["trainNumber"]

    @property
    def output_prefix(self) -> str:
        return self.output_info.output_prefix

    @output_prefix.setter
    def output_prefix(self, val: str) -> None:
        self.output_info.output_prefix = val

    @property
    def printing_extensions(self) -> List[str]:
        return self.output_info.printing_extensions

    @printing_extensions.setter
    def printing_extensions(self, val: List[str]) -> None:
        self.output_info.printing_extensions = val

    @property
    def leading_hadron_bias(self) -> params.LeadingHadronBias:
        # Only calculate the value if we haven't already used it.
        if self._leading_hadron_bias is None:
            # Load this module only if necessary. I'm not moving this function because it makes dependences much messier.
            from jet_hadron.base import analysis_config

            self._leading_hadron_bias = analysis_config.determine_leading_hadron_bias(  # type: ignore
                config = self.config,
                selected_analysis_options = params.SelectedAnalysisOptions(
                    collision_energy = self.collision_energy,
                    collision_system = self.collision_system,
                    event_activity = self.event_activity,
                    leading_hadron_bias = self._leading_hadron_bias_type)
            ).leading_hadron_bias
        # Help out mypy.
        assert self._leading_hadron_bias is not None

        return self._leading_hadron_bias

class JetHBinnedAnalysis(JetHBase):
    """ Jet-hadron analysis object which includes basic binned quantities.

    We store these values separately from the track and jet pt because the eta and phi binning
    is usually constant. In contrast, we usually iterate over the track and jet pt values.

    Attributes:
        eta_bins: List of eta bins.
        phi_bins: List of phi bins.
    """
    def __init__(self, *args, **kwargs):
        # First initialize the base class.
        super().__init__(*args, **kwargs)

        # Additional binning information
        self.eta_bins: List[EtaBin] = self.config["etaBins"]
        self.phi_bins: List[PhiBin] = self.config["phiBins"]

class JetHReactionPlane(JetHBinnedAnalysis):
    """ Jet-hadron analysis object which includes reaction plane dependence.

    Args:
        reaction_plane_orientation (params.ReactionPlaneOrientation): Selected reaction plane angle.
    Attributes:
        reaction_plane_orientation (params.ReactionPlaneOrientation): Selected reaction plane angle.
    """
    def __init__(self,
                 reaction_plane_orientation: params.ReactionPlaneOrientation,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reaction_plane_orientation = reaction_plane_orientation

    def _retrieve_histograms(self, input_hists: Dict[str, Any] = None) -> bool:
        """ Retrieve histograms from a ROOT file.

        Args:
            input_hists: All histograms in a file. Default: None - They will be retrieved.
        Returns:
            bool: True if histograms were retrieved successfully.
        """
        logger.debug(f"input_filename: {self.input_filename}")
        if input_hists is None:
            temp_input_hists = histogram.get_histograms_in_list(
                filename = self.input_filename,
                input_list = self.input_list_name
            )
        else:
            try:
                temp_input_hists = input_hists[self.input_list_name]
            except KeyError as e:
                raise KeyError(f"Available lists: {input_hists}") from e
        self.input_hists = temp_input_hists

        return len(self.input_hists) > 0

    def _setup_projectors(self) -> None:
        """ Setup projectors needed for the analysis. """
        raise NotImplementedError("Must implement the projectors setup in the derived class.")

    def setup(self, input_hists: Dict[str, Any] = None) -> bool:
        """ Setup the jet-hadron analysis object.

        This will handle retrieving histograms and setup the projectors, but further setup is dependent
        on the particular analysis.

        Args:
            input_hists: All histograms in a file. Default: None - In that case, they will be
                retrieved automatically. This optional argument is provided to enable caching
                the open file.
        Returns:
            bool: True if analysis was successfully setup.
        Raises:
            ValueError: If the histograms could not be retrieved.
        """
        result = self._retrieve_histograms(input_hists = input_hists)
        if result is not True:
            raise ValueError("Could not retrieve histograms.")

        self._setup_projectors()

        return True

###################################
# Iterables for binning (with YAML)
###################################
@dataclass(frozen = True)
class AnalysisBin(ABC):
    """ Represent a binned quantity.

    Attributes:
        range: Min and maximum of the bin.
        name: Name of the analysis bin (based on the class name).
    """
    range: params.SelectedRange

    def __str__(self) -> str:
        return str(f"{self.name} Range: ({self.range.min}, {self.range.max})")

    @property
    def name(self) -> str:
        """ Convert class name into capital case. For example: 'JetPtBin' -> 'Jet Pt Bin'. """
        return re.sub("([a-z])([A-Z])", r"\1 \2", self.__class__.__name__)

@dataclass(frozen = True)
class EtaBin(AnalysisBin):
    """ A eta bin, along with the associated eta range.

    We don't need to implement anything else. We just needed to instantiate this with the name
    of the class so that we can differentiate it from other bins.
    """
    ...

@dataclass(frozen = True)
class PhiBin(AnalysisBin):
    """ A phi bin, along with the associated phi range.

    We don't need to implement anything else. We just needed to instantiate this with the name
    of the class so that we can differentiate it from other bins.
    """
    ...

@dataclass(frozen = True)
class PtBin(AnalysisBin, ABC):
    """ Represents a pt bin.

    This object includes a bin number because pt bins have a natural bin ordering that isn't necessarily
    meaningful for other binned quantities.

    Attributes:
        range: Min and maximum of the bin.
        bin: Pt bin. This is meaningful for pt bins because they are ordered.
        name: Name of the pt bin (based on the class name).
    """
    bin: int

    def __str__(self) -> str:
        """ Redefine the string to return the bin number. """
        return str(self.bin)

@dataclass(frozen = True)
class TrackPtBin(PtBin):
    """ A track pt bin, along with the associated track pt range.

    We don't need to implement anything else. We just needed to instantiate this with the name
    of the class so that we can differentiate it from other bins.
    """
    ...

@dataclass(frozen = True)
class JetPtBin(PtBin):
    """ A jet pt bin, along with the associated jet pt range.

    We don't need to implement anything else. We just needed to instantiate this with the name
    of the class so that we can differentiate it from other bins.
    """
    ...

@dataclass(frozen = True)
class PtHardBin(PtBin):
    """ A pt hard bin, along with the train number associated with the bin and the range.

    We don't need to implement anything else. We just needed to instantiate this with the name
    of the class so that we can differentiate it from other bins.
    """
    train_number: int

class AnalysisBins(ABC):
    """ Define an array of analysis bins.

    As an example, consider the example of ``EtaBins``.

    .. code-block:: yaml

        - eta_bins: !EtaBins [0, 0.4, 0.6]

    yields

    .. code-block:: python

        >>> eta_bins == [
        ...     EtaBin(range = params.SelectedRange(0., 0.4)),
        ...     EtaBin(range = params.SelectedRange(0.4, 0.6)),
        ... ]

    Note:
        This is just convenience function for YAML. It isn't round-trip because we would never use write back out.
        This just allow us to define the bins in a compact manner when we write YAML.
    """
    _class: Type[AnalysisBin]

    @classmethod
    def from_yaml(cls, constructor: yaml.Constructor, data: yaml.ruamel.yaml.nodes.SequenceNode) -> List[AnalysisBin]:
        """ Convert input YAML list to set of ``AnalysisBin``. """
        #logger.debug(f"Using representer, {data}")
        values = [constructor.construct_object(v) for v in data.value]
        bins = []
        for val, val_next in zip(values[:-1], values[1:]):
            bins.append(cls._class(range = params.SelectedRange(min = val, max = val_next)))
        return bins

class EtaBins(AnalysisBins):
    """ Define an array of eta bins.

    It reads arrays registered under the tag ``!EtaBins``.
    """
    _class = EtaBin

class PhiBins:
    """ Define an array of eta bins.

    It reads arrays registered under the tag ``!PhiBins``.
    """
    _class = PhiBin

    @classmethod
    def from_yaml(cls, constructor: yaml.Constructor, data: yaml.ruamel.yaml.nodes.SequenceNode) -> List[PhiBin]:
        """ Convert input YAML list to set of ``PtBin``. """
        #logger.debug(f"Using representer, {data}")
        # Extract values
        values = [constructor.construct_object(v) for v in data.value]
        # Scale very thing by a factor of pi for convenience.
        values = [v * np.pi for v in values]
        bins = []
        for val, val_next in zip(values[:-1], values[1:]):
            bins.append(cls._class(range = params.SelectedRange(min = val, max = val_next)))
        return bins

class PtBins(ABC):
    """ Define an array of pt bins.

    As an example, consider the example of ``JetPtBins``.

    .. code-block:: yaml

        - pt_bins: !JetPtBins [5, 11, 21]

    yields

    .. code-block:: python

        >>> pt_bins = [
        ...     JetPtBin(range = (5, 11), bin = 1),
        ...     JetPtBin(range = (11, 21), bin = 2),
        ... ]

    Note:
        This is just convenience function for YAML. It isn't round-trip because we would never use write back out.
        This just allow us to define the bins in a compact manner when we write YAML.
    """
    _class: Type[PtBin]

    @classmethod
    def from_yaml(cls, constructor: yaml.Constructor, data: yaml.ruamel.yaml.nodes.SequenceNode) -> List[PtBin]:
        """ Convert input YAML list to set of ``AnalysisBin``. """
        #logger.debug(f"Using representer, {data}")
        values = [constructor.construct_object(v) for v in data.value]
        bins = []
        for i, (val, val_next) in enumerate(zip(values[:-1], values[1:])):
            bins.append(cls._class(range = params.SelectedRange(min = val, max = val_next), bin = i + 1))
        return bins

class TrackPtBins(PtBins):
    """ Define an array of track pt bins.

    It reads arrays registered under the tag ``!TrackPtBins``.
    """
    _class = TrackPtBin

class JetPtBins(PtBins):
    """ Define an array of jet pt bins.

    It reads arrays registered under the tag ``!JetPtBins``.
    """
    _class = JetPtBin

class PtHardBins:
    """ Define an array of pt hard bins with corresponding pt ranges and train numbers.

    It reads objects registered under the tag ``!PtHardBins``. Loading

    .. code-block:: yaml

        - pt_hard_bin: !PtHardBins
                bins: [5, 11, 21]
                train_numbers:
                    1: 2701
                    2: 2710

    yields

    .. code-block:: python

        >>> pt_hard_bin = [
        ...     PtHardBin(bin = 1, range = SelectedRange(min=5, max=11), train_number=2701),
        ...     PtHardBin(bin = 2, range = SelectedRange(min=11, max=21), train_number=2702),
        ... ]

    Note:
        This is just convenience function for YAML. It isn't round-trip because we would never use write back out.
        This just allow us to define the bins in a compact manner when we write YAML.
    """
    @classmethod
    def from_yaml(cls, constructor: yaml.Constructor, data: yaml.ruamel.yaml.nodes.MappingNode) -> List[PtHardBin]:
        """ Convert input YAML list to set of ``PtHardBin``. """
        # Construct the underlying list and dict to make parsing simpler.
        configuration = {constructor.construct_object(key_node): constructor.construct_object(value_node) for key_node, value_node in data.value}
        # Extract the relevant data
        bins = configuration["bins"]
        train_numbers = configuration["train_numbers"]
        # Create the PtHardBin objects.
        pt_bins = []
        for i, (pt, pt_next) in enumerate(zip(bins[:-1], bins[1:])):
            pt_bins.append(
                PtHardBin(
                    bin = i + 1,
                    range = params.SelectedRange(min = pt, max = pt_next),
                    train_number = train_numbers[i + 1],
                )
            )
        return pt_bins

