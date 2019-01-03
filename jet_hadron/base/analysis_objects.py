#!/usr/bin/env python

""" Utilities for the jet-hadron anaylsis

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

from abc import ABC
from dataclasses import dataclass
import enum
import copy
import logging
import numpy as np
import os
import re
from typing import Any, List, Mapping, Union

from pachyderm import generic_class
from pachyderm import histogram
from pachyderm import utils
from pachyderm import yaml

from jet_hadron.base import params

# Setup logger
logger = logging.getLogger(__name__)

class JetHCorrelationType(enum.Enum):
    """ 1D correlation projection type """
    full_range = 0
    # dPhi specialized
    signal_dominated = 1
    background_dominated = 2
    # dEta specialized
    near_side = 3
    away_side = 4

    def __str__(self) -> str:
        """ Returns the name of the correlation type. """
        return self.name

    def display_str(self) -> str:
        """ Turns "signal_dominated" into "Signal Dominated". """
        # Convert to display name by splitting on camel case
        # For the regex, see: https://stackoverflow.com/a/43898219
        #split_string = re.sub('([a-z])([A-Z])', r'\1 \2', self.name)
        # Capitalize the first letter of every word
        #return split_string.title()
        return self.name.replace("_", " ").title()

    # Handle YAML serialization
    to_yaml = classmethod(yaml.enum_to_yaml)
    from_yaml = classmethod(yaml.enum_from_yaml)

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
        reaction_plane_orientation (params.ReactionPlaneOrientation): Selected event plane angle.
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
        self.output_prefix = config["outputPrefix"]
        self.output_filename = config["outputFilename"]
        # Setup output area
        if not os.path.exists(self.output_prefix):
            os.makedirs(self.output_prefix)

        self.printing_extensions = config["printingExtensions"]
        # Convert the ALICE label if necessary
        alice_label = config["aliceLabel"]
        self.alice_label = params.AliceLabel[alice_label]
        self.train_number = config["trainNumber"]

    @property
    def leading_hadron_bias(self):
        # Only calculate the value if we haven't already used it.
        if self._leading_hadron_bias is None:
            # Load this module only if necessary. I'm not moving this function because it makes dependences much messier.
            from jet_hadron.base import analysis_config

            self._leading_hadron_bias = analysis_config.determine_leading_hadron_bias(
                config = self.config,
                selected_analysis_options = params.SelectedAnalysisOptions(
                    collision_energy = self.collision_energy,
                    collision_system = self.collision_system,
                    event_activity = self.event_activity,
                    leading_hadron_bias = self._leading_hadron_bias_type)
            ).leading_hadron_bias
        return self._leading_hadron_bias

class JetHReactionPlane(JetHBase):
    def __init__(self,
                 reaction_plane_orientation: params.ReactionPlaneOrientation,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reaction_plane_orientation = reaction_plane_orientation

class Observable(object):
    """ Base observable object. Intended to store a HistContainer.

    Args:
        hist (HistContainer): The hist we are interested in.
    """
    def __init__(self, hist = None):
        self.hist = hist

class CorrelationObservable(Observable):
    """ General correlation observable object. Usually used for 2D correlations.

    Args:
        jet_pt_bin (int): Bin of the jet pt of the observable.
        track_pt_bin (int): Bin of the track pt of the observable.
        hist (HistContainer): Associated histogram of the observable. Optional.
    """
    def __init__(self, jet_pt_bin, track_pt_bin, *args, **kwargs):
        """ Initialize the observable """
        super().__init__(*args, **kwargs)

        self.jet_pt_bin = jet_pt_bin
        self.track_pt_bin = track_pt_bin

class CorrelationObservable1D(CorrelationObservable):
    """ For 1D correlation observable object. Can be either dPhi or dEta.

    Args:
        axis (JetHCorrelationAxis): Axis of the 1D observable.
        correlation_type (JetHCorrelationType): Type of the 1D observable.
        jet_pt_bin (int): Bin of the jet pt of the observable.
        track_pt_bin (int): Bin of the track pt of the observable.
        hist (HistContainer): Associated histogram of the observable. Optional.
    """
    def __init__(self, axis, correlation_type, *args, **kwargs):
        # Initialize the base class
        super().__init__(*args, **kwargs)

        self.axis = axis
        self.correlation_type = correlation_type

class ExtractedObservable(object):
    """ For extracted observable such as widths or yields.

   Args:
        jet_pt_bin (int): Bin of the jet pt of the observable.
        track_pt_bin (int): Bin of the track pt of the observable.
        value (number): Extracted value.
        error (number): Error associated with the extracted value.
    """
    def __init__(self, jet_pt_bin, track_pt_bin, value, error):
        self.jet_pt_bin = jet_pt_bin
        self.track_pt_bin = track_pt_bin
        self.value = value
        self.error = error

class HistContainer(object):
    """ Histogram container to allow for normal function access except for those that we choose to overload.

    Args:
        hist (ROOT.TH1, ROOT.THnBase, or similar): Histogram to be stored in the histogram container.
    """
    def __init__(self, hist):
        self.hist = hist

    # Inspired by: https://stackoverflow.com/q/14612442
    def __getattr__(self, attr):
        """ Forwards all requested functions or attributes to the histogram.

        However, other functions or attributes defined in this class are still accessible! This function
        is usually called implicitly by calling the attribute on the HistContainer.

        Args:
            attr (str): Desired attribute of the stored histogram.
        Returns:
            property or function: Requested attrbiute of the stored hist.
        """
        # Uncomment for debugging, as this is far too verbose otherwise
        #logger.debug("attr: {0}".format(attr))

        # Execute the attribute on the hist
        return getattr(self.hist, attr)

    def create_scaled_by_bin_width_hist(self, additional_scale_factor: float = 1.0) -> Any:
        """ Create a new histogram scaled by the bin width. The return histogram is a clone
        of the histogram inside of the container with the scale factor(s) applied.

        One can always assign the result to replace the existing hist (although take care to
        avoid memory leaks!).

        Args:
            additional_scale_factor: Additional scale factor to apply to the scaled hist. Default: 1.0
        Returns:
            ROOT.TH1: Cloned histogram scaled by the calculated scale factor.
        """
        final_scale_factor = self.calculate_final_scale_factor(
            additional_scale_factor = additional_scale_factor
        )

        # Clone hist and scale
        scaled_hist = self.hist.Clone(self.hist.GetName() + "_Scaled")
        scaled_hist.Scale(final_scale_factor)

        return scaled_hist

    def calculate_final_scale_factor(self, additional_scale_factor: float = 1.0) -> float:
        """ Calculate the scale factor to be applied to a hist before plotting.

        The scale factor is determined by multiplying all of the bin widths together. We assumes a
        uniformly binned histogram.

        Args:
            additional_scale_factor: Additional scale factor to apply to the scaled hist. Default: 1.0
        Returns:
            The factor by which the hist should be scaled.
        """
        # The first bin should always exist!
        bin_width_scale_factor = self.hist.GetXaxis().GetBinWidth(1)
        # Because of a ROOT quirk, even a TH1* hist has a Y and Z axis, with 1 bin
        # each. This bin has bin width 1, so it doesn't change anything if we multiply
        # by that bin width. So we just do it for all histograms.
        # This has the benefit that we don't need explicit dependence on an imported
        # ROOT package.
        bin_width_scale_factor *= self.hist.GetYaxis().GetBinWidth(1)
        bin_width_scale_factor *= self.hist.GetZaxis().GetBinWidth(1)

        final_scale_factor = additional_scale_factor / bin_width_scale_factor

        return final_scale_factor

class YAMLStorableObject(object):
    """ Base class for objects which can be represented and stored in YAML. """
    outputFilename = "yamlObject.yaml"

    def __init__(self):
        pass

    @classmethod
    def yamlFilename(cls, prefix, **kwargs):
        """ Determinte the YAML filename given the parameters.

        Args:
            prefix (str): Path to the diretory in which the YAML file is stored.
            kwargs (dict): Formatting dictionary for outputFilename.
        Returns:
            str: The filename of the YAML file which corresponds to the given parameters.
        """
        # Handle arguments
        if isinstance(kwargs["objType"], str):
            kwargs["objType"] = JetHCorrelationType[kwargs["objType"]]

        # Convert to proper name for formatting the string
        formattingArgs = {}
        formattingArgs.update(kwargs)
        formattingArgs["type"] = formattingArgs.pop("objType")

        return os.path.join(prefix, cls.outputFilename.format(**formattingArgs))

    @staticmethod
    def initializeSpecificProcessing(parameters):  # pragma: no cover
        """ Initialization specific processing plugin. Applied to obj immediately after creation from YAML.

        NOTE: This is called on `cls`, but we don't want to modify the object, so
              we define it as a `staticmethod`.

        Args:
            parameters (dict): Parameters which will be used to construct the object.
        Returns:
            dict: Parameters with the initialization specific processing applied.
        """
        return parameters

    @classmethod
    def initFromYAML(cls, *args, **kwargs):
        """ Initialize the object using information from a YAML file.

        Args:
            args (list): Positional arguments to use for intialization. They currently aren't used.
            kwargs (dict): Named arguments to use for initialization. Must contain:
                prefix (str): Path to the diretory in which the YAML file is stored.
                objType (JetHCorrelationType or str): Type of the object.
                jetPtBin (int): Bin of the jet pt of the object.
                trackPtBin (int): Bin of the track pt of the object.
        Returns:
            obj: The object constructed from the YAML file.
        """
        # Handle arguments
        if isinstance(kwargs["objType"], str):
            kwargs["objType"] = JetHCorrelationType[kwargs["objType"]]

        filename = cls.yamlFilename(**kwargs)

        # Check for file.
        if not os.path.exists(filename):
            logger.warning("Requested {objType} {className} ({jetPtBin}, {trackPtBin}) from file {filename}"
                           " does not exist! This container will not be"
                           " initialized".format(objType = str(kwargs["objType"]),
                                                 className = type(cls).__name__,
                                                 jetPtBin = kwargs["jetPtBin"],
                                                 trackPtBin = kwargs["trackPtBin"],
                                                 filename = filename))
            return None

        logger.debug("Loading {objType} {className} ({jetPtBin}, {trackPtBin}) from file"
                     " {filename}".format(objType = str(kwargs["objType"]),
                                          className = type(cls).__name__,
                                          jetPtBin = kwargs["jetPtBin"],
                                          trackPtBin = kwargs["trackPtBin"],
                                          filename = filename))
        parameters = utils.read_YAML(filename = filename)

        # Handle custom data type conversion
        parameters = cls.initializeSpecificProcessing(parameters)

        obj = cls(**parameters)

        return obj

    @staticmethod
    def saveSpecificProcessing(parameters):  # pragma: no cover
        """ Save specific processing plugin. Applied to obj immediately before saving to YAML.

        NOTE: This is called on `self` since we don't have a `cls` instance, but
              we don't want to modify the object, so we define it as a `staticmethod`.

        Args:
            parameters (dict): Parameters which will be saved to the YAML file.
        Returns:
            dict: Parameters with the save specific processing applied.
        """
        return parameters

    def saveToYAML(self, file_access_mode = "w", *args, **kwargs):
        """ Write the object properties to a YAML file.

        Args:
            file_access_mode (str): Mode under which the file should be opened. Defualt: "w"
            args (list): Positional arguments to use for intialization. They currently aren't used.
            kwargs (dict): Named arguments to use for initialization. Must contain:
                prefix (str): Path to the diretory in which the YAML file is stored.
                objType (JetHCorrelationType or str): Type of the object.
                jetPtBin (int): Bin of the jet pt of the object.
                trackPtBin (int): Bin of the track pt of the object.
        """
        # Handle arguments
        if isinstance(kwargs["objType"], str):
            kwargs["objType"] = JetHCorrelationType[kwargs["objType"]]

        # Determine filename
        filename = self.yamlFilename(**kwargs)

        # Use __dict__ so we don't have to explicitly define each field (which is easier to keep up to date!)
        # TODO: This appears to be quite slow. Can we speed this up somehow?
        parameters = copy.deepcopy(self.__dict__)

        # Handle custom data type conversion
        parameters = self.saveSpecificProcessing(parameters)

        #logger.debug("parameters: {}".format(parameters))
        logger.debug("Saving {objType} {className} ({jetPtBin}, {trackPtBin}) to file"
                     " {filename}".format(objType = kwargs["objType"],
                                          className = type(self).__name__,
                                          jetPtBin = kwargs["jetPtBin"],
                                          trackPtBin = kwargs["trackPtBin"],
                                          filename = filename))
        utils.write_YAML(filename = filename,
                         file_access_mode = file_access_mode,
                         parameters = parameters)

class HistArray(YAMLStorableObject):
    """ Represents a histogram's binned data.

    Histograms stored in this class make a number of (soft) assumptions

    - This histogram doesn't include the under-flow and over-flow bins.
    - The binning of the histogram is uniform.

    NOTE:
        The members of this class are stored with leading underscores because we access them
        through properties. It would be cleaner to remove the  "_" before the variables, but for now,
        it is very convenient, since it allows for automatic initialization from YAML. Perhaps
        revise later the names later while maintaining those properties.

    Args:
        _binCenters (np.ndarray): The location of the center of each bin.
        _array (np.ndarray): The values in each bin.
        _errors (np.ndarray): The associated each of the value in each bin.
    """
    # Format of the filename used for storing the hist array.
    outputFilename = "hist_{type}_jetPt{jetPtBin}_trackPt{trackPtBin}.yaml"

    def __init__(self, _binCenters, _array, _errors):
        # Init base class
        super().__init__()

        # Store elements
        self._binCenters = _binCenters
        self._array = _array
        self._errors = _errors

    @classmethod
    def initFromRootHist(cls, hist):
        """ Initialize the hist array from an existing ROOT histogram.

        Args:
            hist (ROOT.TH1): ROOT histogram to be converted.
        Returns:
            HistArray: HistArray created from the passed TH1.
        """
        h = histogram.Histogram1D.from_existing_hist(hist)
        return cls(_binCenters = h.x, _array = h.y, _errors = h.errors)

    @property
    def array(self):
        """ Return array of the histogram data. """
        return self._array

    @property
    def histData(self):
        """ Synonym of array. """
        return self.array

    @property
    def binCenters(self):
        """ Return array of the histogram bin centers. """
        return self._binCenters

    @property
    def x(self):
        """ Synonym of binCenters. """
        return self.binCenters

    @property
    def errors(self):
        """ Return array of the histogram errors. """
        return self._errors

    @staticmethod
    def initializeSpecificProcessing(parameters):
        """ Converts all lists to numpy arrays.

        Args:
            parameters (dict): Parameters which will be used to construct the object.
        Returns:
            dict: Parameters with the initialization specific processing applied.
        """
        # Convert arrays to numpy arrays
        for key, val in parameters.items():
            if isinstance(val, list):
                parameters[key] = np.array(val)

        return parameters

    @staticmethod
    def saveSpecificProcessing(parameters):
        """ Converts all numpy arrays to lists for saving to YAML.

        Args:
            parameters (dict): Parameters which will be saved to the YAML file.
        Returns:
            dict: Parameters with the save specific processing applied.
        """
        # Convert values for storage
        for key, val in parameters.items():
            # Handle numpy arrays to lists
            #logger.debug("Processing key {} with type {} and value {}".format(key, type(val), val))
            if isinstance(val, np.ndarray):
                #logger.debug("Converting list {}".format(val))
                parameters[key] = val.tolist()

        return parameters

class FitContainer(YAMLStorableObject):
    """ Contains information about a particular fit.

    Fits should only be stored if they are valid!

    Args:
        jetPtBin (int): Jet pt bin
        trackPtBin (int): Track pt bin
        fitType (JetHCorrelationType): Type of fit being stored. Usually signal_dominated or background_dominated
        values (dict): Dictionary from minuit.values storing parameter name to value. This is useful to have separately.
        params (dict): Dictionary from minuit.fitarg storing all relevant fit parameters (value, limits, errors,
            etc). This contains the values in values, but it useful to have them available separately, and it would
            take some work to separate the values from the other params.
        covarianceMatrix (dict): Dictionary from minuit.covariance storing the covariance matrix. It is a dict
            from ("param1", "param2") -> value
        errors (dict): Errors associated with the signal dominated and background dominated fit functions. Each
            error has one point for each point in the data.
    """
    # Format of the filename used for storing the fit container.
    # Make the filename accessible even if we don't have a class instance
    outputFilename = "fit_{type}_jetPt{jetPtBin}_trackPt{trackPtBin}.yaml"

    def __init__(self, jetPtBin, trackPtBin, fitType, values, params, covarianceMatrix, errors = None):
        # Init base class
        super().__init__()

        # Handle arguments
        if isinstance(fitType, str):
            fitType = JetHCorrelationType[fitType]

        # Store elements
        self.jetPtBin = jetPtBin
        self.trackPtBin = trackPtBin
        self.fitType = fitType
        self.values = values
        self.params = params
        self.covarianceMatrix = covarianceMatrix
        if errors is None:
            errors = {}
        self.errors = errors

    @staticmethod
    def initializeSpecificProcessing(parameters):
        """ Converts lists in the errors dict into numpy arrays.

        Args:
            parameters (dict): Parameters which will be used to construct the object.
        Returns:
            dict: Parameters with the initialization specific processing applied.
        """
        if "errors" in parameters:
            for k, v in parameters["errors"].items():
                parameters["errors"][k] = np.array(v)
        # NOTE: The enum will be converted in the constructor, so we don't need to handle it here.

        return parameters

    @staticmethod
    def saveSpecificProcessing(parameters):
        """ Convert numpy arrays in the errors dict into lists.

        Args:
            parameters (dict): Parameters which will be saved to the YAML file.
        Returns:
            dict: Parameters with the save specific processing applied.
        """
        for identifier, data in parameters["errors"].items():
            if isinstance(data, np.ndarray):
                # Convert to a normal list so it can be stored
                parameters["errors"][identifier] = data.tolist()
        # Fit type enum to str
        parameters["fitType"] = str(parameters["fitType"])

        return parameters

    def saveToYAML(self, prefix, fileAccessMode = "w", *args, **kwargs):
        """ Write the fit container properties to a YAML file.

        Args:
            prefix (str): Path to the directory where the fit should be stored.
            fileAccessMode (str): Mode under which the file should be opened. Defualt: "w"
            args (list): Additional arguments. They will be forwarded to saveToYAML().
            kwargs (dict): Additional named arguments. `objType`, `jetPtBin`, and `trackPtBin`
                will be overwritten by values stored in the class.
        """
        # Fill in the values from the object.
        kwargs["objType"] = self.fitType
        kwargs["jetPtBin"] = self.jetPtBin
        kwargs["trackPtBin"] = self.trackPtBin

        # Call the base class to handle the actual work.
        super().saveToYAML(prefix = prefix,
                           fileAccessMode = fileAccessMode,
                           *args, **kwargs)

@dataclass(frozen = True)
class PtBin(ABC):
    """ Represent a pt bin.

    Attributes:
        bin: Pt bin.
        range: Min and miximum of the bin.
        name: Name of the pt bin (based on the class name).
    """
    bin: int
    range: params.SelectedRange

    def __str__(self) -> str:
        return str(self.bin)

    @property
    def name(self) -> str:
        """ Convert class name into Captial Case: 'JetPtBin' -> 'Jet Pt Bin'. """
        return re.sub("([a-z])([A-Z])", r"\1 \2", self.__class__.__name__)

@dataclass(frozen = True)
class PtHardBin(PtBin):
    """ A pt hard bin, along with the train number associated with the bin and the range.

    We don't need to implement anything else. We just needed to instantiate this with the name
    of the class so that we can differentiate it from other bins.
    """
    train_number: int

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
        """ Convert input YAML list to set of ``PtHardBin``(s). """
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

class JetPtBins:
    """ Define an array of pt bins.

    It reads arrays registered under the tag ``!JetPtBins``. Loading

    .. code-block:: yaml

        - pt_bins: !JetPtBins [5, 11, 21]

    yields

    .. code-block:: python

        >>> pt_bins = [
        ...     JetPtBin(bin = 1, range = (5, 11)),
        ...     JetPtBin(bin = 2, range = (11, 21)),
        ... ]

    Note:
        This is just convenience function for YAML. It isn't round-trip because we would never use write back out.
        This just allow us to define the bins in a compact manner when we write YAML.
    """
    @classmethod
    def from_yaml(cls, constructor: yaml.Constructor, data: yaml.ruamel.yaml.nodes.SequenceNode) -> List[JetPtBin]:
        """ Convert input YAML list to set of ``JetPtBin``(s). """
        logger.debug(f"Using representer, {data}")
        values = [constructor.construct_object(v) for v in data.value]
        pt_bins = []
        for i, (pt, pt_next) in enumerate(zip(values[:-1], values[1:])):
            pt_bins.append(JetPtBin(bin = i + 1, range = params.SelectedRange(min = pt, max = pt_next)))
        return pt_bins

