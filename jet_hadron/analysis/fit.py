#!/usr/bin/env python

""" Main jet-hadron correlations fit module.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import abc
from abc import ABC
from dataclasses import dataclass
import iminuit.util
import logging
import numpy as np
from pachyderm import histogram
from pachyderm import yaml
import pprint
import sys
from typing import Any, Callable, cast, Dict, List, Optional, Sequence, Tuple, Type, TypeVar, TYPE_CHECKING, Union

import reaction_plane_fit.base

from jet_hadron.base import params
from jet_hadron.base.typing_helpers import Hist

import ROOT

if TYPE_CHECKING:
    from jet_hadron.analysis import extracted

logger = logging.getLogger(__name__)

# Type helpers
FitArguments = Dict[str, Union[bool, float, Tuple[float, float]]]

this_module = sys.modules[__name__]

def print_root_fit_parameters(fit: Any) -> None:
    """ Print out all of the ROOT-based fit parameters. """
    output_parameters = []
    for i in range(0, fit.GetNpar()):
        parameter = fit.GetParameter(i)
        parameter_name = fit.GetParName(i)
        lower_limit = ROOT.Double(0.0)
        upper_limit = ROOT.Double(0.0)
        fit.GetParLimits(i, lower_limit, upper_limit)

        output_parameters.append(f"{i}: {parameter_name} = {parameter} from {lower_limit} - {upper_limit}")

    pprint.pprint(output_parameters)

def fit_1d_mixed_event_normalization(hist: Hist, delta_phi_limits: Sequence[float]) -> ROOT.TF1:
    """ Alternative to determine the mixed event normalization.

    A linear function is fit to the dPhi mixed event normalization for some predefined range.

    Args:
        hist: 1D mixed event histogram to be fit.
        delta_phi_limits: Min and max fit limits in delta phi.
    Returns:
        The ROOT fit function.
    """
    fit_func = ROOT.TF1("mixedEventNormalization1D", "[0] + 0.0*x", delta_phi_limits[0], delta_phi_limits[1])

    # Fit to the given histogram
    # R uses the range defined in the fit function
    # 0 ensures that the fit isn't drawn
    # Q ensures minimum printing
    # + adds the fit to function list to ensure that it is not deleted on the creation of a new fit
    hist.Fit(fit_func, "RIB0")

    # And return the fit
    return fit_func

def fit_2d_mixed_event_normalization(hist: Hist, delta_phi_limits: Sequence[float], delta_eta_limits: Sequence[float]) -> ROOT.TF2:
    """ Alternative to determine the mixed event normalization.

    A linear function is fit to the dPhi-dEta mixed event normalization for some predefined range.

    Args:
        hist: 2D mixed event histogram to be fit.
        delta_phi_limits: Min and max fit limits in delta phi.
        delta_eta_limits: Min and max fit limits in delta eta.
    Returns:
        The ROOT fit function.
    """
    fit_func = ROOT.TF2(
        "mixedEventNormalization2D",
        "[0] + 0.0*x + 0.0*y",
        delta_phi_limits[0], delta_phi_limits[1],
        delta_eta_limits[0], delta_eta_limits[1]
    )

    # Fit to the given histogram
    # R uses the range defined in the fit function
    # 0 ensures that the fit isn't drawn
    # Q ensures minimum printing
    # + adds the fit to function list to ensure that it is not deleted on the creation of a new fit
    hist.Fit(fit_func, "RIB0")

    # And return the fit
    return fit_func

@dataclass
class FitResult(reaction_plane_fit.base.FitResult):
    """ Store the fit result.

    Attributes:
        parameters (list): Names of the parameters used in the fit.
        free_parameters (list): Names of the free parameters used in the fit.
        fixed_parameters (list): Names of the fixed parameters used in the fit.
        values_at_minimum (dict): Contains the values of the fit function at the minimum. Keys are the names
            of parameters, while values are the numerical values at convergence.
        errors_on_parameters (dict): Contains the values of the errors associated with the parameters
            determined via the fit.
        covariance_matrix (dict): Contains the values of the covariance matrix. Keys are tuples
            with (param_name_a, param_name_b), and the values are covariance between the specified parameters.
            Note that fixed parameters are _not_ included in this matrix.
        x: x values where the fit result should be evaluated.
        errors: Store the errors associated with the fit function.
        n_fit_data_points: Number of data points used in the fit.
        minimum_val: Minimum value of the fit when it coverages. This is the chi squared value for a
            chi squared minimization fit.
        nDOF: Number of degrees of freedom. Calculated on request from ``n_fit_data_points`` and ``free_parameters``.
    """
    x: np.array
    errors: np.array
    n_fit_data_points: int
    minimum_val: float

    @property
    def nDOF(self) -> int:
        return self.n_fit_data_points - len(self.free_parameters)

def RPF_result_to_width_fit_result(rpf_result: reaction_plane_fit.base.FitResult,
                                   short_name: str,
                                   subtracted_hist: histogram.Histogram1D,
                                   width_obj: "extracted.ExtractedWidth") -> bool:
    """ Convert from a RPF fit result into a width fit result

    Note:
        This uses a chi squared cost function to calculate the minimum value for the width aspect of the fit.

    Args:
        rpf_result: Result from the reaction plane fit.
        short_name: Shortened version of the attribute name. Should be "ns" (for near side) or "as" (for away side).
        subtracted_hist: Subtracted histogram which this is supposed to fit.
        width_obj: Width object where the fit result will be stored.
    Returns:
        True if the fit result was successfully extracted.
    """
    # NOTE: This function is just a work in progress!!

    # Create chi squared object to calculate the minimum value.
    cost_func = ChiSquared(f = width_obj.fit_object.fit_function, data = subtracted_hist)
    # We want the func_code because it describe will then exclude the x parameter (which is what we want)
    parameters = cost_func.func_code
    fixed_parameters = ["mean", "pedestal"]
    free_parameters = ["width", "amplitude"]
    # Map between our new width fit parameters and those of the RPF.
    parameter_name_map = {
        "width": f"{short_name}_sigma",
        "amplitude": f"{short_name}_amplitude",
    }
    # Need to carefully grab the available values corresponding to the parameters or free_parameters, respectively.
    # NOTE: We cannot just iterate over the dict(s) themselves and check if the keys are in parameters because
    #       the parameters are de-duplicated, and thus the order can be wrong. In particular, for signal fits,
    #       B of the background fit ends up at the end of the dict because all of the other parameters are already
    #       defined for the signal fit. This approach won't have a problem with this, because we retrieve the values
    #       in the order of the parameters of the current fit component.
    # NOTE: The RPF will only have values for the free parameters in our width fit.
    values_at_minimum = {p: rpf_result.values_at_minimum[parameter_name_map[p]] for p in free_parameters}
    errors_on_parameters = {p: rpf_result.errors_on_parameters[parameter_name_map[p]] for p in free_parameters}
    covariance_matrix = {
        (a, b): rpf_result.covariance_matrix[(a, b)] for a in free_parameters for b in free_parameters
    }

    # Store the result
    width_obj.fit_object.fit_result = FitResult(
        parameters = parameters,
        free_parameters = free_parameters,
        fixed_parameters = fixed_parameters,
        values_at_minimum = values_at_minimum,
        errors_on_parameters = errors_on_parameters,
        covariance_matrix = covariance_matrix,
        x = subtracted_hist.x,
        # This will be determine and set below.
        errors = np.array([]),
        n_fit_data_points = len(subtracted_hist.x),
        minimum_val = cost_func(list(values_at_minimum.values())),
    )

    width_obj.fit_object.calculate_errors(x = subtracted_hist.x)

    return True

class ChiSquared:
    """ chi^2 cost function.

    Implemented with some help from the iminuit advanced tutorial. Calling this class will calculate the chi squared.

    Args:
        f: The fit function.
        func_code: Function arguments derived from the fit function. They need to be separately specified
            to allow iminuit to determine the proper arguments.
        data: Data to be used for fitting.
    """
    def __init__(self, f: Callable[..., float], data: histogram.Histogram1D):
        self.f = f
        self.func_code: List[str] = iminuit.util.make_func_code(iminuit.util.describe(self.f)[1:])
        #self.func_code = iminuit.util.make_func_code(iminuit.util.describe(self.f))
        self.data = data

    def __call__(self, *args: List[float]) -> float:
        """ Calculate the chi2. """
        return cast(
            float,
            np.sum(np.power(self.data.y - self.f(self.data.x, *args), 2) / np.power(self.data.errors, 2))
        )

def _validate_user_fit_arguments(default_arguments: FitArguments, user_arguments: FitArguments) -> bool:
    """ Validate the user provided fit arguments.

    Args:
        default_arguments: Default fit arguments.
        user_arguments: User provided fit arguments to be validated.
    Returns:
        True if the user fit arguments that are valid (ie the specified fit variables are also set in the
        default arguments).  It's up to the user to actually update the fit arguments.
    Raises:
        ValueError: If user provides arguments for a parameter that doesn't exist. (Usually a typo).
    """
    # Handle the user arguments
    # First, ensure that all user passed arguments are already in the argument keys. If not, the user probably
    # passed the wrong argument unintentionally.
    for k, v in user_arguments.items():
        # The second condition allows us to fix components that were not previously fixed.
        if k not in default_arguments and k.replace("fix_", "") not in default_arguments:
            raise ValueError(
                f"User argument {k} (with value {v}) is not present in the fit arguments."
                f" Possible arguments: {default_arguments}"
            )

    return True

def fit_with_chi_squared(fit_func: Callable[..., float],
                         arguments: FitArguments, user_arguments: FitArguments,
                         h: histogram.Histogram1D,
                         use_minos: bool = False) -> FitResult:
    """ Fit the given histogram to the given fit function using iminuit.

    Args:
        fit_func: Function to be fit.
        arguments: Required arguments for the fit.
        user_arguments: Arguments to override the default fit arguments.
        h: Histogram to be fit.
        use_minos: If True, minos errors will be calculated.
    Returns:
        Fit result from the fit.
    """
    # Validation
    # We are using a chi squared fit, so the errordef should be 1.
    # We specify it first just in the case wants to override for some reason
    arguments.update({
        "errordef": 1.0,
    })
    # Will raise an exception if the user fit arguments are invalid.
    _validate_user_fit_arguments(default_arguments = arguments, user_arguments = user_arguments)
    # Now, we actually assign the user arguments. We assign them last so we can overwrite any default arguments
    arguments.update(user_arguments)

    logger.debug(f"Minuit args: {arguments}")
    cost_func = ChiSquared(f = fit_func, data = h)
    minuit = iminuit.Minuit(cost_func, **arguments)

    # Perform the fit
    minuit.migrad()
    # Run minos if requested.
    if use_minos:
        logger.info("Running MINOS. This may take a minute...")
        minuit.minos()
    # Just in case (doesn't hurt anything, but may help in a few cases).
    minuit.hesse()

    # Check that the fit is actually good
    if not minuit.migrad_ok():
        raise reaction_plane_fit.base.FitFailed("Minimization failed! The fit is invalid!")

    # Create the fit result
    fixed_parameters: List[str] = [k for k, v in minuit.fixed.items() if v is True]
    # We use the cost function because we want intentionally want to skip "x"
    parameters: List[str] = iminuit.util.describe(cost_func)
    # Can't just use set(parameters) - set(fixed_parameters) because set() is unordered!
    free_parameters: List[str] = [p for p in parameters if p not in set(fixed_parameters)]
    # Store the result
    fit_result = FitResult(
        parameters = parameters,
        free_parameters = free_parameters,
        fixed_parameters = fixed_parameters,
        values_at_minimum = dict(minuit.values),
        errors_on_parameters = dict(minuit.errors),
        covariance_matrix = minuit.covariance,
        x = h.x,
        # These will be calculated below. It's easier to calculate once a FitResult already exists.
        errors = np.array([]),
        n_fit_data_points = len(h.x),
        minimum_val = minuit.fval,
    )

    # Calculate errors.
    fit_result.errors = reaction_plane_fit.base.calculate_function_errors(
        func = fit_func,
        fit_result = fit_result,
        x = fit_result.x
    )

    return fit_result

T_Fit = TypeVar("T_Fit", bound = "Fit")

class Fit(ABC):
    """ Class to direct fitting a histogram to a fit function.

    This allows us to easily store the fit function right alongside the minimization.

    Attributes:
        fit_range: Range used for fitting the data. Values inside of this range will be used.
        user_arguments: User arguments for the fit. Default: None.
        fit_function: Function to be fit.
        fit_result: Result of the fit. Only valid after the fit has been performed.
    """
    def __init__(self, fit_range: params.SelectedRange, user_arguments: Optional[FitArguments] = None):
        # Validation
        if user_arguments is None:
            user_arguments = {}
        self.fit_range = fit_range
        self.user_arguments: FitArguments = user_arguments
        self.fit_function: Callable[..., float]
        self.fit_result: FitResult

    def __call__(self, *args: float, **kwargs: float) -> float:
        """ Call the fit function.

        This is provided for convenience. This way, we can easily evaluate the function while
        still storing the information necessary to perform the entire fit.
        """
        return self.fit_function(*args, **kwargs)

    def calculate_errors(self, x: Optional[np.ndarray] = None) -> np.ndarray:
        """ Calculate the errors on the fit function for the given x values.

        Args:
            x: x values where the fit function error should be evaluated. If not specified,
                the x values over which the fit was performed will be used.
        Returns:
            The fit function error calculated at each x value.
        """
        if x is None:
            x = self.fit_result.x
        return reaction_plane_fit.base.calculate_function_errors(
            func = self.fit_function,
            fit_result = self.fit_result,
            x = x,
        )

    def fit(self, h: histogram.Histogram1D,
            user_arguments: Optional[FitArguments] = None, use_minos: bool = False) -> FitResult:
        """ Fit the given histogram to the stored fit function using iminuit.

        Args:
            h: Background subtracted histogram to be fit.
            user_arguments: Additional user arguments (beyond those already specified when the object
                was created). Default: None.
            use_minos: If True, minos errors will be calculated. Default: False.
        Returns:
            Result of the fit. The user is responsible for storing it in the fit.
        """
        # Validation
        if user_arguments is None:
            user_arguments = {}
        # Copy the user arguments so that we don't modify the arguments passed to the class if when we update them.
        user_fit_arguments = dict(self.user_arguments)
        # Update with any additional user provided arguments
        user_fit_arguments.update(user_arguments)

        # Setup the fit
        hist_for_fit, arguments = self._setup(h = h)

        # Perform the fit by minimizing the chi squared
        fit_result = fit_with_chi_squared(
            fit_func = self.fit_function,
            arguments = arguments, user_arguments = user_fit_arguments,
            h = hist_for_fit, use_minos = use_minos
        )

        return fit_result

    @abc.abstractmethod
    def _setup(self, h: histogram.Histogram1D) -> Tuple[histogram.Histogram1D, FitArguments]:
        """ Setup the histogram and arguments for the fit.

        Args:
            h: Background subtracted histogram to be fit.
        Returns:
            Histogram to use for the fit, default arguments for the fit. Note that the histogram may be range
                restricted or otherwise modified here.
        """
        ...

    @classmethod
    def to_yaml(cls: Type[T_Fit], representer: yaml.Representer, obj: T_Fit) -> yaml.ruamel.yaml.nodes.SequenceNode:
        """ Encode YAML representation. """
        # ``RoundTripRepresenter`` doesn't represent objects directly, so we grab a dict of the
        # members to store those in YAML.
        # NOTE: We must make a copy of the vars. Otherwise, the original object will be modified.
        members = dict(vars(obj))
        # We can't store unbound functions, so we instead set it to the function name
        # (we won't really use this name, but it's useful to store what was used, and
        # we can at least warn if it changed).
        members["fit_function"] = obj.fit_function.__name__
        representation = representer.represent_mapping(
            f"!{cls.__name__}", members
        )

        # Finally, return the represented object.
        return representation

    @classmethod
    def from_yaml(cls: Type[T_Fit], constructor: yaml.Constructor, data: yaml.ruamel.yaml.nodes.MappingNode) -> "Fit":
        """ Decode YAML representation. """
        # First, we construct the class member objects.
        members = {
            constructor.construct_object(key_node): constructor.construct_object(value_node)
            for key_node, value_node in data.value
        }
        # Then we deal with members which require special handling:
        # The fit result isn't set through the constructor, so we grab it and then
        # set it after creating the object.
        fit_result = members.pop("fit_result")
        # The fit function will be set in the fit constructor, so we don't need to use this
        # value to setup the object. However, since this contains the name of the function,
        # we can use it to check if the name of the function that is set in the constructor
        # is the same as the one that we stored. (If they are different, this isn't necessarily
        # a problem, as we sometimes rename functions, but regardless it's good to be notified
        # if that's the case).
        fit_function_name = members.pop("fit_function")

        # Finally, create the object and set the properties as needed.
        obj = cls(**members)
        obj.fit_result = fit_result
        # Sanity check on the fit function name (see above).
        if fit_function_name != obj.fit_function.__name__:
            logger.warning(
                "The stored fit function name from YAML doesn't match the name of the fit function"
                " created in the fit object."
                f" Stored name: {fit_function_name}, object created fit function: {obj.fit_function.__name__}."
                " This may indicate a problem (but is fine if the same function was just renamed)."
            )

        # Now that the object is fully constructed, we can return it.
        return obj

def pedestal(x: float, pedestal: float) -> float:
    """ Pedestal function.

    Note:
        If we defined this via a lambda (which would be trivial), we wouldn't be able to specify the types.
        So we define it separately.

    Args:
        x: Independent variable.
        pedestal: Pedestal value.
    Returns:
        The pedestal value.
    """
    # NOTE: We specify 0 * x so we can get proper array evaluation. If there's no explicit x
    #       dependence, it will only return a single value.
    return 0 * x + pedestal

class PedestalForDeltaEtaBackgroundDominatedRegion(Fit):
    """ Fit a pedestal to the background dominated region of a delta eta hist.

    The initial value of the fit will be determined by the minimum y value of the histogram.

    Attributes:
        fit_range: Range used for fitting the data. Values inside of this range will be used.
        user_arguments: User arguments for the fit. Default: None.
        fit_function: Function to be fit.
        fit_result: Result of the fit. Only valid after the fit has been performed.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Finally, setup the fit function
        self.fit_function = pedestal

    def _setup(self, h: histogram.Histogram1D) -> Tuple[histogram.Histogram1D, FitArguments]:
        """ Setup the histogram and arguments for the fit.

        Args:
            h: Background subtracted histogram to be fit.
        Returns:
            Histogram to use for the fit, default arguments for the fit. Note that the histogram may be range
                restricted or otherwise modified here.
        """
        restricted_range = (
            # For example, -1.2 < h.x < -0.8
            (h.x < -1 * self.fit_range.min) & (h.x > -1 * self.fit_range.max)
            # For example, 0.8 < h.x < 1.2
            | (h.x > self.fit_range.min) & (h.x < self.fit_range.max)
        )
        # Same conditions as above, but we need the bin edges to be inclusive.
        bin_edges_restricted_range = (
            (h.bin_edges <= -1 * self.fit_range.min) & (h.bin_edges >= -1 * self.fit_range.max)
            | (h.bin_edges >= self.fit_range.min) & (h.bin_edges <= self.fit_range.max)
        )
        restricted_hist = histogram.Histogram1D(
            bin_edges = h.bin_edges[bin_edges_restricted_range],
            y = h.y[restricted_range],
            errors_squared = h.errors_squared[restricted_range]
        )

        # Default arguments
        # Use the minimum of the histogram as the starting value.
        min_hist = np.min(restricted_hist.y)
        arguments: FitArguments = {
            "pedestal": min_hist, "error_pedestal": min_hist * 0.1,
            "limit_pedestal": (-1000, 1000),
        }

        return restricted_hist, arguments

def gaussian(x: float, mean: float, width: float) -> float:
    """ Normalized gaussian.

    Args:
        x: Independent variable.
        mean: Mean.
        width: Width.
    Returns:
        Normalized gaussian value.
    """
    return cast(
        float,
        1 / np.sqrt(2 * np.pi * width ** 2) * np.exp(-1 / 2 * ((x - mean) / width) ** 2),
    )

def pedestal_with_extended_gaussian(x: float, mean: float, width: float, amplitude: float, pedestal: float) -> float:
    """ Pedestal + extended (unnormalized) gaussian

    Args:
        x: Independent variable.
        mean: Gaussian mean.
        width: Gaussian width.
        amplitude: Amplitude of the gaussian.
        pedestal: Pedestal value.
    Returns:
        Function value.
    """
    return pedestal + amplitude * gaussian(x = x, mean = mean, width = width)

class FitPedestalWithExtendedGaussian(Fit):
    """ Fit a pedestal + extended (unnormalized) gaussian to a signal peak using iminuit.

    Attributes:
        fit_range: Range used for fitting the data. Values inside of this range will be used.
        user_arguments: User arguments for the fit. Default: None.
        fit_function: Function to be fit.
        fit_result: Result of the fit. Only valid after the fit has been performed.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Finally, setup the fit function
        self.fit_function = pedestal_with_extended_gaussian

    def _setup(self, h: histogram.Histogram1D) -> Tuple[histogram.Histogram1D, FitArguments]:
        """ Setup the histogram and arguments for the fit.

        Args:
            h: Background subtracted histogram to be fit.
        Returns:
            Histogram to use for the fit, default arguments for the fit. Note that the histogram may be range
                restricted or otherwise modified here.
        """
        # Restrict the range so that we only fit within the desired input.
        restricted_range = (h.x > self.fit_range.min) & (h.x < self.fit_range.max)
        restricted_hist = histogram.Histogram1D(
            # We need the bin edges to be inclusive.
            bin_edges = h.bin_edges[(h.bin_edges >= self.fit_range.min) & (h.bin_edges <= self.fit_range.max)],
            y = h.y[restricted_range],
            errors_squared = h.errors_squared[restricted_range]
        )

        # Default arguments required for the fit
        arguments: FitArguments = {
            "pedestal": 0, "limit_pedestal": (-1000, 1000), "error_pedestal": 0.1,
            "amplitude": 1, "limit_amplitude": (0.05, 100), "error_amplitude": 0.1 * 1,
            "mean": 0, "limit_mean": (-0.5, 0.5), "error_mean": 0.05,
            "width": 0.15, "limit_width": (0.05, 0.8), "error_width": 0.1 * 0.15,
        }

        return restricted_hist, arguments

