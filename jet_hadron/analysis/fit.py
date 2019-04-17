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
import scipy.optimize as optimization
import sys
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Type, TypeVar, Union

import reaction_plane_fit.base

from jet_hadron.base import params
from jet_hadron.base.typing_helpers import Hist

import ROOT

logger = logging.getLogger(__name__)

# Type helpers
FitArguments = Dict[str, Union[bool, float, Tuple[float, float]]]

this_module = sys.modules[__name__]

def print_root_fit_parameters(fit) -> None:
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
        return np.sum(np.power(self.data.y - self.f(self.data.x, *args), 2) / np.power(self.data.errors, 2))

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
    # NOTE: mypy doesn't parse this properly some reason. It appears to reverse the inherited arguments for some reason...
    #       It won't show up doing normal type checking, but seems to appear when checking the commit
    fit_result = FitResult(  # type: ignore
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

def fit_pedestal_to_histogram(h: histogram.Histogram1D,
                              fit_arguments: FitArguments, use_minos: bool = False) -> FitResult:
    """ Fit the given histogram to a pedestal.

    Args:
        h: Histogram to be fit.
        fit_arguments: Arguments to override the default fit arguments.
        use_minos: If True, minos errors will be calculated.
    Returns:
        Fit result from the fit.
    """
    # Required arguments for the fit
    arguments: FitArguments = {
        "pedestal": 1, "limit_pedestal": (-0.5, 1000),
    }

    # Perform the fit
    fit_result = fit_with_chi_squared(
        fit_func = lambda x, pedestal: pedestal,
        arguments = arguments, user_arguments = fit_arguments,
        h = h, use_minos = use_minos,
    )

    return fit_result

def fit_pedestal_to_delta_eta_background_dominated_region(h: histogram.Histogram1D,
                                                          fit_range: params.SelectedRange) -> Tuple[float, float]:
    """ Fit a pedestal to a histogram using ``scipy.optimize.curvefit``.

    The initial value of the fit will be determined by the minimum y value of the histogram.

    Args:
        h: Histogram to be fit.
        fit_range: Min and max values within which the fit will be performed.
    Returns:
        (constant, error)
    """
    # TODO: Update to use minuit!
    # For example, -1.2 < h.x < -0.8
    negative_restricted_range = (h.x < -1 * fit_range.min) & (h.x > -1 * fit_range.max)
    # For example, 0.8 < h.x < 1.2
    positive_restricted_range = (h.x > fit_range.min) & (h.x < fit_range.max)
    restricted_range = negative_restricted_range | positive_restricted_range
    constant, covariance_matrix = optimization.curve_fit(
        f = lambda x, c: c,
        xdata = h.x[restricted_range], ydata = h.y[restricted_range], p0 = np.min(h.y[restricted_range]),
        sigma = h.errors[restricted_range],
    )

    # Error reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
    error = float(np.sqrt(np.diag(covariance_matrix)))

    return float(constant), error

@dataclass
class GaussianFitInputs:
    """ Storage for Gaussian fit inputs.

    Attributes:
        mean: Mean value of the Gaussian.
        initial_width: Initial value of the Gaussian fit.
        fit_range: Min and max values within which the fit will be performed.
    """
    mean: float
    initial_width: float
    fit_range: params.SelectedRange

def gaussian(x: float, mean: float, width: float) -> float:
    """ Normalized gaussian.

    Args:
        x: Independent variable.
        mean: Mean.
        width: Width.
    Returns:
        Normalized gaussian value.
    """
    return 1 / np.sqrt(2 * np.pi * width ** 2) * np.exp(-1 / 2 * ((x - mean) / width) ** 2)

def fit_gaussian(h: histogram.Histogram1D,
                 fit_arguments: FitArguments,
                 use_minos: bool = False) -> FitResult:
    """ Fit the given histogram to a normalized gaussian.

    Args:
        h: Histogram to be fit.
        fit_arguments: Arguments to override the default fit arguments.
        use_minos: If True, minos errors will be calculated.
    Returns:
        Fit result from the fit.
    """
    # Required arguments for the fit
    arguments: FitArguments = {
        "mean": 0, "limit_mean": (-0.5, 0.5),
        "width": 0.15, "limit_width": (0.05, 0.8),
    }

    # Perform the fit
    fit_result = fit_with_chi_squared(
        fit_func = gaussian,
        arguments = arguments, user_arguments = fit_arguments,
        h = h, use_minos = use_minos
    )

    return fit_result

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
    @abc.abstractmethod
    def __init__(self, fit_range: params.SelectedRange, user_arguments: Optional[FitArguments] = None):
        self.fit_range: params.SelectedRange
        self.user_arguments: FitArguments
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

    @abc.abstractmethod
    def fit(self, h: histogram.Histogram1D,
            user_arguments: Optional[FitArguments] = None, use_minos: bool = False) -> FitResult:
        """ Fit the given histogram using the defined fit function.

        Args:
            h: Background subtracted histogram to be fit.
            user_arguments: Additional user arguments (beyond those already specified when the object
                was created). Default: None.
            use_minos: If True, minos errors will be calculated. Default: False.
        Returns:
            Result of the fit.
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
                "created in the fit object."
                f"Stored name: {fit_function_name}, object created fit function: {obj.fit_function}."
                "This may indicate a problem (but is fine if the same function was just renamed)."
            )

        # Now that the object is fully constructed, we can return it.
        return obj

class FitPedestalWithExtendedGaussian(Fit):
    """ Fit a pedestal + extended (unnormalized) gaussian to a signal peak using iminuit.

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
        self.fit_function: Callable[..., float] = pedestal_with_extended_gaussian
        self.fit_result: FitResult

    def fit(self, h: histogram.Histogram1D,
            user_arguments: Optional[FitArguments] = None, use_minos: bool = False) -> FitResult:
        """ Fit a gaussian to a signal peak using iminuit.

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

        # Restrict the range so that we only fit within the desired input.
        restricted_range = (h.x > self.fit_range.min) & (h.x < self.fit_range.max)
        restricted_hist = histogram.Histogram1D(
            # We need the bin edges to be inclusive.
            bin_edges = h.bin_edges[(h.bin_edges >= self.fit_range.min) & (h.bin_edges <= self.fit_range.max)],
            y = h.y[restricted_range],
            errors_squared = h.errors_squared[restricted_range]
        )

        # Perform the fit
        # Default arguments required for the fit
        arguments: FitArguments = {
            "pedestal": 1, "limit_pedestal": (-0.5, 1000), "error_pedestal": 0.1 * 1,
            "amplitude": 1, "limit_amplitude": (0.05, 100), "error_amplitude": 0.1 * 1,
            "mean": 0, "limit_mean": (-0.5, 0.5), "error_mean": 0.05,
            "width": 0.15, "limit_width": (0.05, 0.8), "error_width": 0.1 * 0.15,
        }

        # Perform the fit by minimizing the chi squared
        fit_result = fit_with_chi_squared(
            fit_func = self.fit_function,
            arguments = arguments, user_arguments = user_fit_arguments,
            h = restricted_hist, use_minos = use_minos
        )

        return fit_result

