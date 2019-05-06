#!/usr/bin/env python

""" Base interface for event generators.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import abc
from dataclasses import dataclass
import numpy as np
import secrets
from typing import Any, Iterable, Optional, Tuple

# Type helpers
#Event: Tuple["EventProperties", np.ndarray] = Tuple["EventProperties", np.ndarray]
Event = Tuple["EventProperties", np.ndarray]

@dataclass
class EventProperties:
    cross_section: float
    pt_hard: float

class Generator(abc.ABC):
    """ Base generator class.

    Attributes:
        generator: The event generator object.
        sqrt_s: The center of momentum energy in GeV.
        random_seed: Random seed for the generator.
        initialized: True if the generator has been initialized.
    """
    def __init__(self, generator: Any, sqrt_s: float, random_seed: Optional[float] = None):
        # Store the basic properties.
        self.generator = generator
        self.sqrt_s = sqrt_s

        # Determine the random seed
        self.random_seed = self._determine_random_seed(random_seed)

        # Store the state so we can check it later.
        self.initialized = False

    def _determine_random_seed(self, random_seed: Optional[float] = None) -> float:
        """ Determine the random seed.

        If we pass a valid value, it will just be used. If we pass None, then a random seed will be generated.

        Args:
            random_state: Value to help determine the random seed.
        Returns:
            Value if passed, or otherwise a random integer between 0 and 1 billion.
        """
        if random_seed is not None:
            return random_seed

        return secrets.randbelow(1000)

    @abc.abstractmethod
    def setup(self) -> bool:
        ...

    @abc.abstractmethod
    def __call__(self, n_events: int) -> Iterable[Event]:
        """ Generate an event with Pythia 6.

        Args:
            n_events: Number of events to generate.
        Returns:
            Output particles from a particular event.
        """
        ...

