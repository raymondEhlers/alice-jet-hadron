#!/usr/bin/env python

import numpy as np
from typing import Any, Iterable, Optional, Tuple

from jet_hadron.event_gen import generator

import ROOT

DTYPE_EP = np.dtype([('E', np.float64), ('px', np.float64), ('py', np.float64), ('pz', np.float64)])

class Pythia6(generator.Generator):
    """ PYTHIA 6 event generator.

    Defaults to the Perugia 2012 tune (tune number 370).

    Args:
        sqrt_s: Center of momentum energy.
        random_seed: Random seed for the generator. Default: None, which will be totally random.
        tune_number: PYTHIA 6 tune number.
        pt_hard: Pt hard bin values.

    Attributes:
        generator: The event generator object.
        sqrt_s: The center of momentum energy in GeV.
        random_seed: Random seed for the generator.
        initialized: True if the generator has been initialized.
        pt_hard: Pt hard bin values.
    """
    def __init__(self, tune_number: int = 370,
                 pt_hard: Tuple[Optional[float], Optional[float]] = (None, None), *args: Any, **kwargs: Any):
        # Before anything else, ensure that the Pythia6 library. Otherwise, it will crash.
        # LHAPDF is required to be loaded to use Perguia 2012
        ROOT.gSystem.Load("liblhapdf")
        # These pythia libraries are commonly loaded by ALICE, so we emulate them.
        ROOT.gSystem.Load("libEGPythia6")
        ROOT.gSystem.Load("libpythia6_4_28")
        # Sadly, it appears that this is also required. Without it, `_pyr_` isn't defined. After a bit of digging,
        # it appears that this is related to the PYTHIA random number generator interface `PYR`. `AliPythiaRndm` is
        # where it's actually implemented (look for `pyr_`). This could probably be avoided if one builds PYTHIA 6
        # without relying on AliRoot to do perform build.
        ROOT.gSystem.Load("libAliPythia6")

        # Next, setup the base class
        super().__init__(
            ROOT.TPythia6(), *args, **kwargs,
        )
        # Store the other parameters.
        self.pt_hard = pt_hard

        # The setup the generator
        self.initialized = self.setup(tune_number = tune_number)
        # Validate
        if not self.initialized:
            raise RuntimeError("Pythia6 failed to initialize.")

    def tune(self, tune_number: int) -> None:
        """ Set the Pythia tune. """
        if self.initialized is False:
            self.generator.Pytune(tune_number)
        else:
            raise RuntimeError("Cannot change the tune after PYTHIA has been initialized.")

    def _customize_tune(self) -> None:
        """ Provide additional tune customization. """
        ...

    def setup(self, tune_number: Optional[int] = None) -> bool:
        """ Setup the PYTHIA 6 generator.

        Args:
            tune_number: The tune number that should be used to configure Pythia.
        Returns:
            True if setup was successful.
        """
        # Basic setup
        # Pt hard
        if self.pt_hard[0]:
            self.generator.SetCKIN(3, self.pt_hard[0])
        if self.pt_hard[1]:
            self.generator.SetCKIN(4, self.pt_hard[1])
        # Random seed
        self.generator.SetMRPY(1, self.random_seed)

        # Specify or otherwise customize the tune.
        if tune_number:
            self.tune(tune_number)
        # Customize the tune (perhaps further beyond the tune number) if desired
        self._customize_tune()

        # Finally, initialize PYTHIA for pp at the given sqrt(s)
        self.generator.Initialize("cms", "p", "p", self.sqrt_s)

        return True

    def _format_output(self) -> generator.Event:
        """

        """
        # Setup
        status_dtype = DTYPE_EP.descr + [("status_code", np.int32)]

        # Retrieve particles
        particles = self.generator.GetListOfParticles()
        n_particles = particles.GetEntries()
        particles_array = np.empty(n_particles, dtype = status_dtype)

        # Store the particles from pythia. Unfortunately, we have to loop here, so the performance probably
        # isn't going to be amazing.
        # The Pythia particles are 1 indexed, so we start at 1.
        # NOTE: output_index == pythia_index - 1, but we define both for convenience
        for output_index, pythia_index in enumerate(range(1, n_particles + 1)):
            # E, px, py, py, KS (status code)
            particles_array[output_index] = np.array(
                (
                    self.generator.GetP(pythia_index, 4),
                    self.generator.GetP(pythia_index, 1),
                    self.generator.GetP(pythia_index, 2),
                    self.generator.GetP(pythia_index, 3),
                    self.generator.GetK(pythia_index, 1),
                ),
                dtype = status_dtype,
            )

        # Filter out some particles
        # According to the PYTHIA manual: "The ground rule is that codes 1–10 correspond to currently
        # existing partons/particles, while larger codes contain partons/particles which no longer exist,
        # or other kinds of event information."
        filtered_array = particles_array[(particles_array["status_code"] != 0) & (particles_array["status_code"] <= 10)]

        return filtered_array

    def __call__(self, n_events: int) -> Iterable[generator.Event]:
        """ Generate an event with Pythia 6.

        Args:
            n_events: Number of events to generate.
        Returns:
            Output particles from a particular event.
        """
        # TODO: Event properties...
        for i in range(n_events):
            # Call Pyevnt()
            self.generator.GenerateEvent()

            yield self._format_output()

class Pythia6Perugia(Pythia6):
    def _customize_tune(self) -> None:
        """ Use the Perugia 2012 tune.

        Can alternatively be set via the tune number 370.
        """
        # Set the individual parameters
        self.generator.SetMSTJ(11, 5)
        self.generator.SetPARJ(1, 0.085)
        self.generator.SetPARJ(2, 0.20)
        self.generator.SetPARJ(3, 0.92)
        self.generator.SetPARJ(4, 0.043)
        self.generator.SetPARJ(6, 1.0)
        self.generator.SetPARJ(7, 1.0)
        self.generator.SetPARJ(11, 0.35)
        self.generator.SetPARJ(12, 0.40)
        self.generator.SetPARJ(13, 0.54)
        self.generator.SetPARJ(21, 0.33)
        self.generator.SetPARJ(25, 0.70)
        self.generator.SetPARJ(26, 0.135)
        self.generator.SetPARJ(41, 0.45)
        self.generator.SetPARJ(42, 1.0)
        self.generator.SetPARJ(45, 0.86)
        self.generator.SetPARJ(46, 1.0)
        self.generator.SetPARJ(47, 1.0)

