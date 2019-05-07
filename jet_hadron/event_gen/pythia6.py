#!/usr/bin/env python

import numpy as np
import pyjet.utils
from typing import Any, Iterable, Optional, Tuple

from jet_hadron.event_gen import generator

DTYPE_EP = np.dtype([("E", np.float64), ("px", np.float64), ("py", np.float64), ("pz", np.float64)])

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
        # Lazy load ROOT here in an attempt to enable multiprocessing.
        # Unfortunately, it still doesn't appear to work, but I'll leave it here in hopes
        # that it will work some day...
        import ROOT
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
        """ Convert the output from the generator for into a format suitable for further processing.

        Args:
            None.
        Returns:
            Event level information and the input particles.
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
        # NOTE: output_index := pythia_index - 1, but we define both for convenience
        for output_index, pythia_index in enumerate(range(1, n_particles + 1)):
            # Format: E, px, py, py, KS (status code)
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

        # Convert from (E, px, py, pz) -> (pT, eta, phi, mass)
        filtered_array = pyjet.utils.ep2ptepm(filtered_array)

        # Determine event properties
        event_properties = generator.EventProperties(
            cross_section = self.generator.GetPARI(1),
            pt_hard = self.generator.GetVINT(47),
        )

        return event_properties, filtered_array

    def __call__(self, n_events: int) -> Iterable[generator.Event]:
        """ Generate an event with Pythia 6.

        Args:
            n_events: Number of events to generate.
        Returns:
            Generator to provide the requested number of events.
        """
        for i in range(n_events):
            # Call Pyevnt()
            self.generator.GenerateEvent()

            yield self._format_output()

