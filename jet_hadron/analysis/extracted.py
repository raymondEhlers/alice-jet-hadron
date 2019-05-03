#!/usr/bin/env python

""" Handle extracted widths and yields.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

from dataclasses import dataclass

from jet_hadron.base import analysis_objects
from jet_hadron.base import params
from jet_hadron.analysis import fit

@dataclass
class ExtractedYield:
    value: analysis_objects.ExtractedObservable
    central_value: float
    extraction_limit: float

    @property
    def extraction_range(self) -> params.SelectedRange:
        """ Helper to retrieve the extraction range. """
        return params.SelectedRange(
            min = self.central_value - self.extraction_limit,
            max = self.central_value + self.extraction_limit
        )

@dataclass
class ExtractedWidth:
    fit_object: fit.Fit
    fit_args: fit.FitArguments

    @property
    def fit_result(self) -> fit.FitResult:
        """ Helper to retrieve the overall fit result. """
        return self.fit_object.fit_result

    @fit_result.setter
    def fit_result(self, result: fit.FitResult) -> None:
        """ Helper to simplify setting the fit result. """
        self.fit_object.fit_result = result

    @property
    def width(self) -> float:
        """ Helper to retrieve the width and the error on the width. """
        return self.fit_result.values_at_minimum["width"]

    @property
    def mean(self) -> float:
        """ Helper to retrieve the mean (should be fixed). """
        return self.fit_result.values_at_minimum["mean"]

