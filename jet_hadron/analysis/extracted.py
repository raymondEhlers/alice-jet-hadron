#!/usr/bin/env python

""" Handle extracted widths and yields.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

from jet_hadron.base import params
from jet_hadron.analysis import fit

@dataclass
class ExtractedValue:
    properties: Dict[str, Union[float, Tuple[float, float], Any]]

@dataclass
class ExtractedYield(ExtractedValue):
    extraction_range: params.SelectedRange

@dataclass
class ExtractedWidth(ExtractedValue):
    fit_obj: fit.Fit
    fit_args: fit.FitArguments

    #@property
    #def properties(self) -> Any:
    #    """ Redefine properties """
    #    return self.fit_args

    @property
    def fit_result(self) -> fit.FitResult:
        """ Helper to retrieve the overall fit result. """
        return self.fit_obj.fit_result

    @fit_result.setter
    def fit_result(self, result) -> None:
        """ Helper to simplify setting the fit result. """
        self.fit_obj.fit_result = result

    @property
    def width(self) -> float:
        """ Helper to retrieve the width and the error on the width. """
        return self.fit_result.values_at_minimum["width"]

    @property
    def mean(self) -> float:
        """ Helper to retrieve the mean (should be fixed). """
        return self.fit_result.values_at_minimum["mean"]

