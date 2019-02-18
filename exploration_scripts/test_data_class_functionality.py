#!/usr/bin/env python

""" Tests for dataclasses """

import abc
from dataclasses import dataclass
import enum
from typing import Any

Hist = Any

@dataclass
class TestABC(metaclass = abc.ABCMeta):
    hello: str
    world: str

@dataclass
class TestClass:
    hello: str
    world: str
    foo: str

@dataclass
class TestClass2(TestClass):
    foo: str = "foo"
    #world: str = field(default = "world")

@dataclass
class TestClass3(TestClass2):
    world: str = "world"

class JetHCorrelationType(enum.Enum):
    """ 1D correlation projection type """
    full_range = 0
    # delta phi specialized
    signal_dominated = 1
    background_dominated = 2
    # delta eta specialized
    near_side = 3
    away_side = 4

class JetHCorrelationAxis(enum.Enum):
    """ Define the axes of Jet-H 2D correlation hists. """
    delta_phi = 1
    delta_eta = 2

@dataclass
class CorrelationObservable1D:
    """ For 1d correlation observables. """
    hist: Hist
    type: JetHCorrelationType
    axis: JetHCorrelationAxis

@dataclass
class DeltaPhiObservable(CorrelationObservable1D):
    axis: JetHCorrelationAxis = JetHCorrelationAxis.delta_phi

@dataclass
class DeltaPhiSignalDominated(DeltaPhiObservable):
    type: JetHCorrelationType = JetHCorrelationType.signal_dominated
    name: str = "..."

@dataclass
class DeltaPhiBackgroundDominated(DeltaPhiObservable):
    type: JetHCorrelationType = JetHCorrelationType.background_dominated
    name: str = "..."

if __name__ == "__main__":

    t = TestClass(hello = "hello", world = "world", foo = "bar")
    print(f"t.world: {t.world}")

    #t2 = TestClass2(hello = "hello", foo = "bar")
    t2 = TestClass2(hello = "hello", world = "world")
    print(f"t2.world: {t2.world}")

    t3 = TestClass3(hello = "hello")
    print(f"t3.world: {t3.world}")

    # Real world example

    c = CorrelationObservable1D(hist = None, axis = JetHCorrelationAxis.delta_phi, type = JetHCorrelationType.signal_dominated)
    print(f"Observable axis: {c.axis}")
    d = DeltaPhiObservable(hist = None, type = JetHCorrelationType.signal_dominated)
    print(f"DPhi observable axis: {d.axis}")
    e = DeltaPhiSignalDominated(hist = None, name = "...")
    print(f"Signal dominated axis: {e.axis}")

    f = TestABC(hello = "hello", world = "world")
    print(f"ABC: {f.hello}")
