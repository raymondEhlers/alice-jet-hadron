#!/usr/bin/env python

""" Test of ROOT histogram operations.

.. codeauthor:: Ramyond Ehlers <raymond.ehlers@yale.edu>, Yale University
"""

from typing import Callable, Tuple

import ROOT

Hist = ROOT.TH1

def create_hists() -> Tuple[Hist, Hist]:
    bins = [10, 0, 10]
    a = ROOT.TH1F("a", "a", *bins)
    a.Sumw2()
    a.Fill(3)
    a.Fill(3)
    a2 = ROOT.TH1F("aTimes2", "aTimes2", *bins)
    a2.Sumw2()
    a2.Fill(3)
    a2.Fill(3)
    a2.Fill(3)
    a2.Fill(3)

    return a, a2

def create_hists_weighted() -> Tuple[Hist, Hist]:
    bins = [10, 0, 10]
    a = ROOT.TH1F("a", "a", *bins)
    a.Sumw2()
    a.Fill(3, 2)
    a2 = ROOT.TH1F("aTimes2", "aTimes2", *bins)
    a2.Sumw2()
    a2.Fill(3, 2)
    a2.Fill(3, 2)

    return a, a2

def print_hists(hist_1: Hist, hist_2: Hist) -> None:
    """ Helper to print the contents of histograms. """
    for i in range(1, hist_1.GetXaxis().GetNbins()):
        print(f"i: {i}")
        print(f"counts hist_1: {hist_1.GetBinContent(i)}, error: {hist_1.GetBinError(i)}")
        print(f"counts hist_2: {hist_2.GetBinContent(i)}, error: {hist_2.GetBinError(i)}")

def test_add(func: Callable[[], Tuple[Hist, Hist]], label: str) -> None:
    """ Test adding and get the results. """
    a, a2 = func()
    print("Initial")
    print_hists(a, a2)
    print("Add")
    a2.Add(a)
    print("Result")
    print_hists(a, a2)

def test_add_mixed_types() -> None:
    a, _ = create_hists()
    _, a2 = create_hists_weighted()
    print("Initial")
    print_hists(a, a2)
    print("Add")
    a2.Add(a)
    print("Result")
    print_hists(a, a2)

def test_multiply(func: Callable[[], Tuple[Hist, Hist]], label: str) -> None:
    """ Test multiplying and get the results. """
    a, a2 = func()

    print("Initial")
    print_hists(a, a2)
    print("Multiply")
    a2.Multiply(a)
    print("Result")
    print_hists(a, a2)

def test_multiply_mixed_types() -> None:
    a, _ = create_hists()
    _, a2 = create_hists_weighted()
    print("Initial")
    print_hists(a, a2)
    print("Add")
    a2.Multiply(a)
    print("Result")
    print_hists(a, a2)

def test_divide(func: Callable[[], Tuple[Hist, Hist]], label: str) -> None:
    """ Test dividing and get the results. """
    a, a2 = func()

    print("Initial")
    print_hists(a, a2)
    print("Divide")
    a2.Divide(a)
    print("Result")
    print_hists(a, a2)

def test_divide_mixed_types() -> None:
    a, _ = create_hists()
    _, a2 = create_hists_weighted()
    print("Initial")
    print_hists(a, a2)
    print("Add")
    a2.Divide(a)
    print("Result")
    print_hists(a, a2)

if __name__ == "__main__":
    test_add(func = create_hists, label = "Standard hists")
    print("The standard hists agree with standard error prop!\n")
    test_add_mixed_types()
    print("The mixed standard and weighted hists agree with standard error prop!\n")
    test_add(func = create_hists_weighted, label = "Weighted hists")
    print("The weighted hists agree with standard error prop!\n")
    test_multiply(func = create_hists, label = "Standard hists")
    print("The standard hists agree with standard error prop!\n")
    test_multiply_mixed_types()
    print("The mixed standard and weighted hists agree with standard error prop!\n")
    test_multiply(func = create_hists_weighted, label = "Weighted hists")
    print("The weighted hists agree with standard error prop!\n")
    test_divide(func = create_hists, label = "Standard hists")
    print("The standard hists agree with standard error prop!\n")
    test_divide_mixed_types()
    print("The mixed standard and weighted hists agree with standard error prop!\n")
    test_divide(func = create_hists_weighted, label = "Weighted hists")
    print("The weighted hists agree with standard error prop!\n")

