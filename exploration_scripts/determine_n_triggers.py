#!/usr/bin/env python

""" Determine n triggers directly from an AnalysisResults.root file. """

from pachyderm import histogram
from pachyderm.utils import epsilon
from typing import Tuple

import ROOT

def get_trigger_sparse(filename: str, list_name: str, trigger_sparse_name: str) -> Tuple[ROOT.THnSparseF, ROOT.TH1]:
    """ Get trigger sparse from AnalysisResults.root file. """

    hists = histogram.get_histograms_in_file(filename = filename)
    jetH_hists = hists[list_name]
    return jetH_hists[trigger_sparse_name], jetH_hists["fHistCentrality"]

def setup_axis_selections(trigger_sparse) -> None:
    """ Assuming triggers from 0-10% within 20-40 GeV.

    Joel and I have the same axes for the first 3 dimensions.
    """
    # Centrality
    trigger_sparse.GetAxis(0).SetRangeUser(0, 10)
    # EP orientation
    trigger_sparse.GetAxis(2).SetRange(1, trigger_sparse.GetAxis(2).GetNbins())

def joel_calculate_n_trig() -> Tuple[float, float, float, float]:
    trigger_sparse, centrality = get_trigger_sparse(
        filename = "trains/joel/1203/AnalysisResults.root",
        list_name = "Correlations_JetHadTriggerJetsclbias10",
        trigger_sparse_name = "fhnCorr"
    )
    # Jet pt
    trigger_sparse.GetAxis(1).SetRangeUser(20, 40)
    setup_axis_selections(trigger_sparse)
    trigger_spectra = trigger_sparse.Projection(1)

    n_trig = trigger_spectra.GetEntries()
    n_trig_integral_full = trigger_spectra.Integral()
    n_trig_integral_selected = trigger_spectra.Integral(trigger_spectra.FindBin(20 + epsilon), trigger_spectra.FindBin(40 - epsilon))
    centrality_n_entries = centrality.Integral(centrality.FindBin(0 + epsilon), centrality.FindBin(10 - epsilon))

    return n_trig, n_trig_integral_full, n_trig_integral_selected, centrality_n_entries

def mine_calculate_n_trig() -> Tuple[float, float, float, float]:
    trigger_sparse, centrality = get_trigger_sparse(
        filename = "trains/PbPb/3360/AnalysisResults.root",
        list_name = "AliAnalysisTaskJetH_tracks_caloClusters_clusterBias10R2",
        trigger_sparse_name = "fhnTrigger"
    )
    setup_axis_selections(trigger_sparse)
    trigger_spectra = trigger_sparse.Projection(1)

    n_trig = trigger_spectra.GetEntries()
    n_trig_integral_full = trigger_spectra.Integral()
    n_trig_integral_selected = trigger_spectra.Integral(trigger_spectra.FindBin(20 + epsilon), trigger_spectra.FindBin(40 - epsilon))
    centrality_n_entries = centrality.Integral(centrality.FindBin(0 + epsilon), centrality.FindBin(10 - epsilon))

    return n_trig, n_trig_integral_full, n_trig_integral_selected, centrality_n_entries

def compare_n_trig() -> None:
    joel_n_trig, joel_n_trig_integral_full, joel_n_trig_integral_selected, joel_centrality_n_entries = joel_calculate_n_trig()
    mine_n_trig, mine_n_trig_integral_full, mine_n_trig_integral_selected, mine_centrality_n_entries = mine_calculate_n_trig()

    print("I do selections differently, so only my selected value should be compared with Joel's.")
    print("However, all of Joel's values should be consistent due to the way he makes selections.")
    print(f"n_trig: mine: {mine_n_trig}, Joel: {joel_n_trig}")
    print(f"n_trig_integral_full: mine: {mine_n_trig_integral_full}, Joel: {joel_n_trig_integral_full}")
    print(f"n_trig_integral_selected: mine: {mine_n_trig_integral_selected}, Joel: {joel_n_trig_integral_selected}")
    print("For comparison, centrality entries in 0-10%")
    print(f"centrality_n_entries: mine: {mine_centrality_n_entries}, Joel: {joel_centrality_n_entries}")

if __name__ == "__main__":
    compare_n_trig()
