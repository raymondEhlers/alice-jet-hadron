#!/usr/bin/env python

""" Test for TLatex error which doesn't parse a frac definition properly. """

import ROOT

def test_tlatex_error():
    s_fails = r"$\frac{\mathrm{d}N}{d\mathit{p}_{\mathrm{T}}}$"

    #ROOT.TLatex()

    c = ROOT.TCanvas("c", "c")
    tex = ROOT.TMathText()
    tex.DrawMathText(0.1, 0.1, s_fails)
    c.SaveAs("tex_test.pdf")

    # Solution: TLatex is supposed to call TMathText. However, it will explicitly switch back to TLatex if plotting
    # to a PDF... So we really just can't plot to PDF with ROOT and LaTeX labels.

if __name__ == "__main__":
    test_tlatex_error()
