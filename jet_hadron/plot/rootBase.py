#!/usr/bin/env python

# ROOT based plotting utilities. They are kept separately to avoid importing ROOT.
#
# author: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
# date: 29 July 2018

def drawVerticalLine(xValue, ROOT = None):
    """ Draw a vertical line on a hist which will display properly on a log scale.

    Drawing a vertical line with standard functions works fine when the x axis is linear, but it breaks
    when it is log. This function is extracted from a question in
    the [ROOT forums](https://root.cern.ch/phpBB3/viewtopic.php?t=10745) and works for both linear and log
    x axes.

    Note:
        The line that is returned is **not** owned by ROOT to ensure that it isn't lost prematurely.

    Args:
        xValue (float): x value where the vertical line should be drawn.
        ROOT (ROOT): ROOT module currently in use. If `None`, it will import ROOT itself. Default: None
    Returns:
        ROOT.TLine: Vertical line located at the specified value.
    """
    if not ROOT:
        import ROOT

    line = ROOT.TLine()
    lm = ROOT.gPad.GetLeftMargin()
    rm = 1. - ROOT.gPad.GetRightMargin()
    tm = 1. - ROOT.gPad.GetTopMargin()
    bm = ROOT.gPad.GetBottomMargin()
    xndc = (rm - lm) * ((xValue - ROOT.gPad.GetUxmin()) / (ROOT.gPad.GetUxmax() - ROOT.gPad.GetUxmin())) + lm
    line.SetLineWidth(2)
    line.SetLineColor(ROOT.kBlack)

    line.DrawLineNDC(xndc, bm, xndc, tm)

    ROOT.SetOwnership(line, False)

    return line

