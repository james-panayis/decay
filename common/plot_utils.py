import ROOT
import os
import re
import numpy as np
from array import array
import math
import ctypes


class Quiet:
    """Context handler class to quiet errors in a 'with' statement"""

    def __init__(self, level=ROOT.kInfo + 1):
        """Class constructor"""
        #: the level to quiet
        self.level = level

    def __enter__(self):
        """Enter the context"""
        #: the previously set level to be ignored
        self.oldlevel = ROOT.gErrorIgnoreLevel
        ROOT.gErrorIgnoreLevel = self.level

    def __exit__(self, type, value, traceback):
        """Exit the context"""
        ROOT.gErrorIgnoreLevel = self.oldlevel


def plotWithResiduals(rooRealVar,
                      dots,
                      modelCurve,
                      units=str(),
                      nameOfPlot=str(),
                      logAxis=False,
                      removeArtifacts=False,
                      yMin=None,
                      yMax=None,
                      legend=None,
                      typeOfInput=None,
                      rooFitFrame=None,
                      resDenom="model"):
    """
    For plots with residuals.
    """

    if not rooFitFrame:
        rooFitFrame = rooRealVar.frame()
    else:
        rooFitFrame = rooFitFrame.Clone(rooFitFrame.GetName() + "_cloned")

    rooFitFrame.SetTitle("")
    rooFitFrame.GetXaxis().SetTitle("")
    rooFitFrame.GetXaxis().SetLabelSize(0)

    rooFitFrame.GetYaxis().SetTitleSize(0.072)
    rooFitFrame.GetYaxis().SetTitleOffset(1.1)
    rooFitFrame.GetYaxis().SetLabelSize(0.055)
    binwidth = dots.GetErrorXlow(1) + dots.GetErrorXhigh(1)
    binwidth = round(binwidth, -int(math.floor(math.log10(abs(binwidth)) - 2)))
    rooFitFrame.GetYaxis().SetTitle("Candidates per {0} {1}".format(
        binwidth, units))

    rooFitFrame.GetYaxis().CenterTitle()
    rooFitFrame.GetXaxis().CenterTitle()

    xValModel = ctypes.c_double(-1.E30)
    yValModel = ctypes.c_double(-1.E30)
    xValDot = ctypes.c_double(-1.E30)
    yValDot = ctypes.c_double(-1.E30)

    iDotPoint = ROOT.RooRealVar("iDotPoint", "", 0.)
    iModelPoint = ROOT.RooRealVar("iModelPoint", "", 0.)
    iDotError = ROOT.RooRealVar("iDotError", "", 0.)
    # yComb = ROOT.RooFormulaVar("yComb", "", "-1*@0", ROOT.RooArgList(y1RV))
    iResidual = ROOT.RooFormulaVar(
        "yComb", "", "(@0 - @1)/@2",
        ROOT.RooArgList(iDotPoint, iModelPoint, iDotError))

    iBin = 0

    pointsHist = ROOT.RooHist()
    dataHist = ROOT.RooHist()

    while iBin < dots.GetN():
        dots.GetPoint(iBin, xValDot, yValDot)

        iDotPoint.setVal(yValDot.value)
        yValModel = modelCurve.interpolate(xValDot.value)
        iModelPoint.setVal(yValModel)

        if np.isnan(iResidual.getVal()) is not True and yValDot.value > 0:

            if resDenom == "model":
                denominator = math.sqrt(abs(yValModel))
            elif resDenom == "data":
                denominator = float(
                    (dots.GetErrorYlow(iBin) + dots.GetErrorYhigh(iBin)) / 2)
            else:
                print("Unsupported option {}. Use 'model' or 'data'.".format(
                    resDenom))
            iDotError.setVal(denominator)

            residualValue = iResidual.getVal()

            if removeArtifacts:
                if abs(residualValue) > 3:
                    residualValue = 0

            elif residualValue > 10:
                residualValue = 10
            elif residualValue < -10:
                residualValue = -10
            # print("xval = ", xValDot)
            # print(iBin, " = ", iResidual.getVal(), ", @0 = ", yValDot, ", @1 = ",
            #       modelCurve.interpolate(xValDot), ", @2 = ",
            #       float(dots.GetErrorYlow(iBin) + dots.GetErrorYhigh(iBin)))
            pointsHist.addBinWithXYError(
                xValDot.value,
                residualValue,
                0,
                0,
                # dots.GetErrorXlow(iBin),
                # dots.GetErrorXhigh(iBin),
                1,
                1,
            )

            dataHist.addBinWithXYError(
                xValDot.value,
                yValDot.value,
                0,
                0,
                # dots.GetErrorXlow(iBin),
                # dots.GetErrorXhigh(iBin),
                dots.GetErrorYlow(iBin),
                dots.GetErrorYhigh(iBin))

        iBin += 1

    dots.SetFillColor(ROOT.kWhite)
    pointsHist.SetMarkerStyle(dots.GetMarkerStyle())
    pointsHist.SetMarkerSize(dots.GetMarkerSize())
    pointsHist.SetLineWidth(2)
    dataHist.SetFillColor(ROOT.kWhite)
    dataHist.SetMarkerStyle(dots.GetMarkerStyle())
    dataHist.SetMarkerSize(dots.GetMarkerSize())
    dataHist.SetLineWidth(2)

    yMaxDat = dots.GetYaxis().GetXmax()

    if not logAxis:
        if not yMax:
            rooFitFrame.SetAxisRange(0, yMaxDat, "Y")
        else:
            rooFitFrame.GetYaxis().SetRangeUser(0, yMax)

    with Quiet(ROOT.kError):
        # Removes the previous data points (because they have points with 0 values)
        dotsName = dots.GetName()
        rooFitFrame.remove(dotsName)
        dataHist.SetName(dotsName)

        # Add new data points copied from original ones (skipping 0 values)
        rooFitFrame.addPlotable(modelCurve, "L")
        rooFitFrame.addPlotable(dataHist, "P")

    rooFitFrameRes = rooRealVar.frame()

    rooFitFrameRes.SetTitle("")
    rooFitFrameRes.GetXaxis().SetTitle(rooRealVar.GetTitle())
    rooFitFrameRes.GetXaxis().SetTitleSize(0.2)
    rooFitFrameRes.GetXaxis().SetTitleOffset(0.9)

    rooFitFrameRes.GetXaxis().SetTickSize(0.07)
    rooFitFrameRes.GetYaxis().SetTickSize(0.024)

    rooFitFrameRes.GetYaxis().SetTitle("#splitline{Normalized}{ Residuals}")
    rooFitFrameRes.GetYaxis().SetTitleSize(0.16)
    rooFitFrameRes.GetYaxis().SetTitleOffset(0.4)

    # rooFitFrameRes.GetYaxis().CenterTitle()
    rooFitFrameRes.GetXaxis().CenterTitle()

    rooFitFrameRes.GetXaxis().SetLabelSize(0.145)
    rooFitFrameRes.GetYaxis().SetLabelSize(0.120)

    rooFitFrameRes.SetAxisRange(-8, 8, "Y")
    rooFitFrameRes.GetYaxis().SetNdivisions(10)
    rooFitFrameRes.GetYaxis().ChangeLabel(1, -1, 0.)
    rooFitFrameRes.GetYaxis().ChangeLabel(3, -1, 0.)
    rooFitFrameRes.GetYaxis().ChangeLabel(5, -1, 0.)
    rooFitFrameRes.GetYaxis().ChangeLabel(7, -1, 0.)
    rooFitFrameRes.GetYaxis().ChangeLabel(9, -1, 0.)
    rooFitFrameRes.GetYaxis().ChangeLabel(11, -1, 0.)

    xMin = rooFitFrameRes.GetXaxis().GetXmin()
    xMax = rooFitFrameRes.GetXaxis().GetXmax()

    gLine1 = ROOT.TLine(xMin, 6, xMax, 6)
    gLine2 = ROOT.TLine(xMin, 2, xMax, 2)
    gLine3 = ROOT.TLine(xMin, -2, xMax, -2)
    gLine4 = ROOT.TLine(xMin, -6, xMax, -6)
    gLine1.SetLineColor(ROOT.kGreen + 3)
    gLine2.SetLineColor(ROOT.kBlue)
    gLine3.SetLineColor(ROOT.kBlue)
    gLine4.SetLineColor(ROOT.kGreen + 3)

    gLine1.SetLineWidth(2)
    gLine2.SetLineWidth(2)
    gLine3.SetLineWidth(2)
    gLine4.SetLineWidth(2)

    DotError)# print("Number of Y axis bins = ", rooFitFrameRes.GetYaxis().GetNbins() )

    c1 = ROOT.TCanvas("c1", "c1", 700, 640)
    c1.SetBottomMargin(0)
    c1.SetLeftMargin(0.25)
    c1.SetRightMargin(0.05)
    c1.Clear()

    Pad1 = ROOT.TPad("p1" + rooRealVar.GetName(), "p1", 0, 0.277, 1, 1, 0)
    Pad2 = ROOT.TPad("p2" + rooRealVar.GetName(), "p2", 0, 0, 1, 0.276, 0)

    if yMin:
        rooFitFrame.SetMinimum(yMin)

    if logAxis:
        rooFitFrame.SetMinimum(0.05)
        Pad1.SetLogy()

    Pad1.Draw()
    Pad2.Draw()

    Pad1.SetLeftMargin(0.15)
    Pad1.SetBottomMargin(0.02)
    Pad1.SetTopMargin(0.06)

    Pad2.SetLeftMargin(0.15)
    Pad2.SetBottomMargin(0.4)
    Pad2.SetTopMargin(0.01)

    print("Residuals added")
    Pad1.cd()

    rooFitFrame.Draw()
    dataHist.Draw("P SAME")

    if legend:
        legend.Draw()

    Pad2.cd()
    rooFitFrameRes.Draw()
    gLine1.Draw("SAME")
    gLine2.Draw("SAME")
    gLine3.Draw("SAME")
    gLine4.Draw("SAME")
    pointsHist.Draw("P SAME")
    # Pad2.Update()

    print("Data + fit plot added")

    nameOfPlot = nameOfPlot + "WithResiduals.pdf"
    c1.SaveAs(nameOfPlot)
    # c1.Destructor()
    c1 = None
    Pad1 = None
    Pad2 = None


def plotChiSqrdNdf(dots, modelCurve):
    """
    Calculates ch^2/Ndf for plots.
    """

    iBin = 0
    ndof = 0
    chiSquared = 0

    xValModel = ctypes.c_double(-1.E30)
    yValModel = ctypes.c_double(-1.E30)
    xValDot = ctypes.c_double(-1.E30)
    yValDot = ctypes.c_double(-1.E30)

    iDotPoint = ROOT.RooRealVar("iDotPoint", "", 0.)
    iModelPoint = ROOT.RooRealVar("iModelPoint", "", 0.)
    iDotError = ROOT.RooRealVar("iDotError", "", 0.)
    # yComb = ROOT.RooFormulaVar("yComb", "", "-1*@0", ROOT.RooArgList(y1RV))
    iResidual = ROOT.RooFormulaVar(
        "yComb", "", "(@0 - @1)/@2",
        ROOT.RooArgList(iDotPoint, iModelPoint, iDotError))

    while iBin < dots.GetN():
        dots.GetPoint(iBin, xValDot, yValDot)

        iDotPoint.setVal(yValDot.value)
        iModelPoint.setVal(modelCurve.interpolate(xValDot.value))
        iDotError.setVal(
            float((dots.GetErrorYlow(iBin) + dots.GetErrorYhigh(iBin)) / 2))

        if np.isnan(iResidual.getVal()) is not True and float(
                yValDot.value) > 1:

            residualValue = iResidual.getVal()
            chiSquared += float(residualValue**2)
            ndof += 1
        iBin += 1

    return float(chiSquared / ndof)
