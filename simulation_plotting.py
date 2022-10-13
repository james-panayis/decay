# This file should contain all functions to plot ROC curves
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ROOT
from itertools import product
#from lhcbPlotStyle import setLHCbPlotStyle
from scipy import stats

plt.rc("font", **{"family": "serif"})  # , "serif": ["Roman"]})
plt.rc("text", usetex=True)

#lhcbName = setLHCbPlotStyle()

#HISTOGRAM FUNCTIONS
def chiSqrdNdf(histo_one, histo_two):
    i_bin = 1
    chiSqrd = 0
    while i_bin < 1 + histo_one.GetNbinsX():
        chiSqrd += (histo_two.GetBinContent(i_bin) - histo_one.GetBinContent(
            i_bin))**2 / (histo_one.GetBinError(i_bin)**2)
        i_bin += 1
    return chiSqrd / histo_one.GetNbinsX()


def makeHist(name,
             files=None,
             var_name="xgb_output_noTrigger",
             bins=25,
             xmin=None,
             xmax=None,
             cut_string=""):

    Pi_Var = f"Pi{var_name}"
    K_Var = f"K{var_name}"

    infile = ROOT.TFile(f"{files}")
    tree = infile.Get("tree")
    test_rdframe = ROOT.RDataFrame(tree)

    if xmin == None:
        xmin = min(test_rdframe.Min(Pi_Var).GetValue(), test_rdframe.Min(K_Var).GetValue())
    if xmax == None:
        xmax = max(test_rdframe.Max(Pi_Var).GetValue(), test_rdframe.Max(K_Var).GetValue())

    print(f"XMIN: {xmin}")
    print(f"XMAX: {xmax}")

    Pi_cut_string = cut_string + f"{Pi_Var} > {xmin} && {Pi_Var} < {xmax}"
    K_cut_string = cut_string + f"{K_Var} > {xmin} && {K_Var} < {xmax}"

    h_Pi = ROOT.TH1F("h_Pi", "Pi", bins, xmin, xmax)
    tree.Draw(f"{Pi_Var} >> h_Pi",
                   f"({Pi_cut_string})", "")

    h_K = ROOT.TH1F("h_K", "K", bins, xmin, xmax)
    tree.Draw(f"{K_Var} >> h_K",
                   f"({K_cut_string})", "")


    print(f"h_Pi: {h_Pi.Integral()}")
    print(f"h_K: {h_K.Integral()}")

    h_Pi.Scale(1 / h_Pi.Integral())
    h_K.Scale(1 / h_K.Integral())

    y_min = [
        h_Pi.GetBinContent(h_Pi.GetMinimumBin()),
        h_K.GetBinContent(h_K.GetMinimumBin()),
    ]
    y_max = [
        h_Pi.GetBinContent(h_Pi.GetMaximumBin()),
        h_K.GetBinContent(h_K.GetMaximumBin()),
    ]

    y_min_val = min(y_min) / 2 if min(y_min) != 0 else 0.0001

    print("MIN AND MAX: ", y_min_val, max(y_max) )
    h_Pi.GetYaxis().SetRangeUser(y_min_val, max(y_max) + 0.40*max(y_max))
    #h_Pi.GetYaxis().SetRangeUser(y_min_val, 0.4)

    c0 = ROOT.TCanvas(name, "c0", 600, 500)
    Pad = ROOT.TPad("p1full", "p1", 0, 0, 1, 1, 0, 0, 0)
    Pad.SetLeftMargin(0.16)
    Pad.SetBottomMargin(0.15)
    Pad.SetTopMargin(0.06)
    #Pad.SetLogy()
    Pad.Draw()
    Pad.cd()

    h_Pi.GetXaxis().SetTitle(var_name)
    h_Pi.GetYaxis().SetTitle("Events")
    h_Pi.GetXaxis().CenterTitle()
    h_Pi.GetYaxis().CenterTitle()

    h_Pi.SetLineColor(ROOT.kBlue)
    h_K.SetLineColor(ROOT.kRed)
    h_Pi.SetLineStyle(1)
    h_K.SetLineStyle(1)


    h_Pi.Draw("HIST")
    h_K.Draw("HIST same")


    legend = ROOT.TLegend(0.3, 0.7, 0.8, 0.9)
    #legend.SetLegendTextSize(42)
    legend.AddEntry(h_Pi, "Pi ", "lf")
    legend.AddEntry(h_K, "K", "lf")
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.Draw()
    
    c0.Print(f"cache/{name}2.pdf")

    #test_sig_frame = test_rdframe.Filter("label==1").AsNumpy([f"{var_name}"])
    #test_bkg_frame = test_rdframe.Filter("label==0").AsNumpy([f"{var_name}"])


    return


variables = ["U", "UT", "P", "PT", "d"]
for var in variables:
    makeHist(var, 'cache/simulation_task_data.root', var, 100)

