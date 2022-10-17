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

def makeHist(name,
             fileA=None,
             fileB=None,
             var_name="xgb_output_noTrigger",
             bins=25,
             xmin=None,
             xmax=None,
             cut_string=""):

    infileA = ROOT.TFile(f"{fileA}")
    treeA = infileA.Get("Lb_Tuple/DecayTree")
    test_rdframeA = ROOT.RDataFrame(treeA)

    infileB = ROOT.TFile(f"{fileB}")
    treeB = infileB.Get("tree")
    test_rdframeB = ROOT.RDataFrame(treeB)

    print("RUNNING: ", var_name)


    if xmin == None:
        xmin = min(test_rdframeA.Min(var_name).GetValue(), test_rdframeB.Min(var_name).GetValue())
    if xmax == None:
        xmax = max(test_rdframeA.Max(var_name).GetValue(), test_rdframeB.Max(var_name).GetValue())

#    print(f"XMIN: {xmin}")
#    print(f"XMAX: {xmax}")

    h_c = ROOT.TH1F("h_c", "A", bins, xmin, xmax)
    treeA.Draw(f"{var_name} >> h_c",
                   f"{cut_string}", "")
    nq = 2
    xq = np.array([0.0025, 0.9975])
    yq = np.empty(2)

    h_c.GetQuantiles(nq, yq, xq)

#    print("LOOK HERE:", yq[0], yq[1])
    xmin = yq[0]
    xmax = yq[1]
    cut_string += f"{var_name} > {xmin} && {var_name} < {xmax}"

    h_A = ROOT.TH1F("h_A", "A", bins, xmin, xmax)
    treeA.Draw(f"{var_name} >> h_A",
                   f"{cut_string}", "")

    h_B = ROOT.TH1F("h_B", "B", bins, xmin, xmax)
    treeB.Draw(f"{var_name} >> h_B",
                   f"{cut_string}", "")


    print(f"h_A: {h_A.Integral()}")
    print(f"h_B: {h_B.Integral()}")

    if h_A.Integral() == 0 or h_B.Integral() == 0:
        print("VARIABLE FAILED")
        return

    h_A.Scale(1 / h_A.Integral())
    h_B.Scale(1 / h_B.Integral())


    h_A.GetXaxis().SetRangeUser(yq[0], yq[1])

    y_min = [
        h_A.GetBinContent(h_A.GetMinimumBin()),
        h_B.GetBinContent(h_B.GetMinimumBin()),
    ]
    y_max = [
        h_A.GetBinContent(h_A.GetMaximumBin()),
        h_B.GetBinContent(h_B.GetMaximumBin()),
    ]

    y_min_val = min(y_min) / 2 if min(y_min) != 0 else 0.0001

    h_A.GetYaxis().SetRangeUser(y_min_val, max(y_max) + 0.1*max(y_max))

    c0 = ROOT.TCanvas(f"{name}", "c0", 600, 500)
    Pad = ROOT.TPad("p1full", "p1", 0, 0, 1, 1, 0, 0, 0)
    Pad.SetLeftMargin(0.16)
    Pad.SetBottomMargin(0.15)
    Pad.SetTopMargin(0.06)
    #Pad.SetLogy()
    Pad.Draw()
    Pad.cd()

    h_A.GetXaxis().SetTitle(f"{var_name}")
    h_A.GetYaxis().SetTitle("Events")
    h_A.GetXaxis().CenterTitle()
    h_A.GetYaxis().CenterTitle()

    h_A.SetLineColor(ROOT.kBlue)
    h_B.SetLineColor(ROOT.kRed)

    h_A.Draw("HIST")
    h_B.Draw("HIST same")

    legend = ROOT.TLegend(0.3, 0.7, 0.8, 0.9)
    #legend.SetLegendTextSize(42)
    legend.AddEntry(h_A, "Raw Data", "lf")
    legend.AddEntry(h_B, "Signal Only", "lf")
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.Draw()


    c0.Print(f"cache/variables/{name}.pdf")

    return


A = '../../../../../../../work/c/cawhite/public/data/Lb2pKmm_mgUp_2018.root'
B = '../../../../../../../work/c/cawhite/public/data/Lb2pKmm_sim_mgUp_2018.root'

#
variables = [
    'h1_PT',
#    'h2_PT',
#    'mu1_PT',
#    'mu2_PT',
#    'Lb_PT',
#    'Lres_PT',
#    'Lb_ENDVERTEX_X',
#    'Lb_ENDVERTEX_Y',
#    'Lb_ENDVERTEX_Z',
#    'Lb_ENDVERTEX_CHI2',
#    'Lb_ENDVERTEX_NDOF',
#    'Lres_ENDVERTEX_X',
#    'Lres_ENDVERTEX_Y',
#    'Lres_ENDVERTEX_Z',
#    'Lres_ENDVERTEX_CHI2',
#    'Lres_ENDVERTEX_NDOF',
#    'Jpsi_ENDVERTEX_X',
#    'Jpsi_ENDVERTEX_Y',
#    'Jpsi_ENDVERTEX_Z',
#    'Jpsi_ENDVERTEX_CHI2',
#    'Jpsi_ENDVERTEX_NDOF',
]
#
cut = {
    'h1_PT': [None, None], 
#    'h2_PT': [0, 20000],
#    'mu1_PT': [0, 20000],
#    'mu2_PT': [0, 20000],
#    'Lb_PT': [0, 30000],
#    'Lres_PT': [0, 20000],
#    'Lb_ENDVERTEX_X':[-10,10],
#    'Lb_ENDVERTEX_Y':[-10,10],
#    'Lb_ENDVERTEX_Z':[-200,200],
#    'Lb_ENDVERTEX_CHI2':[0, 45],
#    'Lb_ENDVERTEX_NDOF':[4.5,6.5],
#    'Lres_ENDVERTEX_X':[-10,10],
#    'Lres_ENDVERTEX_Y':[-10,10],
#    'Lres_ENDVERTEX_Z':[-200,700],
#    'Lres_ENDVERTEX_CHI2':[0,13],
#    'Lres_ENDVERTEX_NDOF':[0.5,2.5],
#    'Jpsi_ENDVERTEX_X':[-10,10],
#    'Jpsi_ENDVERTEX_Y':[-10,10],
#    'Jpsi_ENDVERTEX_Z':[-200,700],
#    'Jpsi_ENDVERTEX_CHI2':[0,13],
#    'Jpsi_ENDVERTEX_NDOF':[0.5,2.5],
#    'Jpsi_IPCHI2_OWNPV':[],
#    'h1_IPCHI2_OWNPV': [0, ], 
#    'h2_IPCHI2_OWNPV': [0, ],
#    'mu1_IPCHI2_OWNPV': [0, ],
#    'mu2_IPCHI2_OWNPV': [0, ],
#    'Lb_IPCHI2_OWNPV': [0, ],
#    'Lres_IPCHI2_OWNPV': [0, ],
}




infileA = ROOT.TFile("../../../../../../../work/c/cawhite/public/data/Lb2pKmm_mgUp_2018.root")
treeA = infileA.Get("Lb_Tuple/DecayTree")
test_rdframeA = ROOT.RDataFrame(treeA)

variables = test_rdframeA.GetColumnNames()

print(variables.size())
for var in variables:
    makeHist(var, A, B, var, 100, None, None)

