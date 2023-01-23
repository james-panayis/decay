import ROOT
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import sys
import os
sys.path.append("../common")
import common_definitions as cd
import matplotlib as plt
#plt.rcParams['text.latex.preamble'] = [r'\usepackage{bm}']
plt.rc("font", **{"family": "serif"})  # , "serif": ["Roman"]})
plt.rc("text", usetex=True)

letter = "H"
File = ROOT.TFile(f"../cache/D4J/Real_{letter}_D4J.root")
tree = File.Get("tree")

lower_band = 5350.0
#lower_band = 5010.0
upper_band = 5850.0
#upper_band = 6900.0
lb_m = 5619.6
cut = 0.981

Lb_M = ROOT.RooRealVar("Lb_M", "Lb_M", lower_band, upper_band)
xgb_output = ROOT.RooRealVar("xgb_output", "xgb_output", 0, 1)
nsig = ROOT.RooRealVar("nsig", "nsig", 10000, 0, 1000000)
nbkg = ROOT.RooRealVar("nbkg", "nbkg", 10000, 0, 1000000)
nrand = ROOT.RooRealVar("nrand", "nrand", 10000, 0, 1000000)

#sig
mean = ROOT.RooRealVar("mean", "mean of gaussian", 5619.6)
sigma = ROOT.RooRealVar("sigma", "width of gaussian", 1, 0.1, 1000)
sig = ROOT.RooGaussian("gauss", "gaussian PDF", Lb_M, mean, sigma)

#bkg
decay = ROOT.RooRealVar("decay", "decay constant", 0, -0.05, 0.05)
bkg = ROOT.RooExponential("exp", "exponential PDF", Lb_M, decay)

#rand
randmean = ROOT.RooRealVar("randmean", "mean of gaussian", 5510)
randsigma = ROOT.RooRealVar("randsigma", "width of gaussian", 1, 0.5, 1000)
rand = ROOT.RooGaussian("randgauss", "gaussian PDF", Lb_M, randmean, randsigma)

#model
model = ROOT.RooAddPdf("model", "", [sig, bkg, rand], [nsig, nbkg, nrand])
#model = ROOT.RooAddPdf("model", "", [sig,bkg], [nsig, nbkg])
print("made model")

#data
#data = model.generate(ROOT>RooArgSet(Lb_M), 1000)
#print("generated data")

#frame
Lb_Mframe = Lb_M.frame(Title="Fitting")

#dataset = makedata(tree, 0.981, lower_band, upper_band, 100)
#data = RooDataHist("dataset", "dataset", {Lb_M}, Import=tree)

data = ROOT.RooDataSet(
    "data",
    "data",
    ROOT.RooArgSet(Lb_M, xgb_output),
    Import=tree,
    Cut=f"xgb_output>{cut}")

model.fitTo(data)

data.plotOn(Lb_Mframe)
model.plotOn(Lb_Mframe)
model.Print("t")

Lb_Mframe.Draw()

#xframe2 = x.frame(Title = "Gaussian pdf with data")
#data.plotOn(xframe2)
#gauss.plotOn(xframe2)
#
#gauss.fitTo(data)
#mean.Print()
#sigma.Print()
#c = ROOT.TCanvas("rf101_basics", "rf101_basics", 800, 400)
#c.Divide(2)
#c.cd(1)
#ROOT.gPad.SetLeftMargin(0.15)
#xframe.GetYaxis().SetTitleOffset(1.6)
#xframe.Draw()
#c.cd(2)
#ROOT.gPad.SetLeftMargin(0.15)
#xframe2.GetYaxis().SetTitleOffset(1.6)
#xframe2.Draw()
#
#c.SaveAs("rf101_basics.png")
#
#
#
#
#
#
#
#
#
#
#
#
#def makedata(tree, xgb_cut_val, lb, ub, bins):
#    cut_string = f"xgb_output > {xgb_cut_val}"
#    xmin = lb
#    xmax = ub
#    bins = bins
#
#    hist_data = ROOT.TH1F("hist_data", "hist_data", bins, xmin, xmax)
#
#    tree.fill("Lb_M >> hist_data", cut_string, "")
#
#    return hist_data
#
#
#
#
#
#
#
#
