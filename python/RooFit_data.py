import ROOT
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import sys
import os
sys.path.append("../common")
import common_definitions as cd
import plot_utils as pu
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

# residuals
#res_frame = data.plotOn(Lb_M.frame(100), ROOT.RooFit.DataError(ROOT.RooAbsData.SumW2),ROOT.RooFit.Name("Lb_M"))

data.plotOn(Lb_Mframe)
model.plotOn(Lb_Mframe)
model.Print("t")

#data.plotOn(res_frame)

#Lb_Mframe.Draw()
#res_frame.Draw()

c = ROOT.TCanvas("rf101_basics", "rf101_basics", 800, 400)
ROOT.gPad.SetLeftMargin(0.15)
Lb_Mframe.GetYaxis().SetTitleOffset(1.6)
Lb_Mframe.Draw()
ROOT.gPad.SetLeftMargin(0.15)
Lb_Mframe.Draw()

#c.SaveAs("rf101_basics.png")
#
bins = 100
res_frame = data.plotOn(
    Lb_M.frame(bins), ROOT.RooFit.DataError(ROOT.RooAbsData.SumW2),
    ROOT.RooFit.Name("res"))
model.plotOn(res_frame, ROOT.RooFit.LineStyle(ROOT.kDotted),
             ROOT.RooFit.LineColor(ROOT.kGreen + 3), ROOT.RooFit.LineWidth(4),
             ROOT.RooFit.Name("pdf"))

Lb_Mframe.Print()
dots = res_frame.getHist('res')
curve = res_frame.getCurve('pdf')
pu.plotWithResiduals(Lb_M, dots, curve, '', '', False, False, None, None, None,
                     None, Lb_Mframe, 'model')
