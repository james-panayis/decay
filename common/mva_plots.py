# This file should contain all functions to plot ROC curves
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ROOT
from itertools import product
import sys
import common_definitions as cd
import os
from lhcbPlotStyle import setLHCbPlotStyle
from scipy import stats

plt.rc("font", **{"family": "serif"})  # , "serif": ["Roman"]})
plt.rc("text", usetex=True)

lhcbName = setLHCbPlotStyle()

#savingDirectory = "Bc_pimumu_work/roc_curves"
savingDirectory = "mixed_sample_plots"


#ROC CURVE FUNCTIONS
def integrate(x, y):
    # integrates the area under the ROC curve
    sm = 0
    for i in range(1, len(x)):
        h = x[i] - x[i - 1]
        sm += h * (y[i - 1] + y[i]) / 2

    return sm


def plot_roc_curve(sig_tree,
                   bkg_tree,
                   output_one="bdt_output",
                   line_one="OLD",
                   mcweightvar="",
                   plot_name=""):
    #Plots a single ROC curve

    N = 1000
    x = np.linspace(start=0, stop=1, num=N)[::-1]

    if mcweightvar != "":
        sig_total = sum(sig_tree[mcweightvar])
        one_sig_acc = np.array([
            sum(sig_tree.loc[sig_tree[output_one] > x_el][mcweightvar]) /
            sig_total for x_el in x
        ])
    else:
        sig_total = len(sig_tree)
        one_sig_acc = np.array(
            [sum(sig_tree[output_one] > x_el) / sig_total for x_el in x])

    bkg_total = len(bkg_tree)

    one_bkg_rej = np.array(
        [sum(bkg_tree[output_one] < x_el) / bkg_total for x_el in x])

    one_area = integrate(one_sig_acc, one_bkg_rej)

    line1, = plt.plot(
        one_sig_acc,
        one_bkg_rej,
        ls="solid",
        linewidth=3,
        c="k",
        label=f"AUC = {one_area:.6f}")

    plt.xlabel(r"Signal efficiency", fontsize="xx-large")
    plt.ylabel(r"Background rejection", fontsize="xx-large")

    plt.xlim(0, 1.01)
    plt.ylim(0, 1.01)

    plt.tick_params(axis="y", which="major", labelsize="x-large")
    plt.tick_params(axis="x", which="both", labelsize="x-large")
    plt.legend(fontsize="x-large", loc='center left')
    plt.tight_layout()

    output_file_name = f"{savingDirectory}/ROC_{plot_name}.pdf"
    plt.savefig(output_file_name)
    plt.clf()

    print(f"Saved file:  {output_file_name}")


def plot_2roc_curves(sig_tree,
                     bkg_tree,
                     sig_tree2,
                     bkg_tree2,
                     output_one="bdt_output",
                     output_two="bdt_output_Jpsi_psi2s_veto",
                     line_one="OLD",
                     line_two="NEW",
                     mcweightvar="",
                     plot_name=""):

    #Plots two ROC curves against eachother

    N = 1000
    x = np.linspace(start=0, stop=1, num=N)[::-1]

    if mcweightvar != "":
        sig_total = sum(sig_tree[mcweightvar])
        sig_total2 = sum(sig_tree2[mcweightvar])
        one_sig_acc = np.array([
            sum(sig_tree.loc[sig_tree[output_one] > x_el][mcweightvar]) /
            sig_total for x_el in x
        ])
        two_sig_acc = np.array([
            sum(sig_tree2.loc[sig_tree2[output_two] > x_el][mcweightvar]) /
            sig_total2 for x_el in x
        ])
    else:
        sig_total = len(sig_tree)
        sig_total2 = len(sig_tree2)
        one_sig_acc = np.array(
            [sum(sig_tree[output_one] > x_el) / sig_total for x_el in x])
        two_sig_acc = np.array(
            [sum(sig_tree2[output_two] > x_el) / sig_total2 for x_el in x])

    bkg_total = len(bkg_tree)
    bkg_total2 = len(bkg_tree2)

    one_bkg_rej = np.array(
        [sum(bkg_tree[output_one] < x_el) / bkg_total for x_el in x])

    one_area = integrate(one_sig_acc, one_bkg_rej)

    two_bkg_rej = np.array(
        [sum(bkg_tree2[output_two] < x_el) / bkg_total2 for x_el in x])

    two_area = integrate(two_sig_acc, two_bkg_rej)

    line1, = plt.plot(
        one_sig_acc,
        one_bkg_rej,
        ls="solid",
        linewidth=3,
        c="k",
        label=f"{line_one} (AUC = {one_area:.6f})")

    # plt.text(0.1, 0.8, s=f"AUC (OLD)", fontsize="xx-large", color="k")
    # plt.text(0.41, 0.8, s=f"{one_area:.3f}", fontsize="xx-large", color="k")

    line2, = plt.plot(
        two_sig_acc,
        two_bkg_rej,
        ls="dotted",
        linewidth=4,
        c="grey",
        label=f"{line_two} (AUC = {two_area:.6f})")

    #    output_three = "bdt_output_Bc_Jpsipi_psiveto"
    #    line_three = "bdt_output_Bc_Jpsipi_psiveto"
    #
    #
    #    x_long = np.linspace(start=-0.9, stop=1, num=N)[::-1]
    #
    #    three_sig_acc = np.array([
    #        sum(sig_tree.loc[sig_tree[output_three] > x_el]
    #            ["mva_mcweight_nTracks_B_PT"]) / sig_total for x_el in x_long
    #    ])
    #    three_bkg_rej = np.array(
    #        [sum(bkg_tree[output_three] < x_el) / bkg_total for x_el in x_long])
    #
    #    three_area = integrate(three_sig_acc, three_bkg_rej)
    #
    #    line3, = plt.plot(
    #        three_sig_acc,
    #        three_bkg_rej,
    #       ls="dashdot",
    #        linewidth=4,
    #        c="red",
    #        label=f"{line_three} (AUC = {three_area:.6f})")

    # plt.text(0.1, 0.7, s=f"AUC (NEW)", fontsize="xx-large", color="grey")
    # plt.text(0.41, 0.7, s=f"{two_area:.3f}", fontsize="xx-large", color="grey")

    # plt.text(0.14, 0.75, s=r"\textit{\textbf{LHCb}}", fontsize="large", color="k")
    # plt.text(0.1, 0.7, s=r"MC simulation", fontsize="large", color="k")

    plt.xlabel(r"Signal efficiency", fontsize="xx-large")
    plt.ylabel(r"Background rejection", fontsize="xx-large")

    plt.xlim(0, 1.01)
    plt.ylim(0, 1.01)

    plt.tick_params(axis="y", which="major", labelsize="x-large")
    plt.tick_params(axis="x", which="both", labelsize="x-large")
    plt.legend(fontsize="x-large", loc='center left')
    plt.tight_layout()

    output_file_name = f"{savingDirectory}/ROC_{plot_name}.pdf"
    plt.savefig(output_file_name)
    plt.ylim(0.99, 1.01)
    plt.yscale('log')
    plt.legend(fontsize="x-large", loc='upper left')

    plt.subplots_adjust(
        left=0.2, right=0.9, top=0.9, bottom=0.125, wspace=0.2, hspace=0.2)

    output_file_name_log = f"{savingDirectory}/ROC_{plot_name}_log.pdf"
    plt.savefig(output_file_name_log)

    plt.clf()

    print(f"Saved files: {output_file_name}")
    print(f"             {output_file_name_log}")

    return one_area, two_area


def plot_3roc_curves(sig_tree,
                     bkg_tree,
                     sig_tree2,
                     bkg_tree2,
                     sig_tree3,
                     bkg_tree3,
                     output_one="bdt_output",
                     output_two="bdt_output_Jpsi_psi2s_veto",
                     output_three="xgb_output",
                     line_one="OLD",
                     line_two="NEW",
                     line_three="",
                     mcweightvar="",
                     plot_name=""):

    N = 1000
    x = np.linspace(start=0, stop=1, num=N)[::-1]

    if mcweightvar != "":
        sig_total = sum(sig_tree[mcweightvar])
        sig_total2 = sum(sig_tree2[mcweightvar])
        sig_total3 = sum(sig_tree3[mcweightvar])
        one_sig_acc = np.array([
            sum(sig_tree.loc[sig_tree[output_one] > x_el][mcweightvar]) /
            sig_total for x_el in x
        ])
        two_sig_acc = np.array([
            sum(sig_tree2.loc[sig_tree2[output_two] > x_el][mcweightvar]) /
            sig_total2 for x_el in x
        ])
        three_sig_acc = np.array([
            sum(sig_tree3.loc[sig_tree3[output_three] > x_el][mcweightvar]) /
            sig_total3 for x_el in x
        ])
    else:
        sig_total = len(sig_tree)
        sig_total2 = len(sig_tree2)
        sig_total3 = len(sig_tree3)
        one_sig_acc = np.array(
            [sum(sig_tree[output_one] > x_el) / sig_total for x_el in x])
        two_sig_acc = np.array(
            [sum(sig_tree2[output_two] > x_el) / sig_total2 for x_el in x])
        three_sig_acc = np.array(
            [sum(sig_tree3[output_three] > x_el) / sig_total3 for x_el in x])

    bkg_total = len(bkg_tree)
    bkg_total2 = len(bkg_tree2)
    bkg_total3 = len(bkg_tree3)

    one_bkg_rej = np.array(
        [sum(bkg_tree[output_one] < x_el) / bkg_total for x_el in x])

    one_area = integrate(one_sig_acc, one_bkg_rej)

    two_bkg_rej = np.array(
        [sum(bkg_tree2[output_two] < x_el) / bkg_total2 for x_el in x])

    two_area = integrate(two_sig_acc, two_bkg_rej)

    three_bkg_rej = np.array(
        [sum(bkg_tree3[output_three] < x_el) / bkg_total3 for x_el in x])

    three_area = integrate(three_sig_acc, three_bkg_rej)

    line1, = plt.plot(
        one_sig_acc,
        one_bkg_rej,
        ls="dotted",
        linewidth=3,
        c="red",
        label=f"{line_one} (AUC = {one_area:.6f})")

    # plt.text(0.1, 0.8, s=f"AUC (OLD)", fontsize="xx-large", color="k")
    # plt.text(0.41, 0.8, s=f"{one_area:.3f}", fontsize="xx-large", color="k")

    line2, = plt.plot(
        two_sig_acc,
        two_bkg_rej,
        ls="dotted",
        linewidth=4,
        c="grey",
        label=f"{line_two} (AUC = {two_area:.6f})")

    line3, = plt.plot(
        three_sig_acc,
        three_bkg_rej,
        ls="dotted",
        linewidth=4,
        c="blue",
        label=f"{line_three} (AUC = {three_area:.6f})")

    # plt.text(0.1, 0.7, s=f"AUC (NEW)", fontsize="xx-large", color="grey")
    # plt.text(0.41, 0.7, s=f"{new_area:.3f}", fontsize="xx-large", color="grey")

    # plt.text(0.14, 0.75, s=r"\textit{\textbf{LHCb}}", fontsize="large", color="k")
    # plt.text(0.1, 0.7, s=r"MC simulation", fontsize="large", color="k")

    plt.xlabel(r"Signal efficiency", fontsize="xx-large")
    plt.ylabel(r"Background rejection", fontsize="xx-large")

    plt.xlim(0, 1.01)
    plt.ylim(0, 1.01)

    plt.tick_params(axis="y", which="major", labelsize="x-large")
    plt.tick_params(axis="x", which="both", labelsize="x-large")
    plt.legend(fontsize="x-large", loc='center left')
    plt.tight_layout()

    output_file_name = f"{savingDirectory}/ROC_{plot_name}.pdf"
    plt.savefig(output_file_name)
    plt.ylim(0.99, 1.01)
    plt.yscale('log')
    plt.legend(fontsize="x-large", loc='upper left')

    plt.subplots_adjust(
        left=0.2, right=0.9, top=0.9, bottom=0.125, wspace=0.2, hspace=0.2)

    output_file_name_log = f"{savingDirectory}/ROC_{plot_name}_log.pdf"
    plt.savefig(output_file_name_log)

    plt.clf()

    print(f"Saved files: {output_file_name}")
    print(f"             {output_file_name_log}")


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
             tree_test=None,
             tree_train=None,
             var_name="xgb_output_noTrigger",
             bins=25,
             xmin=None,
             xmax=None,
             weight_name="1",
             cut_string=""):

    test_rdframe = ROOT.RDataFrame(tree_test)
    train_rdframe = ROOT.RDataFrame(tree_train)

    if xmin == None:
        xmin = test_rdframe.Min(var_name).GetValue()
    if xmax == None:
        xmax = test_rdframe.Max(var_name).GetValue()

    print(f"XMIN: {xmin}")
    print(f"XMAX: {xmax}")

    cut_string += f" && {var_name} > {xmin} && {var_name} < {xmax}"

    h_test_sig = ROOT.TH1F("h_test_sig", "test_sig", bins, xmin, xmax)
    tree_test.Draw(f"{var_name} >> h_test_sig",
                   f"(label==1 {cut_string})*{weight_name}", "")

    h_test_bkg = ROOT.TH1F("h_test_bkg", "test_bkg", bins, xmin, xmax)
    tree_test.Draw(f"{var_name} >> h_test_bkg",
                   f"(label==0 {cut_string})*{weight_name}", "")

    h_train_sig = ROOT.TH1F("h_train_sig", "train_sig", bins, xmin, xmax)
    tree_train.Draw(f"{var_name} >> h_train_sig",
                    f"(label==1 {cut_string})*{weight_name}", "")

    h_train_bkg = ROOT.TH1F("h_train_bkg", "train_bkg", bins, xmin, xmax)
    tree_train.Draw(f"{var_name} >> h_train_bkg",
                    f"(label==0 {cut_string})*{weight_name}", "")

    print(f"h_test_sig: {h_test_sig.Integral()}")
    print(f"h_test_bkg: {h_test_bkg.Integral()}")
    print(f"h_train_sig: {h_train_sig.Integral()}")
    print(f"h_train_bkg: {h_train_bkg.Integral()}")

    h_test_sig.Scale(1 / h_test_sig.Integral())
    h_test_bkg.Scale(1 / h_test_bkg.Integral())
    h_train_sig.Scale(1 / h_train_sig.Integral())
    h_train_bkg.Scale(1 / h_train_bkg.Integral())

    y_min = [
        h_test_sig.GetBinContent(h_test_sig.GetMinimumBin()),
        h_test_bkg.GetBinContent(h_test_bkg.GetMinimumBin()),
        h_train_sig.GetBinContent(h_train_sig.GetMinimumBin()),
        h_train_bkg.GetBinContent(h_train_bkg.GetMinimumBin())
    ]
    y_max = [
        h_test_sig.GetYaxis().GetXmax(),
        h_test_bkg.GetYaxis().GetXmax(),
        h_train_sig.GetYaxis().GetXmax(),
        h_train_bkg.GetYaxis().GetXmax()
    ]

    y_min_val = min(y_min) / 2 if min(y_min) != 0 else 0.0001

    h_test_sig.GetYaxis().SetRangeUser(y_min_val, max(y_max))

    c0 = ROOT.TCanvas(name, "c0", 600, 500)
    Pad = ROOT.TPad("p1full", "p1", 0, 0, 1, 1, 0, 0, 0)
    Pad.SetLeftMargin(0.16)
    Pad.SetBottomMargin(0.15)
    Pad.SetTopMargin(0.06)
    Pad.SetLogy()
    Pad.Draw()
    Pad.cd()

    h_test_sig.GetXaxis().SetTitle(var_name)
    h_test_sig.GetYaxis().SetTitle("Events")
    h_test_sig.GetXaxis().CenterTitle()
    h_test_sig.GetYaxis().CenterTitle()

    h_test_sig.SetLineColor(ROOT.kBlue)
    h_test_bkg.SetLineColor(ROOT.kRed)
    h_train_sig.SetLineColor(ROOT.kBlue)
    h_train_bkg.SetLineColor(ROOT.kRed)

    h_train_sig.SetLineStyle(ROOT.kDotted)
    h_train_bkg.SetLineStyle(ROOT.kDotted)
    h_test_sig.SetLineStyle(1)
    h_test_bkg.SetLineStyle(1)

    h_test_sig.SetFillStyle(3365)
    h_test_bkg.SetFillStyle(3352)
    h_test_sig.SetFillColorAlpha(ROOT.kBlue, 0.3)
    h_test_bkg.SetFillColorAlpha(ROOT.kRed, 0.3)
    h_test_sig.SetLineColorAlpha(ROOT.kBlue, 0.6)
    h_test_bkg.SetLineColorAlpha(ROOT.kRed, 0.6)

    h_train_sig.SetMarkerColorAlpha(ROOT.kBlue, 0.8)
    h_train_bkg.SetMarkerColorAlpha(ROOT.kRed, 0.8)

    h_test_sig.Draw("HIST")
    h_test_bkg.Draw("HIST same")
    h_train_sig.Draw("E same")
    h_train_bkg.Draw("E same")

    Kol_bkg = h_test_bkg.KolmogorovTest(h_train_bkg)
    Kol_sig = h_test_sig.KolmogorovTest(h_train_sig)
    print("KS SIG :")
    print(Kol_sig)
    print("KS BKG :")
    print(Kol_bkg)

    legend = ROOT.TLegend(0.3, 0.7, 0.8, 0.9)
    #legend.SetLegendTextSize(42)
    legend.AddEntry(h_test_sig, "Test Signal", "lf")
    legend.AddEntry(h_test_bkg, "Test Background", "lf")
    legend.AddEntry(h_train_sig, "Train Signal", "p")
    legend.AddEntry(h_train_bkg, "Train Background", "p")
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.Draw()

    t = ROOT.TText(
        0.17, 0.05,
        f"KS test for sig.(bkg.) = {Kol_sig:0.4f} ({Kol_bkg:0.4f}) ")
    t.SetTextSize(0.04)
    t.Draw()

    c0.Print(f"{name}")
    if var_name in ["xgb_output!<---------------"]:
        sig_chiSqrd = chiSqrdNdf(h_test_sig, h_train_sig)
        bkg_chiSqrd = chiSqrdNdf(h_test_bkg, h_train_bkg)

        print("SIG Chi Squared:")
        print(sig_chiSqrd)
        print("BKG Chi Squared:")
        print(bkg_chiSqrd)

    test_sig_frame = test_rdframe.Filter("label==1").AsNumpy([f"{var_name}"])
    test_bkg_frame = test_rdframe.Filter("label==0").AsNumpy([f"{var_name}"])
    train_sig_frame = train_rdframe.Filter("label==1").AsNumpy([f"{var_name}"])
    train_bkg_frame = train_rdframe.Filter("label==0").AsNumpy([f"{var_name}"])

    sig_ks = stats.kstest(train_sig_frame[f"{var_name}"],
                          test_sig_frame[f"{var_name}"])

    bkg_ks = stats.kstest(train_bkg_frame[f"{var_name}"],
                          test_bkg_frame[f"{var_name}"])

    print("KS SIG python:")
    print(sig_ks)
    print("KS BKG python:")
    print(bkg_ks)

    return Kol_sig, Kol_bkg, sig_ks, bkg_ks


# Correlation Plot


def draw_correlation(df=None, label='', filename=''):
    '''
    Produce correlation plots from pandas dataframe
    '''
    if df is None: return
    print(df.columns)

    df = df.drop(columns=[])

    print(df.columns)

    corr = df.corr()

    f, a = plt.subplots()
    a.imshow(corr)

    ncol = len(df.columns)

    a.set_xticks(np.arange(ncol))
    a.set_yticks(np.arange(ncol))

    a.set_xticklabels(df.columns)
    a.set_yticklabels(df.columns)

    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)

    plt.setp(
        a.get_xticklabels(), rotation=20, ha="right", rotation_mode="anchor")

    for i in range(ncol):
        for j in range(ncol):
            t = a.text(
                i,
                j,
                '%.2f' % corr.values[i, j],
                ha="center",
                va="center",
                color="w",
                fontsize=6)
    plt.title(label)
    plt.savefig(filename)
    print('\nCreated ' + filename + '\n')
    #plt.show()
    plt.clf()
    return


def divide_cut_plot(filename,
                    variable,
                    original_cut,
                    cut_string,
                    plot_name,
                    save_dir=".",
                    bins=50,
                    xmin=0,
                    xmax=4000):
    '''
    Produces a graph showing the result of the cut variable histogram for the chosen variable divided by the full histogram
    '''

    rfile = ROOT.TFile(f"{filename}")
    rtree = rfile.Get("tree")

    full_entries = rtree.GetEntries(f"{variable} && {original_cut}")
    cut_entries = rtree.GetEntries(f"{variable} && {cut_string}")
    divided_entries = cut_entries / full_entries

    full_hist = ROOT.TH1F("full_hist", "full", bins, xmin, xmax)
    cut_hist = ROOT.TH1F("cut_hist", "cut", bins, xmin, xmax)

    rtree.Draw(f"{variable} >> full_hist", f"{original_cut}", "")
    rtree.Draw(f"{variable} >> cut_hist", f"{cut_string}", "")

    full_hist.SetDefaultSumw2()
    cut_hist.SetDefaultSumw2()

    cut_hist.Divide(full_hist)

    c0 = ROOT.TCanvas(plot_name, "c0", 600, 500)
    Pad = ROOT.TPad("p1full", "p1", 0, 0, 1, 1, 0, 0, 0)
    Pad.SetLeftMargin(0.16)
    Pad.SetBottomMargin(0.15)
    Pad.SetTopMargin(0.06)
    Pad.Draw()
    Pad.cd()

    cut_hist.SetLineColor(ROOT.kRed)
    cut_hist.Draw("E")

    line = ROOT.TLine(xmin, divided_entries, xmax, divided_entries)
    line.SetLineColor(ROOT.kBlue)
    line.SetLineStyle(4)
    line.SetLineWidth(4)

    line.Draw("SAME")

    cut_hist.GetXaxis().SetTitle(variable)
    cut_hist.GetYaxis().SetTitle("cut / full entries")
    cut_hist.GetXaxis().CenterTitle()
    cut_hist.GetYaxis().CenterTitle()

    c0.Print(f"{save_dir}/{plot_name}.pdf")

    print(f"plot made: {save_dir}/{plot_name}.pdf")
