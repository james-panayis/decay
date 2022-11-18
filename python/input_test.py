import ROOT
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from simple_mva_xgboost import *
import sys
import os
sys.path.append("../common")
import common_definitions as cd
from mva_plots import plot_roc_curve, makeHist, plot_2roc_curves, plot_3roc_curves, draw_correlation
# plt.rcParams['text.latex.preamble'] = [r'\usepackage{bm}']
plt.rc("font", **{"family": "serif"})  # , "serif": ["Roman"]})
plt.rc("text", usetex=True)

#THIS NEEDS TO CHANGE!!!!!!!!!!!!!!!#
LbSignalSel = "(Lb_BKGCAT == 0 || Lb_BKGCAT == 10 || Lb_BKGCAT == 50)"
#This has the same effect as the selection above:
#BcSignalSel_full = "abs(B_TRUEID) == 541 && abs(muplus_TRUEID) == 13 && muplus_TRUEID == -muminus_TRUEID && (B_BKGCAT == 20 || B_BKGCAT == 40) && abs(hadron_TRUEID) == 211 && abs(dimuon_TRUEID) == 443"

columns = [
    "Lb_PT",
    "Lb_P",
    "Lb_ENDVERTEX_CHI2",
    "Lb_IPCHI2_OWNPV",
    "Lb_TAUCHI2",
    "Lb_FD_OWNPV",
    "Lb_OWNPV_Y",
    "Lb_IP_OWNPV",
    "Jpsi_PT",
    "Jpsi_P",
    "Jpsi_ENDVERTEX_CHI2",
    "Jpsi_IPCHI2_OWNPV",
    "Jpsi_TAUCHI2",
    "Jpsi_FD_OWNPV",
    "Jpsi_OWNPV_Y",
    "Jpsi_ORIVX_CHI2",
    "Jpsi_IP_OWNPV",
    "Lres_PT",
    "Lres_P",
    "Lres_ENDVERTEX_CHI2",
    "Lres_IPCHI2_OWNPV",
    "Lres_TAUCHI2",
    "Lres_FD_OWNPV",
    "Lres_OWNPV_Y",
    "Lres_ORIVX_CHI2",
    "h1_PT",
    "h1_P",
    "h1_IPCHI2_OWNPV",
    "h1_OWNPV_Y",
    "h2_PT",
    "h2_P",
    "h2_IPCHI2_OWNPV",
    "h2_OWNPV_Y",
    "mu1_OWNPV_Y",
    "mu2_OWNPV_Y",
]

input_columns = [
    "Lb_PT",
    "Lb_P",
    "Lb_ENDVERTEX_CHI2",
    "Lb_IPCHI2_OWNPV",
    "Lb_TAUCHI2",
    "Lb_FD_OWNPV",
    "Lb_OWNPV_Y",
    "Lb_IP_OWNPV",
    "Jpsi_PT",
    "Jpsi_P",
    "Jpsi_ENDVERTEX_CHI2",
    "Jpsi_IPCHI2_OWNPV",
    "Jpsi_TAUCHI2",
    "Jpsi_FD_OWNPV",
    "Jpsi_OWNPV_Y",
    "Jpsi_ORIVX_CHI2",
    "Jpsi_IP_OWNPV",
    "Lres_PT",
    "Lres_P",
    "Lres_ENDVERTEX_CHI2",
    "Lres_IPCHI2_OWNPV",
    "Lres_TAUCHI2",
    "Lres_FD_OWNPV",
    "Lres_OWNPV_Y",
    "Lres_ORIVX_CHI2",
    "h1_PT",
    "h1_P",
    "h1_IPCHI2_OWNPV",
    "h1_OWNPV_Y",
    "h2_PT",
    "h2_P",
    "h2_IPCHI2_OWNPV",
    "h2_OWNPV_Y",
    "mu1_OWNPV_Y",
    "mu2_OWNPV_Y",
]

#derived_inputs = ["max(muplus_PT,muminus_PT)"]
#["max(max(B_DOCA_muplus_hadron,B_DOCA_muminus_hadron),B_DOCA_muplus_muminus)"] +
# + ["abs(muplus_P-muminus_P)"]

more_columns = ["Lb_M", "Jpsi_M"]
additional_columns = [
    #    "muplus_P",
    #    "muminus_P",
    #    #"bdt_output",
    #    #"bdt_output_Jpsi_psi2s_veto",
    #    #"bdt_output_Bc_Jpsipi_psiveto",
    "survive_trigger_selection",
    "survive_full_fiducial",
    "nTracks"
] + more_columns

if __name__ == '__main__':
    year = "2016"
    polarity = "Up"
    data = "Lb2pKmm"

    X_test, X_train = load_test_train(
        f"../cache/{data}_{year}_{polarity}_Xtrain.root",
        f"../cache/{data}_{year}_{polarity}_Xtest.root", "tree", "tree")

    y_test = X_test[["label"]]
    y_train = X_train[["label"]]

    print("INPUT TRAIN AND TEST LOADED")

    scan_outputs_file = "inputs_scan.json"
    scan_outputs_list = list()

    i = 0
    columns_to_loop = input_columns
    while i < len(columns_to_loop):
        new_columns = []
        x = 0
        while x < len(columns_to_loop):
            if x != i:
                new_columns.append(columns_to_loop[x])
            x += 1
        print(f"LOOPING ON: {columns_to_loop[i]}")
        print(f"LENGTH OF NEW ARRAY: {len(new_columns)}")

        plot_name = f"TO_DELETE_2"
        name = f"../cache/{plot_name}_{data}_{year}_{polarity}_xgboost"
        model = train_xgb_classifier(
            X_train[new_columns], X_test[new_columns], y_train, y_test,
            X_train["mva_mcweight_nTracks_B_PT"],
            X_test["mva_mcweight_nTracks_B_PT"], f"{name}_model")

        predict_and_save(model, X_test, X_train, new_columns, "xgb_output",
                         "xgb_output", f"{name}_test", f"{name}_train")

        file_test = ROOT.TFile(f"{name}_test.root")
        tree_test = file_test.Get("tree")

        file_train = ROOT.TFile(f"{name}_train.root")
        tree_train = file_train.Get("tree")

        Kol_sig, Kol_bkg, sig_ks, bkg_ks = makeHist(
            f"../Histograms/{plot_name}.pdf", tree_test, tree_train,
            "xgb_output", 25, 0, 1, "1")

        test, train = load_test_train(f"{name}_test.root",
                                      f"{name}_train.root", "tree", "tree")

        ROC_test, ROC_train = plot_2roc_curves(
            test.loc[test.label == 1], test.loc[test.label == 0],
            train.loc[train.label == 1], train.loc[train.label == 0],
            "xgb_output", "xgb_output", "test", "train", "", f"{plot_name}")

        scan_outputs_list.append(
            pd.DataFrame({
                "removed_variable": [columns_to_loop[i]],
                "best_iteration": [model.best_iteration],
                "best_score": [model.best_score],
                "KS_sig": [Kol_sig],
                "KS_bkg": [Kol_bkg],
                "ROC_test": [ROC_test],
                "ROC_train": [ROC_train],
                "Delta_ROC": [0.976403 - ROC_test],
            }))
        i += 1

    merged_scan_outputs = pd.concat(
        scan_outputs_list, sort=False, ignore_index=True).sort_values(
            by=["Delta_ROC"], ascending=False)
    merged_scan_outputs["score"] = merged_scan_outputs[
        "Delta_ROC"] * 100 / merged_scan_outputs[["Delta_ROC"]].max().values[0]
    merged_scan_outputs.to_json(scan_outputs_file)
