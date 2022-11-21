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

selection_files = ["TriggerSelection", "Preselection", "FiducialSelection"]


def hyp_train_xgb_classifier(X_train,
                             X_test,
                             y_train,
                             y_test,
                             sample_weight=None,
                             sample_weight_eval_set=None,
                             name=None,
                             max_depth=3,
                             eta=0.8,
                             gamma=0.03,
                             subsample=0.4):
    """
    This function receives sets of training and testing data (eventually with event weights) and returns a xgboost classifier trained with the provided training data sample.
    @param X_train the training sample in pandas format
    @param y_train the label of the training data in pandas format
    @param X_test the testing sample in pandas format
    @param y_train the label of the testing data in pandas format 
    @param sample_weight fit parameter
    @param sample_weight_eval_set fit parameter
    
    """
    #tests the data and saves the output under name

    xgb_param = {
        'max_depth': max_depth,
        'eta': eta,
        'gamma': gamma,
        'subsample': subsample,
        #'max_bin': 256,
        #'objective': 'binary:logistic',
        #'nthread': 4,
        #'eval_metric': 'auc',
        'n_estimators': 500,
        #'use_label_encoder':False,
    }

    model = xgb.XGBClassifier()
    model.set_params(**xgb_param)
    model.fit(
        X_train,
        y_train,
        #        sample_weight=sample_weight,
        #        sample_weight_eval_set=[sample_weight_eval_set],
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=10)

    assert model.n_classes_ == 2
    model.save_model(f"{name}.json")
    #./Bc_pimumu_work/models/{name}.json")

    print("EARLY STOPPING RESULTS:")
    print(f"BEST N TREE LIMIT: {model.best_ntree_limit}")
    print(f"BEST SCORE: {model.best_score}")
    print(f"BEST ITERATION: {model.best_iteration}")

    return model


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

MAX_DEPTH = [3]
ETA = [1.0]
#GAMMA = [0.03]
SUBSAMPLE = [0.4]

#MAX_DEPTH = [1,2,3,4,5]
#ETA = [ 0.7, 0.8, 0.9, 1.0, 1.1]
GAMMA = [0.01, 0.03, 0.1, 0.25, 0.5]
#SUBSAMPLE = [0.2, 0.3, 0.4, 0.5, 0.6]

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

    scan_output_file = "scan_gamma_diff_output.json"
    scan_output_list = list()

    #SWAP THIS FOR HYPERVARIABLE TEST
    for md in MAX_DEPTH:
        for eta in ETA:
            for gamma in GAMMA:
                for sub in SUBSAMPLE:

                    plot_name = f"TO_DELETE"
                    name = f"../cache/{plot_name}_{data}_{year}_{polarity}_xgboost"
                    model = hyp_train_xgb_classifier(
                        X_train[input_columns], X_test[input_columns], y_train,
                        y_test, X_train["mva_mcweight_nTracks_B_PT"],
                        X_test["mva_mcweight_nTracks_B_PT"], f"{name}_model",
                        md, eta, gamma, sub)

                    predict_and_save(model, X_test, X_train, input_columns,
                                     "xgb_output", "xgb_output",
                                     f"{name}_test", f"{name}_train")

                    print(
                        f"CREATED MODEL FOR: MAX_DEPTH={md}, ETA={eta}, GAMMA={gamma}, SUBSAMPLE={sub}"
                    )

                    file_test = ROOT.TFile(f"{name}_test.root")
                    tree_test = file_test.Get("tree")

                    file_train = ROOT.TFile(f"{name}_train.root")
                    tree_train = file_train.Get("tree")

                    Kol_sig, Kol_bkg, sig_ks, bkg_ks = makeHist(
                        f"../Histograms/{plot_name}.pdf", tree_test,
                        tree_train, "xgb_output", 25, 0, 1, "1")

                    test, train = load_test_train(f"{name}_test.root",
                                                  f"{name}_train.root", "tree",
                                                  "tree")

                    ROC_test, ROC_train = plot_2roc_curves(
                        test.loc[test.label == 1], test.loc[test.label == 0],
                        train.loc[train.label == 1],
                        train.loc[train.label == 0], "xgb_output",
                        "xgb_output", "test", "train", "", f"{plot_name}")

                    scan_output_list.append(
                        pd.DataFrame({
                            # "MAX_DEPTH": [md],
                            # "ETA": [eta],
                            "GAMMA": [gamma],
                            # "SUBSAMPLE": [sub],
                            "best_iteration": [model.best_iteration],
                            "best_score": [model.best_score],
                            "KS_sig": [Kol_sig],
                            "KS_bkg": [Kol_bkg],
                            "ROC_test": [ROC_test],
                            "ROC_train": [ROC_train],
                            "Delta_ROC": [0.976403 - ROC_test],
                        }))

    merged_scan_output = pd.concat(
        scan_output_list, sort=False, ignore_index=True).sort_values(
            by=["Delta_ROC"], ascending=False)
    merged_scan_output["score"] = merged_scan_output[
        "Delta_ROC"] * 100 / merged_scan_output[["Delta_ROC"]].max().values[0]
    merged_scan_output.to_json(scan_output_file)
