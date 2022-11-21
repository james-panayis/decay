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
    #    "Lb_P",
    #    "Lb_ENDVERTEX_CHI2",
    #    "Lb_IPCHI2_OWNPV",
    #    "Lb_TAUCHI2",
    #    "Lb_FD_OWNPV",
    #    "Lb_OWNPV_Y",
    "Lb_IP_OWNPV",
    #    "Jpsi_PT",
    "Jpsi_P",
    "Jpsi_ENDVERTEX_CHI2",
    #    "Jpsi_IPCHI2_OWNPV",
    #    "Jpsi_TAUCHI2",
    "Jpsi_FD_OWNPV",
    #    "Jpsi_OWNPV_Y",
    #    "Jpsi_ORIVX_CHI2",
    "Jpsi_IP_OWNPV",
    #    "Lres_PT",
    #    "Lres_P",
    "Lres_ENDVERTEX_CHI2",
    "Lres_IPCHI2_OWNPV",
    "Lres_TAUCHI2",
    "Lres_FD_OWNPV",
    #    "Lres_OWNPV_Y",
    #    "Lres_ORIVX_CHI2",
    "h1_PT",
    "h1_P",
    "h1_IPCHI2_OWNPV",
    #    "h1_OWNPV_Y",
    "h2_PT",
    "h2_P",
    "h2_IPCHI2_OWNPV",
    #    "h2_OWNPV_Y",
    #    "mu1_OWNPV_Y",
    #    "mu2_OWNPV_Y",
]

#derived_inputs = ["max(muplus_PT,muminus_PT)"]
#["max(max(B_DOCA_muplus_hadron,B_DOCA_muminus_hadron),B_DOCA_muplus_muminus)"] +
# + ["abs(muplus_P-muminus_P)"]

selections = [
    "survive_trigger_selection",
    "survive_full_fiducial",
    "survive_preselection",
    "survive_PT_preselection",
    "survive_track_pid_fiducial",
    "survive_HLT1_trigger_selection",
    "survive_HLT2_trigger_selection",
    "survive_fiducial",
]

more_columns = ["Lb_M", "Jpsi_M"]
additional_columns = [
    #    "muplus_P",
    #    "muminus_P",
    #    #"bdt_output",
    #    #"bdt_output_Jpsi_psi2s_veto",
    #    #"bdt_output_Bc_Jpsipi_psiveto",
    "nTracks"
] + more_columns + selections

if __name__ == '__main__':

    #NEED TO CREATE THESE SELECTIONS!!!!!!!!!!
    baseline_selection = "survive_preselection && survive_full_fiducial && survive_trigger_selection"
    sig_selection = f"{baseline_selection} && {LbSignalSel}"
    bkg_selection = f"{baseline_selection} && {cd.training_sideband}"

    #sig_selection = f"{LbSignalSel}"
    #sig_selection = "true"
    #bkg_selection = f"{cd.training_sideband}"

    year = "2016"
    polarity = "Up"
    data = "Lb2pKmm"

    if doesXGBInputExist(data, year, polarity, False) == False:
        print("STARTING...")
        sig_sample_name = "Lb2pKmm_sim_mgUp_2016"
        bkg_sample_name = "Lb2pKmm_mgUp_2016"

        sig_file_name = cd.samples[sig_sample_name]["ntuples"]
        bkg_file_name = cd.samples[bkg_sample_name]["ntuples"]

        sig_chain = ROOT.TChain("tree")
        sig_chain.Add(sig_file_name)

        bkg_chain = ROOT.TChain("Lb_Tuple/DecayTree")
        bkg_chain.Add(bkg_file_name)

        i_file_name = os.path.basename(sig_file_name)

        for selection in selection_files:
            sig_friend_chain = ROOT.TChain('tree')
            sig_sel_name = f"../selections/{selection}_{sig_sample_name}.root"
            print(sig_sel_name)
            sig_friend_chain.Add(sig_sel_name)
            print(f"ADDED SIG {selection}")
            sig_chain.AddFriend(sig_friend_chain)

        sig_rdframe = ROOT.RDataFrame(sig_chain)
        sig_frame = pd.DataFrame(
            data=sig_rdframe.Filter(sig_selection).AsNumpy(columns +
                                                           additional_columns))

        for selection in selection_files:
            bkg_friend_chain = ROOT.TChain('tree')
            bkg_sel_name = f"../selections/{selection}_{bkg_sample_name}.root"
            bkg_friend_chain.Add(bkg_sel_name)
            print(f"ADDED BKG {selection}")
            bkg_chain.AddFriend(bkg_friend_chain)

        bkg_rdframe = ROOT.RDataFrame(bkg_chain)
        bkg_frame = pd.DataFrame(
            data=bkg_rdframe.Filter(bkg_selection).AsNumpy(columns +
                                                           additional_columns))

        #        bkg_frame["abs(muplus_P-muminus_P)"] = abs(bkg_frame["muplus_P"] -
        #                                                   bkg_frame["muminus_P"])

        #STARTING XGBOOST

        sig_label = pd.DataFrame(
            np.ones(shape=sig_frame.shape[0]), columns=["label"])
        bkg_label = pd.DataFrame(
            np.zeros(shape=bkg_frame.shape[0]), columns=["label"])
        #sig_weights = sig_frame["mva_mcweight_nTracks_B_PT"]
        sig_weights = pd.DataFrame(np.ones(shape=sig_frame.shape[0]))
        bkg_weights = pd.DataFrame(np.ones(shape=bkg_frame.shape[0]))

        #+ ["abs(muplus_P-muminus_P)"]
        train_data = pd.concat([sig_frame[columns], bkg_frame[columns]],
                               sort=False,
                               ignore_index=True)
        train_label = pd.concat([sig_label, bkg_label],
                                sort=False,
                                ignore_index=True)
        train_weights = pd.concat([sig_weights, bkg_weights],
                                  sort=False,
                                  ignore_index=True)
        #+["abs(muplus_P-muminus_P)"]
        train_data = pd.concat([
            sig_frame[columns + additional_columns],
            bkg_frame[columns + additional_columns]
        ],
                               sort=False,
                               ignore_index=True)

        #        train_data[
        #            "max(max(B_DOCA_muplus_hadron,B_DOCA_muminus_hadron),B_DOCA_muplus_muminus)"] = train_data[
        #                [
        #                    "B_DOCA_muplus_hadron", "B_DOCA_muminus_hadron",
        #                    "B_DOCA_muplus_muminus"
        #                ]].max(axis=1)
        #        train_data["max(muplus_PT,muminus_PT)"] = train_data[[
        #            "muplus_PT", "muminus_PT"
        #        ]].max(axis=1)

        train_data["label"] = train_label["label"]
        train_data["mva_mcweight_nTracks_B_PT"] = train_weights

        X_train, X_test, y_train, y_test = split_data(train_data, train_label)

        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")

        print(X_train.columns)

        #SAVE THE SPLIT DATA HERE

        X_train_dict = X_train.to_dict("list")
        X_train_dict = {
            key: np.array(X_train_dict[key])
            for key in X_train_dict.keys()
        }

        X_test_dict = X_test.to_dict("list")
        X_test_dict = {
            key: np.array(X_test_dict[key])
            for key in X_test_dict.keys()
        }

        df_Xtrain = ROOT.RDF.MakeNumpyDataFrame(X_train_dict)
        df_Xtest = ROOT.RDF.MakeNumpyDataFrame(X_test_dict)

        #WILL WANT TO CHANGE THE DIRECTORIES HERE!!!!!!

        df_Xtrain.Snapshot('tree',
                           f'../cache/{data}_{year}_{polarity}_Xtrain.root')
        df_Xtest.Snapshot('tree',
                          f'../cache/{data}_{year}_{polarity}_Xtest.root')

    else:
        #REMEMBER TO LOAD IF ALREADY THERE

        X_test, X_train = load_test_train(
            f"../cache/{data}_{year}_{polarity}_Xtrain.root",
            f"../cache/{data}_{year}_{polarity}_Xtest.root", "tree", "tree")

        y_test = X_test[["label"]]
        y_train = X_train[["label"]]

        print("INPUT TRAIN AND TEST LOADED")

    name = f"{data}_{year}_{polarity}_xgboost_"
    model = train_xgb_classifier(
        X_train[input_columns], X_test[input_columns], y_train, y_test,
        X_train["mva_mcweight_nTracks_B_PT"],
        X_test["mva_mcweight_nTracks_B_PT"], f"{name}")

    Xtest, Xtrain = predict_and_save(
        model, X_test, X_train, input_columns, "xgb_output", "xgb_output",
        f"../cache/{name}_test", f"../cache/{name}_train")

    file_test = ROOT.TFile(f"../cache/{name}_test.root")
    tree_test = file_test.Get("tree")

    file_train = ROOT.TFile(f"../cache/{name}_train.root")
    tree_train = file_train.Get("tree")

    #Plotting Functions#

    plot_name = "Initial"

    #Histogram
    makeHist(f"{plot_name}.pdf", tree_test, tree_train, "xgb_output", 25, 0, 1,
             "1")

    #Correlations
    draw_correlation(Xtest.loc[Xtest.label == 1][input_columns + more_columns],
                     'Correlations', f'{plot_name}_input_correlations_sig.pdf')

    draw_correlation(Xtest.loc[Xtest.label == 0][input_columns + more_columns],
                     'Correlations', f'{plot_name}_input_correlations_bkg.pdf')

    #ROC Curves
    from_cache = "../cache/"

    Xtest1, Xtrain1 = load_test_train(
        f"{from_cache}Lb2pKmm_2016_Up_xgboost__test.root",
        f"{from_cache}Lb2pKmm_2016_Up_xgboost__train.root", "tree", "tree")

    plot_2roc_curves(
        Xtest1.loc[Xtest.label == 1], Xtest1.loc[Xtest.label == 0],
        Xtrain1.loc[Xtrain1.label == 1], Xtrain1.loc[Xtrain1.label == 0],
        "xgb_output", "xgb_output", "Test", "Train", "", "Initial")
