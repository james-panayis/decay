import ROOT
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from simple_mva_xgboost import *
import sys
import os
sys.path.append("../common")
import common_definitions as cd
# plt.rcParams['text.latex.preamble'] = [r'\usepackage{bm}']
plt.rc("font", **{"family": "serif"})  # , "serif": ["Roman"]})
plt.rc("text", usetex=True)
from mva_testing_functions import *

selection_files = ["TriggerSelection", "FiducialSelection", "Preselection"]
sig_training_antisideband = "(Lb_M > 5350 && Lb_M < 5850)"
bkg_sig_sel = f"(Lb_BKGCAT == 0 || Lb_BKGCAT == 10 || Lb_BKGCAT == 50)"
bkg_training_sideband = "!(Lb_M > 5350 && Lb_M < 5850)"

#vetoes = "abs(Jpsi_M - 3096.9) > 100 && abs(Jpsi_M - 3686.097) > 100 && abs(Jpsi_M - 1020)>30"

vetoes = " \
!((Jpsi_M*Jpsi_M)/1000000 > 8.0 &&  (Jpsi_M*Jpsi_M)/1000000 < 11) &&  \
!((Jpsi_M*Jpsi_M)/1000000 > 12.5 && (Jpsi_M*Jpsi_M)/1000000 < 15) &&  \
!((Jpsi_M*Jpsi_M)/1000000 > 0.98 && (Jpsi_M*Jpsi_M)/1000000 < 1.10) "

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
    #"h1_OWNPV_Y",
    "h2_PT",
    "h2_P",
    "h2_IPCHI2_OWNPV",
    #    "h2_OWNPV_Y",
    #"mu1_OWNPV_Y",
    #    "mu2_OWNPV_Y",
    #    "max(OWNPV_Y)"
]

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
additional_columns = ["nTracks"] + more_columns + selections

chosen_selections = selections

Make_single_TTData = False
Make_mixed_TTData = True
Make_Model = True
Make_XGBFile = True  #NOTE Make_Model Must be true for this to work
Make_Graphs = True
Input_Test = False
Hypervariable = False
Selection = False

if __name__ == '__main__':

    data = ["Lb2pKmm"]
    years = [2016, 2017, 2018]
    polarity = ["Up", "Dn"]

    for dat in data:
        for year in years:
            for pol in polarity:
                name = "H"
                baseline = "survive_trigger_selection"
                # && survive_full_fiducial && survive_track_pid_fiducial"
                # && survive_preselection && survive_PT_preselection && survive_full_fiducial"

                sig_sel = f"{sig_training_antisideband} && {bkg_sig_sel} && {vetoes} && {baseline}"
                bkg_sel = f"{bkg_training_sideband} && {vetoes} && {baseline}"

                if Make_single_TTData == True:
                    MakeTestTrainData(name, dat, year, pol, selection_files,
                                      sig_sel, bkg_sel, columns, input_columns,
                                      additional_columns, True)

                if Make_mixed_TTData == True:
                    X_train, X_test, y_train, y_test = TrainXGBs(
                        name, years, polarity, input_columns,
                        additional_columns, bkg_sel, sig_sel, selection_files)

                if Make_Model == True:
                    #                    X_train, X_test, y_train, y_test = MakeTestTrainData(
                    #                        name, dat, year, pol, selection_files, sig_sel,
                    #                        bkg_sel, input_columns, additional_columns, False)

                    model = CreateModel(name, X_train, X_test, y_train, y_test,
                                        input_columns)
                    if Make_XGBFile == True:
                        Xtest, Xtrain = predict_and_save(
                            model, X_test, X_train, input_columns,
                            "xgb_output", "xgb_output", f"XGB_{name}_test",
                            f"XGB_{name}_train")

                        if Make_Graphs == True:
                            MakeROC(name, Xtest, Xtrain)
                            MakeHist(name)
                            MakeCorr(name, Xtest, Xtrain, input_columns,
                                     more_columns)

                if Input_Test == True:

                    InputTesting(f"{name}_Input_testing", input_columns, 1.0,
                                 f"XGB_{name}_test", f"XGB_{name}_train")

                if Hypervariable == True:
                    print("STARTING HYPERVARIABLES")
                    MAX_DEPTH = [2, 3]
                    ETA = [0.6, 0.7]
                    GAMMA = [0.03]  # dont need to change this
                    SUBSAMPLE = [0.4, 0.5, 0.6]

                    HypervariableTest(name, MAX_DEPTH, ETA, GAMMA, SUBSAMPLE,
                                      input_columns, 1.0, f"XGB_{name}_test",
                                      f"XGB_{name}_train")

                if Selection == True:
                    print("STARTING SELECTIONS")
                    SelectionTest(f"{name}_preselections", selection_files,
                                  selections, sig_sel, bkg_sel, dat, year, pol,
                                  input_columns)
