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
    "h1_OWNPV_Y",
    "h2_PT",
    "h2_P",
    "h2_IPCHI2_OWNPV",
    #    "h2_OWNPV_Y",
    "mu1_OWNPV_Y",
    #    "mu2_OWNPV_Y",
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

Make_TTData = False
Make_Model = False
Make_XGBFile = False  #NOTE Make_Model Must be true for this to work
Input_Test = False
Hypervariable = False
Selection = True

if __name__ == '__main__':

    data = ["Lb2pKmm"]
    years = [2016]
    polarity = ["Up"]

    for dat in data:
        for year in years:
            for pol in polarity:
                name = "TESTING"
                baseline = "True"
                sig_sel = f"(Lb_BKGCAT == 0 || Lb_BKGCAT == 10 || Lb_BKGCAT == 50)"
                bkg_sel = f"{cd.training_sideband}"

                if Make_TTData == True:
                    MakeTestTrainData(name, dat, year, pol, selection_files,
                                      sig_sel, bkg_sel, input_columns,
                                      additional_columns, True)

                if Make_Model == True:
                    X_train, X_test, y_train, y_test = MakeTestTrainData(
                        "TESTING", dat, year, pol, selection_files, sig_sel,
                        bkg_sel, input_columns, additional_columns, False)

                    model = CreateModel(name, X_train, X_test, y_train, y_test,
                                        input_columns)
                    if Make_XGBFile == True:
                        Xtest, Xtrain = predict_and_save(
                            model, X_test, X_train, input_columns,
                            "xgb_output", "xgb_output", f"XGB_{name}_test",
                            f"XGB_{name}_train")

                if Input_Test == True:

                    InputTesting("Input_testing", input_columns, 1.0,
                                 f"XGB_{name}_test", f"XGB_{name}_train")

                if Hypervariable == True:
                    MAX_DEPTH = [3]
                    ETA = [0.8]
                    GAMMA = [0.03]
                    SUBSAMPLE = [0.4]

                    HypervariableTest("max_depth", MAX_DEPTH, ETA, GAMMA,
                                      SUBSAMPLE, input_columns, 1.0,
                                      f"XGB_{name}_test", f"XGB_{name}_train")

                if Selection == True:
                    SelectionTest("preselections", selection_files, selections,
                                  sig_sel, bkg_sel, dat, year, pol,
                                  input_columns)
