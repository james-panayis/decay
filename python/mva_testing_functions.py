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


##########################################################################################
#CREATING XGBOOST DATA, MODELS AND GRAPHS
def MakeTestTrainData(name,
                      data,
                      year,
                      polarity,
                      selection_files,
                      sig_selection,
                      bkg_selection,
                      columns,
                      input_columns,
                      additional_columns,
                      retrain=False):
    if doesXGBInputExist(data, year, polarity, retrain) == False:
        sig_sample_name = f"Lb2pKmm_sim_mg{polarity}_{year}"
        bkg_sample_name = f"Lb2pKmm_mg{polarity}_{year}"

        sig_file_name = cd.samples[sig_sample_name]["ntuples"]
        bkg_file_name = cd.samples[bkg_sample_name]["ntuples"]

        sig_chain = ROOT.TChain("tree")
        sig_chain.Add(sig_file_name)

        bkg_chain = ROOT.TChain("Lb_Tuple/DecayTree")
        bkg_chain.Add(bkg_file_name)

        for selection in selection_files:
            sig_friend_chain = ROOT.TChain('tree')
            sig_sel_name = f"../selections/{selection}_{sig_sample_name}.root"
            sig_friend_chain.Add(sig_sel_name)
            sig_chain.AddFriend(sig_friend_chain)

        sig_rdframe = ROOT.RDataFrame(sig_chain)
        sig_frame = pd.DataFrame(
            data=sig_rdframe.Filter(sig_selection).AsNumpy(columns +
                                                           additional_columns))

        for selection in selection_files:
            bkg_friend_chain = ROOT.TChain('tree')
            bkg_sel_name = f"../selections/{selection}_{bkg_sample_name}.root"
            bkg_friend_chain.Add(bkg_sel_name)
            bkg_chain.AddFriend(bkg_friend_chain)

        bkg_rdframe = ROOT.RDataFrame(bkg_chain)
        bkg_frame = pd.DataFrame(
            data=bkg_rdframe.Filter(bkg_selection).AsNumpy(columns +
                                                           additional_columns))

        print("ADDED SELECTION FILES")

        sig_label = pd.DataFrame(
            np.ones(shape=sig_frame.shape[0]), columns=["label"])
        bkg_label = pd.DataFrame(
            np.zeros(shape=bkg_frame.shape[0]), columns=["label"])

        train_label = pd.concat([sig_label, bkg_label],
                                sort=False,
                                ignore_index=True)

        train_data = pd.concat([
            sig_frame[columns + additional_columns],
            bkg_frame[columns + additional_columns]
        ],
                               sort=False,
                               ignore_index=True)

        train_data["label"] = train_label["label"]

        train_data["max(OWNPV_Y)"] = train_data[[
            "h1_OWNPV_Y", "h2_OWNPV_Y", "mu1_OWNPV_Y", "mu2_OWNPV_Y"
        ]].max(axis=1)

        X_train, X_test, y_train, y_test = split_data(train_data, train_label)

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

        df_Xtrain.Snapshot('tree',
                           f"../cache/TestTrainData/{name}_Xtrain.root")
        df_Xtest.Snapshot('tree', f"../cache/TestTrainData/{name}_Xtest.root")

        print(f"CREATED {name}")

    else:
        print("DATA SET ALREADY EXISTS")
        print("RETRIEVING...")

        X_test, X_train = load_test_train(
            f"../cache/TestTrainData/{name}_Xtrain.root",
            f"../cache/TestTrainData/{name}_Xtest.root", "tree", "tree")

        y_test = X_test[["label"]]
        y_train = X_train[["label"]]

    return X_train, X_test, y_train, y_test


def CreateModel(model_name, X_train, X_test, y_train, y_test, input_columns):
    model = train_xgb_classifier(X_train[input_columns], X_test[input_columns],
                                 y_train, y_test,
                                 f"../cache/models/{model_name}")
    return model


def MakeROC(plot_name, Xtest, Xtrain):
    from_cache = "../cache/"

    #Xtest1, Xtrain1 = load_test_train(
    #    f"{from_cache}{name}_test.root",
    #    f"{from_cache}{name}_train.root", "tree", "tree")

    plot_2roc_curves(Xtest.loc[Xtest.label == 1], Xtest.loc[Xtest.label == 0],
                     Xtrain.loc[Xtrain.label == 1],
                     Xtrain.loc[Xtrain.label == 0], "xgb_output", "xgb_output",
                     "Test", "Train", "", plot_name)


def MakeHist(plot_name):
    file_test = ROOT.TFile(f"../cache/XGB_data/XGB_{plot_name}_test.root")
    tree_test = file_test.Get("tree")

    file_train = ROOT.TFile(f"../cache/XGB_data/XGB_{plot_name}_train.root")
    tree_train = file_train.Get("tree")

    makeHist(f"{plot_name}.pdf", tree_test, tree_train, "xgb_output", 25, 0, 1,
             "1")


def MakeCorr(plot_name, Xtest, Xtrain, input_columns, more_columns):

    draw_correlation(Xtest.loc[Xtest.label == 1][input_columns + more_columns],
                     'Correlations', f'{plot_name}_input_correlations_sig.pdf')

    draw_correlation(Xtest.loc[Xtest.label == 0][input_columns + more_columns],
                     'Correlations', f'{plot_name}_input_correlations_bkg.pdf')


##########################################################################################
#PRESELECTION TESTING
def SelectionTest(filename, selection_files, selection_columns, const_sig_sel,
                  const_bkg_sel, data, year, pol, input_columns):

    sig_sample_name = f"{data}_sim_mg{pol}_{year}"
    bkg_sample_name = f"{data}_mg{pol}_{year}"

    sig_file_name = cd.samples[sig_sample_name]["ntuples"]
    bkg_file_name = cd.samples[bkg_sample_name]["ntuples"]

    sig_chain = ROOT.TChain("tree")
    sig_chain.Add(sig_file_name)

    bkg_chain = ROOT.TChain("Lb_Tuple/DecayTree")
    bkg_chain.Add(bkg_file_name)

    sig_chain.Add(sig_file_name)
    bkg_chain.Add(bkg_file_name)

    for selection in selection_files:
        sig_friend_chain = ROOT.TChain('tree')
        sig_sel_name = f"../selections/{selection}_{sig_sample_name}.root"
        sig_friend_chain.Add(sig_sel_name)
        sig_chain.AddFriend(sig_friend_chain)

    for selection in selection_files:
        bkg_friend_chain = ROOT.TChain('tree')
        bkg_sel_name = f"../selections/{selection}_{bkg_sample_name}.root"
        bkg_friend_chain.Add(bkg_sel_name)
        bkg_chain.AddFriend(bkg_friend_chain)

    sig_selection = const_sig_sel
    bkg_selection = const_bkg_sel

    sig_rdframe = ROOT.RDataFrame(sig_chain)
    sig_frame = pd.DataFrame(
        data=sig_rdframe.Filter(const_sig_sel).AsNumpy(input_columns))

    bkg_rdframe = ROOT.RDataFrame(bkg_chain)
    bkg_frame = pd.DataFrame(
        data=bkg_rdframe.Filter(const_bkg_sel).AsNumpy(input_columns))

    scan_outputs_file = f"../cache/preselection_tests/{filename}.json"
    scan_outputs_list = list()

    sig_ord_len = len(sig_frame)
    bkg_ord_len = len(bkg_frame)

    i = 0
    while i < len(selection_columns):

        print(f"{selection_columns[i]} filtering")
        sig_frame2 = pd.DataFrame(
            data=sig_rdframe.Filter(
                f"{sig_selection} && {selection_columns[i]}").AsNumpy(
                    input_columns))
        bkg_frame2 = pd.DataFrame(
            data=bkg_rdframe.Filter(
                f"{bkg_selection} && {selection_columns[i]}").AsNumpy(
                    input_columns))

        scan_outputs_list.append(
            pd.DataFrame({
                "selection_choice": [f"{selection_columns[i]}"],
                "signal_removed": [1 - len(sig_frame2) / sig_ord_len],
                "background_removed": [1 - len(bkg_frame2) / bkg_ord_len],
                "FoM":
                [len(sig_frame2) / (len(sig_frame2) + len(bkg_frame2))**0.5],
            }))
        i += 1

    merged_scan_outputs = pd.concat(
        scan_outputs_list, sort=False, ignore_index=True).sort_values(
            by=["FoM"], ascending=False)
    merged_scan_outputs.to_json(scan_outputs_file)

    print("COMPLETE")
    return


###########################################################################################
#HYPERVARIABLE TESTING


def hyp_train_xgb_classifier(X_train,
                             X_test,
                             y_train,
                             y_test,
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
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=10)

    assert model.n_classes_ == 2
    model.save_model(f"TEMP.json")

    return model


def HypervariableTest(filename, MAX_DEPTH, ETA, GAMMA, SUBSAMPLE,
                      input_columns, BEST_ROC, test_name, train_name):

    X_test, X_train = load_test_train(f"../cache/XGB_data/{test_name}.root",
                                      f"../cache/XGB_data/{train_name}.root",
                                      "tree", "tree")

    y_test = X_test[["label"]]
    y_train = X_train[["label"]]

    scan_output_file = f"../cache/hypervariable_tests/scan_{filename}_diff_output.json"
    scan_output_list = list()

    #SWAP THIS FOR HYPERVARIABLE TEST
    for md in MAX_DEPTH:
        for eta in ETA:
            for gamma in GAMMA:
                for sub in SUBSAMPLE:
                    print(f"Depth: {md}, Eta: {eta}, Sub: {sub}")
                    plot_name = f"TO_DELETE"
                    name = f"TEMP"
                    model = hyp_train_xgb_classifier(
                        X_train[input_columns], X_test[input_columns], y_train,
                        y_test, f"{name}_model", md, eta, gamma, sub)

                    predict_and_save(model, X_test, X_train, input_columns,
                                     "xgb_output", "xgb_output",
                                     f"{name}_test", f"{name}_train")

                    file_test = ROOT.TFile(
                        f"../cache/XGB_data/{name}_test.root")
                    tree_test = file_test.Get("tree")

                    file_train = ROOT.TFile(
                        f"../cache/XGB_data/{name}_train.root")
                    tree_train = file_train.Get("tree")

                    Kol_sig, Kol_bkg, sig_ks, bkg_ks = makeHist(
                        f"../Histograms/{plot_name}.pdf", tree_test,
                        tree_train, "xgb_output", 25, 0, 1, "1")

                    test, train = load_test_train(
                        f"../cache/XGB_data/{name}_test.root",
                        f"../cache/XGB_data/{name}_train.root", "tree", "tree")

                    ROC_test, ROC_train = plot_2roc_curves(
                        test.loc[test.label == 1], test.loc[test.label == 0],
                        train.loc[train.label == 1],
                        train.loc[train.label == 0], "xgb_output",
                        "xgb_output", "test", "train", "", f"{plot_name}")

                    scan_output_list.append(
                        pd.DataFrame({
                            "MAX_DEPTH": [md],
                            "ETA": [eta],
                            "GAMMA": [gamma],
                            "SUBSAMPLE": [sub],
                            "best_iteration": [model.best_iteration],
                            "best_score": [model.best_score],
                            "KS_sig": [Kol_sig],
                            "KS_bkg": [Kol_bkg],
                            "ROC_test": [ROC_test],
                            "ROC_train": [ROC_train],
                            "Delta_ROC": [BEST_ROC - ROC_test],
                        }))

    merged_scan_output = pd.concat(
        scan_output_list, sort=False, ignore_index=True).sort_values(
            by=["Delta_ROC"], ascending=False)
    merged_scan_output["score"] = merged_scan_output[
        "Delta_ROC"] * 100 / merged_scan_output[["Delta_ROC"]].max().values[0]
    merged_scan_output.to_json(scan_output_file)

    return


###########################################################################################
#Input Testing


def InputTesting(filename, input_columns, BEST_ROC, test_name, train_name):

    X_test, X_train = load_test_train(f"../cache/XGB_data/{test_name}.root",
                                      f"../cache/XGB_data/{train_name}.root",
                                      "tree", "tree")

    y_test = X_test[["label"]]
    y_train = X_train[["label"]]

    scan_outputs_file = f"../cache/input_testing/{filename}.json"
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
        print(f"{i}/{len(columns_to_loop)}")
        plot_name = f"TEMP"
        name = f"TEMP"
        model = train_xgb_classifier(X_train[new_columns], X_test[new_columns],
                                     y_train, y_test, f"{name}_model")

        predict_and_save(model, X_test, X_train, new_columns, "xgb_output",
                         "xgb_output", f"{name}_test", f"{name}_train")

        file_test = ROOT.TFile(f"../cache/XGB_data/{name}_test.root")
        tree_test = file_test.Get("tree")

        file_train = ROOT.TFile(f"../cache/XGB_data/{name}_train.root")
        tree_train = file_train.Get("tree")

        Kol_sig, Kol_bkg, sig_ks, bkg_ks = makeHist(
            f"../Histograms/{plot_name}.pdf", tree_test, tree_train,
            "xgb_output", 25, 0, 1, "1")

        test, train = load_test_train(f"../cache/XGB_data/{name}_test.root",
                                      f"../cache/XGB_data/{name}_train.root",
                                      "tree", "tree")

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
                "Delta_ROC": [BEST_ROC - ROC_test],
            }))
        i += 1

    merged_scan_outputs = pd.concat(
        scan_outputs_list, sort=False, ignore_index=True).sort_values(
            by=["Delta_ROC"], ascending=False)
    merged_scan_outputs["score"] = merged_scan_outputs[
        "Delta_ROC"] * 100 / merged_scan_outputs[["Delta_ROC"]].max().values[0]
    merged_scan_outputs.to_json(scan_outputs_file)

    return


######################################################################################


def TrainXGBs(Name, Years, polarities, input_columns, additional_columns,
              bkg_selection, sig_selection, selection_files):
    """
    Returns a XGBoost classifier model which is saved in the xgb_models file
    Also saves a snapshot of the test and train data after a prediction and saves this in a cache file
    @param do_years - list of years to iterate over
    @param version - name of version in string format 
    """

    bkg_dataframe = list()
    bkg_events = 0

    sig_dataframe = list()

    final_sig_dataframe = list()
    sig_events = 0

    for year in Years:
        for polarity in polarities:

            sig_sample_name = f"Lb2pKmm_sim_mg{polarity}_{year}"
            bkg_sample_name = f"Lb2pKmm_mg{polarity}_{year}"

            sig_file_name = cd.samples[sig_sample_name]["ntuples"]
            bkg_file_name = cd.samples[bkg_sample_name]["ntuples"]

            sig_chain = ROOT.TChain("tree")
            sig_chain.Add(sig_file_name)

            #sig_file = uproot.open("sig_file_name")
            #bkg_file = uproot.open("bkg_file_name")

            bkg_chain = ROOT.TChain("Lb_Tuple/DecayTree")
            bkg_chain.Add(bkg_file_name)

            for selection in selection_files:
                sig_friend_chain = ROOT.TChain('tree')
                sig_sel_name = f"../selections/{selection}_{sig_sample_name}.root"
                sig_friend_chain.Add(sig_sel_name)
                sig_chain.AddFriend(sig_friend_chain)

            #sig_rdframe = ROOT.RDataFrame(sig_chain)
            sig_rdframe = ROOT.RDataFrame(sig_chain)
            sig_frame = pd.DataFrame(
                data=sig_rdframe.Filter(sig_selection).AsNumpy(
                    input_columns + additional_columns))

            for selection in selection_files:
                bkg_friend_chain = ROOT.TChain('tree')
                bkg_sel_name = f"../selections/{selection}_{bkg_sample_name}.root"
                bkg_friend_chain.Add(bkg_sel_name)
                bkg_chain.AddFriend(bkg_friend_chain)

            bkg_rdframe = ROOT.RDataFrame(bkg_chain)
            bkg_frame = pd.DataFrame(
                data=bkg_rdframe.Filter(bkg_selection).AsNumpy(
                    input_columns + additional_columns))

            bkg_frame["year"] = year
            if polarity == "Up":
                bkg_frame["polarity"] = 1
            else:
                bkg_frame["polarity"] = 0
            bkg_dataframe.append(bkg_frame)

            sig_frame["year"] = year
            if polarity == "Up":
                sig_frame["polarity"] = 1
            else:
                sig_frame["polarity"] = 0
            sig_dataframe.append(sig_frame)

            print(f"ADDED {year} {polarity}")

    bkg_dataframe = pd.concat(bkg_dataframe, sort=False, ignore_index=True)
    sig_dataframe = pd.concat(sig_dataframe, sort=False, ignore_index=True)

    print("Starting Training")
    print(f"Preselected: {bkg_events} = {bkg_dataframe.shape[0]}")
    print(f"Preselected: {sig_events} = {sig_dataframe.shape[0]}")

    sig_dataframe["label"] = 1
    bkg_dataframe["label"] = 0

    train_data = pd.concat([sig_dataframe, bkg_dataframe],
                           sort=False,
                           ignore_index=True)

    X_train, X_test, y_train, y_test = split_data(train_data,
                                                  train_data["label"])

    #make the model CHANGE THIS! MAYBE JUST RETURN X and Y here
    #    model = train_xgb_classifier(
    #        X_train[input_variables], X_test[input_variables], y_train, y_test,
    #        X_train["mcweightvar"], X_test["mcweightvar"], f"{Name}")
    #
    #    print("MODEL CREATED")
    #
    #    predict_and_save(model, X_test, X_train, input_variables, 'xgb_output',
    #                     'xgb_output', f"{Name}_test", f"{Name}_train")
    #    print(f"File saved: cache/xgb_samples/{year}_test")
    #    print(f"File saved: cache/xgb_samples/{year}_train")
    #
    return X_train, X_test, y_train, y_test
