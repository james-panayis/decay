import pandas as pd
import matplotlib.pyplot as plt
import ROOT
from itertools import product
import numpy as np
import sys
sys.path.append("../common")
import common_definitions as cd
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split

# For the error propagation when adding or dividing histograms
ROOT.TH1.SetDefaultSumw2()

plt.rc("font", **{"family": "serif"})  # , "serif": ["Roman"]})
plt.rc("text", usetex=True)

#------Training Tools---------------------------------------------------------


def train_xgb_classifier(X_train,
                         X_test,
                         y_train,
                         y_test,
                         sample_weight=None,
                         sample_weight_eval_set=None,
                         name=None):
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
        'max_depth': 2,
        'eta': 0.7,
        #'gamma': 0.03,
        'subsample': 0.4,
        #'max_bin': 256,
        #'objective': 'binary:logistic',
        #'nthread': 4,
        #'eval_metric': 'auc',
        'n_estimators': 500,
        #'use_label_encoder':False,
    }

    param = {
        'max_depth': 5,
        'eta': 1,
        'gamma': 0.03,
        'subsample': 0.5,
        'max_bin': 20,
        'objective': 'binary:logistic',
        'nthread': 4,
        'eval_metric': 'auc',
        'n_estimators': 100,
        'use_label_encoder': False,
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


def predict_and_save(model, X_test, X_train, columns, test_tree_output_name,
                     train_tree_output_name, test_data_name, train_data_name):
    '''
    Takes the training sample, testing sample and the model. Predicts test and train, appends to train and test samples, converts to a root file and saves under train_data_name and test_data_name respectively. Returns the numpy versions of X-test and X_train with the predictions columns on.
    
    @param model - this is a xgboost model, which can be created using the train_xgb_classifier function
    @param X_train - the training sample in pandas format
    @param X_test - the testing sample in pandas format 
    @param columns - list of column titles
    '''

    pred_train = model.predict_proba(
        X_train[columns],
        validate_features=True,
        iteration_range=(0, model.best_iteration + 1))
    pred_test = model.predict_proba(
        X_test[columns],
        validate_features=True,
        iteration_range=(0, model.best_iteration + 1))

    X_train[train_tree_output_name] = pred_train[:, 1]
    X_test[test_tree_output_name] = pred_test[:, 1]

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

    df_train = ROOT.RDF.MakeNumpyDataFrame(X_train_dict)
    df_test = ROOT.RDF.MakeNumpyDataFrame(X_test_dict)

    #save file
    df_train.Snapshot('tree', f'{train_data_name}.root')
    df_test.Snapshot('tree', f'{test_data_name}.root')

    return X_test, X_train


def training_func_direct(data):
    #trains the data using the old method
    '''
    This method ignores all splitting of data and all use of classifiers and creates a model directly from the data. Returns a XGB model to use for in predictions.
    @Params data - data given in an XGBoost dMatrix form
    '''
    evallist = [(data, 'train')]
    num_round = 22

    param = {
        'max_depth': 5,
        'eta': 1,
        'gamma': 0.03,
        'subsample': 0.5,
        'max_bin': 20,
        'objective': 'binary:logistic',
        'nthread': 4,
        'eval_metric': 'auc',
    }

    bst = xgb.train(param, data, num_round, evallist)

    return bst


def split_data(data, label):
    #splits up the data
    '''
    Splitting the data in half without shuffling
    (If you want a shuffle you can use train_test_split in the scipy library)
    returns a split data (X_train and X_test) and a matching split label (y_train and y_test)
    @params data - data in a pandas dataframe
    @params label - in a pandas dataframe

    '''
    #    X_sig = data.loc[data.label == 1]
    #    X_bkg = data.loc[data.label == 0]
    #
    #    X_sig_train = X_sig.head(n=int(X_sig.shape[0] / 2))
    #    X_sig_test = X_sig.tail(n=int(X_sig.shape[0] / 2))
    #
    #    X_bkg_train = X_bkg.head(n=int(X_bkg.shape[0] / 2))
    #    X_bkg_test = X_bkg.tail(n=int(X_bkg.shape[0] / 2))
    #
    #    X_train = pd.concat([X_sig_train, X_bkg_train],
    #                        sort=False,
    #                        ignore_index=True)
    #    X_test = pd.concat([X_sig_test, X_bkg_test], sort=False, ignore_index=True)
    #
    #    y_train = X_train["label"]
    #    y_test = X_test["label"]
    #
    # train_test_splite method

    X = data
    Y = label
    seed = 42
    test_size = 0.5
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=test_size, shuffle=True, random_state=seed)

    return X_train, X_test, y_train, y_test


def load_test_train(testfile, trainfile, test_tree, train_tree):
    '''
    Loads the already produced test and train data files and converts them to pa
ndas dataframes.
    
    @param testfile - name of test file
    @param trainfile - name of train file
    @param test_tree - tree where test is stored
    @param train_tree - tree where train is stored

    '''

    #loads the test and train data in pandas format
    test_RDF = ROOT.RDataFrame(test_tree, testfile)
    train_RDF = ROOT.RDataFrame(train_tree, trainfile)

    train_panda = pd.DataFrame(train_RDF.AsNumpy())
    test_panda = pd.DataFrame(test_RDF.AsNumpy())

    #train_panda = train_panda.loc[train_panda.mva_mcweight_nTracks_B_PT<2]
    #test_panda = test_panda.loc[test_panda.mva_mcweight_nTracks_B_PT<2]

    return test_panda, train_panda


#----Saving-and-Loading-Tools-----------------------------------------------

SelectionDirectory = '../cache/'


def getOutFileNamePrefix():
    return f"{SelectionDirectory}"


def getOutTupleName():
    return 'XGBOutput'


def doesXGBInputExist(dataname, year, polarity, recalculate=False):
    '''
    Checks if test file is present in the given folder
    @params dataname - name of the data file in a string form, without directory, year or polarity
    @params year - year of data in string form
    @params polarity - polarity of data in string form
    @recalculate - Bool, if true will reproduce data even if its found in the given directory
    '''

    filenameXtest = f"{getOutFileNamePrefix()}{dataname}_{year}_{polarity}_Xtest.root"
    filenameXtrain = f"{getOutFileNamePrefix()}{dataname}_{year}_{polarity}_Xtrain.root"
    if recalculate == True or not (os.path.exists(filenameXtest)):
        return False
    else:
        return True
