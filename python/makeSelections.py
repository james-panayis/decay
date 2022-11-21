import pandas as pd
import matplotlib.pyplot as plt
import ROOT
import numpy as np
import xgboost as xgb
import sys
import os
sys.path.append("../common")
import common_definitions as cd

import sys

sys.path.append('common/')
sys.path.append('python/')

#import triggerselection

#selections=[triggerselection.addTriggerSelection]


def makedataname(data, year, polarity):
    name = '{}_mg{}_{}.root'.format(data, polarity, year)
    return name


def makesimname(data, year, polarity):
    name = '{}_sim_mg{}_{}.root'.format(data, polarity, year)
    return name


#TRIGGER
def passTriggerSelection(tree, filename):

    rootdata = ROOT.RDataFrame(tree, f'../../data/{filename}')
    print('Data Loaded')

    columns = [
        'Lb_L0MuonDecision_TOS', 'Lb_Hlt1TrackMVADecision_TOS',
        'Lb_Hlt1TrackMuonDecision_TOS', 'Lb_Hlt2Topo2BodyDecision_TOS',
        'Lb_Hlt2Topo3BodyDecision_TOS', 'Lb_Hlt2Topo4BodyDecision_TOS',
        'Lb_Hlt2TopoMu2BodyDecision_TOS', 'Lb_Hlt2TopoMu3BodyDecision_TOS',
        'Lb_Hlt2TopoMu4BodyDecision_TOS', 'Lb_Hlt2DiMuonDetachedDecision_TOS'
    ]
    data = pd.DataFrame(rootdata.AsNumpy(columns))
    print("converted")

    result = pd.DataFrame()
    result['survive_trigger_selection'] = np.ones(shape=data.shape[0])
    result['trigger_selection_killed_by'] = np.zeros(shape=data.shape[0])
    result['survive_L0_trigger_selection'] = np.ones(shape=data.shape[0])
    result['survive_HLT1_trigger_selection'] = np.ones(shape=data.shape[0])
    result['survive_HLT2_trigger_selection'] = np.ones(shape=data.shape[0])
    print("Results columns made")

    i = 0
    while (i < data.shape[0]):
        if data['Lb_L0MuonDecision_TOS'][
                i] == 0:  #and data['Lb_L0DiMuonDecision_TOS == 0:
            result['survive_trigger_selection'][i] = 0
            result['survive_L0_trigger_selection'][i] = 0
            result['trigger_selection_killed_by'][i] += 1
        if data['Lb_Hlt1TrackMVADecision_TOS'][i] == 0 and data[
                'Lb_Hlt1TrackMuonDecision_TOS'][i] == 0:
            result['survive_trigger_selection'][i] = 0
            result['survive_HLT1_trigger_selection'][i] = 0
            result['trigger_selection_killed_by'][i] += 2
        if data['Lb_Hlt2Topo2BodyDecision_TOS'][i] == 0 and data[
                'Lb_Hlt2Topo3BodyDecision_TOS'][i] == 0 and data[
                    'Lb_Hlt2Topo4BodyDecision_TOS'][i] == 0 and data[
                        'Lb_Hlt2TopoMu2BodyDecision_TOS'][i] == 0 and data[
                            'Lb_Hlt2TopoMu3BodyDecision_TOS'][i] == 0 and data[
                                'Lb_Hlt2TopoMu4BodyDecision_TOS'][
                                    i] == 0 and data[
                                        'Lb_Hlt2DiMuonDetachedDecision_TOS'][
                                            i] == 0:
            result['survive_trigger_selection'][i] = 0
            result['survive_HLT2_trigger_selection'][i] = 0
            result['trigger_selection_killed_by'][i] += 4
        i += 1
        if (i % 100000 == 0):
            print(i)

    result_dict = result.to_dict("list")
    result_dict = {
        key: np.array(result_dict[key])
        for key in result_dict.keys()
    }

    df_result = ROOT.RDF.MakeNumpyDataFrame(result_dict)

    df_result.Snapshot('tree', f'../selections/TriggerSelection_{filename}')

    return


def passPreselection(tree, filename):

    print("MAKING PRESELECTIONS")
    rootdata = ROOT.RDataFrame(tree, f'../../data/{filename}')
    print('Data Loaded')

    columns = [
        'Lb_PT',
        'h1_PT',
        'h2_PT',
        'mu1_PT',
        'mu2_PT',
        'nTracks',
        'mu1_P',
        'mu1_PZ',
        'mu2_P',
        'mu2_PZ',
        'h1_P',
        'h1_PZ',
        'h2_P',
        'h2_PZ',
    ]
    data = pd.DataFrame(rootdata.AsNumpy(columns))
    print("converted")

    result = pd.DataFrame()
    result['survive_preselection'] = np.ones(shape=data.shape[0])
    result['preselection_killed_by'] = np.zeros(shape=data.shape[0])
    result['survive_PT_preselection'] = np.ones(shape=data.shape[0])
    result['survive_track_pid_fiducial'] = np.zeros(shape=data.shape[0])
    result['survive_full_fiducial'] = np.zeros(shape=data.shape[0])
    print("result columns made")

    i = 0
    while (i < data.shape[0]):
        if data['Lb_PT'][i] <= 220:
            result['survive_preselection'][i] = 0
            result['preselection_killed_by'][i] += 1
            result['survive_PT_preselection'][i] = 0
        if data['h1_PT'][i] <= 100:
            result['survive_preselection'][i] = 0
            result['preselection_killed_by'][i] += 2
            result['survive_PT_preselection'][i] = 0
        if data['h2_PT'][i] <= 72:
            result['survive_preselection'][i] = 0
            result['preselection_killed_by'][i] += 4
            result['survive_PT_preselection'][i] = 0
        if data['mu1_PT'][i] <= 150:
            result['survive_preselection'][i] = 0
            result['preselection_killed_by'][i] += 8
            result['survive_PT_preselection'][i] = 0
        if data['mu2_PT'][i] <= 120:
            result['survive_preselection'][i] = 0
            result['preselection_killed_by'][i] += 16
            result['survive_PT_preselection'][i] = 0
        if data['nTracks'][i] > 0 and data['nTracks'][i] < 500:
            mu1_ETA = 0.5 * ROOT.TMath.Log(
                (data['mu1_P'][i] + data['mu1_PZ'][i]) /
                (data['mu1_P'][i] - data['mu1_PZ'][i]))
            mu2_ETA = 0.5 * ROOT.TMath.Log(
                (data['mu2_P'][i] + data['mu2_PZ'][i]) /
                (data['mu2_P'][i] - data['mu2_PZ'][i]))
            h1_ETA = 0.5 * ROOT.TMath.Log(
                (data['h1_P'][i] + data['h1_PZ'][i]) /
                (data['h1_P'][i] - data['h1_PZ'][i]))
            h2_ETA = 0.5 * ROOT.TMath.Log(
                (data['h2_P'][i] + data['h2_PZ'][i]) /
                (data['h2_P'][i] - data['h2_PZ'][i]))

            if data['mu1_P'][i] > 3400 and data['mu1_P'][
                    i] < 27000 and mu1_ETA > 1.9 and mu1_ETA < 4.9 and data[
                        'mu2_P'][i] > 3300 and data['mu2_P'][
                            i] < 250000 and mu2_ETA > 1.9 and mu2_ETA < 4.9 and data[
                                'h1_P'][i] > 1575 and data['h1_P'][
                                    i] < 180000 and h1_ETA > 1.9 and h1_ETA < 4.9 and data[
                                        'h2_P'][i] > 1550 and data['h2_P'][
                                            i] < 140000 and h2_ETA > 1.9 and h2_ETA < 4.9:

                result['survive_track_pid_fiducial'][i] = 1

                if data['nTracks'][i] < 300:
                    result['survive_full_fiducial'][i] = 1

        i += 1
        if (i % 100000 == 0):
            print(i)

    result_dict = result.to_dict("list")
    result_dict = {
        key: np.array(result_dict[key])
        for key in result_dict.keys()
    }

    df_result = ROOT.RDF.MakeNumpyDataFrame(result_dict)

    df_result.Snapshot('tree', f'../selections/Preselection_{filename}')

    return


def passFiducialSelection(tree, filename):

    rootdata = ROOT.RDataFrame(tree, f'../../data/{filename}')
    print('Data Loaded')

    columns = ['nTracks', 'Lb_PT', 'Lb_OWNPV_NDOF']
    data = pd.DataFrame(rootdata.AsNumpy(columns))
    print("converted")

    result = pd.DataFrame()
    result['survive_fiducial'] = np.ones(shape=data.shape[0])
    result['fiducial_killed_by'] = np.zeros(shape=data.shape[0])
    print("Results columns made")

    i = 0
    while (i < data.shape[0]):
        if data['nTracks'][i] >= 495:
            result['survive_fiducial'][i] = 0
            result['fiducial_killed_by'][i] += 1
        if data['Lb_PT'][i] >= 27000:
            result['survive_fiducial'][i] = 0
            result['fiducial_killed_by'][i] += 1
        if data['Lb_OWNPV_NDOF'][i] >= 300:
            result['survive_fiducial'][i] = 0
            result['fiducial_killed_by'][i] += 1

        i += 1
        if (i % 100000 == 0):
            print(i)

    result_dict = result.to_dict("list")
    result_dict = {
        key: np.array(result_dict[key])
        for key in result_dict.keys()
    }

    df_result = ROOT.RDF.MakeNumpyDataFrame(result_dict)

    df_result.Snapshot('tree', f'../selections/FiducialSelection_{filename}')

    return


############################################################
selections = [
    passPreselection,
    passFiducialSelection,
]  #'passTriggerSelection', #passFiducialSelection
DataType = ['Lb2pKmm', 'Lb2pKmm_sim']
years = [2016, 2017, 2018]
polarities = ['Up', 'Dn']

for year in years:
    for polarity in polarities:
        for selection in selections:
            namedata = makedataname('Lb2pKmm', year, polarity)
            namesim = makesimname('Lb2pKmm', year, polarity)
            print(f"Making selection for {year} {polarity}")
            selection('tree', namesim)
            selection('Lb_Tuple/DecayTree', namedata)
            print(f"complete {year} {polarity}")

#passTriggerSelection('Lb_Tuple/DecayTree', 'Lb2pKmm_mgUp_2018.root')
