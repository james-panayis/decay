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

print(xgb.__version__)

name = "H"
model_name = f"{name}.json"

#(Lb_M > 5350 && Lb_M < 5850) &&
#set_sel = "abs(Jpsi_M - 3096.9) > 100 && abs(Jpsi_M - 3686.097) > 100 && abs(Jpsi_M - 1020)>30"

#sig_training_antisideband = "(Lb_M > 5350 && Lb_M < 5850)"
sig_sel = f"(Lb_BKGCAT == 0 || Lb_BKGCAT == 10 || Lb_BKGCAT == 50)"
#bkg_training_sideband = "!(Lb_M > 5350 && Lb_M < 5850)"

#vetoes = "abs(Jpsi_M - 3096.9) > 100 && abs(Jpsi_M - 3686.097) > 100 && abs(Jpsi_M - 1020)>30"

vetoes = " \
!((Jpsi_M*Jpsi_M)/1000000 > 8.0 &&  (Jpsi_M*Jpsi_M)/1000000 < 11) &&  \
!((Jpsi_M*Jpsi_M)/1000000 > 12.5 && (Jpsi_M*Jpsi_M)/1000000 < 15) &&  \
!((Jpsi_M*Jpsi_M)/1000000 > 0.98 && (Jpsi_M*Jpsi_M)/1000000 < 1.10) "

sel = "survive_trigger_selection"

bkg_selection = f"{vetoes} && {sel}"
sig_selection = f"{vetoes} && {sig_sel} && {sel}"

#sel = set_sel
#sel = f"{set_sel}"
# && survive_trigger_selection"
# && survive_preselection && survive_PT_preselection && survive_full_fiducial"

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
    #    "h1_PIDp",
    #    "h2_PIDK",
    #    "mu1_PIDmu",
    #    "mu2_PIDmu",
    #    "h1_ProbNNp",
    #    "h2_ProbNNk",
    #    "mu1_ProbNNmu",
    #    "mu2_ProbNNmu",
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
    #    "max(OWNPV_Y)",
]

#derived_inputs = ["max(muplus_PT,muminus_PT)"]
#["max(max(B_DOCA_muplus_hadron,B_DOCA_muminus_hadron),B_DOCA_muplus_muminus)"] +
# + ["abs(muplus_P-muminus_P)"]

selections = [
    "survive_trigger_selection",
    #    "survive_full_fiducial",
    #    "survive_preselection",
    #    "survive_PT_preselection",
    #    "survive_track_pid_fiducial",
    #    "survive_HLT1_trigger_selection",
    #    "survive_HLT2_trigger_selection",
    #    "survive_fiducial",
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

#LOAD FULL DATA SET
#ADJUST THIS FOR FULL DATA SET

data = ["Lb2pKmm"]
years = [2016, 2017, 2018]
polarities = ["Up", "Dn"]

#bkg_dataframe = list()
bkg_events = 0

#sig_dataframe = list()

final_sig_dataframe = list()
sig_events = 0

cols = input_columns + selections + ["eventNumber"] + ["Lb_M"] + ["runNumber"]
cols2 = input_columns + selections + ["eventNumber"] + ["Lb_M"] + ["year"] + [
    "polarity"
] + ["runNumber"]
bkg_dataframe = pd.DataFrame(columns=cols2)
sig_dataframe = pd.DataFrame(columns=cols2)

for year in years:
    for polarity in polarities:

        sig_sample_name = f"Lb2pKmm_sim_mg{polarity}_{year}"
        bkg_sample_name = f"Lb2pKmm_mg{polarity}_{year}"

        sig_file_name = cd.samples[sig_sample_name]["ntuples"]
        bkg_file_name = cd.samples[bkg_sample_name]["ntuples"]

        sig_chain = ROOT.TChain("tree")
        sig_chain.Add(sig_file_name)

        print('Created sig chain')

        #sig_file = uproot.open("sig_file_name")
        #bkg_file = uproot.open("bkg_file_name")

        bkg_chain = ROOT.TChain("Lb_Tuple/DecayTree")
        bkg_chain.Add(bkg_file_name)

        print('Created bkg chain')

        #for selection in selection_files:
        #    sig_friend_chain = ROOT.TChain('tree')
        #    sig_sel_name = f"../selections/{selection}_{sig_sample_name}.root"
        #    sig_friend_chain.Add(sig_sel_name)
        #    sig_chain.AddFriend(sig_friend_chain)
        for selection in selection_files:
            print('Added selection', selection)
            sig_sel_name = f"../selections/{selection}_{sig_sample_name}.root"
            sig_chain.AddFriend('tree', sig_sel_name)

        print('Added sig friends')

        #sig_rdframe = ROOT.RDataFrame(sig_chain)
        sig_rdframe = ROOT.RDataFrame(sig_chain)

        print('Made sig RDataFrame')

        sig_rdfilter = sig_rdframe.Filter(sig_selection)
        #        sig_tree = sig_rdfilter.Get(tree_name)

        print('Made sig filtered RDataFrame')
        sig_numpy = sig_rdfilter.AsNumpy(cols)

        print('Made sig numpy array')

        #sig_frame = pd.DataFrame(
        #    data=sig_rdframe.Filter(sig_selection).AsNumpy(
        # columns + selections + ["eventNumber"] + ["Lb_M"]))
        sig_frame = pd.DataFrame(data=sig_numpy)

        print('Made sig pd frame')

        for selection in selection_files:
            print('Added selection', selection)
            bkg_sel_name = f"../selections/{selection}_{bkg_sample_name}.root"
            bkg_chain.AddFriend('tree', bkg_sel_name)

        print('Added bkg friends')

        #bkg_rdframe = ROOT.RDataFrame(bkg_chain)
        bkg_rdframe = ROOT.RDataFrame(bkg_chain)

        print('Made bkg RDataFrame')

        bkg_rdfilter = bkg_rdframe.Filter(bkg_selection)
        #        bkg_tree = bkg_rdfilter.Get(tree_name)

        print('Made bkg filtered RDataFrame')
        cols = columns + selections + ["eventNumber"] + ["Lb_M"]
        bkg_numpy = bkg_rdfilter.AsNumpy(cols)

        bkg_frame = pd.DataFrame(data=bkg_numpy)

        print('Made bkg dataframe')

        #        for selection in selection_files:
        #            print( 'Added selection', selection )
        #            #bkg_friend_chain = ROOT.TChain('tree')
        #            bkg_sel_name = f"../selections/{selection}_{bkg_sample_name}.root"
        #            #bkg_friend_chain.Add(bkg_sel_name)
        #            bkg_chain.AddFriend('tree', bkg_sel_name)
        #
        #        print( 'Added bkg friends' )
        #
        #        bkg_rdframe = ROOT.RDataFrame(bkg_chain)
        #        print( 'Made bkg RDataFrame' )
        #
        #        bkg_frame = pd.DataFrame(
        #            data=bkg_rdframe.Filter(bkg_selection).AsNumpy(
        #                columns + selections + ["eventNumber"] + ["Lb_M"]))
        #
        #        print( 'Made bkg data frame' )

        bkg_frame["year"] = year
        if polarity == "Up":
            bkg_frame["polarity"] = 1
        else:
            bkg_frame["polarity"] = 0

        sig_frame["year"] = year
        if polarity == "Up":
            sig_frame["polarity"] = 1
        else:
            sig_frame["polarity"] = 0

        print(f"ADDED {year} {polarity}")

        bkg_dataframe = pd.concat([bkg_dataframe, bkg_frame],
                                  sort=False,
                                  ignore_index=True)
        sig_dataframe = pd.concat([sig_dataframe, sig_frame],
                                  sort=False,
                                  ignore_index=True)

print('Concatenated files')

#sample_name = "Lb2pKmm_sim_mgUp_2016"
#sample_file_name = cd.samples[sample_name]["ntuples"]
#sample_chain = ROOT.TChain("tree")
#sample_chain.Add(sample_file_name)
#
##ADD SELECTIONS
#for selection in selection_files:
#    sample_friend_chain = ROOT.TChain('tree')
#    sample_sel_name = f"../selections/{selection}_{sample_name}.root"
#    sample_friend_chain.Add(sample_sel_name)
#    sample_chain.AddFriend(sample_friend_chain)
#    print(f"added: {selection}")
#
#
##MAKE CUTS
#sample_rdframe = ROOT.RDataFrame(sample_chain)
#sample_frame = pd.DataFrame(data=sample_rdframe.Filter(sel).AsNumpy(columns + ["eventNumber"]+["Lb_M"]))
#
##ADD DEFINITIONS OF VARIABLES
#sample_frame["max(OWNPV_Y)"] = sample_frame[["h1_OWNPV_Y", "h2_OWNPV_Y","mu1_OWNPV_Y","mu2_OWNPV_Y"]].max(axis=1)
#
#

#load model
model = xgb.XGBClassifier()
model.load_model(f"../cache/models/{model_name}")
print(model.best_iteration + 1)

#APPLY MODEL
sig_prediction = model.predict_proba(
    sig_dataframe[input_columns],
    validate_features=True,
    iteration_range=(0, model.best_iteration + 1))
bkg_prediction = model.predict_proba(
    bkg_dataframe[input_columns],
    validate_features=True,
    iteration_range=(0, model.best_iteration + 1))

#Make new data array
sim_data = {
    "eventNumber": sig_dataframe["eventNumber"],
    "runNumber": sig_dataframe["runNumber"],
    "Lb_M": sig_dataframe["Lb_M"],
    "xgb_output": sig_prediction[:, 1],
    "year": sig_dataframe["year"],
    "polarity": sig_dataframe["polarity"]
    #    "h1_PIDp": sig_dataframe["h1_PIDp"],
    #    "h2_PIDK": sig_dataframe["h2_PIDK"],
    #    "mu1_PIDmu": sig_dataframe["mu1_PIDmu"],
    #    "mu2_PIDmu": sig_dataframe["mu2_PIDmu"],
    #    "h1_ProbNNp": sig_dataframe["h1_ProbNNp"],
    #    "h2_ProbNNk": sig_dataframe["h2_ProbNNk"],
    #    "mu1_ProbNNmu": sig_dataframe["mu1_ProbNNmu"],
    #    "mu2_ProbNNmu": sig_dataframe["mu2_ProbNNmu"]
}
output = pd.DataFrame(sim_data)

output_dict = output.to_dict("list")
output_dict = {key: np.array(output_dict[key]) for key in output_dict.keys()}

df_output = ROOT.RDF.MakeNumpyDataFrame(output_dict)
df_output.Snapshot('tree', f'../cache/D4J/Sim_{name}_D4J.root')

real_data = {
    "event": bkg_dataframe["eventNumber"],
    "runNumber": bkg_dataframe["runNumber"],
    "Lb_M": bkg_dataframe["Lb_M"],
    "xgb_output": bkg_prediction[:, 1],
    "year": bkg_dataframe["year"],
    "polarity": bkg_dataframe["polarity"]
    #    "h1_PIDp": bkg_dataframe["h1_PIDp"],
    #    "h2_PIDK": bkg_dataframe["h2_PIDK"],
    #    "mu1_PIDmu": bkg_dataframe["mu1_PIDmu"],
    #    "mu2_PIDmu": bkg_dataframe["mu2_PIDmu"],
    #    "h1_ProbNNp": bkg_dataframe["h1_ProbNNp"],
    #    "h2_ProbNNk": bkg_dataframe["h2_ProbNNk"],
    #    "mu1_ProbNNmu": bkg_dataframe["mu1_ProbNNmu"],
    #    "mu2_ProbNNmu": bkg_dataframe["mu2_ProbNNmu"]
}
real_output = pd.DataFrame(real_data)

real_output_dict = real_output.to_dict("list")
real_output_dict = {
    key: np.array(real_output_dict[key])
    for key in real_output_dict.keys()
}

df_real_output = ROOT.RDF.MakeNumpyDataFrame(real_output_dict)
df_real_output.Snapshot('tree', f'../cache/D4J/Real_{name}_D4J.root')
#FIND Best working point using FoM
