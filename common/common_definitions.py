# Selection without PID and BDT cuts
baseSelection_mc = "survive_PT_preselection && survive_full_fiducial && survive_trigger_selection "
baseSelection_data = "survive_preselection==1 && survive_full_fiducial && survive_trigger_selection "

# Other selection
basicSelection = "survive_trigger_selection==1 && survive_preselection==1 && bdt_output>0.4 && "
muonSelection = "muplus_ProbNNmu > 0.2 && muminus_ProbNNmu > 0.2 && " + \
                 "muplus_isMuon == 1 && muminus_isMuon == 1 && hadron_isMuon == 0 "

isSignalSel = "(B_BKGCAT == 0 || B_BKGCAT == 10 || B_BKGCAT == 50) "

jpsi_PDG_M = 3096.9
Lb_PDG_M = 5619.6 
B_DTF_dimuon_M_res_jpsipi = 7.3
B_DTF_dimuon_M_res_Dstpi = 6.7
B_DTF_dimuon_M_res_jpsik = 6.6






sideband_min = 5 * B_DTF_dimuon_M_res_Dstpi
sideband_max = 10 * B_DTF_dimuon_M_res_Dstpi

#pimumu_sideband = " && abs(B_DTF_dimuon_M - {0}) > {1} && abs(B_DTF_dimuon_M - {0}) < {2} ".format(
#    Dst_PDG_M, sideband_min, sideband_max)


B_PDG_M = 5279.34
Bc_PDG_M = 6274.47
B_M_res_jpsipi = 18.54
B_M_res_Dstpi = 19.77
B_M_res_jpsik = 17.48

B_M_sigregion = 3 * B_M_res_Dstpi

pimumu_B_M_sigregion = " && abs(B_M - {0}) < {1} ".format(
    B_PDG_M, B_M_sigregion)

#pimumu_nosigbox = " !((abs(B_DTF_dimuon_M - {0}) < {1}) && (abs(B_M - {2}) < {3})) ".format(
#    Dst_PDG_M, sideband_three, B_PDG_M, B_M_sigregion)

################# FILL THIS IN ########################################

fullLb_M_range = " && Lb_M > !!!!! && Lb_M < !!!!! "

training_sideband = "Lb_M > 4750 && !(Lb_M > 5250 && Lb_M < 5725) && abs(dimuon_M - 3096.9) > 100 && abs(dimuon_M - 3686.097) > 100"


################ LIST OF ALL SAMPLES WE WILL USE ######################

inputDirectory = '../../../../../../work/c/cawhite/public/data/'

samples = dict()

#samples["Lb2pKmm_mgDn_2016"] = {
#    "ntuples": inputDirectory + "Lb2pKmm_mgDn_2016.root",
#    "tree": "Lb_Tuple/DecayTree"
#}

#samples["Lb2pKmm_mgDn_2017"] = {
#    "ntuples": inputDirectory + "Lb2pKmm_mgDn_2017.root",
#    "tree": (TTree*)_file0->Get("Lb_Tuple/DecayTree")
#}
#
#samples["Lb2pKmm_mgDn_2018"] = {
#    "ntuples": inputDirectory + "Lb2pKmm_mgDn_2018.root",
#    "tree": (TTree*)_file0->Get("Lb_Tuple/DecayTree")
#}
#
samples["Lb2pKmm_mgUp_2016"] = {
    "ntuples": inputDirectory + "Lb2pKmm_mgUp_2016.root",
    "tree": "Lb_Tuple/DecayTree"
}
#
#samples["Lb2pKmm_mgUp_2017"] = {
#    "ntuples": inputDirectory + "Lb2pKmm_mgUp_2017.root",
#    "tree": (TTree*)_file0->Get("Lb_Tuple/DecayTree")
#}
#
#samples["Lb2pKmm_mgUp_2018"] = {
#    "ntuples": inputDirectory + "Lb2pKmm_mgUp_2018.root",
#    "tree": (TTree*)_file0->Get("Lb_Tuple/DecayTree")
#}
#
#samples["Lb2pKmm_sim_mgDn_2016"] = {
#    "ntuples": inputDirectory + "Lb2pKmm_sim_mgDn_2016.root",
#    "tree": "tree"
#}

samples["Lb2pKmm_sim_mgDn_2017"] = {
    "ntuples": inputDirectory + "Lb2pKmm_sim_mgDn_2017.root",
    "tree": "tree"
}

samples["Lb2pKmm_sim_mgDn_2018"] = {
    "ntuples": inputDirectory + "Lb2pKmm_sim_mgDn_2018.root",
    "tree": "tree"
}

samples["Lb2pKmm_sim_mgUp_2016"] = {
    "ntuples": inputDirectory + "Lb2pKmm_sim_mgUp_2016.root",
    "tree": "tree"
}

samples["Lb2pKmm_sim_mgUp_2017"] = {
    "ntuples": inputDirectory + "Lb2pKmm_sim_mgUp_2017.root",
    "tree": "tree"
}

samples["Lb2pKmm_sim_mgUp_2018"] = {
    "ntuples": inputDirectory + "Lb2pKmm_sim_mgUp_2018.root",
    "tree": "tree"
}

###################################################################




dimuon_M_xlim_Dst = [1900, 2100]
dimuon_M_xlim_jpsi = [3000, 3200]
Dstpi_sel = muonSelection + "&& hadron_ProbNNpi > 0.2 && hadron_ProbNNk < 0.05 "
jpsipi_sel = muonSelection + "&& hadron_ProbNNpi > 0.2 && hadron_ProbNNk < 0.05 "  # && tight_jpsi_region==1
jpsik_sel = muonSelection + "&& hadron_ProbNNk > 0.4 "  # && tight_jpsi_region==1

years = [2016, 2017, 2018]

branches = [
    "B_M", "B_DTF_dimuon_M", "dimuon_M", "runNumber", "eventNumber",
    "B_BKGCAT", "bdt_output", "bdt_output_Jpsi_psi2s_veto", "Polarity"
]

new_branches = [
    "B_M", "B_DTF_J_psi_1S_M[0]", "dimuon_M", "runNumber", "eventNumber",
    "B_BKGCAT", "bdt_output", "bdt_output_Jpsi_psi2s_veto", "Polarity"
]

pid_branches = [
    "muplus_ProbNNmu", "muminus_ProbNNmu", "muplus_isMuon", "muminus_isMuon",
    "hadron_isMuon", "hadron_ProbNNpi", "hadron_ProbNNk"
]

friends = {
    "MVAOutput": "mvaoutput",
    "preselection": "preselection",
    "triggerselection": "triggerselection",
    #    "kinematicselection": "kinematicselection",
    "pidselection": "pidselection",
    "geometry": "geometry",
}

