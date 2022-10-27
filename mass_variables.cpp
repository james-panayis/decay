#include "TFile.h" 
#include "TTree.h"
#include "TH1D.h"
#include "TLorentzVector.h" 
#include "TCanvas.h"

#include "particlefromtree.hpp" 

#include <fmt/format.h>
#include <fmt/compile.h>
#include <fmt/color.h>

void mass_combinations( const std::string filename = "../data/Lb2pKmm_mgUp_2018.root",
			const std::string treename = "Lb_Tuple/DecayTree",
			const std::string outputfile = "cache/mass.root" ){ 

  //TFile* root_file = TFile::Open( filename.c_str() );
  //TTree* root_tree = (TTree*) root_file->Get( treename.c_str() );
  const auto root_file = std::make_unique<TFile>(filename.c_str());
  const auto root_tree = std::unique_ptr<TTree>(dynamic_cast<TTree*>(root_file->Get(treename.c_str())));
  
  root_tree->SetBranchStatus("*",0);

  Particle<Double_t> mu1( "mu1", root_tree.get() );
  Particle<Double_t> mu2( "mu2", root_tree.get() );
  Particle<Double_t> h1 ( "h1" , root_tree.get() );
  Particle<Double_t> h2 ( "h2" , root_tree.get() );

  constexpr double kaon_mass = 493.677;
  constexpr double muon_mass = 105.658;
  constexpr double pion_mass = 139.570;
  
  //TFile* output_file = new TFile(outputfile.c_str(),"RECREATE");
  //TTree* output_tree = new TTree("Masses","Masses");
  const auto output_file = std::make_unique<TFile>(outputfile.c_str(),"RECREATE");
  const auto output_tree = std::make_unique<TTree>("Masses","Masses");
    
  Double_t mPMu1, mPMu2, mKMu1, mKMu2; 

  output_tree->Branch("Mass_pmu1", &mPMu1, "Mass_pmu1/D"); 
  output_tree->Branch("Mass_pmu2", &mPMu2, "Mass_pmu2/D"); 
  output_tree->Branch("Mass_Kmu1", &mKMu1, "Mass_Kmu1/D"); 
  output_tree->Branch("Mass_Kmu2", &mKMu2, "Mass_Kmu2/D"); 

  Double_t mPKMu1, mPKMu2;
  output_tree->Branch("Mass_pKmu1", &mPKMu1, "Mass_pKmu1/D"); 
  output_tree->Branch("Mass_pKmu2", &mPKMu2, "Mass_pKmu2/D");
  
  Double_t mPKMuMu; 
  output_tree->Branch("Mass_pKmumu", &mPKMuMu, "Mass_pKmumu/D"); 

  Double_t mKK_PtoK, mKKMuMu_PtoK; 
  output_tree->Branch("Mass_KK_p_to_K", &mKK_PtoK, "Mass_KK_p_to_K/D"); 
  output_tree->Branch("Mass_KKmumu_p_to_K", &mKKMuMu_PtoK, "Mass_KKmumu_p_to_K/D"); 

  Double_t mPPi_KtoPi, mPPiMuMu_KtoPi;
  output_tree->Branch("Mass_ppi_K_to_pi", &mPPi_KtoPi, "Mass_ppi_K_to_pi/D"); 
  output_tree->Branch("Mass_ppimumu_K_to_pi", &mPPiMuMu_KtoPi, "Mass_ppimumu_K_to_pi/D"); 
  
  
  Double_t mKPi_PtoPi, mKPiMuMu_PtoPi;
  output_tree->Branch("Mass_Kpi_p_to_pi", &mKPi_PtoPi, "Mass_Kpi_p_to_pi/D");
  output_tree->Branch("Mass_Kpimumu_p_to_pi", &mKPiMuMu_PtoPi, "Mass_Kpimumu_p_to_pi/D");
  
  Double_t mKPi_PtoKandKtoPi, mKPiMuMu_PtoKandKtoPi;
  output_tree->Branch("Mass_Kpi_p_to_K_and_K_to_pi", &mKPi_PtoKandKtoPi, "Mass_Kpi_p_to_K_and_K_to_pi/D");
  output_tree->Branch("Mass_Kpimumu_p_to_K_and_K_to_pi", &mKPiMuMu_PtoKandKtoPi, "Mass_Kpimumu_p_to_K_and_K_to_pi/D");
  
  fmt::print("{}, entries = {}\n", root_tree->GetName(), root_tree->GetEntries());
  
  //std::cout 
  //  << root_tree->GetName() << ", entries = " 
  //  << root_tree->GetEntries() << std::endl;

  for ( Long64_t i = 0; i < root_tree->GetEntries(); i++ ){ 
    root_tree->GetEntry(i);
    
    TLorentzVector pp1  = h1.getVec(); 
    TLorentzVector pK2  = h2.getVec();
    TLorentzVector pmu1 = mu1.getVec(); 
    TLorentzVector pmu2 = mu2.getVec();
    
    // 2-body cross-combinations
    mPMu1 = ( pp1 + pmu1 ).M();
    mPMu2 = ( pp1 + pmu2 ).M();
    mKMu1 = ( pK2 + pmu1 ).M();
    mKMu2 = ( pK2 + pmu2 ).M();

    // 3-body cross-combinations
    mPKMu1 = ( pp1 + pK2 + pmu1 ).M();
    mPKMu2 = ( pp1 + pK2 + pmu2 ).M();

    // 4-body combination
    mPKMuMu = ( pp1 + pK2 + pmu1 + pmu2 ).M(); 

    // Swap mass hypothesis on h1/h2
    TLorentzVector pK1  = h1.getHypothesis( kaon_mass );
    TLorentzVector ppi1 = h1.getHypothesis( pion_mass );
    TLorentzVector ppi2 = h2.getHypothesis( pion_mass );
    
    mKK_PtoK     = ( pK1 + pK2 ).M();
    mKKMuMu_PtoK = ( pK1 + pK2 + pmu1 + pmu2 ).M(); 

    mPPi_KtoPi     = ( pp1 + ppi2 ).M();
    mPPiMuMu_KtoPi = ( pp1 + ppi2 + pmu1 + pmu2 ).M();
    
    mKPi_PtoPi = ( ppi1 + pK2 ).M();
    mKPiMuMu_PtoPi = ( ppi1 + pK2 + pmu1 + pmu2).M();
    
    mKPi_PtoKandKtoPi = ( pK1 + ppi2 ).M();
    mKPiMuMu_PtoKandKtoPi = ( pK1 + ppi2 + pmu1 + pmu2).M();

    output_tree->Fill();
  }

  fmt::print("{}, entries = {}\n", output_tree->GetName(), output_tree->GetEntries());
  
  //std::cout 
  //  << output_tree->GetName() << ", entries = " 
  //  << output_tree->GetEntries() << std::endl;

  output_file->cd();
  output_tree->Write();
  //output_file->Close();
  
  return; 
}

int main()
{
  mass_combinations();
}
