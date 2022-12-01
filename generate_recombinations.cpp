#include "TFile.h" 
#include "TTree.h"
#include "TH1D.h"
#include "Math/Vector4D.h" 
#include "TCanvas.h"
#include "ROOT/RDataFrame.hxx"

#include "particlefromtree.hpp" 

#include <fmt/format.h>
#include <fmt/compile.h>
#include <fmt/color.h>

#include <array>
#include <string_view>

using namespace std::literals;

constexpr double z_mass  = 0;
constexpr double e_mass  = 0.51099895000;
constexpr double mu_mass = 105.6583755;
constexpr double pi_mass = 139.57039;
constexpr double k_mass  = 493.677;
constexpr double p_mass  = 938.27208816;

constexpr std::array masses{z_mass, e_mass, mu_mass, pi_mass, k_mass, p_mass};

constexpr std::array names {"0"sv, "e"sv,  "mu"sv,  "pi"sv,  "k"sv,  "p"sv};

static_assert(masses.size() == names.size());

void mass_combinations( const std::string filename   = "../data/Lb2pKmm_mgUp_2018.root",
                        const std::string treename   = "Lb_Tuple/DecayTree",
                        const std::string outputfile = "cache/mass.root" )
{ 
  ROOT::EnableImplicitMT();

  const auto root_file = std::make_unique<TFile>(filename.c_str());
  const auto root_tree = std::unique_ptr<TTree>(dynamic_cast<TTree*>(root_file->Get(treename.c_str())));

  const auto entry_count = root_tree->GetEntries();

  fmt::print("{}, entries = {}\n", root_tree->GetName(), entry_count);

  root_tree->SetBranchStatus("*",0);

  const Particle<double> a( "h1" , root_tree.get() ); // p
  const Particle<double> b( "h2" , root_tree.get() ); // K
  const Particle<double> c( "mu1", root_tree.get() ); // mu
  const Particle<double> d( "mu2", root_tree.get() ); // mu

  const auto output_file = std::make_unique<TFile>(outputfile.c_str(), "RECREATE");
  const auto output_tree = std::make_unique<TTree>("Masses", "Masses");

  std::array<std::array<std::array<std::array<double, 6>, 6>, 6>, 6> vals;

  for (std::uint32_t ap = 0; ap < names.size(); ++ap)
  for (std::uint32_t bp = 0; bp < names.size(); ++bp)
  for (std::uint32_t cp = 0; cp < names.size(); ++cp)
  for (std::uint32_t dp = 0; dp < names.size(); ++dp)
  {
    const auto name = fmt::format("{}{}{}{}", names[ap], names[bp], names[cp], names[dp]);

    output_tree->Branch(name.c_str(), &vals[ap][bp][cp][dp], fmt::format("{}{}", name, "/D").c_str());
  }

  for ( std::int64_t i = 0; i < root_tree->GetEntries(); ++i )
  //for ( std::int64_t i = 0; i < 30000; ++i )
  {
    if (i % 5'000 == 0)
      fmt::print("Working on entry {}/{} ({}%)\n", i, entry_count, static_cast<double>(i) / static_cast<double>(entry_count) * 100.0);

    root_tree->GetEntry(i);

    auto for_mass = [&](const Particle<double> particle, const std::uint32_t p)
    {
      if (p == 0)
        return ROOT::Math::XYZTVector{};
      else
        return particle.getHypothesis(masses[p]);
    };
        

    for (std::uint32_t ap = 0; ap < names.size(); ++ap)
    for (std::uint32_t bp = 0; bp < names.size(); ++bp)
    for (std::uint32_t cp = 0; cp < names.size(); ++cp)
    for (std::uint32_t dp = 0; dp < names.size(); ++dp)
      vals[ap][bp][cp][dp] = (for_mass(a, ap) + for_mass(b, bp) + for_mass(c, cp) + for_mass(d, dp)).M();

    output_tree->Fill();
  }

  fmt::print("{}, entries = {}\n", output_tree->GetName(), output_tree->GetEntries());

  output_file->cd();
  output_tree->Write();

  return; 
}

int main()
{
  mass_combinations();
}
