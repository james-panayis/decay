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
//constexpr double e_mass  = 0.51099895000; //unused
constexpr double mu_mass = 105.6583755;
constexpr double pi_mass = 139.57039;
constexpr double k_mass  = 493.677;
constexpr double p_mass  = 938.27208816;

constexpr std::array masses{z_mass, mu_mass, pi_mass, k_mass, p_mass};

constexpr std::array names {"0"sv, "mu"sv,  "pi"sv,  "k"sv,  "p"sv};

static_assert(masses.size() == names.size());

// runs a function for every possible recombination of daughter particles
constexpr void for_all_recombinations(auto func)
{
  static_assert(std::invocable<decltype(func), std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t>, "for_all_recombinations must be passed a function which takes 4 std::uint32_t inputs.");

  for (std::uint32_t ap = 0; ap < names.size(); ++ap)
  for (std::uint32_t bp = 0; bp < names.size(); ++bp)
  for (std::uint32_t cp = 0; cp < names.size(); ++cp)
  for (std::uint32_t dp = 0; dp < names.size(); ++dp)
  {
    // skip if only 1 or 0 particles
    {
      int p_count = 0;

      p_count += ap != 0;
      p_count += bp != 0;
      p_count += cp != 0;
      p_count += dp != 0;
    
      if (p_count <= 1)
        continue;
    }

    // skip if combined particle would have charge of +-2
    if (ap == 0 && bp != 0 && cp == 0 && dp != 0)
      continue;

    if (ap != 0 && bp == 0 && cp != 0 && dp == 0)
      continue;

    func(ap, bp, cp, dp);
  }
}

int main()
{
  ROOT::EnableImplicitMT();

  const std::string filename   = "../data/Lb2pKmm_mgUp_2018.root";
  const std::string treename   = "Lb_Tuple/DecayTree";
  const std::string outputfile = "cache/mass.root";

  const auto root_file = std::make_unique<TFile>(filename.c_str());
  const auto root_tree = std::unique_ptr<TTree>(dynamic_cast<TTree*>(root_file->Get(treename.c_str())));

  const auto entry_count = root_tree->GetEntries();

  fmt::print("Using {} entries from tree {} in file {}\n", entry_count, root_tree->GetName(), root_file->GetName());

  root_tree->SetBranchStatus("*",0);

  const Particle<double> a( "h1" , root_tree.get() ); // p
  const Particle<double> b( "h2" , root_tree.get() ); // K
  const Particle<double> c( "mu1", root_tree.get() ); // mu
  const Particle<double> d( "mu2", root_tree.get() ); // mu

  const auto output_file = std::make_unique<TFile>(outputfile.c_str(), "RECREATE");
  const auto output_tree = std::make_unique<TTree>("Masses", "Masses");

  std::array<std::array<std::array<std::array<double, 5>, 5>, 5>, 5> vals;

  for_all_recombinations([&](const std::uint32_t ap, const std::uint32_t bp, const std::uint32_t cp, const std::uint32_t dp)
  {
    const auto name = fmt::format("{}_{}_{}_{}", names[ap], names[bp], names[cp], names[dp]);

    output_tree->Branch(name.c_str(), &vals[ap][bp][cp][dp], fmt::format("{}{}", name, "/D").c_str());
  });

  for ( std::int64_t i = 0; i < entry_count; ++i )
  {
    if (i % 5'000 == 0)
      fmt::print("Working on entry {}/{} ({}%)\n", i, entry_count, static_cast<double>(i) / static_cast<double>(entry_count) * 100.0);

    root_tree->GetEntry(i);

    const auto hypotheses = [&]
    {
      std::array<std::array<ROOT::Math::XYZTVector, 5>, 4> out;

      const std::array particles{a, b, c, d};

      for (std::uint32_t p = 0; p < 4; ++p)
        for (std::uint32_t m = 0; m < 5; ++m)
          if (m == 0)
            out[p][m] = ROOT::Math::XYZTVector{};
          else
            out[p][m] = particles[p].getHypothesis(masses[m]);

      return out;
    }();

    for_all_recombinations([&](const std::uint32_t ap, const std::uint32_t bp, const std::uint32_t cp, const std::uint32_t dp)
    {
      vals[ap][bp][cp][dp] = (hypotheses[0][ap] + hypotheses[1][bp] + hypotheses[2][cp] + hypotheses[3][dp]).M();
    });

    output_tree->Fill();
  }

  fmt::print("Created file {} with {} entries in tree {}\n", output_file->GetName(), output_tree->GetEntries(), output_tree->GetName());

  output_file->cd();
  output_tree->Write();

  return EXIT_SUCCESS;
}
