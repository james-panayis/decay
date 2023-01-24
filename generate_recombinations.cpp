#include "particlefromtree.hpp" 

#include "TFile.h" 
#include "TTree.h"
#include "ROOT/RDataFrame.hxx"

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

constexpr std::array names {"0"sv, "mu"sv, "pi"sv, "k"sv, "p"sv};

static_assert(masses.size() == names.size());

constexpr std::size_t preds_count{names.size()};

// runs a function for every possible recombination of daughter particles
constexpr void for_all_recombinations(auto func)
{
  static_assert(std::invocable<decltype(func), std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t>, "for_all_recombinations must be passed a function which takes 4 std::uint32_t inputs.");

  for (std::uint32_t ap = 0; ap < preds_count; ++ap)
  for (std::uint32_t bp = 0; bp < preds_count; ++bp)
  for (std::uint32_t cp = 0; cp < preds_count; ++cp)
  for (std::uint32_t dp = 0; dp < preds_count; ++dp)
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

  const auto output_file = std::make_unique<TFile>("cache/mass.root", "RECREATE");
  const auto output_tree = std::make_unique<TTree>("tree", "tree");

  std::array<std::array<std::array<std::array<double, preds_count>, preds_count>, preds_count>, preds_count> vals;

  for_all_recombinations([&](const std::uint32_t ap, const std::uint32_t bp, const std::uint32_t cp, const std::uint32_t dp)
  {
    const auto name = fmt::format("{}_{}_{}_{}", names[ap], names[bp], names[cp], names[dp]);

    output_tree->Branch(name.c_str(), &vals[ap][bp][cp][dp], (name + "/D").c_str());
  });

  bool propagate_uid{true};

  Long64_t uid;

  constexpr std::array infilenames = std::to_array({"../data/Lb2pKmm_mgUp_2016_UID.root",
                                                    "../data/Lb2pKmm_mgDn_2016_UID.root",
                                                    "../data/Lb2pKmm_mgUp_2017_UID.root",
                                                    "../data/Lb2pKmm_mgDn_2017_UID.root",
                                                    "../data/Lb2pKmm_mgUp_2018_UID.root",
                                                    "../data/Lb2pKmm_mgDn_2018_UID.root"});

  for (std::size_t f = 0; f < infilenames.size(); ++f)
  {
    const auto input_file = std::make_unique<TFile>(infilenames[f]);
    const auto input_tree = std::shared_ptr<TTree>(dynamic_cast<TTree*>(input_file->Get("tree")));

    if (input_tree.get() == nullptr)
    {
      fmt::print(fmt::emphasis::bold | fg(fmt::color::red), "ERROR: file {} contains no tree called \"tree\"\n", input_file->GetName());

      return EXIT_FAILURE;
    }

    const auto entry_count = input_tree->GetEntries();

    fmt::print("Using {} entries from tree \"{}\" in file \"{}\"\n", entry_count, input_tree->GetName(), input_file->GetName());

    input_tree->SetBranchStatus("*", false); // default all branches to not be read

    const std::array<Particle, 4> particles{{
      {"h1" , input_tree}, // p
      {"h2" , input_tree}, // K
      {"mu1", input_tree}, // mu
      {"mu2", input_tree}  // mu
    }};

    // Propagate UID variable to output file if present in input files
    if (propagate_uid)
    {
      auto found = std::make_unique<std::uint32_t>();

      input_tree->SetBranchStatus("UID", true, found.get());

      if (*found)
      {
        input_tree->SetBranchAddress("UID", &uid);

        if (f == 0)
          output_tree->Branch("UID", &uid, "UID/L");
      }
      else
      {
        if (f == 0)
        {
          propagate_uid = false;

          fmt::print(fmt::emphasis::bold | fg(fmt::color::yellow), "WARNING: No UIDs found in first input file {}, so none will be present in output file.\n", input_file->GetName());
        }
        else
        {
          fmt::print(fmt::emphasis::bold | fg(fmt::color::red), "ERROR: No UIDs found in input file {}, though previous input files did contain UIDs\n", input_file->GetName());

          return EXIT_FAILURE;
        }
      }
    }

    for (std::int64_t i = 0; i < entry_count; ++i)
    {
      if (i % 10'000 == 0 && i)
        fmt::print("Working on entry {}/{} ({}% completed)\n", i, entry_count, static_cast<double>(i) / static_cast<double>(entry_count) * 100.0);

      input_tree->GetEntry(i);

      const auto hypotheses = [&]
      {
        std::array<std::array<ROOT::Math::XYZTVector, preds_count>, 4> out;

        for (std::uint32_t p = 0; p < 4; ++p)
          for (std::uint32_t m = 0; m < preds_count; ++m)
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

    fmt::print("Finished with input file {}\n\n", input_file->GetName());
  }

  fmt::print(fg(fmt::color::green), "Created file \"{}\" with {} entries in tree \"{}\"\n", output_file->GetName(), output_tree->GetEntries(), output_tree->GetName());

  output_file->cd();
  output_tree->Write();

  return EXIT_SUCCESS;
}
