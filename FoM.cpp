#include "root.hpp"
#include "threading.hpp"

#include <fmt/format.h>
#include <fmt/compile.h>
#include <fmt/color.h>

#include <algorithm>
#include <fstream>

// Picks a cut value that optimizes a figure of merit (FoM) 

/*
Let S be the number of signal events correctly identified.
Let N be the total number of events identified.
The decays (approximately) model a Poisson distribution since they are independent and consistent.
Thus the number of decays that occur in a fixed period is an unbiased estimator of the variance of that statistic.
Significance is S/(standard deviation) = S/sqrt(N).
We know N (by applying the MVA to the actual data).
The `efficiency' is the fraction of actual decays that are correctly identified.
S is equal to the number of Lambda_b particles produced x the fraction of these particles that decay into pKmumu x the efficiency.
Thus S is proportional to the efficiency.
We estimate the efficiency using the efficiency on the simulated decays.
*/

// read data from file and return a sorted vector of the predictions within the lb_mass window
auto get_data(const movency::root::file file)
{ 
  const auto prediction_values{file.uncompress<double>("nn_output")};
  const auto lb_masses        {file.uncompress<double>("Lb_M")};

  std::vector<double> out;

  for (std::size_t j = 0; j < prediction_values.size(); ++j)
    if (lb_masses[j] > 5569.6 && lb_masses[j] < 5669.6)
      out.emplace_back(prediction_values[j]);

  std::ranges::sort(out, std::ranges::greater{});

  return out;
}

int main()
{
  // sorted real and simulated predictions
  std::vector<double> real;
  std::vector<double> simu;

  loop_threaded([&](const std::size_t n) {
      if (n == 0)
        real = get_data({"./cache/Real_D4J.root"});
      else
        simu = get_data({"./cache/Sim_D4J.root"});
    }, 2);

  const auto original_simu_size = simu.size();

  double max_FoM {0.0};
  double best_cut{0.0};

  constexpr auto log_name{"./cache/FoM_results.txt"};

  std::ofstream log{log_name};

  constexpr auto cut_count{100000}; // how many cuts (between 0 and 1)
  constexpr auto digits{static_cast<int>(std::log10(cut_count) + 1)}; // how many digits needed to represent the cuts

  log << fmt::format("CUT,{: >{}}FoM\n", "", digits);

  for (int i = 0; i < cut_count; ++i)
  {
    const double cut = static_cast<double>(i) / cut_count;

    // get rid of elements below cut
    real.resize(static_cast<std::size_t>(std::ranges::lower_bound(real, cut, std::ranges::greater{}) - real.begin()));
    simu.resize(static_cast<std::size_t>(std::ranges::lower_bound(simu, cut, std::ranges::greater{}) - simu.begin()));

    const auto efficiency = static_cast<double>(simu.size()) / static_cast<double>(original_simu_size);
    const auto N          = static_cast<double>(real.size());

    if (N == 0)
      break;

    const auto FoM = efficiency / std::sqrt(N);

    log << fmt::format("{:0<.{}f}, {:0<.6f}\n", cut, digits, FoM);

    if (FoM > max_FoM)
    {
      max_FoM  = FoM;
      best_cut = cut;
    }
  }

  log << "\nBest:\n\n";

  const auto best_string = fmt::format("CUT: {}, FoM: {}\n", best_cut, max_FoM);

  if (max_FoM == 0.0)
  {
    fmt::print(fmt::emphasis::bold | fg(fmt::color::red), "ERROR: no best cut or maximum figure of merit\n");
    log << "n/a\n";
  }
  else
    log << best_string;

  fmt::print("Figure of merit calculations complete.\nBest:\n      {}Full output in file: {}\n", best_string, log_name);

  return EXIT_SUCCESS;
}

