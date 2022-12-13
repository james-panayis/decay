#include "root.hpp"

#include <fmt/format.h>
#include <fmt/compile.h>
#include <fmt/color.h>

#include <algorithm>
#include <fstream>


int main()
{
  std::vector<double> real;
  std::vector<double> simu;

  {
    movency::root::file file_real{"../data/E_D4J.root"};
    movency::root::file file_simu{"../data/SIM_E_D4J.root"};

    const auto prediction_real{file_real.uncompress<double>("xgb_output")};
    const auto prediction_simu{file_simu.uncompress<double>("xgb_output")};

    const auto lbmass_real{file_real.uncompress<double>("Lb_M")};
    const auto lbmass_simu{file_simu.uncompress<double>("Lb_M")};

    for (std::size_t j = 0; j < prediction_real.size(); ++j)
      if (lbmass_real[j] > 5569.6 && lbmass_real[j] < 5669.6)
        real.emplace_back(prediction_real[j]);

    for (std::size_t j = 0; j < prediction_simu.size(); ++j)
      if (lbmass_simu[j] > 5569.6 && lbmass_simu[j] < 5669.6)
        simu.emplace_back(prediction_simu[j]);
  }

  const auto original_simu_size = simu.size();

  double max_FoM {0.0};
  double best_cut{0.0};

  const auto log_name{"cache/FoM_results.txt"};

  std::ofstream log{log_name};

  constexpr auto cut_count{5000};
  constexpr auto digits{static_cast<int>(std::log10(cut_count) + 1)};

  log << fmt::format("CUT,{: >{}}FoM\n", "", digits);

  for (int i = 0; i < cut_count; ++i)
  {
    const double cut = static_cast<double>(i) / cut_count;

    // get rid of elements below cut
    std::erase_if(real, [=](double v){ return v < cut; });
    std::erase_if(simu, [=](double v){ return v < cut; });

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
    fmt::print(fmt::emphasis::bold | fg(fmt::color::red), "ERROR: no best cut or maximum figure of merit");
    log << "n/a\n";
  }
  else
    log << best_string;

  fmt::print("Figure of merit calculations complete.\nBest:\n{}Full output in file {}\n", best_string, log_name);

  return EXIT_SUCCESS;
}

