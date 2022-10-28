#include "TCanvas.h"
#include "TGraph.h"
#include "TMultiGraph.h"
#include "TH1D.h"
#include <ROOT/RDataFrame.hxx>

#include <fmt/format.h>
#include <fmt/compile.h>
#include <fmt/color.h>

#include<memory>
#include<array>
#include<algorithm>

auto compare_variables()
{
  ROOT::EnableImplicitMT();

  ROOT::RDataFrame simurdf{"tree", "../data/Lb2pKmm_sim_mgUp_2018.root"};
  ROOT::RDataFrame realrdf{"Lb_Tuple/DecayTree", "../data/Lb2pKmm_mgUp_2018.root"};

  auto simucols = simurdf.GetColumnNames();
  auto realcols = realrdf.GetColumnNames();

  // columns present in both simulated and real data
  std::vector<std::string> cols;

  for (auto& realcol : realcols)
  {
    bool found{false};

    for (auto& simucol : simucols)
    {
      if (simucol == realcol)
      {
        found = true;

        cols.emplace_back(realcol);

        break;
      }
    }

    if (!found)
      fmt::print("{} in real data but not simulated data\n\n", realcol);
  }

  fmt::print("columns in real data: {}, simulated data: {}, common to both: {}\n", realcols.size(), simucols.size(), cols.size());

  std::vector<std::pair<double, std::string>> scores{};

  auto canvas = std::make_unique<TCanvas>("canvas", "canvas", 1750, 1000);

  for (auto& col : cols)
  {
    /*if (realrdf.GetColumnType(col) != "float")
    {
      std::cout << "Column contains " << realrdf.GetColumnType(col) << " instead of float values; skipping\n";

      continue;
    }

    auto simudata = simurdf.Take<float>(col);*/

    fmt::print("working on variable: {}\n", col);

    auto simumin = simurdf.Min(col).GetValue();
    auto simumax = simurdf.Max(col).GetValue();
    auto realmin = realrdf.Min(col).GetValue();
    auto realmax = realrdf.Max(col).GetValue();

    if (simumin > realmax || simumax < realmin)
    {
      fmt::print("No data overlap between simulated and real data; skipping\n");

      continue;
    }

    auto min = std::min(simumin, realmin);
    auto max = std::max(simumax, realmax);

    constexpr int bin_count{400};

    TH1D tempsimuhist{*simurdf.Histo1D({col.c_str(), col.c_str(), bin_count, min, max}, col)};
    TH1D temprealhist{*realrdf.Histo1D({col.c_str(), col.c_str(), bin_count, min, max}, col)};

    if (tempsimuhist.Integral() == 0 || temprealhist.Integral() == 0)
    {
      fmt::print("Histogram has 0 integral. Integral of simulated data: {}. Integral of real data: {}\n", tempsimuhist.Integral(), temprealhist.Integral());

      continue;
    }

    constexpr std::array<double, 2> boundaries{0.05, 0.95};

    std::array<double, 2> simuquantiles{};
    std::array<double, 2> realquantiles{};

    tempsimuhist.GetQuantiles(2, simuquantiles.data(), boundaries.data());
    temprealhist.GetQuantiles(2, realquantiles.data(), boundaries.data());

    simumin = simuquantiles[0];
    simumax = simuquantiles[1];
    realmin = realquantiles[0];
    realmax = realquantiles[1];

    if (simumin > realmax || simumax < realmin)
    {
      fmt::print("No data overlap between simulated and real data reduced by quantiles; skipping\n");

      continue;
    }

    min = std::min(simumin, realmin);
    max = std::max(simumax, realmax);

    TH1D simuhist{*simurdf.Histo1D({col.c_str(), col.c_str(), bin_count, min, max}, col)};
    TH1D realhist{*realrdf.Histo1D({col.c_str(), col.c_str(), bin_count, min, max}, col)};

    simuhist.SetLineColor(kBlue);
    realhist.SetLineColor(kRed);

    if (simuhist.Integral() == 0 || realhist.Integral() == 0)
    {
      fmt::print("Histogram reduced by quantiles has 0 integral. Integral of simulated data (before reduction): {} ({}). Integral of real data (before reduction): {} ({})\n", simuhist.Integral(), tempsimuhist.Integral(), realhist.Integral(), temprealhist.Integral());

      continue;
    }

    simuhist.Scale(1.0 / simuhist.Integral());
    realhist.Scale(1.0 / realhist.Integral());
    tempsimuhist.Scale(1.0 / tempsimuhist.Integral());
    temprealhist.Scale(1.0 / temprealhist.Integral());

    const auto simuymax = simuhist.GetBinContent(simuhist.GetMaximumBin());
    const auto realymax = realhist.GetBinContent(realhist.GetMaximumBin());

    const auto ymax = 1.1 * std::max(simuymax, realymax);

    simuhist.GetYaxis()->SetRangeUser(0, ymax);
    realhist.GetYaxis()->SetRangeUser(0, ymax);

    simuhist.Draw("HIST");
    realhist.Draw("HIST SAME");

    canvas->SaveAs(fmt::format("cache/compare_var_{}.png", col).c_str());

    double simucumulative{0};
    double realcumulative{0};

    double score{0};

    for (int i = 1; i <= bin_count; ++i)
    {
      simucumulative += simuhist.GetBinContent(i);
      realcumulative += realhist.GetBinContent(i);

      score += std::max(simucumulative, realcumulative) - std::min(simucumulative, realcumulative);
    }

    if (std::max(simucumulative, realcumulative) - std::min(simucumulative, realcumulative) > 0.000000001)
      fmt::print(fmt::emphasis::bold | fg(fmt::color::red), "ERROR: cumulative real and simulated graphs do not end on same value for variable {}. real: {}, simulated: {}\n", col, realcumulative, simucumulative);

    scores.emplace_back(score, col);
  }

  fmt::print("\n{} graphs generated\n", scores.size());

  std::ranges::sort(scores);

  for (auto& p : scores)
    fmt::print("{} : {}\n", p.first, p.second);
}

int main()
{
  compare_variables();
}

