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

auto compare_variables()
{
  //ROOT::EnableImplicitMT();

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

  //auto canvas = std::make_unique<TCanvas>("canvas", "canvas", 3500, 2000);
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

    TH1D tempsimuhist{*simurdf.Histo1D({col.c_str(), col.c_str(), 100, min, max}, col)};
    TH1D temprealhist{*realrdf.Histo1D({col.c_str(), col.c_str(), 100, min, max}, col)};

    constexpr std::array<double, 2> boundaries{0.0025, 0.9975};

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

    TH1D simuhist{*simurdf.Histo1D({col.c_str(), col.c_str(), 100, min, max}, col)};
    TH1D realhist{*realrdf.Histo1D({col.c_str(), col.c_str(), 100, min, max}, col)};

    simuhist.SetLineColor(kBlue);
    realhist.SetLineColor(kRed);

    simuhist.DrawNormalized("");
    realhist.DrawNormalized("SAME");

    canvas->SaveAs(fmt::format("cache/compare_var_{}.png", col).c_str());
  }








  //mg_p->Add(simugraph);

  //mg_p->Draw("a");

  /*
  auto mg_p = new TMultiGraph;

  realtree->Print();

  realtree->Draw("nMuonTracks>>htemp");

  TH1F* hist = (TH1F*)gDirectory->Get("htemp");

  hist->Print();

  auto g1_p = new TGraph;

  g1_p->SetHistogram(hist);

  g1_p->Print();

  g1_p->Draw("a");

  //mg_p->Add(g1_p);

  //mg_p->Draw("a");


  //mg_p->Draw("a");
  */

/*
  std::cout << std::is_same_v<decltype(simutree), decltype(realtree)> << '\n';
  std::cout << std::is_pointer_v<decltype(simutree)>  << '\n';
  std::cout << typeid(simutree).name() << '\n';
  std::cout << typeid(realtree).name() << '\n';
  std::cout << typeid(TTree).name() << '\n';
  std::cout << typeid(TTree*).name() << '\n';
  std::cout << typeid(TObject*).name() << '\n';
  */
}

int main()
{
  compare_variables();
}

