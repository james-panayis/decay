#include "TCanvas.h"
#include "TGraph.h"
#include "TMultiGraph.h"

#include<memory>

void roottest()
{
  auto canvas = std::make_unique<TCanvas>("canvas", "canvas", 3500, 2000);

  {
    TMultiGraph mg_p{};

    auto gr_p_K  = std::make_unique<TGraph>("cache/out.csv", "%lg,%lg");
    auto gr_p_pi = std::make_unique<TGraph>("cache/out.csv", "%*lg,%*lg,%lg,%lg");

    gr_p_K ->SetLineColor(kRed);
    gr_p_pi->SetLineColor(kBlue);

    mg_p.Add(gr_p_K.release());
    mg_p.Add(gr_p_pi.release());

    mg_p.Draw("a");

    canvas->SaveAs("cache/toy_simulation_p.png");
  }

  {
    TMultiGraph mg_pt{};

    auto gr_pt_K  = std::make_unique<TGraph>("cache/out.csv", "%*lg,%*lg,%*lg,%*lg,%lg,%lg");
    auto gr_pt_pi = std::make_unique<TGraph>("cache/out.csv", "%*lg,%*lg,%*lg,%*lg,%*lg,%*lg,%lg,%lg");

    gr_pt_K ->SetLineColor(kRed);
    gr_pt_pi->SetLineColor(kBlue);

    mg_pt.Add(gr_pt_K.release());
    mg_pt.Add(gr_pt_pi.release());

    mg_pt.Draw("a");

    canvas->SaveAs("cache/toy_simulation_pt.png");
  }

  {
    TMultiGraph mg_ip{};

    auto gr_ip_K  = std::make_unique<TGraph>("cache/out.csv", "%*lg,%*lg,%*lg,%*lg,%*lg,%*lg,%*lg,%*lg,%lg,%lg");
    auto gr_ip_pi = std::make_unique<TGraph>("cache/out.csv", "%*lg,%*lg,%*lg,%*lg,%*lg,%*lg,%*lg,%*lg,%*lg,%*lg,%lg,%lg");

    gr_ip_K ->SetLineColor(kRed);
    gr_ip_pi->SetLineColor(kBlue);

    mg_ip.Add(gr_ip_K.release());
    mg_ip.Add(gr_ip_pi.release());

    mg_ip.Draw("a");

    canvas->SaveAs("cache/toy_simulation_ip.png");
  }
}

int main()
{
  roottest();
}
