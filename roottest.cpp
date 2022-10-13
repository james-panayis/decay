void roottest()
{
  auto mg_p    = new TMultiGraph;

  auto gr_p_K  = new TGraph("cache/out.csv", "%lg,%lg");
  auto gr_p_pi = new TGraph("cache/out.csv", "%*lg,%*lg,%lg,%lg");

  gr_p_K ->SetLineColor(kRed);
  gr_p_pi->SetLineColor(kBlue);

  mg_p->Add(gr_p_K);
  mg_p->Add(gr_p_pi);

  mg_p->Draw("a");
  return;



  auto mg_pt    = new TMultiGraph;

  auto gr_pt_K  = new TGraph("cache/out.csv", "%*lg,%*lg,%*lg,%*lg,%lg,%lg");
  auto gr_pt_pi = new TGraph("cache/out.csv", "%*lg,%*lg,%*lg,%*lg,%*lg,%*lg,%lg,%lg");

  gr_pt_K ->SetLineColor(kRed);
  gr_pt_pi->SetLineColor(kBlue);

  mg_pt->Add(gr_pt_K);
  mg_pt->Add(gr_pt_pi);

  mg_pt->Draw("a");



  auto mg_ip    = new TMultiGraph;

  auto gr_ip_K  = new TGraph("cache/out.csv", "%*lg,%*lg,%*lg,%*lg,%*lg,%*lg,%*lg,%*lg,%lg,%lg");
  auto gr_ip_pi = new TGraph("cache/out.csv", "%*lg,%*lg,%*lg,%*lg,%*lg,%*lg,%*lg,%*lg,%*lg,%*lg,%lg,%lg");

  gr_ip_K ->SetLineColor(kRed);
  gr_ip_pi->SetLineColor(kBlue);

  mg_ip->Add(gr_ip_K);
  mg_ip->Add(gr_ip_pi);

  mg_ip->Draw("a");
}
