import ROOT
from array import array


def setLHCbPlotStyle():

    ROOT.gROOT.SetBatch(ROOT.kTRUE)
    ROOT.gROOT.SetStyle("Plain")
    lhcbStyle = ROOT.TStyle("lhcbStyle", "LHCb plot style")

    # define names for colours
    black = 1
    red = 2
    green = 3
    blue = 4
    yellow = 5
    magenta = 6
    cyan = 7
    purple = 9

    lhcbFont = 132  # Old LHCb style: 62;
    lhcbWidth = 2  # Old LHCb style: 3.00;
    lhcbTSize = 0.06

    lhcbStyle.SetErrorX(0)
    #  don't suppress the error bar along X

    lhcbStyle.SetFillColor(1)
    lhcbStyle.SetFillStyle(1001)
    # solid
    lhcbStyle.SetFrameFillColor(0)
    lhcbStyle.SetFrameBorderMode(0)
    lhcbStyle.SetPadBorderMode(0)
    lhcbStyle.SetPadColor(0)
    lhcbStyle.SetCanvasBorderMode(0)
    lhcbStyle.SetCanvasColor(0)
    lhcbStyle.SetStatColor(0)
    lhcbStyle.SetLegendBorderSize(0)
    lhcbStyle.SetLegendFont(132)

    # If you want the usual gradient palette (blue . red)
    lhcbStyle.SetPalette(1)
    # If you want colors that correspond to gray scale in black and white:
    colors = array('i', [0, 5, 7, 3, 6, 2, 4, 1])
    lhcbStyle.SetPalette(8, colors)

    # set the paper & margin sizes
    lhcbStyle.SetPaperSize(20, 26)
    lhcbStyle.SetPadTopMargin(0.05)
    lhcbStyle.SetPadRightMargin(0.05)
    # increase for colz plots
    lhcbStyle.SetPadBottomMargin(0.16)
    lhcbStyle.SetPadLeftMargin(0.14)

    # use large fonts
    lhcbStyle.SetTextFont(lhcbFont)
    lhcbStyle.SetTextSize(lhcbTSize)
    lhcbStyle.SetLabelFont(lhcbFont, "x")
    lhcbStyle.SetLabelFont(lhcbFont, "y")
    lhcbStyle.SetLabelFont(lhcbFont, "z")
    lhcbStyle.SetLabelSize(lhcbTSize, "x")
    lhcbStyle.SetLabelSize(lhcbTSize, "y")
    lhcbStyle.SetLabelSize(lhcbTSize, "z")
    lhcbStyle.SetTitleFont(lhcbFont)
    lhcbStyle.SetTitleFont(lhcbFont, "x")
    lhcbStyle.SetTitleFont(lhcbFont, "y")
    lhcbStyle.SetTitleFont(lhcbFont, "z")
    lhcbStyle.SetTitleSize(1.2 * lhcbTSize, "x")
    lhcbStyle.SetTitleSize(1.2 * lhcbTSize, "y")
    lhcbStyle.SetTitleSize(1.2 * lhcbTSize, "z")

    # use medium bold lines and thick markers
    lhcbStyle.SetLineWidth(lhcbWidth)
    lhcbStyle.SetFrameLineWidth(lhcbWidth)
    lhcbStyle.SetHistLineWidth(lhcbWidth)
    lhcbStyle.SetFuncWidth(lhcbWidth)
    lhcbStyle.SetGridWidth(lhcbWidth)
    lhcbStyle.SetLineStyleString(2, "[12 12]")
    # postscript dashes
    lhcbStyle.SetMarkerStyle(20)
    lhcbStyle.SetMarkerSize(1.0)

    # label offsets
    lhcbStyle.SetLabelOffset(0.010, "X")
    lhcbStyle.SetLabelOffset(0.010, "Y")

    # by default, do not display histogram decorations:
    lhcbStyle.SetOptStat(0)
    #lhcbStyle.SetOptStat("emr");  # show only nent -e , mean - m , rms -r
    # full opts at http:#root.cern.ch/root/html/TStyle.html#TStyle:SetOptStat
    lhcbStyle.SetStatFormat("6.3g")
    # specified as c printf options
    lhcbStyle.SetOptTitle(0)
    lhcbStyle.SetOptFit(0)
    #lhcbStyle.SetOptFit(1011); # order is probability, Chi2, errors, parameters
    #titles
    lhcbStyle.SetTitleOffset(0.95, "X")
    lhcbStyle.SetTitleOffset(0.95, "Y")
    lhcbStyle.SetTitleOffset(1.2, "Z")
    lhcbStyle.SetTitleFillColor(0)
    lhcbStyle.SetTitleStyle(0)
    lhcbStyle.SetTitleBorderSize(0)
    lhcbStyle.SetTitleFont(lhcbFont, "title")
    lhcbStyle.SetTitleX(0.0)
    lhcbStyle.SetTitleY(1.0)
    lhcbStyle.SetTitleW(1.0)
    lhcbStyle.SetTitleH(0.05)

    # look of the statistics box:
    lhcbStyle.SetStatBorderSize(0)
    lhcbStyle.SetStatFont(lhcbFont)
    lhcbStyle.SetStatFontSize(0.05)
    lhcbStyle.SetStatX(0.9)
    lhcbStyle.SetStatY(0.9)
    lhcbStyle.SetStatW(0.25)
    lhcbStyle.SetStatH(0.15)

    # put tick marks on top and RHS of plots
    lhcbStyle.SetPadTickX(1)
    lhcbStyle.SetPadTickY(1)

    # histogram divisions: only 5 in x to avoid label overlaps
    lhcbStyle.SetNdivisions(505, "x")
    lhcbStyle.SetNdivisions(510, "y")

    ROOT.gROOT.SetStyle("lhcbStyle")
    ROOT.gROOT.ForceStyle()

    # add LHCb label
    lhcbName = ROOT.TPaveText(ROOT.gStyle.GetPadLeftMargin() + 0.05,
                              0.87 - ROOT.gStyle.GetPadTopMargin(),
                              ROOT.gStyle.GetPadLeftMargin() + 0.20,
                              0.95 - ROOT.gStyle.GetPadTopMargin(), "BRNDC")
    lhcbName.AddText("LHCb")
    lhcbName.SetFillColor(0)
    lhcbName.SetTextAlign(12)
    lhcbName.SetBorderSize(0)

    lhcbLabel = ROOT.TText()
    lhcbLabel.SetTextFont(lhcbFont)
    lhcbLabel.SetTextColor(1)
    lhcbLabel.SetTextSize(lhcbTSize)
    lhcbLabel.SetTextAlign(12)

    lhcbLatex = ROOT.TLatex()
    lhcbLatex.SetTextFont(lhcbFont)
    lhcbLatex.SetTextColor(1)
    lhcbLatex.SetTextSize(lhcbTSize)
    lhcbLatex.SetTextAlign(12)

    ROOT.gROOT.SetStyle("lhcbStyle")
    ROOT.gROOT.ForceStyle()

    print("-------------------------")
    print("Set LHCb Style - Feb 2012")
    print("-------------------------")

    return lhcbName
