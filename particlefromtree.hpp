#pragma once

#include "TFile.h" 
#include "TTree.h" 
#include "Math/Vector4D.h" 

#include <string>
#include <memory>

class Particle
{
public:
  Particle(const std::string name, std::shared_ptr<TTree> tree)
    : name_(name), tree_(tree)
  {
    tree_->SetBranchStatus ((name_ + "_PX").c_str(), true);
    tree_->SetBranchStatus ((name_ + "_PY").c_str(), true);
    tree_->SetBranchStatus ((name_ + "_PZ").c_str(), true);
    
    tree_->SetBranchAddress((name_ + "_PX").c_str(), &PX_);
    tree_->SetBranchAddress((name_ + "_PY").c_str(), &PY_);
    tree_->SetBranchAddress((name_ + "_PZ").c_str(), &PZ_);
  }

  Particle(const Particle&) = delete; // non-copyable and non-moveable

  ~Particle()
  {
    tree_->SetBranchStatus ((name_ + "_PX").c_str(), false);
    tree_->SetBranchStatus ((name_ + "_PY").c_str(), false);
    tree_->SetBranchStatus ((name_ + "_PZ").c_str(), false);
  }

  ROOT::Math::XYZTVector getHypothesis(const double M) const
  { 
    return {PX_, PY_, PZ_, std::sqrt(PX_*PX_ + PY_*PY_ + PZ_*PZ_ + M*M)};
  }

private:
  
  const std::string name_;

  const std::shared_ptr<TTree> tree_;
  
  double PX_;
  double PY_;
  double PZ_;
};
