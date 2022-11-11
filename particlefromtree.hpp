#pragma once

#include "Math/Vector4D.h" 
#include "TVector3.h" 

#include "TFile.h" 
#include "TTree.h" 

#include <string>

template<class T>
class Particle
{
public:
  Particle(std::string name);
  
  Particle(std::string name, TTree* tree);

  void setTree(TTree* tree);

  ROOT::Math::XYZTVector getVec() const;

  ROOT::Math::XYZTVector getHypothesis(const double M) const;

  std::string name_;
  
  
private:
  
  T M_;
  T PX_;
  T PY_;
  T PZ_;
  T PE_;
};

template<class T>
class Vertex
{
public:
  Vertex(std::string name);
  
  Vertex(std::string name, TTree* tree); 
  
  void setTree(TTree* tree);

  TVector3 getPos() const;

  std::string name_; 

private:

  T X_;
  T Y_;
  T Z_;
};



template<class T>
Particle<T>::Particle(std::string name) : 
  name_(name) {} 

template<class T>
Particle<T>::Particle(std::string name, TTree* tree) : 
  name_(name)
{
  setTree(tree);
} 

template<class T>
void Particle<T>::setTree(TTree* tree)
{
  
  tree->SetBranchStatus( (name_ + "_PX").c_str(), 1 );
  tree->SetBranchStatus( (name_ + "_PY").c_str(), 1 );
  tree->SetBranchStatus( (name_ + "_PZ").c_str(), 1 );
  tree->SetBranchStatus( (name_ + "_PE").c_str(), 1 );
  
  tree->SetBranchAddress( (name_ + "_PX").c_str(), &PX_ );
  tree->SetBranchAddress( (name_ + "_PY").c_str(), &PY_ );
  tree->SetBranchAddress( (name_ + "_PZ").c_str(), &PZ_ );
  tree->SetBranchAddress( (name_ + "_PE").c_str(), &PE_ );
  
  return;
}

template<class T>
ROOT::Math::XYZTVector Particle<T>::getVec() const
{
  return ROOT::Math::XYZTVector( PX_, PY_, PZ_, PE_ );
}


template<class T>
ROOT::Math::XYZTVector 
Particle<T>::getHypothesis( const double M ) const
{ 
  ROOT::Math::XYZTVector result;

  result.SetXYZT( PX_, PY_, PZ_, std::sqrt(PX_*PX_ + PY_*PY_ + PZ_*PZ_ + M*M) );
  
  return result;
}

template<class T>
Vertex<T>::Vertex( std::string name ) : 
  name_(name) {}

template<class T>
Vertex<T>::Vertex(std::string name, TTree* tree) : 
  name_(name)
{
  setTree(tree);
}

template<class T>
void Vertex<T>::setTree(TTree* tree)
{
  
  tree->SetBranchStatus( (name_ + "_X").c_str(), 1 );
  tree->SetBranchStatus( (name_ + "_Y").c_str(), 1 );
  tree->SetBranchStatus( (name_ + "_Z").c_str(), 1 );
  
  tree->SetBranchAddress( (name_ + "_X").c_str(), &X_ );
  tree->SetBranchAddress( (name_ + "_Y").c_str(), &Y_ );
  tree->SetBranchAddress( (name_ + "_Z").c_str(), &Z_ );
  
  return;
}

template<class T>
TVector3 Vertex<T>::getPos() const
{
  return TVector3( X_, Y_, Z_ );
}
