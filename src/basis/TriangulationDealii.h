#ifndef dftefeTriangulationDealii_h
#define dftefeTriangulationDealii_h

#include "TriangulationBase.h"

namespace dftefe {

namespace basis
{

<template unsigned int dim>
class TriangulationDealii : public  TriangulationBase
{
    public :
    TriangulationDealii();
    ~TriangulationDealii();
    
     TriangulationBase & getTriangulationObject()  override;
     void     refineGlobal (const unsigned int times=1) override;
    
     void     coarsenGlobal (const unsigned int times=1) override;
    
     void     clearUserFlags () override;
    
     void     executeCoarseningAndRefinement () override;
     unsigned int     nLocallOwnedActiveCells () const  override;
    size_type     nGlobalActiveCells () const override ;
    size_type     locallyOwnedSubdomain () const override;
     std::vector< size_type >     getBoundaryIds () const override;

      TriaCellBase  &   beginActive (const unsigned int level=0) const override;
    
     TriaCellBase &     endActive (const unsigned int level) const override;
    
    
    unsigned int getDim() const override;
    
    
    private :
    
    dealii::parallel::distributed::Triangulation<dim> triangDealii;
    
    std::vector<TriaCellDealii<dim> > triaVectorCell;
    
    
    
}; // end of class TriangulationDealii

}// end of namespace basis

} // end of namespace dftefe
#endif
