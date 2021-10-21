#ifndef dftefeTriangulationBase_h
#define dftefeTriangulationBase_h

#include <utils/TypeConfig.h>
#include "TriaCellBase.h"
namespace dftefe {

namespace basis
{
/**
 * @brief An abstract class for the triangulation class. The derived class specialises this class to dealii and otehr specialisations if required.
 **/
  class TriangulationBase
{
    public :
    TriangulationBase();
    ~TriangulationBase();
    
    
    virtual TriangulationBase & getTriangulationObject() = 0;
    virtual void     refineGlobal (const unsigned int times=1) = 0;
    
    virtual void     coarsenGlobal (const unsigned int times=1) = 0;
    
    virtual void     clearUserFlags () = 0;
    
    virtual void     executeCoarseningAndRefinement () = 0;
    virtual unsigned int     nLocallOwnedActiveCells () const = 0 ;
    virtual size_type     nGlobalActiveCells () const = 0 ;
    virtual size_type     locallyOwnedSubdomain () const = 0;
    virtual std::vector< size_type >     getBoundaryIds () const = 0;

    virtual  TriaCellBase  &   beginActive () const = 0;
    
    virtual TriaCellBase &     endActive () const = 0;
    
    
    virtual  unsigned int getDim() const = 0 ;
    
    
      
}; // end of class TriangulationBase
} // end of namespace basis
}// end of namespace dftefe
#endif
