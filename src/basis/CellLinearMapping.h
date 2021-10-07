#ifndef dftefeCellLinearMapping_h
#define dftefeCellLinearMapping_h

#include <utils/Point.h>

namespace dftefe {

  class CellLinearMapping {

    CellLinearMapping();
    ~CellLinearMapping();



    virtual
      void * 
      getData() const = 0;




  }; // end of class CellLinearMapping

} // end of dftefe namespace

#endif
