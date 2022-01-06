#ifndef dftefeIntegrate_h
#define dftefeIntegrate_h

#include "CellQuadratureContainer.h"
#include <utils/ScalarSpatialFunction.h>

namespace dftefe
{
  namespace quadrature
  {
    void
    integrate(const utils::ScalarSpatialFunction<double> &function,
              const CellQuadratureContainer &cellQuadratureContainer,
              double &                       integral);

  } // end of namespace quadrature

} // end of namespace dftefe


#endif // dftefeIntegrate_h
