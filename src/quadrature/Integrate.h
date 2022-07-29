#ifndef dftefeIntegrate_h
#define dftefeIntegrate_h

#include "QuadratureRuleContainer.h"
#include <utils/ScalarSpatialFunction.h>

namespace dftefe
{
  namespace quadrature
  {
    void
    integrate(const utils::ScalarSpatialFunction<double> &function,
              const QuadratureRuleContainer &quadratureRuleContainer,
              double &                       integral);

  } // end of namespace quadrature

} // end of namespace dftefe


#endif // dftefeIntegrate_h
