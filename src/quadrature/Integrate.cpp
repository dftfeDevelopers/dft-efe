#include "Integrate.h"
#include <utils/TypeConfig.h>

namespace dftefe
{
  namespace quadrature
  {
    void
    integrate(const utils::ScalarSpatialFunction<double> &function,
              const CellQuadratureContainer &cellQuadratureContainer,
              double &                       integral)
    {
      const std::vector<dftefe::utils::Point> &realPoints =
        cellQuadratureContainer.getRealPoints();
      const size_type numQuadraturePoints =
        cellQuadratureContainer.nQuadraturePoints();
      const std::vector<double> &JxW = cellQuadratureContainer.getJxW();
      const std::vector<double> &functionValues = function(realPoints);
      integral                                  = 0.0;
      for (unsigned int i = 0; i < numQuadraturePoints; ++i)
        integral += functionValues[i] * JxW[i];
    }
  } // end of namespace quadrature
} // end of namespace dftefe
