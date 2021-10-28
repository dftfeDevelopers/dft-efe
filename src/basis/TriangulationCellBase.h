#ifndef dftefeTriangulationCellBase_h
#define dftefeTriangulationCellBase_h

#include <utils/Point.h>
#include <utils/TypeConfig.h>
#include "CellMappingBase.h"

#include <memory>
namespace dftefe
{
  namespace basis
  {
    /**
     * @brief An abstract class for an geometric cell.
     * This is done to prevent the template (as required by deal.ii objects)
     * to propagate all across the code,
     *
     **/
    class TriangulationCellBase
    {
    public:
        TriangulationCellBase();
      virtual ~TriangulationCellBase();

      virtual void
      getVertices( std::vector<utils::Point> & outputDftefePoints  ) const = 0;

      virtual void
      getVertex(size_type i, utils::Point &outputDftefePoint) const = 0;

      virtual size_type
      getId() const = 0;

      virtual bool
      isPointInside(const utils::Point &point) const = 0;

      virtual bool
      isAtBoundary(const unsigned int i) const = 0;

      virtual bool
      isAtBoundary() const = 0;

      virtual unsigned int
      getDim() const = 0;

        
        
      virtual void
      getParametricPoint(const Point &realPoint,
                         const CellMappingBase &      cellMapping,
                         Point  &parametricPoint ) const = 0;

      virtual void
      getRealPoint(const Point  &parametricPoint,
                   const CellMappingBase &      cellMapping,
                   Point  &realPoint) const = 0;

    }; // end of class TriaCellBase
  }    // end of namespace basis

} // end of namespace dftefe
#endif // dftefeTriaCellBase_h
