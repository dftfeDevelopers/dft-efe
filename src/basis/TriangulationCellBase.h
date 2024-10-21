#ifndef dftefeTriangulationCellBase_h
#define dftefeTriangulationCellBase_h

#include <utils/TypeConfig.h>
#include <utils/Point.h>
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
      virtual ~TriangulationCellBase() = default;
      virtual void
      getVertices(std::vector<utils::Point> &points) const = 0;

      virtual void
      getVertex(size_type i, utils::Point &point) const = 0;

      virtual size_type
      getId() const = 0;

      virtual bool
      isPointInside(const utils::Point &point) const = 0;

      virtual bool
      isAtBoundary(const unsigned int i) const = 0;

      virtual bool
      isAtBoundary() const = 0;

      virtual size_type
      getDim() const = 0;

      virtual double
      diameter() const = 0;

      virtual void
      center(dftefe::utils::Point &centerPoint) const = 0;

      virtual void
      setRefineFlag() = 0;

      virtual void
      clearRefineFlag() = 0;

      virtual double
      minimumVertexDistance() const = 0;

      virtual double
      distanceToUnitCell(dftefe::utils::Point &parametricPoint) const = 0;

      virtual void
      getParametricPoint(const utils::Point &   realPoint,
                         const CellMappingBase &cellMapping,
                         utils::Point &         parametricPoint) const = 0;

      virtual void
      getRealPoint(const utils::Point &   parametricPoint,
                   const CellMappingBase &cellMapping,
                   utils::Point &         realPoint) const = 0;

    }; // end of class TriaCellBase
  }    // end of namespace basis

} // end of namespace dftefe
#endif // dftefeTriaCellBase_h
