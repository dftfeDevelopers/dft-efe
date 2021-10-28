#ifndef dftefeTriaCellBase_h
#define dftefeTriaCellBase_h

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
    class TriaCellBase
    {
    public:
      TriaCellBase();
      virtual ~TriaCellBase();

      virtual std::vector<std::shared_ptr<Point>>
      getVertices() const = 0;

      virtual std::shared_ptr<Point>
      getVertex(size_type i) const = 0;

      virtual size_type
      getId() const = 0;

      virtual bool
      isPointInside(std::shared_ptr<const Point> point) const = 0;

      virtual bool
      isAtBoundary(const unsigned int i) const = 0;

      virtual bool
      isAtBoundary() const = 0;

      virtual void
      setRefineFlag() = 0;

      virtual void
      clearRefineFlag() = 0;

      virtual void
      setCoarsenFlag() = 0;

      virtual void
      clearCoarsenFlag() = 0;

      virtual bool
      isActive() const = 0;

      virtual bool
      isLocallyOwned() const = 0;

      virtual bool
      isGhost() const = 0;

      virtual bool
      isArtificial() const = 0;

      virtual int
      getDim() const = 0;

      virtual std::shared_ptr<Point>
      getParametricPoint(std::shared_ptr<const Point> realPoint,
                         const CellMappingBase &      cellMapping) const = 0;

      virtual std::shared_ptr<Point>
      getRealPoint(std::shared_ptr<const Point> parametricPoint,
                   const CellMappingBase &      cellMapping) const = 0;

    }; // end of class TriaCellBase
  }    // end of namespace basis

} // end of namespace dftefe
#endif // dftefeTriaCellBase_h
