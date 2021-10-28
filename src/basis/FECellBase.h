#ifndef dftefeFECellBase_h
#define dftefeFECellBase_h

#include <utils/Point.h>
#include <utils/TypeConfig.h>
#include "TriaCellBase.h"
#include "CellMappingBase.h"

#include <memory>

namespace dftefe
{
  namespace basis
  {
    /**
     * @brief An abstract class for a finite element cell (can be of any dimension)
     * This is created primarily to be a wrapper around deal.ii cells, so as to
     * avoid the cascading of template parameters.
     **/

    class FECellBase : public TriaCellBase
    {
    public:
      FECellBase();
      ~FECellBase();

      virtual std::vector<std::shared_ptr<Point>>
      getVertices() const override = 0;

      virtual std::shared_ptr<Point>
      getVertex(size_type i) const override = 0;

      virtual std::vector<std::shared_ptr<Point>>
      getNodalPoints() const override = 0;

      virtual size_type
      getId() const override = 0;

      virtual bool
      isPointInside(std::shared_ptr<const Point> point) const override = 0;

      virtual bool
      isAtBoundary(const unsigned int i) const override = 0;

      virtual bool
      isAtBoundary() const override = 0;

      virtual void
      setRefineFlag() override = 0;

      virtual void
      clearRefineFlag() override = 0;

      virtual void
      setCoarsenFlag() override = 0;

      virtual void
      clearCoarsenFlag() override = 0;

      virtual bool
      isActive() const override = 0;

      virtual bool
      isLocallyOwned() const override = 0;

      virtual bool
      isGhost() const override = 0;

      virtual bool
      isArtificial() const override = 0;

      virtual int
      getDim() const override = 0;

      virtual std::shared_ptr<Point>
      getParametricPoint(std::shared_ptr<const Point> realPoint,
                         const CellMappingBase &cellMapping) const override = 0;

      virtual std::shared_ptr<Point>
      getRealPoint(std::shared_ptr<const Point> parametricPoint,
                   const CellMappingBase &      cellMapping) const override = 0;

      virtual global_size_type
      getLocalToGlobalDoFId(size_type i) const override = 0;

      virtual size_type
      getFEOrder() const override = 0;


    }; // end of class FECellBase
  }    // end of namespace basis

} // end of namespace dftefe
#endif // dftefeFECellBase_h
