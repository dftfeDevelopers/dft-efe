#ifndef dftefeTriaCellDealII_h
#define dftefeTriaCellDealII_h

#include <utils/Point.h>
#include <utils/TypeConfig.h>
#include "CellMappingBase.h"
#include "TriaCellBase.h"

#include <memory>
namespace dftefe
{
  namespace basis
  {
    /**
     * @brief An interface to deal.ii geometric cell 
     **/
    template <unsigned int DIM>
    class TriaCellDealII : public TriaBase
    {
    public:
      TriaCellDealII();
      ~TriaCellDealII();

      std::vector<std::shared_ptr<Point>>
      getVertices() const override;

      std::shared_ptr<Point>
      getVertex(size_type i) const override;

      size_type
      getId() const override;

      bool
      isPointInside(std::shared_ptr<const Point> point) const override;

      bool
      isAtBoundary(const unsigned int i) const override;

      bool
      isAtBoundary() const override;

      void
      setRefineFlag() override;

      void
      clearRefineFlag() override;

      void
      setCoarsenFlag() override;

      void
      clearCoarsenFlag() override;

      bool
      isActive() const override;

      bool
      isLocallyOwned() const override;

      bool
      isGhost() const override;

      bool
      isArtificial() const override;

      int
      getDim() const override;

      std::shared_ptr<Point>
      getParametricPoint(std::shared_ptr<const Point> realPoint,
                         const CellMappingBase &        cellMapping) const override;

      std::shared_ptr<Point>
      getRealPoint(std::shared_ptr<const Point> parametricPoint,
                   const CellMappingBase &        cellMapping) const override;

    }; // end of class TriaCellDealII
  }    // end of namespace basis

} // end of namespace dftefe
#endif // dftefeTriaCellDealII_h
