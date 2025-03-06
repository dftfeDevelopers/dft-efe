#ifndef dftefeFECellBase_h
#define dftefeFECellBase_h

#include <utils/Point.h>
#include <utils/TypeConfig.h>
#include <basis/TriangulationCellBase.h>
#include <basis/CellMappingBase.h>

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

    class FECellBase : public TriangulationCellBase
    {
    public:
      virtual ~FECellBase() = default;
      virtual void
      getVertices(std::vector<utils::Point> &points) const = 0;

      virtual void
      getVertex(size_type i, utils::Point &point) const = 0;

      virtual std::vector<std::shared_ptr<utils::Point>>
      getNodalPoints() const = 0;

      virtual size_type
      getId() const = 0;

      virtual bool
      isPointInside(const utils::Point &point) const = 0;

      virtual bool
      isAtBoundary(const unsigned int i) const = 0;

      virtual bool
      isAtBoundary() const = 0;

      virtual bool
      hasPeriodicNeighbor(const unsigned int i) const = 0;

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

      virtual size_type
      getDim() const = 0;

      virtual void
      getParametricPoint(const utils::Point &   realPoint,
                         const CellMappingBase &cellMapping,
                         utils::Point &         parametricPoint) const = 0;

      virtual void
      getRealPoint(const utils::Point &   parametricPoint,
                   const CellMappingBase &cellMapping,
                   utils::Point &         realPoint) const = 0;

      virtual void
      cellNodeIdtoGlobalNodeId(std::vector<global_size_type> &vecId) const = 0;

      virtual size_type
      getFaceBoundaryId(size_type faceId) const = 0;

      virtual void
      getFaceDoFGlobalIndices(
        size_type                      faceId,
        std::vector<global_size_type> &vecNodeId) const = 0;

      virtual size_type
      getFEOrder() const = 0;


    }; // end of class FECellBase
  }    // end of namespace basis

} // end of namespace dftefe
#endif // dftefeFECellBase_h
