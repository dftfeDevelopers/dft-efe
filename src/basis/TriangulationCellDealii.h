#ifndef dftefeTriangulationCellDealii_h
#define dftefeTriangulationCellDealii_h

#include <utils/Point.h>
#include <utils/TypeConfig.h>
#include "CellMappingBase.h"
#include "TriangulationCellBase.h"

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <memory>
namespace dftefe
{
  namespace basis
  {
    /**
     * @brief An interface to deal.ii geometric cell
     **/
    template <unsigned int dim>
    class TriangulationCellDealii : public TriangulationCellBase
    {
      /*
       * typedefs
       */
    public:
      using DealiiTriangulationCellIterator =
        typename dealii::Triangulation<dim>::active_cell_iterator;

    public:
      TriangulationCellDealii(DealiiTriangulationCellIterator dealiiCellIter);
      ~TriangulationCellDealii();

      void
      getVertices(std::vector<utils::Point> &points) const override;

      void
      getVertex(size_type i, utils::Point &point) const override;

      size_type
      getId() const override;

      bool
      isPointInside(const utils::Point &point) const override;

      bool
      isAtBoundary(const unsigned int i) const override;

      bool
      isAtBoundary() const override;

      bool
      hasPeriodicNeighbor(const unsigned int i) const override;

      unsigned int
      getDim() const override;

      double
      diameter() const override;

      void
      center(dftefe::utils::Point &centerPoint) const override;

      void
      setRefineFlag() override;
      /*
       * \todo
       * TODO : Should implement the cellMapping before implementation
       */

      void
      clearRefineFlag() override;

      double
      minimumVertexDistance() const override;

      double
      distanceToUnitCell(dftefe::utils::Point &parametricPoint) const override;

      void
      getParametricPoint(const dftefe::utils::Point &realPoint,
                         const CellMappingBase &     cellMapping,
                         dftefe::utils::Point &parametricPoint) const override;

      /*
       * \todo
       * TODO : Should implement the cellMapping before implementation
       */
      void
      getRealPoint(const utils::Point &   parametricPoint,
                   const CellMappingBase &cellMapping,
                   utils::Point &         realPoint) const override;

      DealiiTriangulationCellIterator &
      getCellIterator();

    private:
      DealiiTriangulationCellIterator d_cellItr;

    }; // end of class TriaCellDealii
  }    // end of namespace basis

} // end of namespace dftefe
#include "TriangulationCellDealii.t.cpp"
#endif // dftefeTriaCellDealii_h
