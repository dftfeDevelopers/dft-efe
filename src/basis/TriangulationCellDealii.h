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
      using DealiiCellIter =
        typename dealii::Triangulation<dim>::active_cell_iterator;

    public:
      TriangulationCellDealii(DealiiCellIter dealiiCellIter);
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

      unsigned int
      getDim() const override;

      /*
       * \todo
       * TODO : Should implement the cellMapping before implementation
       */

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

      DealiiCellIter &
      getCellIterator();
      //      const DealiiCellIter &
      //      getCellIterator() const;

    private:
      DealiiCellIter d_cellItr;

    }; // end of class TriaCellDealii
  }    // end of namespace basis

} // end of namespace dftefe
#include "TriangulationCellDealii.t.cpp"
#endif // dftefeTriaCellDealii_h
