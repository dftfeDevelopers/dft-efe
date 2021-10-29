#ifndef dftefeTriangulationCellDealii_h
#define dftefeTriangulationCellDealii_h

#include <utils/Point.h>
#include <utils/TypeConfig.h>
#include "CellMappingBase.h"
#include "TriangulationCellBase.h"

#include <deal.II/grid/tria.h>

#include <memory>
namespace dftefe
{
  namespace basis
  {
    /**
     * @brief An interface to deal.ii geometric cell
     **/
    template <unsigned int dim>
    class TrianfulationCellDealii : public TriangulationCellBase
    {
    public:
      TrianfulationCellDealii(
        dealii::Triangulation<dim>::active_cell_iterator dealiiCellIter);
      ~TrianfulationCellDealii();

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
      getParametricPoint(const Point &          realPoint,
                         const CellMappingBase &cellMapping,
                         Point &                parametricPoint) const override;

      /*
       * \todo
       * TODO : Should implement the cellMapping before implementation
       */
      void
      getRealPoint(const Point &          parametricPoint,
                   const CellMappingBase &cellMapping,
                   Point &                realPoint) const override;


    private:
      dealii::Triangulation<dim>::active_cell_iterator d_cellItr;

    }; // end of class TriaCellDealii
  }    // end of namespace basis

} // end of namespace dftefe
#endif // dftefeTriaCellDealii_h
