#include <utils/Exceptions.h>
#include "TriangulationCellDealii.h"
#include <deal.II/base/geometry_info.h>


namespace dftefe
{
  namespace basis
  {
    template <unsigned int dim>
    TrianfulationCellDealii<dim>::TrianfulationCellDealii(
      dealii::Triangulation<dim>::active_cell_iterator dealiiCellIter)
      : d_cellItr(dealiiCellIter)
    {}

    template <unsigned int dim>
    TrianfulationCellDealii<dim>::~TrianfulationCellDealii()
    {}

    template <unsigned int dim>
    void
    TrianfulationCellDealii<dim>::getVertices(
      std::vector<utils::Point> &points) const
    {
      const unsigned int nVertices =
        dealii::GeometryInfo<dim>::vertices_per_cell;
      points.resize(nVertices);
      for (unsigned int iVertex = 0; iVertex < nVertices; iVertex++)
        {
          pointsDealii[iVertex] = d_cellItr->vertex(iVertex);
          utils::convertToDftefePoint<dim>(d_cellItr->vertex(iVertex),
                                           points[iVertex]);
        }
    }

    template <unsigned int dim>
    void
    TrianfulationCellDealii<dim>::getVertex(size_type     i,
                                            utils::Point &point) const
    {
      utils::convertToDftefePoint<dim>(d_cellItr->vertex(i), outputDftefePoint);
    }

    template <unsigned int dim>
    size_type
    TrianfulationCellDealii<dim>::getId() const
    {
      utils::throwException(
        false, "getId() in TriangulationCellDeaii not yet implemented.");
    }

    template <unsigned int dim>
    bool
    TrianfulationCellDealii<dim>::isPointInside(const utils::Point &point) const
    {
      dealii::Point<dim, double> dealiiPoint;
      utils::convertToDealiiPoint<dim>(point, dealiiPoint);
      return d_cellItr->point_inside(dealiiPoint);
    }


    template <unsigned int dim>
    bool
    TrianfulationCellDealii<dim>::isAtBoundary(const unsigned int i) const
    {
      return d_cellItr->at_boundary(i);
    }

    template <unsigned int dim>
    bool
    TrianfulationCellDealii<dim>::isAtBoundary() const
    {
      return d_cellItr->at_boundary();
    }

    template <unsigned int dim>
    unsigned int
    TrianfulationCellDealii<dim>::getDim()
    {
      return dim;
    }

    template <unsigned int dim>
    void
    TrianfulationCellDealii<dim>::getParametricPoint(
      const Point &          realPoint,
      const CellMappingBase &cellMapping,
      Point &                parametricPoint) const
    {
      utils::throwException(
        false,
        "getParametricPoint() in TriangulationCellDeaii not yet implemented.");
    }

    template <unsigned int dim>
    void
    TrianfulationCellDealii<dim>::getRealPoint(
      const Point &          parametricPoint,
      const CellMappingBase &cellMapping,
      Point &                realPoint) const
    {
      utils::throwException(
        false, "getRealPoint() in TriangulationCellDeaii not yet implemented.");
    }

  } // namespace basis
} // namespace dftefe
