#include <utils/Exceptions.h>
#include "DealiiConversions.h"
#include <deal.II/base/geometry_info.h>


namespace dftefe
{
  namespace basis
  {
    template <size_type dim>
    TriangulationCellDealii<dim>::TriangulationCellDealii(
      DealiiTriangulationCellIterator dealiiCellIter)
      : d_cellItr(dealiiCellIter)
    {}

    template <size_type dim>
    TriangulationCellDealii<dim>::~TriangulationCellDealii()
    {}

    template <size_type dim>
    void
    TriangulationCellDealii<dim>::getVertices(
      std::vector<utils::Point> &points) const
    {
      const size_type nVertices =
        dealii::GeometryInfo<dim>::vertices_per_cell;
      points.resize(nVertices, utils::Point(dim, 0.0));
      std::vector<dealii::Point<dim, double>> pointsDealii;
      pointsDealii.resize(nVertices);
      for (size_type iVertex = 0; iVertex < nVertices; iVertex++)
        {
          pointsDealii[iVertex] = d_cellItr->vertex(iVertex);
          convertToDftefePoint<dim>(d_cellItr->vertex(iVertex),
                                    points[iVertex]);
        }
    }

    template <size_type dim>
    void
    TriangulationCellDealii<dim>::getVertex(size_type     i,
                                            utils::Point &point) const
    {
      convertToDftefePoint<dim>(d_cellItr->vertex(i), point);
    }

    template <size_type dim>
    size_type
    TriangulationCellDealii<dim>::getId() const
    {
      utils::throwException(
        false, "getId() in TriangulationCellDeaii not yet implemented.");
      return 0;
    }

    template <size_type dim>
    bool
    TriangulationCellDealii<dim>::isPointInside(const utils::Point &point) const
    {
      dealii::Point<dim, double> dealiiPoint;
      convertToDealiiPoint<dim>(point, dealiiPoint);
      return d_cellItr->point_inside(dealiiPoint);
    }


    template <size_type dim>
    bool
    TriangulationCellDealii<dim>::isAtBoundary(const size_type i) const
    {
      return d_cellItr->at_boundary(i);
    }

    template <size_type dim>
    bool
    TriangulationCellDealii<dim>::isAtBoundary() const
    {
      return d_cellItr->at_boundary();
    }

    template <size_type dim>
    bool
    TriangulationCellDealii<dim>::hasPeriodicNeighbor(
      const size_type i) const
    {
      return d_cellItr->has_periodic_neighbor(i);
    }

    template <size_type dim>
    size_type
    TriangulationCellDealii<dim>::getDim() const
    {
      return dim;
    }
    template <size_type dim>
    double
    TriangulationCellDealii<dim>::diameter() const
    {
      return d_cellItr->diameter();
    }

    template <size_type dim>
    void
    TriangulationCellDealii<dim>::center(
      dftefe::utils::Point &centerPoint) const
    {
      dealii::Point<dim, double> dealiiPoint;
      dealiiPoint = d_cellItr->center();
      convertToDftefePoint<dim>(dealiiPoint, centerPoint);
    }

    template <size_type dim>
    void
    TriangulationCellDealii<dim>::setRefineFlag()
    {
      d_cellItr->set_refine_flag();
    }

    template <size_type dim>
    void
    TriangulationCellDealii<dim>::clearRefineFlag()
    {
      d_cellItr->clear_refine_flag();
    }

    template <size_type dim>
    double
    TriangulationCellDealii<dim>::minimumVertexDistance() const
    {
      return d_cellItr->minimum_vertex_distance();
    }

    template <size_type dim>
    double
    TriangulationCellDealii<dim>::distanceToUnitCell(
      dftefe::utils::Point &parametricPoint) const
    {
      dealii::Point<dim, double> dealiiPoint;
      convertToDealiiPoint<dim>(parametricPoint, dealiiPoint);
      return dealii::GeometryInfo<dim>::distance_to_unit_cell(dealiiPoint);
    }

    template <size_type dim>
    void
    TriangulationCellDealii<dim>::getParametricPoint(
      const dftefe::utils::Point &realPoint,
      const CellMappingBase &     cellMapping,
      dftefe::utils::Point &      parametricPoint) const
    {
      bool isPointInside;
      cellMapping.getParametricPoint(realPoint,
                                     *this,
                                     parametricPoint,
                                     isPointInside);
    }

    template <size_type dim>
    void
    TriangulationCellDealii<dim>::getRealPoint(
      const utils::Point &   parametricPoint,
      const CellMappingBase &cellMapping,
      utils::Point &         realPoint) const
    {
      utils::throwException(
        false, "getRealPoint() in TriangulationCellDeaii not yet implemented.");
    }

    // TODO removed const qualifier
    template <size_type dim>
    typename TriangulationCellDealii<dim>::DealiiTriangulationCellIterator &
    TriangulationCellDealii<dim>::getCellIterator()
    {
      return d_cellItr;
    }

    //    template <size_type dim>
    //    const typename dealii::Triangulation<dim>::active_cell_iterator &
    //    TriangulationCellDealii<dim>::getCellIterator() const
    //    {
    //      return d_cellItr;
    //    }

  } // namespace basis
} // namespace dftefe
