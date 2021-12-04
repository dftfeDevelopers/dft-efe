#include <utils/Exceptions.h>
#include <utils/DealiiConversions.h>
#include <deal.II/base/geometry_info.h>


namespace dftefe
{
  namespace basis
  {
    template <unsigned int dim>
    TriangulationCellDealii<dim>::TriangulationCellDealii(
      DealiiCellIter dealiiCellIter)
      : d_cellItr(dealiiCellIter)
    {}

    template <unsigned int dim>
    TriangulationCellDealii<dim>::~TriangulationCellDealii()
    {}

    template <unsigned int dim>
    void
    TriangulationCellDealii<dim>::getVertices(
      std::vector<utils::Point> &points) const
    {
      const unsigned int nVertices =
        dealii::GeometryInfo<dim>::vertices_per_cell;
      points.resize(nVertices, utils::Point(dim, 0.0));
      std::vector<dealii::Point<dim, double>> pointsDealii;
      pointsDealii.resize(nVertices);
      for (unsigned int iVertex = 0; iVertex < nVertices; iVertex++)
        {
          pointsDealii[iVertex] = d_cellItr->vertex(iVertex);
          utils::convertToDftefePoint<dim>(d_cellItr->vertex(iVertex),
                                           points[iVertex]);
        }
    }

    template <unsigned int dim>
    void
    TriangulationCellDealii<dim>::getVertex(size_type     i,
                                            utils::Point &point) const
    {
      utils::convertToDftefePoint<dim>(d_cellItr->vertex(i), point);
    }

    template <unsigned int dim>
    size_type
    TriangulationCellDealii<dim>::getId() const
    {
      utils::throwException(
        false, "getId() in TriangulationCellDeaii not yet implemented.");
      return 0;
    }

    template <unsigned int dim>
    bool
    TriangulationCellDealii<dim>::isPointInside(const utils::Point &point) const
    {
      dealii::Point<dim, double> dealiiPoint;
      utils::convertToDealiiPoint<dim>(point, dealiiPoint);
      return d_cellItr->point_inside(dealiiPoint);
    }


    template <unsigned int dim>
    bool
    TriangulationCellDealii<dim>::isAtBoundary(const unsigned int i) const
    {
      return d_cellItr->at_boundary(i);
    }

    template <unsigned int dim>
    bool
    TriangulationCellDealii<dim>::isAtBoundary() const
    {
      return d_cellItr->at_boundary();
    }

    template <unsigned int dim>
    unsigned int
    TriangulationCellDealii<dim>::getDim() const
    {
      return dim;
    }

    template <unsigned int dim>
    void
    TriangulationCellDealii<dim>::getParametricPoint(
      const dftefe::utils::Point &   realPoint,
      const CellMappingBase &cellMapping,
      dftefe::utils::Point &         parametricPoint) const
    {

      bool  isPointInside ;
      cellMapping.getParametricPoint( realPoint, *this, parametricPoint, isPointInside);

    }

    template <unsigned int dim>
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
    template <unsigned int dim>
    typename dealii::Triangulation<dim>::active_cell_iterator &
    TriangulationCellDealii<dim>::getCellIterator()
    {
      return d_cellItr;
    }

//    template <unsigned int dim>
//    const typename dealii::Triangulation<dim>::active_cell_iterator &
//    TriangulationCellDealii<dim>::getCellIterator() const
//    {
//      return d_cellItr;
//    }

  } // namespace basis
} // namespace dftefe
