#include <utils/Exceptions.h>
#include "TriangulationCellDealii.h"
#include <deal.II/base/geometry_info.h>


namespace dftefe
{
  namespace basis
  {
      <template unsigned int dim>
      TrianfulationCellDealii<dim>::TrianfulationCellDealii(dealii::Triangulation<dim>::active_cell_iterator dealiiCellIter):
  d_cellItr( dealiiCellIter)
      {
          
      }
  
      <template unsigned int dim>
      TrianfulationCellDealii<dim>::~TrianfulationCellDealii()
      {
          
      }
  
  <template unsigned int dim>
      void TrianfulationCellDealii<dim>::getVertices(std::vector<utils::Point> & outputDftefePoints ) const
      {
          const unsigned int no_of_vertices = dealii::GeometryInfo< dim >::vertices_per_cell;
          std::vector<dealii::Point<dim,double>> vertices_coordinates;
                  vertices_coordinates.resize(no_of_vertices);
          

          outputDftefePoints.resize(no_of_vertices);
          for( unsigned int iVertex = 0; iVertex < no_of_vertices ; iVertex++)
          {
              vertices_coordinates[iVertex] =  d_cellItr->vertex(iVertex);
              utils::convertToDftefePoint<dim>(vertices_coordinates[iVertex] ,outputDftefePoints[iVertex] );
          }
          
          
      }
  
  <template unsigned int dim>
  void TrianfulationCellDealii<dim>::getVertex(size_type i, utils::Point & outputDftefePoint) const
  {
      dealii::Point<dim,double> dealiiPoint;
      
      dealiiPoint = d_cellItr->vertex(i);
      utils::convertToDftefePoint<dim>(dealiiPoint ,outputDftefePoint );
      
  }
  
  <template unsigned int dim>
  size_type
  TrianfulationCellDealii<dim>::getId() const
  {
      
  }

  <template unsigned int dim>
  bool
  TrianfulationCellDealii<dim>::isPointInside(const utils::Point &point) const
  {
      dealii::Point<dim,double> dealiiPoint;
      utils::convertToDealiiPoint<dim>(point, dealiiPoint);
      return (d_cellItr->point_inside(dealiiPoint) );
  }
  
  
  <template unsigned int dim>
  bool
  TrianfulationCellDealii<dim>::isAtBoundary(const unsigned int i) const
  {
      return d_cellItr->at_boundary(i);
  }

  <template unsigned int dim>
  bool
  TrianfulationCellDealii<dim>::isAtBoundary() const
  {
      return d_cellItr->at_boundary();
  }
  <template unsigned int dim>
  unsigned int
  TrianfulationCellDealii<dim>::getDim()
  {
      return dim;
  }
  
  <template unsigned int dim>
  void
  TrianfulationCellDealii<dim>::getParametricPoint(const Point &realPoint,
                    const CellMappingBase &      cellMapping,
                    Point  &parametricPoint ) const
  {
      
  }
  
  <template unsigned int dim>
  void
  TrianfulationCellDealii<dim>::getRealPoint(const Point  &parametricPoint,
               const CellMappingBase &      cellMapping,
               Point  &realPoint) const
  {
      
  }
  
  }
}
