
#include "TriangulationCellDealii.h"
#include "DealiiConversions.h"
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_values.h>

namespace dftefe
{
  namespace basis
  {
    template <unsigned int dim>
    LinearCellMappingDealii<dim>::LinearCellMappingDealii()
      : d_mappingDealii()
      , d_fe(1)
    {}


    template <unsigned int dim>
    LinearCellMappingDealii<dim>::~LinearCellMappingDealii()
    {}


    template <unsigned int dim>
    void
    LinearCellMappingDealii<dim>::getJxW(
      const TriangulationCellBase &            triaCellBase,
      const std::vector<dftefe::utils::Point> &paramPoints,
      const std::vector<double> &              weights,
      std::vector<double> &                    valuesJxW) const
    {
      TriangulationCellDealii<dim> triaCellDealii =
        dynamic_cast<const TriangulationCellDealii<dim> &>(triaCellBase);
      // dealii::FE_Q<dim>                       fe(1);
      std::vector<dealii::Point<dim, double>> quadPointsDealii;
      quadPointsDealii.resize(paramPoints.size());
      convertToDealiiPoint<dim>(paramPoints, quadPointsDealii);
      dealii::Quadrature<dim> quadRuleDealii(quadPointsDealii, weights);
      dealii::FEValues<dim>   fe_values(d_fe,
                                      quadRuleDealii,
                                      dealii::update_values |
                                        dealii::update_JxW_values);
      auto                    cellItr = triaCellDealii.getCellIterator();
      fe_values.reinit(cellItr);

      for (unsigned int iQuad = 0; iQuad < quadRuleDealii.size(); iQuad++)
        {
          valuesJxW[iQuad] = fe_values.JxW(iQuad);
        }
    }

    template <unsigned int dim>
    void
    LinearCellMappingDealii<dim>::getParametricPoint(
      const dftefe::utils::Point & realPoint,
      const TriangulationCellBase &triaCellBase,
      dftefe::utils::Point &       parametricPoint,
      bool &                       isPointInside) const
    {
      TriangulationCellDealii<dim> triaCellDealii =
        dynamic_cast<const TriangulationCellDealii<dim> &>(triaCellBase);
      typename dealii::Triangulation<dim>::active_cell_iterator cellDealii =
        triaCellDealii.getCellIterator();

      dealii::Point<dim, double> dealiiParametricPoint, dealiiRealPoint;
      convertToDealiiPoint<dim>(realPoint, dealiiRealPoint);
      isPointInside = true;
      try
        {
          dealiiParametricPoint =
            d_mappingDealii.transform_real_to_unit_cell(cellDealii,
                                                        dealiiRealPoint);
        }
      catch (typename dealii::Mapping<dim>::ExcTransformationFailed)
        {
          isPointInside = dealii::GeometryInfo<dim>::is_inside_unit_cell(
            dealiiParametricPoint);
        }
      convertToDftefePoint<dim>(dealiiParametricPoint, parametricPoint);
    }


    template <unsigned int dim>
    void
    LinearCellMappingDealii<dim>::getParametricPoints(
      const std::vector<dftefe::utils::Point> &realPoints,
      const TriangulationCellBase &            triaCellBase,
      std::vector<utils::Point> &              parametricPoints,
      std::vector<bool> &                      arePointsInside) const
    {
      TriangulationCellDealii<dim> triaCellDealii =
        dynamic_cast<const TriangulationCellDealii<dim> &>(triaCellBase);
      typename dealii::Triangulation<dim>::active_cell_iterator cellDealii =
        triaCellDealii.getCellIterator();

      const size_type numPoints = realPoints.size();
      parametricPoints.resize(numPoints, utils::Point(dim));
      std::vector<dealii::Point<dim, double>> dealiiParametricPoints(numPoints);
      std::vector<dealii::Point<dim, double>> dealiiRealPoints(numPoints);
      convertToDealiiPoint<dim>(realPoints, dealiiRealPoints);
      d_mappingDealii.transform_points_real_to_unit_cell(
        cellDealii, dealiiRealPoints, dealiiParametricPoints);
      arePointsInside = std::vector<bool>(numPoints, true);
      for (unsigned int i = 0; i < numPoints; ++i)
        {
          if (dealiiParametricPoints[i][0] ==
              std::numeric_limits<double>::infinity())
            {
              arePointsInside[i] = false;
            }
        }
      convertToDftefePoint<dim>(dealiiParametricPoints, parametricPoints);
    }

    template <unsigned int dim>
    void
    LinearCellMappingDealii<dim>::getRealPoint(
      const dftefe::utils::Point & parametricPoint,
      const TriangulationCellBase &triaCellBase,
      dftefe::utils::Point &       realPoint) const
    {
      TriangulationCellDealii<dim> triaCellDealii =
        dynamic_cast<const TriangulationCellDealii<dim> &>(triaCellBase);
      typename dealii::Triangulation<dim>::active_cell_iterator cellDealii =
        triaCellDealii.getCellIterator();

      dealii::Point<dim, double> dealiiParametricPoint, dealiiRealPoint;
      convertToDealiiPoint<dim>(parametricPoint, dealiiParametricPoint);
      dealiiRealPoint =
        d_mappingDealii.transform_unit_to_real_cell(cellDealii,
                                                    dealiiParametricPoint);
      convertToDftefePoint<dim>(dealiiRealPoint, realPoint);
    }

    template <unsigned int dim>
    void
    LinearCellMappingDealii<dim>::getRealPoints(
      const std::vector<dftefe::utils::Point> &parametricPoints,
      const TriangulationCellBase &            triaCellBase,
      std::vector<dftefe::utils::Point> &      realPoints) const
    {
      TriangulationCellDealii<dim> triaCellDealii =
        dynamic_cast<const TriangulationCellDealii<dim> &>(triaCellBase);
      typename dealii::Triangulation<dim>::active_cell_iterator cellDealii =
        triaCellDealii.getCellIterator();

      const size_type numPoints = parametricPoints.size();
      realPoints.resize(numPoints, utils::Point(dim));
      std::vector<dealii::Point<dim, double>> dealiiParametricPoints(numPoints);
      std::vector<dealii::Point<dim, double>> dealiiRealPoints(numPoints);
      convertToDealiiPoint<dim>(parametricPoints, dealiiParametricPoints);
      for (unsigned int i = 0; i < numPoints; ++i)
        {
          dealiiRealPoints[i] = d_mappingDealii.transform_unit_to_real_cell(
            cellDealii, dealiiParametricPoints[i]);
        }
      convertToDftefePoint<dim>(dealiiRealPoints, realPoints);
    }



  } // end of namespace basis

} // end of namespace dftefe
