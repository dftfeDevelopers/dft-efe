
#include "TriangulationCellDealii.h"
#include <utils/DealiiConversions.h>
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_values.h>

namespace dftefe
{
  namespace basis
  {
    template <unsigned int dim>
    LinearCellMappingDealii<dim>::LinearCellMappingDealii()
      : mapping()
    {}


    template <unsigned int dim>
    LinearCellMappingDealii<dim>::~LinearCellMappingDealii()
    {}


    template <unsigned int dim>
    void
    LinearCellMappingDealii<dim>::getParametricPoint(
      const dftefe::utils::Point &         realPoint,
      const TriangulationCellBase &triaCellBase,
      dftefe::utils::Point &               parametricPoint,
      bool &                       isPointInside) const
    {
      TriangulationCellDealii<dim> triaCellDealii =
        dynamic_cast<const TriangulationCellDealii<dim>&>(triaCellBase);

      typename dealii::Triangulation<dim>::active_cell_iterator cellDealii =
        triaCellDealii.getCellIterator();

      dealii::Point<dim,double> P_ref, dealiiRealPoint;
      dftefe::utils::convertToDealiiPoint<dim>(realPoint, dealiiRealPoint);
      P_ref = mapping.transform_real_to_unit_cell(cellDealii, dealiiRealPoint);
      isPointInside = dealii::GeometryInfo<dim>::is_inside_unit_cell(P_ref);
    }

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
      dealii::FE_Q<dim>               fe(1);
      std::vector<dealii::Point<dim,double>> quadPointsDealii;
      quadPointsDealii.resize(paramPoints.size());
      utils::convertToDealiiPoint<dim>(paramPoints, quadPointsDealii);
      dealii::Quadrature<dim> quadRuleDealii(quadPointsDealii, weights);
      dealii::FEValues<dim>   fe_values(
        fe, quadRuleDealii, dealii::update_values | dealii::update_JxW_values);
      auto cellItr = triaCellDealii.getCellIterator();
      fe_values.reinit(cellItr);

      for (unsigned int iQuad = 0; iQuad < quadRuleDealii.size(); iQuad++)
        {
          valuesJxW[iQuad] = fe_values.JxW(iQuad);
        }
    }


    template <unsigned int dim>
    void
    LinearCellMappingDealii<dim>::getParametricPoints(const std::vector<dftefe::utils::Point> &realPoints,
                        const TriangulationCellBase &            triaCellBase,
                        std::vector<utils::Point> &parametricPoints,
                        std::vector<bool> &arePointsInside) const
    {
      DFTEFE_AssertWithMsg(
        false,
        "getParametricPoints() not implemented in LinearCellMappingDealii ");

    }

    template <unsigned int dim>
    void
    LinearCellMappingDealii<dim>::getRealPoint(const dftefe::utils::Point & parametricPoint,
                 const TriangulationCellBase &triaCellBase,
                 dftefe::utils::Point &       realPoint) const
    {
      DFTEFE_AssertWithMsg(
        false,
        "getRealPoint() not implemented in LinearCellMappingDealii ");

    }

    template <unsigned int dim>
    void
    LinearCellMappingDealii<dim>::getRealPoints(
      const std::vector<dftefe::utils::Point> &parametricPoints,
      const TriangulationCellBase &            triaCellBase,
      std::vector<dftefe::utils::Point> &      realPoints) const
    {
      DFTEFE_AssertWithMsg(
        false,
        "getRealPoints() not implemented in LinearCellMappingDealii ");

    }



  } // end of namespace basis

} // end of namespace dftefe
