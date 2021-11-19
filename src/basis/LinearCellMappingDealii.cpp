#include "LinearCellMappingDealii.h"
#include "TriangulationCellDealii.h"
#include "TriangulationCellDealii.h"
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_values.h>

namespace dftefe {
  namespace basis
  {
    template <unsigned int dim>
    LinearCellMappingDealii<dim>::LinearCellMappingDealii():
      mapping()
    {

    }
    template <unsigned int dim>
    void
    LinearCellMappingDealii<dim>::getParametricPoint(const utils::Point &realPoint,
                                                const TriangulationCellBase &triaCellBase,
                                                utils::Point &      parametricPoint,
                                                bool isPointInside) const
    {

      TriangulationCellDealii<dim> triaCellDealii =  dynamic_cast<TriangulationCellDealii<dim>>(triaCellBase);

      typename dealii::Triangulation<dim>::active_cell_iterator cellDealii = triaCellDealii.getCellIterator();

      utils::Point P_ref(dim,0.0);
      P_ref = mapping.transform_real_to_unit_cell(cellDealii,realPoint);
       isPointInside = dealii::GeometryInfo<dim>::is_inside_unit_cell(P_ref);

    }

    template <unsigned int dim>
    void LinearCellMappingDealii<dim>::getJxW(const TriangulationCellBase &triaCellBase,
                                    const std::vector<utils::Point> &paramPoints,
                                    const std::vector<double> &weights,
                                    std::vector<double> & valuesJxW)
    {
      TriangulationCellDealii<dim> triaCellDealii =  dynamic_cast<TriangulationCellDealii<dim>>(triaCellBase);
      dealii::FE_Q<dim> fe(1);
      std::vector<dealii::Point<dim>> quadPointsDealii;
      quadPointsDealii.resize(paramPoints.size());
      utils::convertToDealiiPoint(paramPoints,quadPointsDealii);
      dealii::Quadrature<dim> quadRuleDealii(quadPointsDealii, weights);
      dealii::FEValues<dim> fe_values(fe, quadRuleDealii, dealii::update_values |
                                         dealii::update_JxW_values);
      fe_values.reinit(triaCellDealii.getCellIterator());

      for (unsigned int iQuad = 0; iQuad < quadRuleDealii.size(); iQuad++)
        {
          valuesJxW[iQuad] = fe_values.JxW(iQuad);
        }
    }


  } // end of namespace basis

} // end of namespace dftefe