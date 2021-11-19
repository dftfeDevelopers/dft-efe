#ifndef dftefeLinearCellMappingDealii_h
#define dftefeLinearCellMappingDealii_h

#include "CellMappingBase.h"
#include "TriangulationCellBase.h"
#include <vector>
#include <utils/Point.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/fe/mapping_q1.h>
namespace dftefe {
  namespace basis
  {

    template <unsigned int dim>
    class LinearCellMappingDealii : public CellMappingBase
    {


      LinearCellMappingDealii();

      ~LinearCellMappingDealii();

      void getJxW(const TriangulationCellBase &triaCellBase, const std::vector<utils::Point> &paramPoints,
             const       std::vector<double> &weights,
             std::vector<double> & valuesJxW);
      void
      getParametricPoint(const utils::Point &realPoint,
                         const TriangulationCellBase &triaCellBase,
                         utils::Point &      parametricPoint,
                         bool isPointInside) const;

      void
      getParametricPoints(const std::vector<utils::Point> & realPoints,
                          const TriangulationCellBase &triaCellBase,
                          std::vector<utils::Point> &      parametricPoints) const;

      void
      getRealPoint(const utils::Point &parametricPoint,
                   const TriangulationCellBase &triaCellBase,
                   utils::Point &      realPoint) const;


      void
      getRealPoints(const std::vector<utils::Point> &parametricPoints,
                    const TriangulationCellBase &triaCellBase,
                    std::vector<utils::Point> &      realPoints) const;

    private :

      dealii::MappingQ1<dim> mapping;


    }; // end of class LinearCellMappingDealii

  } // end of namespace basis
} // end of namespace dftefe

#endif /* dftefeLinearCellMappingDealii_hpp */
