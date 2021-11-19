#ifndef dftefeLinearCellMappingDealii_h
#define dftefeLinearCellMappingDealii_h

#include "CellMappingBase.h"
#include "TriangulationCellBase.h"
#include <vector>
#include <utils/Point.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/fe/mapping_q1.h>
namespace dftefe
{
  namespace basis
  {
    template <unsigned int dim>
    class LinearCellMappingDealii : public CellMappingBase
    {
    public:
      LinearCellMappingDealii();
      ~LinearCellMappingDealii();

      void
      getJxW(const TriangulationCellBase &            triaCellBase,
             const std::vector<dftefe::utils::Point> &paramPoints,
             const std::vector<double> &              weights,
             std::vector<double> &                    valuesJxW) const override;
      void
      getParametricPoint(const dftefe::utils::Point & realPoint,
                         const TriangulationCellBase &triaCellBase,
                         utils::Point &               parametricPoint,
                         bool &isPointInside) const override;

      void
      getParametricPoints(const std::vector<dftefe::utils::Point> &realPoints,
                          const TriangulationCellBase &            triaCellBase,
                          std::vector<utils::Point> &parametricPoints,
                          std::vector<bool> &arePointsInside) const override;

      void
      getRealPoint(const dftefe::utils::Point & parametricPoint,
                   const TriangulationCellBase &triaCellBase,
                   dftefe::utils::Point &       realPoint) const override;

      void
      getRealPoints(
        const std::vector<dftefe::utils::Point> &parametricPoints,
        const TriangulationCellBase &            triaCellBase,
        std::vector<dftefe::utils::Point> &      realPoints) const override;

    private:
      dealii::MappingQ1<dim> mapping;


    }; // end of class LinearCellMappingDealii

  } // end of namespace basis
} // end of namespace dftefe

#endif /* dftefeLinearCellMappingDealii_h */
