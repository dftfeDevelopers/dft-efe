#ifndef dftefeCellQuadratureContainer_h
#define dftefeCellQuadratureContainer_h

#include "QuadratureRule.h"
#include <utils/TypeConfig.h>
#include <utils/Point.h>
#include <basis/TriangulationBase.h>
#include <basis/CellMappingBase.h>
#include <memory>

namespace dftefe
{
  namespace quadrature
  {
    /**
     * This class stores the quadrature points and corresponding JxW in each
     * cell. This supports adaptive quadrature i.e each cell can have different
     * quadrature rules. Further each cell can have arbitrary quadrature rule.
     */
    class CellQuadratureContainer
    {
    public:
      /**
       * @brief Constructor for assigning each cell with a common quadrature rule.
       * @param[in] quadratureRule The quadrature rule specifying the quad
       * points in the parametric coordinates with its corresponding weights.
       * @param[in] triangulation The triangulation that has information on the
       * cell and its vertices
       * @param[in] cellMapping cellMapping provides the the information on how
       * the cell in real space is mapped to its parametric coordinates.  This
       * is required to calculate the JxW values at each quad point
       */
      CellQuadratureContainer(
        std::shared_ptr<const QuadratureRule>           quadratureRule,
        std::shared_ptr<const basis::TriangulationBase> triangulation,
        const basis::CellMappingBase &                  cellMapping);

      /**
       * @brief Constructor for assigning each cell with a common quadrature rule.
       * @param[in] quadratureRule The quadrature rule specifying the quad
       * points in the parametric coordinates with its corresponding weights.
       * @param[in] triangulation The triangulation that has information on the
       * cell and its vertices
       * @param[in] cellMapping cellMapping provides the the information on how
       * the cell in real space is mapped to its parametric coordinates.  This
       * is required to calculate the JxW values at each quad point
       */
      CellQuadratureContainer(
        std::vector<std::shared_ptr<const QuadratureRule>> quadratureRule,
        std::shared_ptr<const basis::TriangulationBase>    triangulation,
        const basis::CellMappingBase &                     cellMapping);

      /**
       * @brief Function that returns a vector containing the real coordinates of the quad points in all cells
       *
       * @returns  a vector of dftefe::utils::Point
       */
      const std::vector<dftefe::utils::Point> &
      getRealPoints() const;

      /**
       * @brief Function that returns a vector containing the real coordinates of the
       * quad points in cell corresponding to the cellId
       *
       * @param[n] cellId the id to the cell
       * @returns  a vector of dftefe::utils::Point
       */
      std::vector<dftefe::utils::Point>
      getCellRealPoints(const unsigned int cellId) const;

      /**
       * @brief Function that returns a vector containing the real coordinates of the
       * quad points in cell corresponding to the cellId
       *
       * @param[n] cellId the id to the cell
       * @returns  a vector of dftefe::utils::Point
       */
      const std::vector<dftefe::utils::Point> &
      getCellParametricPoints(const unsigned int cellId) const;

      /**
       * @brief Function that returns a vector containing the weight of the
       * quad points in cell corresponding to the cellId
       *
       * @param[n] cellId the id to the cell
       * @returns  a vector of weights double
       */
      const std::vector<double> &
      getCellQuadratureWeights(const unsigned int cellId) const;

      /**
       * @brief Function that returns a vector containing the real coordinates of the
       * quad points in cell corresponding to the cellId
       *
       * @returns  a vector of dftefe::utils::Point
       */
      const std::vector<double> &
      getJxW() const;

      /**
       * @brief Function that returns a vector containing the JxW of the
       * quad points in cell corresponding to the cellId
       *
       * @param[n] cellId the id to the cell
       * @returns  a vector of JxW (double)
       */
      std::vector<double>
      getCellJxW(const unsigned int cellId) const;

      /**
       * @brief Function that returns the handle to quadrature rule corresponding to the
       * the cell Id
       *
       * @param[n] cellId the id to the cell
       * @returns  QuadratureRule
       */
      const QuadratureRule &
      getQuadratureRule(const unsigned int cellId) const;

      /**
       * @brief  A function to return the total number of quadrature points in all the cells
       *
       * @returns  the number of quadrature points in all the cells
       */
      size_type
      nQuadraturePoints() const;

      /**
       * @brief A function to returns the number of quadrature points in cell corresponding to the
       * the cell Id
       *
       * @param[n] cellId the id to the cell
       * @returns  number of quadrature points
       */
      size_type
      nCellQuadraturePoints(const unsigned int cellId) const;


    private:
      std::vector<std::shared_ptr<const QuadratureRule>> d_quadratureRuleVec;
      std::vector<size_type>                             d_numCellQuadPoints;
      std::vector<size_type>                             d_cellQuadStartIds;
      std::vector<dftefe::utils::Point>                  d_realPoints;
      std::vector<double>                                d_JxW;
      unsigned int                                       d_dim;
      size_type                                          d_numQuadPoints;
    };
  } // end of namespace quadrature

} // end of namespace dftefe

#endif // dftefeCellQuadratureContainer_h
