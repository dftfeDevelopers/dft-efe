#ifndef dftefeCellQuadratureContainer_h
#define dftefeCellQuadratureContainer_h

#include "QuadratureRule.h"
#include <utils/TypeConfig.h>
#include <utils/Point.h>
#include <utils/ScalarSpatialFunction.h>
#include <basis/TriangulationBase.h>
#include <basis/CellMappingBase.h>
#include <basis/ParentToChildCellsManagerBase.h>
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
       * @brief Constructor for assigning each cell with different quadrature rule for each cell.
       * @param[in] quadratureRulevec vector of quadratureRule pointers
       * specifying the quadrature rule for each cell
       * @param[in] triangulation The triangulation that has information on the
       * cell and its vertices
       * @param[in] cellMapping cellMapping provides the the information on how
       * the cell in real space is mapped to its parametric coordinates.  This
       * is required to calculate the JxW values at each quad point
       */
      CellQuadratureContainer(
        std::vector<std::shared_ptr<const QuadratureRule>> quadratureRuleVec,
        std::shared_ptr<const basis::TriangulationBase>    triangulation,
        const basis::CellMappingBase &                     cellMapping);

      /**
       * @brief Constructor for creating an adaptive quadrature rule in each cell based
       * user-defined functions
       * @param[in] baseQuadratureRule The base quadrature rule to be used in
       * constructing the adaptive quadrature rule
       * @param[in] triangulation The triangulation that has information on the
       * cell and its vertices
       * @param[in] cellMapping cellMapping object that provides the the
       * information on how the cell in real space is mapped to its parametric
       * coordinates. This is required to calculate the JxW values at each quad
       * point
       */
      CellQuadratureContainer(
        std::shared_ptr<const QuadratureRule>           baseQuadratureRule,
        std::shared_ptr<const basis::TriangulationBase> triangulation,
        const basis::CellMappingBase &                  cellMapping,
        basis::ParentToChildCellsManagerBase &parentToChildCellsManager,
        std::vector<std::shared_ptr<const utils::ScalarSpatialFunctionReal>>
                                   functions,
        const std::vector<double> &tolerances,
        const std::vector<double> &integralThresholds,
        const double               smallestCellVolume = 1e-12,
        const unsigned int         maxRecursion       = 100);

      /**
       * @brief Returns the number of cells in the quadrature container
       * @returns  number of cells in the quadrature container
       */
      size_type
      nCells() const;

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
       * @param[in] cellId the id to the cell
       * @returns  a vector of dftefe::utils::Point
       */
      std::vector<dftefe::utils::Point>
      getCellRealPoints(const unsigned int cellId) const;

      /**
       * @brief Function that returns a vector containing the real coordinates of the
       * quad points in cell corresponding to the cellId
       *
       * @param[in] cellId the id to the cell
       * @returns  a vector of dftefe::utils::Point
       */
      const std::vector<dftefe::utils::Point> &
      getCellParametricPoints(const unsigned int cellId) const;

      /**
       * @brief Function that returns a vector containing the weight of the
       * quad points in cell corresponding to the cellId
       *
       * @param[in] cellId the id to the cell
       * @returns  a vector of weights double
       */
      const std::vector<double> &
      getCellQuadratureWeights(const unsigned int cellId) const;

      /**
       * @brief Function that returns a vector containing the Jacobian times quadrature weight
       * for all the quad points across all the cells in the triangulation
       *
       * @returns vectors of Jacobian times quadrature weight
       */
      const std::vector<double> &
      getJxW() const;

      /**
       * @brief Function that returns a vector containing the Jacobian times weight of the
       * quad points in cell corresponding to the cellId
       *
       * @param[in] cellId the id to the cell
       * @returns  a vector (double) of Jacobian times weight
       */
      std::vector<double>
      getCellJxW(const unsigned int cellId) const;

      /**
       * @brief Function that returns the handle to quadrature rule corresponding to the
       * the cell Id
       *
       * @param[in] cellId the id to the cell
       * @returns  Const reference to QuadratureRule
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
       * @param[in] cellId the id to the cell
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
      size_type                                          d_numCells;
      bool d_storeJacobianInverse;
    };
  } // end of namespace quadrature

} // end of namespace dftefe

#endif // dftefeCellQuadratureContainer_h
