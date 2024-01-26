#ifndef dftefeQuadratureRuleContainer_h
#define dftefeQuadratureRuleContainer_h

#include <quadrature/QuadratureRule.h>
#include <quadrature/QuadratureAttributes.h>
#include <utils/TypeConfig.h>
#include <utils/Point.h>
#include <utils/ScalarSpatialFunction.h>
#include <basis/TriangulationBase.h>
#include <basis/CellMappingBase.h>
#include <basis/ParentToChildCellsManagerBase.h>
#include <memory>
#include <utils/MPITypes.h>
#include <utils/MPIWrapper.h>
#include <quadrature/Defaults.h>
namespace dftefe
{
  namespace quadrature
  {
    /**
     * This class stores the quadrature points and corresponding JxW in each
     * cell. This supports adaptive quadrature i.e each cell can have different
     * quadrature rules. Further each cell can have arbitrary quadrature rule.
     */
    class QuadratureRuleContainer
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
      QuadratureRuleContainer(
        const QuadratureRuleAttributes &      quadratureRuleAttributes,
        std::shared_ptr<const QuadratureRule> quadratureRule,
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
      QuadratureRuleContainer(
        const QuadratureRuleAttributes &quadratureRuleAttributes,
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
      QuadratureRuleContainer(
        const QuadratureRuleAttributes &      quadratureRuleAttributes,
        std::shared_ptr<const QuadratureRule> baseQuadratureRule,
        std::shared_ptr<const basis::TriangulationBase> triangulation,
        const basis::CellMappingBase &                  cellMapping,
        basis::ParentToChildCellsManagerBase &parentToChildCellsManager,
        std::vector<std::shared_ptr<const utils::ScalarSpatialFunctionReal>>
                                   functions,
        const std::vector<double> &absoluteTolerances,
        const std::vector<double> &relativeTolerances,
        const std::vector<double> &integralThresholds,
        const double               smallestCellVolume = 1e-12,
        const unsigned int         maxRecursion       = 100);

      /**
       * @brief Constructor for creating a subdivided quadrature rule in each cell based
       * user-defined functions and a reference quadrature. So one provides a
       * minimum and maximum order (or number of 1D gauss points) and maimum
       * number of copies of the orders one wants to go. Also one has a
       * reference quadrature which can
       * 1. either be constructed from highly refined tensor structures of GAUSS
       * or GLL. from spatally variable quadrature rules like GAUSS_VARIABLE,
       * GLL_VARIABLE, or ADAPTIVE. Care should be taken that for ADAPTIVE it is
       * assumed that the quadrature is optimal and hence only cells with high
       * quadrature density are traversed. Whereas for all others all the cells
       * are traversed. Then the {order, copy} pair with minimum quad points per
       * cell is chosen. Then again the maximum of this points over all
       * processors are taken. In one statement the condition can be written as
       * sup_{processors} inf_{all pairs in a processor} (QuadPoints in Cells
       * statisfing the given tolerances w.r.t. the reference quadrature)
       * @param[in] order1DMin The minimum gauss 1D number of points from which
       * to start
       * @param[in] order1DMax The maximum gauss 1D number of points upto which
       * to iterate
       * @param[in] copies1DMax The maimum number of copies (starting from 1) to
       * be done at each iteration of order.
       * @param[in] triangulation The triangulation that has information on the
       * cell and its vertices
       * @param[in] cellMapping cellMapping object that provides the the
       * information on how the cell in real space is mapped to its parametric
       * coordinates. This is required to calculate the JxW values at each quad
       * point
       */
      QuadratureRuleContainer(
        const QuadratureRuleAttributes &quadratureRuleAttributes,
        const size_type                 order1DMin,
        const size_type                 order1DMax,
        const size_type                 copies1DMax,
        std::shared_ptr<const basis::TriangulationBase> triangulation,
        const basis::CellMappingBase &                  cellMapping,
        std::vector<std::shared_ptr<const utils::ScalarSpatialFunctionReal>>
                                   functions,
        const std::vector<double> &absoluteTolerances,
        const std::vector<double> &relativeTolerances,
        const quadrature::QuadratureRuleContainer
          &                        quadratureRuleContainerReference,
        const utils::mpi::MPIComm &comm);

      /**
       * @brief Returns the underlying QuadratureRuleAttributes
       * @returns const reference to the QuadratureAttributes
       */
      const QuadratureRuleAttributes &
      getQuadratureRuleAttributes() const;

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

      /**
       * @brief A function to return the starting index of the quadrature point of each cell
       *
       * @returns  vector storing the starting index of the quadrature point of each cell
       */
      const std::vector<size_type> &
      getCellQuadStartIds() const;

      /**
       * @brief A function to return the starting index of the quadrature point of a given cell
       *
       * @param[in] cellId index of the cell
       * @returns  the starting index of the quadrature point of the cell
       */
      size_type
      getCellQuadStartId(const size_type cellId) const;

      std::shared_ptr<const basis::TriangulationBase>
      getTriangulation() const;

      const basis::CellMappingBase &
      getCellMapping() const;

    private:
      const QuadratureRuleAttributes &d_quadratureRuleAttributes;
      std::vector<std::shared_ptr<const QuadratureRule>> d_quadratureRuleVec;
      std::vector<size_type>                             d_numCellQuadPoints;
      std::vector<size_type>                             d_cellQuadStartIds;
      std::vector<dftefe::utils::Point>                  d_realPoints;
      std::vector<double>                                d_JxW;
      unsigned int                                       d_dim;
      size_type                                          d_numQuadPoints;
      size_type                                          d_numCells;
      bool                                               d_storeJacobianInverse;
      std::shared_ptr<const basis::TriangulationBase>    d_triangulation;
      const basis::CellMappingBase &                     d_cellMapping;
    };
  } // end of namespace quadrature

} // end of namespace dftefe

#endif // dftefeQuadratureRuleContainer_h
