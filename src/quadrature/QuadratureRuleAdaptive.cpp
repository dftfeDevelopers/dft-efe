#include <utils/Exceptions.h>
#include "QuadratureRuleAdaptive.h"
#include <numeric>
#include <functional>
#include <algorithm>

namespace dftefe
{
  namespace quadrature
  {
    namespace
    {
      void
      updateAdaptiveQuadratureRule(
        const basis::TriangulationCellBase &currentCell,
        const basis::TriangulationCellBase &globalCell,
        const QuadratureRule &              baseQuadratureRule,
        const basis::CellMappingBase &      cellMapping,
        const std::vector<double> &         currentCellJxW,
        std::vector<utils::Point> &         adaptiveQuadPoints,
        std::vector<double> &               adaptiveQuadWeights)
      {
        const size_type    numberBaseQuadPoints = baseQuadratureRule.nPoints();
        const unsigned int dim                  = baseQuadratureRule.getDim();
        const std::vector<utils::Point> &baseQuadratureRuleParametricPoints =
          baseQuadratureRule.getPoints();
        const std::vector<double> &baseQuadratureRuleWeights =
          baseQuadratureRule.getWeights();

        std::vector<utils::Point> realQuadPoints(numberBaseQuadPoints,
                                                 utils::Point(dim, 0.0));
        cellMapping.getRealPoints(baseQuadratureRuleParametricPoints,
                                  currentCell,
                                  realQuadPoints);

        std::vector<utils::Point> parametricQuadPointsGlobalCell(
          numberBaseQuadPoints, utils::Point(dim, 0.0));

        std::vector<bool> arePointsInside(numberBaseQuadPoints, false);
        cellMapping.getParametricPoints(realQuadPoints,
                                        globalCell,
                                        parametricQuadPointsGlobalCell,
                                        arePointsInside);

        bool areAllPointsInside = false;
        if (std::all_of(arePointsInside.begin(),
                        arePointsInside.end(),
                        [](bool x) { return x; }))
          {
            areAllPointsInside = true;
          }

        utils::throwException(
          areAllPointsInside,
          "In the construction of the adaptive quadrature,"
          "one or more quadrature point in a child cell is found"
          "to be outside the global cell.");

        std::vector<double> weightsOne(numberBaseQuadPoints, 1.0);
        std::vector<double> globalCellJacobian(numberBaseQuadPoints, 0.0);
        cellMapping.getJxW(globalCell,
                           parametricQuadPointsGlobalCell,
                           weightsOne,
                           globalCellJacobian);

        std::vector<double> weightsGlobalCell(numberBaseQuadPoints, 0.0);
        for (unsigned int iPoint; iPoint < numberBaseQuadPoints; ++iPoint)
          {
            weightsGlobalCell[iPoint] =
              currentCellJxW[iPoint] / globalCellJacobian[iPoint];
          }

        adaptiveQuadPoints.insert(adaptiveQuadPoints.end(),
                                  parametricQuadPointsGlobalCell.begin(),
                                  parametricQuadPointsGlobalCell.end());

        adaptiveQuadWeights.insert(adaptiveQuadWeights.end(),
                                   weightsGlobalCell.begin(),
                                   weightsGlobalCell.end());
      }


      bool
      haveIntegralsConverged(
        const std::vector<double> &             parentCellIntegralValues,
        const std::vector<double> &             parentCellIntegralThresholds,
        const std::vector<std::vector<double>> &childCellsIntegralValues,
        const std::vector<double> &             tolerances)
      {
        bool            returnValue     = true;
        const size_type numberFunctions = parentCellIntegralValues.size();
        const size_type numberChildren  = childCellsIntegralValues.size();
        for (unsigned int iFunction = 0; iFunction < numberFunctions;
             ++iFunction)
          {
            const double parentIntegral = parentCellIntegralValues[iFunction];
            if (fabs(parentIntegral) > parentCellIntegralThresholds[iFunction])
              {
                double sumChildIntegrals = 0.0;
                for (unsigned int iChild = 0; iChild < numberChildren; ++iChild)
                  {
                    sumChildIntegrals +=
                      childCellsIntegralValues[iChild][iFunction];
                  }

                const double diff = fabs(sumChildIntegrals - parentIntegral);
                if (diff > tolerances[iFunction])
                  {
                    returnValue = false;
                    break;
                  }
              }
          }

        return returnValue;
      }

      inline void
      recursiveIntegrate(
        const basis::TriangulationCellBase &parentCell,
        const std::vector<double> &         parentCellIntegralValues,
        const std::vector<double> &         parenCellIntegralThresholds,
        const double                        parentVolume,
        const std::vector<double> &         tolerances,
        const double                        smallestCellVolume,
        int &                               recursionLevel,
        const int                           maxRecursion,
        std::vector<std::shared_ptr<const ScalarFunction>> functions,
        const basis::TriangulatioCellBase &                globalCell,
        const QuadratureRule &                             baseQuadratureRule,
        const basis::CellMappingBase &                     cellMapping,
        const std::vector<double> &                        parentCellJxW,
        std::vector<utils::Point> &                        adaptiveQuadPoints,
        std::vector<double> &                              adaptiveQuadWeights)

      {
        //
        const size_type    numberBaseQuadPoints = baseQuadratureRule.nPoints();
        const size_type    numberFunctions      = functions.size();
        const unsigned int dim                  = baseQuadratureRule.getDim();
        const std::vector<utils::Point> &baseQuadratureRuleParametricPoints =
          baseQuadratureRule.getPoints();

        const std::vector<double> &baseQuadratureRuleWeights =
          baseQuadratureRule.getWeights();

        if (parentVolume < smallestCellVolume || recursionLevel > maxRecursion)
          {
            updateAdaptiveQuadratureRule(parentCell,
                                         globalCell,
                                         baseQuadratureRule,
                                         cellMapping,
                                         parentCellJxW,
                                         adaptiveQuadPoints,
                                         adaptiveQuadWeights);
          }

        else
          {
            const int                        numberChildren = 8;
            std::vector<std::vector<double>> childCellsIntegralValues(
              numberChildren, std::vector<double>(numberFunctions, 0.0));

            std::vector<std::vector<double>> childCellsJxW(
              numberChildren, std::vector<double>(numberBaseQuadPoints, 0.0));

            std::vector<double> childCellsVolume(numberChildren, 0.0);

            std::vector<std::shared_ptr<const basis::TriangulationCellBase>>
              childCells(numberChildren);

            createChildCells(parentCell, childCells);

            for (unsigned int iChild = 0; iChild < numberChildren; iChild++)
              {
                basis::TriangulationCellBase &childCell = *(childCells[iChild]);

                std::vector<utils::Point> realQuadPoints(numberBaseQuadPoints,
                                                         utils::Point(dim,
                                                                      0.0));
                cellMapping.getRealPoints(baseQuadratureRuleParametricPoints,
                                          childCell,
                                          realQuadPoints);

                std::vector<double> &childCellJxW = childCellsJxW[iChild];
                cellMapping.getJxW(childCell,
                                   baseQuadratureRuleParametricPoints,
                                   baseQuadratureRuleWeights,
                                   childCellJxW);

                childCellsVolume[iChild] =
                  std::accumulate(childJxW.begin(), childJxW.end(), 0.0);

                for (unsigned int iFunction = 0; iFunction < numberFunctions;
                     ++iFunction)
                  {
                    std::shared_ptr<const ScalarFunction> function =
                      functions[iFunction];
                    std::vector<double> functionValues(numberBaseQuadPoints,
                                                       0.0);
                    function->getValue(realQuadPoints, functionValues);
                    childCellsIntegralValues[iChild][iFunction] =
                      std::inner_product(functionValues.begin(),
                                         functionValues.end(),
                                         childCellJxW.begin(),
                                         0.0);
                  }
              }

            bool convergenceFlag =
              haveIntegralsConverged(parentCellIntegralValues,
                                     parentCellIntegralThresholds,
                                     chilCellsIntegralValues,
                                     tolerances);
            if (convergenceFlag)
              {
                updateAdaptiveQuadratureRule(parentCell,
                                             globalCell,
                                             baseQuadratureRule,
                                             cellMapping,
                                             parentCellJxW,
                                             adaptiveQuadPoints,
                                             adaptiveQuadWeights);

                destroyChildCells(childCells);
              }

            else
              {
                for (unsigned int iChild = 0; iChild < numberChildren; ++iChild)
                  {
                    recursionLevel = recursionLevel + 1;
                    recursiveIntegrate(childCells[iChild],
                                       childCellsIntegralValues[iChild],
                                       parentCellIntegralThresholds,
                                       childCellsVolume[iChild],
                                       tolerances,
                                       smallestCellVolume,
                                       recursionLevel,
                                       maxRecursion,
                                       functions,
                                       globalCell,
                                       baseQuadratureRule,
                                       cellMapping,
                                       childCellsJxW[iChild],
                                       adaptiveQuadPoints,
                                       adaptiveQuadWeights)

                      destroyChildCell(childCells[iChild]);
                  }
              }
          }
      }


    } // namespace

    QuadratureRuleAdaptive::QuadratureRuleAdaptive(
      const basis::TriangulationCellBase &               cell,
      const QuadratureRule &                             baseQuadratureRule,
      const basis::CellMappingBase &                     cellMapping,
      std::vector<std::shared_ptr<const ScalarFunction>> functions,
      const std::vector<double> &                        tolerances,
      const std::vector<double> &                        integralThresholds,
      const double       smallestCellVolume /*= 1e-12*/,
      const unsigned int maxRecursion /*= 100*/)
    {
      d_dim                = baseQuadratureRule.getDim();
      d_isTensorStructured = false;
      d_points.resize(0, utils::Point(d_dim, 0.0));
      d_weights.resize(0);

      const size_type numberBaseQuadPoints = baseQuadratureRule.nPoints();
      const size_type numberFunctions      = functions.size();

      const std::vector<utils::Point> &baseQuadratureRuleParametricPoints =
        baseQuadratureRule.getPoints();

      const std::vector<double> &baseQuadratureRuleWeights =
        baseQuadratureRule.getWeights();

      std::vector<double> cellJxW(numberBaseQuadPoints, 0.0);
      cellMapping.getJxW(cell,
                         baseQuadratureRuleParametricPoints,
                         baseQuadratureRuleWeights,
                         cellJxW);

      std::vector<utils::Point> realQuadPoints(numberBaseQuadPoints,
                                               utils::Point(d_dim, 0.0));
      cellMapping.getRealPoints(baseQuadratureRuleParametricPoints,
                                cell,
                                realQuadPoints);

      const double cellVolume =
        std::accumulate(cellJxW.begin(), cellJxW.end(), 0.0);

      std::vector<double> integralValues(numberFunctions, 0.0);
      for (unsigned int iFunction = 0; iFunction < numberFunctions; ++iFunction)
        {
          std::vector<double> functionValues(numberBaseQuadPoints, 0.0);
          function->getValue(realQuadPoints, functionValues);
          integralValues[iFunction] = std::inner_product(functionValues.begin(),
                                                         functionValues.end(),
                                                         cellJxW.begin(),
                                                         0.0);
        }

      int recursionLevel = 0;
      recursiveIntegrate(cell,
                         integralValues,
                         integralThresholds,
                         cellVolume,
                         tolerances,
                         smallestCellVolume,
                         recursionLevel,
                         maxRecursion,
                         functions,
                         cell,
                         baseQuadratureRule,
                         cellMapping,
                         cellJxW,
                         d_points,
                         d_weights);

      d_numPoints = d_weights.size();
    }


  } // end of namespace quadrature
} // end of namespace dftefe
