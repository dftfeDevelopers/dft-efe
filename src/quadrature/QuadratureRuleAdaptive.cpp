#include <utils/Exceptions.h>
#include <quadrature/QuadratureRuleAdaptive.h>
#include <numeric>
#include <functional>
#include <algorithm>
#include <iomanip>

namespace dftefe
{
  namespace quadrature
  {
    namespace
    {
      //
      // Returns power of integer raised to a positive integer
      // C++ standard library deals only with floats and doubles
      size_type
      intPowPositiveInt(int base, unsigned int exp)
      {
        size_type result = 1;
        for (;;)
          {
            if (exp & 1)
              result *= base;
            exp >>= 1;
            if (!exp)
              break;
            base *= base;
          }

        return result;
      }

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
        for (unsigned int iPoint = 0; iPoint < numberBaseQuadPoints; ++iPoint)
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
        const std::vector<double> &         parentCellIntegralThresholds,
        const double                        parentVolume,
        const std::vector<double> &         tolerances,
        const double                        smallestCellVolume,
        const unsigned int                  recursionLevel,
        const unsigned int                  maxRecursion,
        std::vector<std::shared_ptr<const utils::ScalarSpatialFunctionReal>>
                                              functions,
        const basis::TriangulationCellBase &  globalCell,
        const QuadratureRule &                baseQuadratureRule,
        const basis::CellMappingBase &        cellMapping,
        basis::ParentToChildCellsManagerBase &parentToChildCellsManager,
        const std::vector<double> &           parentCellJxW,
        std::vector<utils::Point> &           adaptiveQuadPoints,
        std::vector<double> &                 adaptiveQuadWeights,
        std::vector<double> &                 integrals)

      {
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

            for (unsigned int iFunction = 0; iFunction < numberFunctions;
                 ++iFunction)
              integrals[iFunction] += parentCellIntegralValues[iFunction];
          }

        else
          {
            const size_type numberChildren = intPowPositiveInt(2, dim);
            std::vector<std::vector<double>> childCellsIntegralValues(
              numberChildren, std::vector<double>(numberFunctions, 0.0));

            std::vector<std::vector<double>> childCellsJxW(
              numberChildren, std::vector<double>(numberBaseQuadPoints, 0.0));

            std::vector<double> childCellsVolume(numberChildren, 0.0);

            std::vector<std::shared_ptr<const basis::TriangulationCellBase>>
              childCells =
                parentToChildCellsManager.createChildCells(parentCell);

            utils::throwException(
              numberChildren == childCells.size(),
              "The number of child cells created by ParentToChildCellsManager"
              "should be 2^dim");

            for (unsigned int iChild = 0; iChild < numberChildren; iChild++)
              {
                const basis::TriangulationCellBase &childCell =
                  *(childCells[iChild]);

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

                childCellsVolume[iChild] = std::accumulate(childCellJxW.begin(),
                                                           childCellJxW.end(),
                                                           0.0);

                for (unsigned int iFunction = 0; iFunction < numberFunctions;
                     ++iFunction)
                  {
                    std::shared_ptr<const utils::ScalarSpatialFunctionReal>
                                        function = functions[iFunction];
                    std::vector<double> functionValues =
                      (*function)(realQuadPoints);
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
                                     childCellsIntegralValues,
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

                for (unsigned int iFunction = 0; iFunction < numberFunctions;
                     ++iFunction)
                  integrals[iFunction] += parentCellIntegralValues[iFunction];
              }

            else
              {
                for (unsigned int iChild = 0; iChild < numberChildren; ++iChild)
                  {
                    const unsigned int recursionLevelNext = recursionLevel + 1;
                    recursiveIntegrate(*(childCells[iChild]),
                                       childCellsIntegralValues[iChild],
                                       parentCellIntegralThresholds,
                                       childCellsVolume[iChild],
                                       tolerances,
                                       smallestCellVolume,
                                       recursionLevelNext,
                                       maxRecursion,
                                       functions,
                                       globalCell,
                                       baseQuadratureRule,
                                       cellMapping,
                                       parentToChildCellsManager,
                                       childCellsJxW[iChild],
                                       adaptiveQuadPoints,
                                       adaptiveQuadWeights,
                                       integrals);
                  }
              }

            // delete the last set of child cells created
            parentToChildCellsManager.popLast();
          }
      }


    } // namespace

    QuadratureRuleAdaptive::QuadratureRuleAdaptive(
      const basis::TriangulationCellBase &  cell,
      const QuadratureRule &                baseQuadratureRule,
      const basis::CellMappingBase &        cellMapping,
      basis::ParentToChildCellsManagerBase &parentToChildCellsManager,
      std::vector<std::shared_ptr<const utils::ScalarSpatialFunctionReal>>
                                 functions,
      const std::vector<double> &tolerances,
      const std::vector<double> &integralThresholds,
      const double               smallestCellVolume /*= 1e-12*/,
      const unsigned int         maxRecursion /*= 100*/)
    {
      d_dim                = baseQuadratureRule.getDim();
      d_isTensorStructured = false;
      d_num1DPoints        = 0;
      d_1DPoints.resize(0, utils::Point(0));
      d_1DWeights.resize(0);
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

      std::vector<double> classicalIntegralValues(numberFunctions, 0.0);
      for (unsigned int iFunction = 0; iFunction < numberFunctions; ++iFunction)
        {
          std::shared_ptr<const utils::ScalarSpatialFunctionReal> function =
            functions[iFunction];
          std::vector<double> functionValues = (*function)(realQuadPoints);
          classicalIntegralValues[iFunction] = std::inner_product(
            functionValues.begin(), functionValues.end(), cellJxW.begin(), 0.0);
        }

      int                 recursionLevel = 0;
      std::vector<double> adaptiveIntegralValues(numberFunctions, 0.0);
      recursiveIntegrate(cell,
                         classicalIntegralValues,
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
                         parentToChildCellsManager,
                         cellJxW,
                         d_points,
                         d_weights,
                         adaptiveIntegralValues);

      d_numPoints = d_weights.size();
    }


  } // end of namespace quadrature
} // end of namespace dftefe
