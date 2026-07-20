#include <utils/Exceptions.h>
#include <quadrature/QuadratureRuleAdaptive.h>
#include <utils/MemoryStorage.h>
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
      intPowPositiveInt(int base, dftefe::size_type exp)
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
        const size_type numberBaseQuadPoints = baseQuadratureRule.nPoints();
        const dftefe::size_type          dim = baseQuadratureRule.getDim();
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
        for (dftefe::size_type iPoint = 0; iPoint < numberBaseQuadPoints;
             ++iPoint)
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
        const std::vector<double> &             absoluteTolerances,
        const std::vector<double> &             relativeTolerances)
      {
        bool            returnValue     = true;
        const size_type numberFunctions = parentCellIntegralValues.size();
        const size_type numberChildren  = childCellsIntegralValues.size();
        for (dftefe::size_type iFunction = 0; iFunction < numberFunctions;
             ++iFunction)
          {
            const double parentIntegral = parentCellIntegralValues[iFunction];
            if (fabs(parentIntegral) >
                parentCellIntegralThresholds[iFunction] /
                  (fabs(parentIntegral) + QuadratureRuleAdaptiveDefaults::
                                            INTEGRAL_THRESHOLDS_NORMALIZATION))
              {
                double sumChildIntegrals = 0.0;
                for (dftefe::size_type iChild = 0; iChild < numberChildren;
                     ++iChild)
                  {
                    sumChildIntegrals +=
                      childCellsIntegralValues[iChild][iFunction];
                  }

                const double diff = fabs(sumChildIntegrals - parentIntegral);
                if (diff > std::max(absoluteTolerances[iFunction],
                                    fabs(parentIntegral) *
                                      relativeTolerances[iFunction]))
                  {
                    returnValue = false;
                    break;
                  }
              }
          }

        return returnValue;
      }

      struct CellWork
      {
        // root cells co-own with triangulation (shared_ptr copy from cells vec)
        // child cells naturally owning (from createChildCells)
        // globalCell is always a root mesh cell — raw ptr, always valid
        std::shared_ptr<const basis::TriangulationCellBase> cell;
        const basis::TriangulationCellBase *                globalCell;
        size_type                                           cellIndex;
        std::vector<double>                                 integralValues;
        double                                              volume;
        size_type                                           recursionLevel;
      };

      inline void
      recursiveIntegrate(
        const basis::TriangulationCellBase &parentCell,
        const std::vector<double> &         parentCellIntegralValues,
        const std::vector<double> &         parentCellIntegralThresholds,
        const double                        parentVolume,
        const std::vector<double> &         absoluteTolerances,
        const std::vector<double> &         relativeTolerances,
        const double                        smallestCellVolume,
        const dftefe::size_type             recursionLevel,
        const dftefe::size_type             maxRecursion,
        std::vector<std::shared_ptr<const utils::ScalarSpatialFunctionReal>>
                                              functions,
        const basis::TriangulationCellBase &  globalCell,
        const QuadratureRule &                baseQuadratureRule,
        const basis::CellMappingBase &        cellMapping,
        basis::ParentToChildCellsManagerBase &parentToChildCellsManager,
        const std::vector<double> &           parentCellJxW,
        std::vector<utils::Point> &           adaptiveQuadPoints,
        std::vector<double> &                 adaptiveQuadWeights,
        std::vector<double> &                 integrals,
        std::map<std::string, double> &       timer)

      {
        const size_type numberBaseQuadPoints = baseQuadratureRule.nPoints();
        const size_type numberFunctions      = functions.size();
        const dftefe::size_type          dim = baseQuadratureRule.getDim();
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

            for (dftefe::size_type iFunction = 0; iFunction < numberFunctions;
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

            auto start = std::chrono::high_resolution_clock::now();
            std::vector<std::shared_ptr<const basis::TriangulationCellBase>>
              childCells =
                parentToChildCellsManager.createChildCells(parentCell);
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration =
              std::chrono::duration_cast<std::chrono::microseconds>(stop -
                                                                    start);
            timer["Child Cell Creation"] += duration.count();

            utils::throwException(
              numberChildren == childCells.size(),
              "The number of child cells created by ParentToChildCellsManager"
              "should be 2^dim");

            for (dftefe::size_type iChild = 0; iChild < numberChildren;
                 iChild++)
              {
                const basis::TriangulationCellBase &childCell =
                  *(childCells[iChild]);

                start = std::chrono::high_resolution_clock::now();
                std::vector<utils::Point> realQuadPoints(numberBaseQuadPoints,
                                                         utils::Point(dim,
                                                                      0.0));
                cellMapping.getRealPoints(baseQuadratureRuleParametricPoints,
                                          childCell,
                                          realQuadPoints);
                stop = std::chrono::high_resolution_clock::now();
                duration =
                  std::chrono::duration_cast<std::chrono::microseconds>(stop -
                                                                        start);
                timer["Cell Mapping real"] += duration.count();

                std::vector<double> &childCellJxW = childCellsJxW[iChild];
                start = std::chrono::high_resolution_clock::now();
                cellMapping.getJxW(childCell,
                                   baseQuadratureRuleParametricPoints,
                                   baseQuadratureRuleWeights,
                                   childCellJxW);

                stop = std::chrono::high_resolution_clock::now();
                duration =
                  std::chrono::duration_cast<std::chrono::microseconds>(stop -
                                                                        start);
                timer["Cell Mapping jxw"] += duration.count();

                childCellsVolume[iChild] = std::accumulate(childCellJxW.begin(),
                                                           childCellJxW.end(),
                                                           0.0);

                start = std::chrono::high_resolution_clock::now();
                for (dftefe::size_type iFunction = 0;
                     iFunction < numberFunctions;
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
                stop = std::chrono::high_resolution_clock::now();
                duration =
                  std::chrono::duration_cast<std::chrono::microseconds>(stop -
                                                                        start);
                timer["Function Eval"] += duration.count();
              }

            bool convergenceFlag =
              haveIntegralsConverged(parentCellIntegralValues,
                                     parentCellIntegralThresholds,
                                     childCellsIntegralValues,
                                     absoluteTolerances,
                                     relativeTolerances);

            if (convergenceFlag)
              {
                updateAdaptiveQuadratureRule(parentCell,
                                             globalCell,
                                             baseQuadratureRule,
                                             cellMapping,
                                             parentCellJxW,
                                             adaptiveQuadPoints,
                                             adaptiveQuadWeights);

                for (dftefe::size_type iFunction = 0;
                     iFunction < numberFunctions;
                     ++iFunction)
                  integrals[iFunction] += parentCellIntegralValues[iFunction];
              }

            else
              {
                for (dftefe::size_type iChild = 0; iChild < numberChildren;
                     ++iChild)
                  {
                    const dftefe::size_type recursionLevelNext =
                      recursionLevel + 1;
                    recursiveIntegrate(*(childCells[iChild]),
                                       childCellsIntegralValues[iChild],
                                       parentCellIntegralThresholds,
                                       childCellsVolume[iChild],
                                       absoluteTolerances,
                                       relativeTolerances,
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
                                       integrals,
                                       timer);
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
      const std::vector<double> &    absoluteTolerances,
      const std::vector<double> &    relativeTolerances,
      const std::vector<double> &    integralThresholds,
      std::map<std::string, double> &timer,
      const double                   smallestCellVolume /*= 1e-12*/,
      const dftefe::size_type        maxRecursion /*= 100*/)
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
      for (dftefe::size_type iFunction = 0; iFunction < numberFunctions;
           ++iFunction)
        {
          std::shared_ptr<const utils::ScalarSpatialFunctionReal> function =
            functions[iFunction];
          std::vector<double> functionValues = (*function)(realQuadPoints);
          classicalIntegralValues[iFunction] = std::inner_product(
            functionValues.begin(), functionValues.end(), cellJxW.begin(), 0.0);
        }

      size_type           recursionLevel = 0;
      std::vector<double> adaptiveIntegralValues(numberFunctions, 0.0);
      recursiveIntegrate(cell,
                         classicalIntegralValues,
                         integralThresholds,
                         cellVolume,
                         absoluteTolerances,
                         relativeTolerances,
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
                         adaptiveIntegralValues,
                         timer);

      d_numPoints = d_weights.size();
    }


    template <utils::MemorySpace memorySpace>
    std::vector<QuadratureRule>
    QuadratureRuleAdaptive::QuadratureRuleAdaptiveBFS(
      const std::vector<std::shared_ptr<const basis::TriangulationCellBase>>
        &                                   cells,
      const QuadratureRule &                baseQuadratureRule,
      const basis::CellMappingBase &        cellMapping,
      basis::ParentToChildCellsManagerBase &parentToChildCellsManager,
      std::vector<std::shared_ptr<const utils::ScalarSpatialFunctionReal>>
                                     functions,
      const std::vector<double> &    absoluteTolerances,
      const std::vector<double> &    relativeTolerances,
      const std::vector<double> &    integralThresholds,
      std::map<std::string, double> &timer,
      const double                   smallestCellVolume,
      const dftefe::size_type        maxRecursion)
    {
      const size_type numberCells          = cells.size();
      const size_type numberBaseQuadPoints = baseQuadratureRule.nPoints();
      const size_type numberFunctions      = functions.size();
      const size_type dim                  = baseQuadratureRule.getDim();
      const size_type numberChildren       = intPowPositiveInt(2, dim);

      const std::vector<utils::Point> &baseParamPts =
        baseQuadratureRule.getPoints();
      const std::vector<double> &baseWeights = baseQuadratureRule.getWeights();

      std::vector<std::vector<utils::Point>> cellAdaptiveQuadPoints(
        numberCells);
      std::vector<std::vector<double>> cellAdaptiveQuadWeights(numberCells);

      auto evalFunctions = [&](const std::vector<utils::Point> &allRealPoints,
                               size_type                        totalPoints,
                               size_type                        iFunction,
                               std::vector<double> &functionValues) {
        std::vector<double> flatCoordsHostVec(totalPoints * dim, 0.0);
        for (size_type iPoint = 0; iPoint < totalPoints; ++iPoint)
          for (size_type d = 0; d < dim; ++d)
            flatCoordsHostVec[iPoint * dim + d] = allRealPoints[iPoint][d];
        utils::MemoryStorage<double, memorySpace> flatCoordsStorage(
          totalPoints * dim);
        utils::MemoryStorage<double, memorySpace> outputStorage(totalPoints);
        flatCoordsStorage.copyFrom(flatCoordsHostVec, totalPoints * dim, 0, 0);
        functions[iFunction]->template eval<memorySpace>(
          totalPoints, flatCoordsStorage.data(), outputStorage.data());
        outputStorage.copyTo(functionValues, totalPoints, 0, 0);
      };

      // Depth 0: batch eval for all mesh cells
      std::vector<utils::Point>        allRootRealPoints;
      std::vector<std::vector<double>> rootCellJxW(
        numberCells, std::vector<double>(numberBaseQuadPoints, 0.0));
      std::vector<double> rootCellVolume(numberCells, 0.0);

      allRootRealPoints.reserve(numberCells * numberBaseQuadPoints);
      for (size_type iCell = 0; iCell < numberCells; ++iCell)
        {
          std::vector<utils::Point> cellRealPoints(numberBaseQuadPoints,
                                                   utils::Point(dim, 0.0));
          cellMapping.getRealPoints(baseParamPts,
                                    *cells[iCell],
                                    cellRealPoints);
          cellMapping.getJxW(*cells[iCell],
                             baseParamPts,
                             baseWeights,
                             rootCellJxW[iCell]);
          rootCellVolume[iCell] = std::accumulate(rootCellJxW[iCell].begin(),
                                                  rootCellJxW[iCell].end(),
                                                  0.0);
          allRootRealPoints.insert(allRootRealPoints.end(),
                                   cellRealPoints.begin(),
                                   cellRealPoints.end());
        }

      std::vector<std::vector<double>> rootCellIntegralValues(
        numberCells, std::vector<double>(numberFunctions, 0.0));
      for (size_type iFunction = 0; iFunction < numberFunctions; ++iFunction)
        {
          std::vector<double> functionValues;
          evalFunctions(allRootRealPoints,
                        numberCells * numberBaseQuadPoints,
                        iFunction,
                        functionValues);
          for (size_type iCell = 0; iCell < numberCells; ++iCell)
            {
              const size_type offset = iCell * numberBaseQuadPoints;
              rootCellIntegralValues[iCell][iFunction] =
                std::inner_product(functionValues.begin() + offset,
                                   functionValues.begin() + offset +
                                     numberBaseQuadPoints,
                                   rootCellJxW[iCell].begin(),
                                   0.0);
            }
        }

      // Seed BFS queue
      std::vector<CellWork> currentLevelWork;
      currentLevelWork.reserve(numberCells);
      for (size_type iCell = 0; iCell < numberCells; ++iCell)
        currentLevelWork.push_back({cells[iCell],
                                    cells[iCell].get(),
                                    iCell,
                                    rootCellIntegralValues[iCell],
                                    rootCellVolume[iCell],
                                    0});

      while (!currentLevelWork.empty())
        {
          std::vector<CellWork> nextLevelWork;

          std::vector<CellWork *> terminalCellWork, nonTerminalCellWork;
          for (auto &cellWork : currentLevelWork)
            {
              if (cellWork.volume < smallestCellVolume ||
                  cellWork.recursionLevel > maxRecursion)
                terminalCellWork.push_back(&cellWork);
              else
                nonTerminalCellWork.push_back(&cellWork);
            }

          for (auto *cellWork : terminalCellWork)
            {
              std::vector<double> cellJxW(numberBaseQuadPoints, 0.0);
              cellMapping.getJxW(*cellWork->cell,
                                 baseParamPts,
                                 baseWeights,
                                 cellJxW);
              updateAdaptiveQuadratureRule(
                *cellWork->cell,
                *cellWork->globalCell,
                baseQuadratureRule,
                cellMapping,
                cellJxW,
                cellAdaptiveQuadPoints[cellWork->cellIndex],
                cellAdaptiveQuadWeights[cellWork->cellIndex]);
            }

          if (nonTerminalCellWork.empty())
            break;

          const size_type numberNonTerminalCells = nonTerminalCellWork.size();

          std::vector<
            std::vector<std::shared_ptr<const basis::TriangulationCellBase>>>
            childCellsPerParent(numberNonTerminalCells);
          for (size_type iParent = 0; iParent < numberNonTerminalCells;
               ++iParent)
            childCellsPerParent[iParent] =
              parentToChildCellsManager.createChildCells(
                *nonTerminalCellWork[iParent]->cell);

          std::vector<std::vector<std::vector<utils::Point>>>
            childCellRealPoints(
              numberNonTerminalCells,
              std::vector<std::vector<utils::Point>>(
                numberChildren,
                std::vector<utils::Point>(numberBaseQuadPoints,
                                          utils::Point(dim, 0.0))));
          std::vector<std::vector<std::vector<double>>> childCellsJxW(
            numberNonTerminalCells,
            std::vector<std::vector<double>>(
              numberChildren, std::vector<double>(numberBaseQuadPoints, 0.0)));
          std::vector<std::vector<double>> childCellsVolume(
            numberNonTerminalCells, std::vector<double>(numberChildren, 0.0));

          std::vector<utils::Point> allChildRealPoints;
          allChildRealPoints.reserve(numberNonTerminalCells * numberChildren *
                                     numberBaseQuadPoints);
          for (size_type iParent = 0; iParent < numberNonTerminalCells;
               ++iParent)
            for (size_type iChild = 0; iChild < numberChildren; ++iChild)
              {
                cellMapping.getRealPoints(baseParamPts,
                                          *childCellsPerParent[iParent][iChild],
                                          childCellRealPoints[iParent][iChild]);
                cellMapping.getJxW(*childCellsPerParent[iParent][iChild],
                                   baseParamPts,
                                   baseWeights,
                                   childCellsJxW[iParent][iChild]);
                childCellsVolume[iParent][iChild] =
                  std::accumulate(childCellsJxW[iParent][iChild].begin(),
                                  childCellsJxW[iParent][iChild].end(),
                                  0.0);
                allChildRealPoints.insert(
                  allChildRealPoints.end(),
                  childCellRealPoints[iParent][iChild].begin(),
                  childCellRealPoints[iParent][iChild].end());
              }

          const size_type totalChildPoints =
            numberNonTerminalCells * numberChildren * numberBaseQuadPoints;
          std::vector<std::vector<std::vector<double>>>
            childCellsIntegralValues(
              numberNonTerminalCells,
              std::vector<std::vector<double>>(
                numberChildren, std::vector<double>(numberFunctions, 0.0)));

          for (size_type iFunction = 0; iFunction < numberFunctions;
               ++iFunction)
            {
              std::vector<double> functionValues;
              evalFunctions(allChildRealPoints,
                            totalChildPoints,
                            iFunction,
                            functionValues);
              for (size_type iParent = 0; iParent < numberNonTerminalCells;
                   ++iParent)
                for (size_type iChild = 0; iChild < numberChildren; ++iChild)
                  {
                    const size_type offset =
                      (iParent * numberChildren + iChild) *
                      numberBaseQuadPoints;
                    childCellsIntegralValues[iParent][iChild][iFunction] =
                      std::inner_product(functionValues.begin() + offset,
                                         functionValues.begin() + offset +
                                           numberBaseQuadPoints,
                                         childCellsJxW[iParent][iChild].begin(),
                                         0.0);
                  }
            }

          for (size_type iParent = 0; iParent < numberNonTerminalCells;
               ++iParent)
            {
              if (haveIntegralsConverged(
                    nonTerminalCellWork[iParent]->integralValues,
                    integralThresholds,
                    childCellsIntegralValues[iParent],
                    absoluteTolerances,
                    relativeTolerances))
                {
                  std::vector<double> parentJxW(numberBaseQuadPoints, 0.0);
                  cellMapping.getJxW(*nonTerminalCellWork[iParent]->cell,
                                     baseParamPts,
                                     baseWeights,
                                     parentJxW);
                  updateAdaptiveQuadratureRule(
                    *nonTerminalCellWork[iParent]->cell,
                    *nonTerminalCellWork[iParent]->globalCell,
                    baseQuadratureRule,
                    cellMapping,
                    parentJxW,
                    cellAdaptiveQuadPoints[nonTerminalCellWork[iParent]
                                             ->cellIndex],
                    cellAdaptiveQuadWeights[nonTerminalCellWork[iParent]
                                              ->cellIndex]);
                }
              else
                {
                  for (size_type iChild = 0; iChild < numberChildren; ++iChild)
                    nextLevelWork.push_back(
                      {childCellsPerParent[iParent][iChild],
                       nonTerminalCellWork[iParent]->globalCell,
                       nonTerminalCellWork[iParent]->cellIndex,
                       childCellsIntegralValues[iParent][iChild],
                       childCellsVolume[iParent][iChild],
                       nonTerminalCellWork[iParent]->recursionLevel + 1});
                }
            }

          currentLevelWork = std::move(nextLevelWork);
        }

      std::vector<QuadratureRule> result;
      result.reserve(numberCells);
      for (size_type iCell = 0; iCell < numberCells; ++iCell)
        result.emplace_back(dim,
                            cellAdaptiveQuadPoints[iCell],
                            cellAdaptiveQuadWeights[iCell]);
      return result;
    }

    template std::vector<QuadratureRule>
    QuadratureRuleAdaptive::QuadratureRuleAdaptiveBFS<utils::MemorySpace::HOST>(
      const std::vector<std::shared_ptr<const basis::TriangulationCellBase>> &,
      const QuadratureRule &,
      const basis::CellMappingBase &,
      basis::ParentToChildCellsManagerBase &,
      std::vector<std::shared_ptr<const utils::ScalarSpatialFunctionReal>>,
      const std::vector<double> &,
      const std::vector<double> &,
      const std::vector<double> &,
      std::map<std::string, double> &,
      const double,
      const dftefe::size_type);

#ifdef DFTEFE_WITH_DEVICE
    template std::vector<QuadratureRule>
    QuadratureRuleAdaptive::QuadratureRuleAdaptiveBFS<
      utils::MemorySpace::DEVICE>(
      const std::vector<std::shared_ptr<const basis::TriangulationCellBase>> &,
      const QuadratureRule &,
      const basis::CellMappingBase &,
      basis::ParentToChildCellsManagerBase &,
      std::vector<std::shared_ptr<const utils::ScalarSpatialFunctionReal>>,
      const std::vector<double> &,
      const std::vector<double> &,
      const std::vector<double> &,
      std::map<std::string, double> &,
      const double,
      const dftefe::size_type);
#endif

  } // end of namespace quadrature
} // end of namespace dftefe
