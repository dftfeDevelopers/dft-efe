#include <basis/TriangulationBase.h>
#include <basis/TriangulationDealiiSerial.h>
#include <basis/CellMappingBase.h>
#include <basis/LinearCellMappingDealii.h>
#include <basis/ParentToChildCellsManagerBase.h>
#include <basis/ParentToChildCellsManagerDealii.h>
#include <quadrature/QuadratureRule.h>
#include <quadrature/QuadratureRuleGauss.h>
#include <quadrature/QuadratureRuleContainer.h>
#include <utils/Point.h>
#include <utils/TypeConfig.h>
#include <utils/Function.h>
#include <utils/ScalarSpatialFunction.h>
#include <utils/LogModX.h>
#include <vector>
#include <memory>

void
printVertices(const std::vector<dftefe::utils::Point> &points)
{
  const unsigned int N = points.size();
  for (unsigned int i = 0; i < N; ++i)
    std::cout << points[i][0] << "\t" << points[i][1] << "\t" << points[i][2]
              << std::endl;
}

int
main()
{
  std::shared_ptr<dftefe::basis::TriangulationBase> triangulationBase =
    std::make_shared<dftefe::basis::TriangulationDealiiSerial<3>>();

  std::vector<unsigned int>         subdivisions = {1, 1, 1};
  std::vector<bool>                 isPeriodicFlags(3, false);
  std::vector<dftefe::utils::Point> domainVectors(3,
                                                  dftefe::utils::Point(3, 0.0));
  domainVectors[0][0] = 10.0;
  domainVectors[1][1] = 10.0;
  domainVectors[2][2] = 10.0;
  triangulationBase->initializeTriangulationConstruction();
  triangulationBase->createUniformParallelepiped(subdivisions,
                                                 domainVectors,
                                                 isPeriodicFlags);
  triangulationBase->finalizeTriangulationConstruction();
  std::cout << triangulationBase->nLocalCells() << std::endl;

  dftefe::basis::ParentToChildCellsManagerBase *parentToChildCellsManager =
    new dftefe::basis::ParentToChildCellsManagerDealii<3>();

  dftefe::basis::CellMappingBase *mapping =
    new dftefe::basis::LinearCellMappingDealii<3>();

  std::shared_ptr<dftefe::quadrature::QuadratureRule> quadratureRuleGauss =
    std::make_shared<dftefe::quadrature::QuadratureRuleGauss>(3, 4);
  const std::vector<dftefe::utils::Point> &parametricPoints =
    quadratureRuleGauss->getPoints();
  const std::vector<double> &weights = quadratureRuleGauss->getWeights();

  const unsigned int numGaussQuadPoints = parametricPoints.size();
  for (unsigned int iQuad = 0; iQuad < numGaussQuadPoints; ++iQuad)
    {
      std::cout << iQuad << ": " << parametricPoints[iQuad][0] << " "
                << parametricPoints[iQuad][1] << " "
                << parametricPoints[iQuad][2] << " " << weights[iQuad]
                << std::endl;
    }

  dftefe::quadrature::QuadratureRuleAttributes quadAttr(dftefe::quadrature::QuadratureFamily::GAUSS,true,4);

  dftefe::quadrature::QuadratureRuleContainer quadratureRuleContainer(
    quadAttr, quadratureRuleGauss, triangulationBase, *mapping);

  std::vector<std::shared_ptr<dftefe::basis::TriangulationCellBase>>::iterator it =
    triangulationBase->beginLocal();
  unsigned int iCell = 0;
  for (; it != triangulationBase->endLocal(); ++it)
    {
      std::cout << "Printing for iCell: " << iCell << std::endl;
      const std::vector<dftefe::utils::Point> points =
        quadratureRuleContainer.getCellRealPoints(iCell);
      printVertices(points);
      iCell++;
    }

  std::vector<std::shared_ptr<const dftefe::utils::ScalarSpatialFunctionReal>>
    functions(1, std::make_shared<dftefe::utils::LogModX>(0));

  std::vector<double> tolerances(1, 1e-3);
  std::vector<double> integralThresholds(1, 1e-3);
  const double        smallestCellVolume = 1e-14;
  const unsigned int  maxRecursion       = 100;

  dftefe::quadrature::QuadratureRuleAttributes quadAttrAdaptive(dftefe::quadrature::QuadratureFamily::ADAPTIVE,false);

  dftefe::quadrature::QuadratureRuleContainer adaptiveQuadratureContainer(
    quadAttrAdaptive,
    quadratureRuleGauss,
    triangulationBase,
    *mapping,
    *parentToChildCellsManager,
    functions,
    tolerances,
    tolerances,
    integralThresholds,
    smallestCellVolume,
    maxRecursion);

  const dftefe::size_type numAdaptiveQuadPoints =
    adaptiveQuadratureContainer.nQuadraturePoints();
  std::cout << "Number of adaptive quad points:" << numAdaptiveQuadPoints
            << std::endl;
  // for (; it != triangulationBase->endLocal(); ++it)
  //{
  //  std::vector<dftefe::utils::Point> points(0, dftefe::utils::Point(3, 0.0));
  //  (*it)->getVertices(points);
  //  printVertices(points);

  //  std::vector<std::shared_ptr<const dftefe::basis::TriangulationCellBase>>
  //    childCells = parentToChildCellsManager->createChildCells(*(*it));

  //  std::cout << "Printing for iCell: " << iCell << std::endl;
  //  std::cout << "==========================================================="
  //    << std::endl;
  //  std::cout << "Number child cells: " << childCells.size() << std::endl;
  //  for (unsigned int iChild = 0; iChild < childCells.size(); ++iChild)
  //  {
  //    std::vector<dftefe::utils::Point> points(0,
  //        dftefe::utils::Point(3,
  //          0.0));
  //    childCells[iChild]->getVertices(points);
  //    std::cout << "Printing for iChild: " << iChild << std::endl;
  //    std::cout
  //      << "------------------------------------------------------------"
  //      << std::endl;
  //    printVertices(points);
  //  }

  //  parentToChildCellsManager->popLast();
  //  iCell++;
  //}

  delete mapping;
  delete parentToChildCellsManager;
}
