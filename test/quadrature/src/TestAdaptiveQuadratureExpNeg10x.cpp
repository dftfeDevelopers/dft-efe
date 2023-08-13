#include <basis/TriangulationBase.h>
#include <basis/TriangulationDealiiSerial.h>
#include <basis/CellMappingBase.h>
#include <basis/LinearCellMappingDealii.h>
#include <basis/ParentToChildCellsManagerBase.h>
#include <basis/ParentToChildCellsManagerDealii.h>
#include <quadrature/QuadratureRule.h>
#include <quadrature/QuadratureRuleGauss.h>
#include <quadrature/QuadratureRuleContainer.h>
#include <quadrature/Integrate.h>
#include <utils/Point.h>
#include <utils/TypeConfig.h>
#include <utils/Function.h>
#include <utils/ScalarSpatialFunction.h>
#include <utils/ExpModX.h>
#include <vector>
#include <memory>
#include <iomanip>

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
  domainVectors[0][0] = 1.0;
  domainVectors[1][1] = 1.0;
  domainVectors[2][2] = 1.0;
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
    std::make_shared<dftefe::quadrature::QuadratureRuleGauss>(3, 6);

  unsigned int iCell = 0;

  std::vector<std::shared_ptr<const dftefe::utils::ScalarSpatialFunctionReal>>
    functions(1, std::make_shared<dftefe::utils::ExpModX>(0,-10.0));

  std::vector<double> tolerances(1, 1e-10);
  std::vector<double> integralThresholds(1, 1e-14);
  const double        smallestCellVolume = 1e-14;
  const unsigned int  maxRecursion       = 100;

  dftefe::quadrature::QuadratureRuleAttributes quadAttr(dftefe::quadrature::QuadratureFamily::ADAPTIVE,false);
  dftefe::quadrature::QuadratureRuleContainer adaptiveQuadratureContainer(
      quadAttr,
      quadratureRuleGauss,
      triangulationBase,
      *mapping,
      *parentToChildCellsManager,
      functions,
      tolerances,
      integralThresholds,
      smallestCellVolume,
      maxRecursion);

  double integral = 0.0;
  dftefe::quadrature::integrate(*functions[0],
      adaptiveQuadratureContainer,
      integral);
  
  std::ofstream out;
  out.open("outTestAdaptiveQuadratureExpNeg10x");
  out << std::setprecision(16);
  out << "Values: ";
  out << integral;
  out.close();
  //auto printFile = [&out](const int& n) { out << " " << n; };
  //std::cout << "Values:";
  //auto printStd = [](const int& n) { std::cout << " " << n; };
  
  delete mapping;
  delete parentToChildCellsManager;
}
