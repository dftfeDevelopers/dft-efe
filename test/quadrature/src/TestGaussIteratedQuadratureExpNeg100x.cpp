#include <basis/TriangulationBase.h>
#include <basis/TriangulationDealiiParallel.h>
#include <basis/CellMappingBase.h>
#include <basis/LinearCellMappingDealii.h>
#include <basis/ParentToChildCellsManagerBase.h>
#include <basis/ParentToChildCellsManagerDealii.h>
#include <quadrature/QuadratureRule.h>
#include <quadrature/QuadratureRuleGauss.h>
#include <quadrature/QuadratureRuleGaussIterated.h>
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
#include <utils/MPITypes.h>
#include <utils/MPIWrapper.h>

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

  int mpiInitFlag = 0;
  dftefe::utils::mpi::MPIInitialized(&mpiInitFlag);
  if(!mpiInitFlag)
  {
      dftefe::utils::mpi::MPIInit(NULL, NULL);
  }
  
  dftefe::utils::mpi::MPIComm mpi_communicator = dftefe::utils::mpi::MPICommWorld;

  std::shared_ptr<dftefe::basis::TriangulationBase> triangulationBase =
    std::make_shared<dftefe::basis::TriangulationDealiiParallel<3>>(mpi_communicator);

  std::vector<unsigned int>         subdivisions = {10, 10, 10};
  std::vector<bool>                 isPeriodicFlags(3, false);
  std::vector<dftefe::utils::Point> domainVectors(3,
      dftefe::utils::Point(3, 0.0));
  domainVectors[0][0] = 5.0;
  domainVectors[1][1] = 5.0;
  domainVectors[2][2] = 5.0;

  std::vector<double> origin(0);
  origin.resize(3);
  for(unsigned int i = 0 ; i < 3 ; i++)
    origin[i] = -domainVectors[i][i]*0.5;

  triangulationBase->initializeTriangulationConstruction();
  triangulationBase->createUniformParallelepiped(subdivisions,
      domainVectors,
      isPeriodicFlags);
  triangulationBase->shiftTriangulation(dftefe::utils::Point(origin));
  triangulationBase->finalizeTriangulationConstruction();
  std::cout << triangulationBase->nLocallyOwnedCells() << std::endl;

  dftefe::basis::ParentToChildCellsManagerBase *parentToChildCellsManager =
    new dftefe::basis::ParentToChildCellsManagerDealii<3>();

  dftefe::basis::CellMappingBase *mapping =
    new dftefe::basis::LinearCellMappingDealii<3>();

  std::shared_ptr<dftefe::quadrature::QuadratureRule> quadratureRuleGauss =
    std::make_shared<dftefe::quadrature::QuadratureRuleGauss>(3, 6);

  unsigned int iCell = 0;

  std::vector<std::shared_ptr<const dftefe::utils::ScalarSpatialFunctionReal>>
    functions(1, std::make_shared<dftefe::utils::ExpModX>(0,-100.0));

  std::vector<double> tolerances(1, 1e-10);
  std::vector<double> integralThresholds(1, 1e-10);
  const double        smallestCellVolume = 1e-12;
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
      tolerances,
      integralThresholds,
      smallestCellVolume,
      maxRecursion);

  std::cout << "Quadrature Adaptive totoal points: "<< adaptiveQuadratureContainer.nQuadraturePoints()<<std::endl;

  dftefe::quadrature::QuadratureRuleAttributes quadAttr1(dftefe::quadrature::QuadratureFamily::GAUSS_SUBDIVIDED,true);
    dftefe::quadrature::QuadratureRuleContainer gaussIteratedQuadratureContainer(
      quadAttr1,
      6,
      20,
      4,
      triangulationBase,
      *mapping,
      functions,
      tolerances,
      tolerances,
      adaptiveQuadratureContainer,
      mpi_communicator);

  double integral = 0.0,mpiReducedIntegral = 0.0;
  dftefe::quadrature::integrate(*functions[0],
      gaussIteratedQuadratureContainer,
      integral);

  unsigned int quadPoints = gaussIteratedQuadratureContainer.nQuadraturePoints(),
        mpiReducedQuadPoints = 0.0;
  dftefe::utils::mpi::MPIAllreduce<dftefe::utils::MemorySpace::HOST>(
        &quadPoints,
        &mpiReducedQuadPoints,
        1,
        dftefe::utils::mpi::MPIUnsigned,
        dftefe::utils::mpi::MPISum,
        mpi_communicator);

  dftefe::utils::mpi::MPIAllreduce<dftefe::utils::MemorySpace::HOST>(
        &integral,
        &mpiReducedIntegral,
        1,
        dftefe::utils::mpi::MPIDouble,
        dftefe::utils::mpi::MPISum,
        mpi_communicator);

  quadPoints = adaptiveQuadratureContainer.nQuadraturePoints();
  unsigned int mpiReducedQuadPointsReference = 0.0;
  dftefe::utils::mpi::MPIAllreduce<dftefe::utils::MemorySpace::HOST>(
        &quadPoints,
        &mpiReducedQuadPointsReference,
        1,
        dftefe::utils::mpi::MPIUnsigned,
        dftefe::utils::mpi::MPISum,
        mpi_communicator);

  double integralReference = 0.0,mpiReducedIntegralReference = 0.0;
  dftefe::quadrature::integrate(*functions[0],
      adaptiveQuadratureContainer,
      integralReference);

  dftefe::utils::mpi::MPIAllreduce<dftefe::utils::MemorySpace::HOST>(
        &integralReference,
        &mpiReducedIntegralReference,
        1,
        dftefe::utils::mpi::MPIDouble,
        dftefe::utils::mpi::MPISum,
        mpi_communicator);
  
  int rank;
  dftefe::utils::mpi::MPICommRank(mpi_communicator, &rank);
  if(rank == 0)
  {
    std::ofstream out;
    out.open("outTestGaussIteratedQuadratureExpNeg100x");
    out << std::setprecision(16);
    out << "NumTotalQuadPoints (GaussIterated/Adaptive): ";
    out << mpiReducedQuadPoints <<"/" << mpiReducedQuadPointsReference <<std::endl ;
    out << "Values (GaussIterated/Adaptive): ";
    out << mpiReducedIntegral <<"/" << mpiReducedIntegralReference <<std::endl ;
    out.close();
    //auto printFile = [&out](const int& n) { out << " " << n; };
    //std::cout << "Values:";
    //auto printStd = [](const int& n) { std::cout << " " << n; };
  }
  
  delete mapping;
  delete parentToChildCellsManager;

  int mpiFinalFlag = 0;
  dftefe::utils::mpi::MPIFinalized(&mpiFinalFlag);
  if(!mpiFinalFlag)
  {
    dftefe::utils::mpi::MPIFinalize();
  }
}
