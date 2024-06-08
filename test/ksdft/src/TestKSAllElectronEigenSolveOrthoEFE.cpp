#include <basis/TriangulationBase.h>
#include <basis/TriangulationDealiiParallel.h>
#include <basis/CellMappingBase.h>
#include <basis/LinearCellMappingDealii.h>
#include <basis/CFEBasisDofHandlerDealii.h>
#include <basis/CFEBasisDataStorageDealii.h>
#include <basis/EFEBasisDofHandlerDealii.h>
#include <basis/EFEBasisDataStorageDealii.h>
#include <basis/FEBasisOperations.h>
#include <basis/CFEConstraintsLocalDealii.h>
#include <basis/FEBasisManager.h>
#include <quadrature/QuadratureAttributes.h>
#include <quadrature/QuadratureRuleGauss.h>
#include <quadrature/QuadratureRuleContainer.h>
#include <quadrature/QuadratureValuesContainer.h>
#include <basis/FECellWiseDataOperations.h>
#include <utils/Point.h>
#include <utils/TypeConfig.h>
#include <utils/MemorySpaceType.h>
#include <utils/MemoryStorage.h>
#include <vector>
#include <cmath>
#include <memory>
#include <linearAlgebra/LinearSolverFunction.h>
#include <electrostatics/PoissonLinearSolverFunctionFE.h>
#include <linearAlgebra/LinearAlgebraProfiler.h>
#include <linearAlgebra/CGLinearSolver.h>
#include <ksdft/ElectrostaticAllElectronFE.h>
#include <ksdft/KineticFE.h>
#include <ksdft/ExchangeCorrelationFE.h>
#include <ksdft/KohnShamOperatorContextFE.h>
#include <ksdft/KohnShamEigenSolver.h>
#include <basis/OrthoEFEOverlapInverseOpContextGLL.h>
#include <utils/PointChargePotentialFunction.h>
#include <ksdft/DensityCalculator.h>
#include <atoms/AtomSevereFunction.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/dofs/dof_tools.h>

#include <iostream>

using namespace dftefe;
const utils::MemorySpace Host = utils::MemorySpace::HOST;

// e- charge density
double rho1sOrbital(const dftefe::utils::Point &point, const std::vector<dftefe::utils::Point> &origin)
{
  double ret = 0;
  for (unsigned int i = 0 ; i < origin.size() ; i++ )
  {
    double r = 0;
    for (unsigned int j = 0 ; j < point.size() ; j++ )
    {
      r += std::pow((point[j]-origin[i][j]),2);
    }
    r = std::sqrt(r);
    ret += (1/M_PI)*exp(-2*r);
  }
  return ret;
}


// basisDofHandler for wave function
// basisDofHandler for b_smear
// single adaptive Quadrature with b_smear (highest) - same for Hamiltonian

int main()
{
  std::cout<<" Entering test kohn sham eigensolve ortho enrichment\n";
  //initialize MPI

  int mpiInitFlag = 0;
  utils::mpi::MPIInitialized(&mpiInitFlag);
  if(!mpiInitFlag)
  {
    utils::mpi::MPIInit(NULL, NULL);
  }

  utils::mpi::MPIComm comm = utils::mpi::MPICommWorld;

    // Get the rank of the process
  int rank;
  utils::mpi::MPICommRank(comm, &rank);

    // Get nProcs
    int numProcs;
    utils::mpi::MPICommSize(comm, &numProcs);

  int blasQueue = 0;
  int lapackQueue = 0;
  std::shared_ptr<linearAlgebra::blasLapack::BlasQueue
    <Host>> blasQueuePtr = std::make_shared
      <linearAlgebra::blasLapack::BlasQueue
        <Host>>(blasQueue);
  std::shared_ptr<linearAlgebra::blasLapack::LapackQueue
    <Host>> lapackQueuePtr = std::make_shared
      <linearAlgebra::blasLapack::LapackQueue
        <Host>>(lapackQueue);
  std::shared_ptr<linearAlgebra::LinAlgOpContext
    <Host>> linAlgOpContext = 
    std::make_shared<linearAlgebra::LinAlgOpContext
    <Host>>(blasQueuePtr, lapackQueuePtr);

  // Set up Triangulation
  const unsigned int dim = 3;
  double xmax = 24.0;
  double ymax = 24.0;
  double zmax = 24.0;
  double rc = 0.6;
  double hMin = 1e6;
  size_type maxIter = 2e7;
  double absoluteTol = 1e-10;
  double relativeTol = 1e-12;
  double divergenceTol = 1e10;
  double refineradius = 3*rc;
  unsigned int feOrder = 3;
  unsigned int num1DGaussSize = 4;
  unsigned int num1DGllSize = 4;

  double    smearingTemperature = 500.0;
  double    fermiEnergyTolerance = 1e-10;
  double    fracOccupancyTolerance = 1e-3;
  double    eigenSolveResidualTolerance = 1e-3;
  size_type chebyshevPolynomialDegree = 50;
  size_type maxChebyshevFilterPass = 100;
  size_type numWantedEigenvalues = 15;
  size_type numElectrons = 1;
  double nuclearCharge = -1.0;
  
  // Set up Triangulation
    std::shared_ptr<basis::TriangulationBase> triangulationBase =
        std::make_shared<basis::TriangulationDealiiParallel<dim>>(comm);
  std::vector<unsigned int>         subdivisions = {15, 15, 15};
  std::vector<bool>                 isPeriodicFlags(dim, false);
  std::vector<utils::Point> domainVectors(dim, utils::Point(dim, 0.0));

  domainVectors[0][0] = xmax;
  domainVectors[1][1] = ymax;
  domainVectors[2][2] = zmax;

  std::vector<double> origin(0);
  origin.resize(dim);
  for(unsigned int i = 0 ; i < dim ; i++)
    origin[i] = -domainVectors[i][i]*0.5;

  // initialize the triangulation
  triangulationBase->initializeTriangulationConstruction();
  triangulationBase->createUniformParallelepiped(subdivisions,
                                                 domainVectors,
                                                 isPeriodicFlags);
  triangulationBase->shiftTriangulation(utils::Point(origin));
    triangulationBase->finalizeTriangulationConstruction();

    char* dftefe_path = getenv("DFTEFE_PATH");
    std::string sourceDir;
    // if executes if a non null value is returned
    // otherwise else executes
    if (dftefe_path != NULL) 
    {
      sourceDir = (std::string)dftefe_path + "/test/ksdft/src/";
    }
    else
    {
      utils::throwException(false,
                            "dftefe_path does not exist!");
    }
    std::string atomDataFile = "SingleAtomData.in";
    std::string inputFileName = sourceDir + atomDataFile;

  std::fstream fstream;
  fstream.open(inputFileName, std::fstream::in);
  
    // read the input file and create atomSymbolVec vector and atom coordinates vector.
    std::vector<utils::Point> atomCoordinatesVec;
    std::vector<double> coordinates;
    coordinates.resize(dim,0.);
    std::vector<std::string> atomSymbolVec;
    std::string symbol;
    atomSymbolVec.resize(0);
    std::string line;
    while (std::getline(fstream, line)){
        std::stringstream ss(line);
        ss >> symbol; 
        for(unsigned int i=0 ; i<dim ; i++){
            ss >> coordinates[i]; 
        }
        atomCoordinatesVec.push_back(coordinates);
        atomSymbolVec.push_back(symbol);
    }
    utils::mpi::MPIBarrier(comm);
        
    std::map<std::string, std::string> atomSymbolToFilename;
    for (auto i:atomSymbolVec )
    {
        atomSymbolToFilename[i] = sourceDir + i + ".xml";
    }

    std::vector<std::string> fieldNames{"vnuclear","orbital"};
    std::vector<std::string> metadataNames{ "symbol", "Z", "charge", "NR", "r" };
    std::shared_ptr<atoms::AtomSphericalDataContainer>  atomSphericalDataContainer = 
        std::make_shared<atoms::AtomSphericalDataContainer>(atomSymbolToFilename,
                                                        fieldNames,
                                                        metadataNames);

    std::string vTotal = "vnuclear";
    std::string orbital = "orbital";
    double atomPartitionTolerance = 1e-6;

    std::vector<double> atomChargesVec(atomCoordinatesVec.size(), nuclearCharge);
    std::vector<double> smearedChargeRadiusVec(atomCoordinatesVec.size(),rc);

  int flag = 1;
  int mpiReducedFlag = 1;
  bool radiusRefineFlag = true;
  while(mpiReducedFlag)
  {
    flag = 0;
    auto triaCellIter = triangulationBase->beginLocal();
    for( ; triaCellIter != triangulationBase->endLocal(); triaCellIter++)
    {
      radiusRefineFlag = false;
      (*triaCellIter)->clearRefineFlag();
      utils::Point centerPoint(dim, 0.0); 
      (*triaCellIter)->center(centerPoint);
      for ( unsigned int i=0 ; i<atomCoordinatesVec.size() ; i++)
      {
        double dist = 0;
        for (unsigned int j = 0 ; j < dim ; j++ )
        {
          dist += std::pow((centerPoint[j]-atomCoordinatesVec[i][j]),2);
        }
        dist = std::sqrt(dist);
        if(dist < refineradius)
          radiusRefineFlag = true;
      }
      if (radiusRefineFlag && (*triaCellIter)->diameter() > hMin)
      {
        (*triaCellIter)->setRefineFlag();
        flag = 1;
      }
    }
    triangulationBase->executeCoarseningAndRefinement();
    triangulationBase->finalizeTriangulationConstruction();
    // Mpi_allreduce that all the flags are 1 (mpi_max)
    int err = utils::mpi::MPIAllreduce<Host>(
      &flag,
      &mpiReducedFlag,
      1,
      utils::mpi::MPIInt,
      utils::mpi::MPIMax,
      comm);
    std::pair<bool, std::string> mpiIsSuccessAndMsg =
      utils::mpi::MPIErrIsSuccessAndMsg(err);
    utils::throwException(mpiIsSuccessAndMsg.first,
                          "MPI Error:" + mpiIsSuccessAndMsg.second);
  }

  // Make orthogonalized EFE basis

  // 1. Make EnrichmentClassicalInterface object for Pristine enrichment
  // 2. Make CFEBasisDataStorageDealii object for Rhs (ADAPTIVE with GAUSS and fns are N_i^2 - make quadrulecontainer), overlapmatrix (GAUSS)
  // 3. Make EnrichmentClassicalInterface object for Orthogonalized enrichment
  // 4. Input to the EFEBasisDofHandler(eci, feOrder) 
  // 5. Make EFEBasisDataStorage with input as quadratureContainer.

  std::shared_ptr<basis::EnrichmentClassicalInterfaceSpherical
                          <double, Host, dim>>
                          enrichClassIntfceTotalPot = std::make_shared<basis::EnrichmentClassicalInterfaceSpherical
                          <double, Host, dim>>(triangulationBase,
                          atomSphericalDataContainer,
                          atomPartitionTolerance,
                          atomSymbolVec,
                          atomCoordinatesVec,
                          vTotal,
                          comm);

    // Set up the vector of scalarSpatialRealFunctions for adaptive quadrature
    std::vector<std::shared_ptr<const utils::ScalarSpatialFunctionReal>> functionsVec(0);
    unsigned int numfun = 2;
    functionsVec.resize(numfun); // Enrichment Functions
    std::vector<double> absoluteTolerances(numfun), relativeTolerances(numfun);
    std::vector<double> integralThresholds(numfun);
    for ( unsigned int i=0 ;i < functionsVec.size() ; i++ )
    {
        functionsVec[i] = std::make_shared<atoms::AtomSevereFunction<dim>>(        
            enrichClassIntfceTotalPot->getEnrichmentIdsPartition(),
            atomSphericalDataContainer,
            atomSymbolVec,
            atomCoordinatesVec,
            vTotal,
            i);
        absoluteTolerances[i] = 1e-4;
        relativeTolerances[i] = 1e-4;
        integralThresholds[i] = 1e-10;
    }

    double smallestCellVolume = 1e-12;
    unsigned int maxRecursion = 1000;

    //Set up quadAttr for Rhs and OverlapMatrix

    quadrature::QuadratureRuleAttributes quadAttrAdaptive(quadrature::QuadratureFamily::ADAPTIVE,false);

    quadrature::QuadratureRuleAttributes quadAttrGll(quadrature::QuadratureFamily::GLL,true,num1DGllSize);

    // Set up base quadrature rule for adaptive quadrature 

    std::shared_ptr<quadrature::QuadratureRule> baseQuadRule =
      std::make_shared<quadrature::QuadratureRuleGauss>(dim, num1DGaussSize);

    std::shared_ptr<basis::CellMappingBase> cellMapping = std::make_shared<basis::LinearCellMappingDealii<dim>>();
    std::shared_ptr<basis::ParentToChildCellsManagerBase> parentToChildCellsManager = std::make_shared<basis::ParentToChildCellsManagerDealii<dim>>();

    std::shared_ptr<quadrature::QuadratureRuleContainer> quadRuleContainerAdaptive =
      std::make_shared<quadrature::QuadratureRuleContainer>
      (quadAttrAdaptive, 
      baseQuadRule, 
      triangulationBase, 
      *cellMapping, 
      *parentToChildCellsManager,
      functionsVec,
      absoluteTolerances,
      relativeTolerances,
      integralThresholds,
      smallestCellVolume,
      maxRecursion);

    // Set the CFE basis manager and handler for bassiInterfaceCoeffcient distributed vector
  std::shared_ptr<const basis::FEBasisDofHandler<double, Host,dim>> cfeBasisDofHandler =  
   std::make_shared<basis::CFEBasisDofHandlerDealii<double, Host,dim>>(triangulationBase, feOrder, comm);

  basis::BasisStorageAttributesBoolMap basisAttrMap;
  basisAttrMap[basis::BasisStorageAttributes::StoreValues] = true;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradient] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreHessian] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreOverlap] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradNiGradNj] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreJxW] = true;

    // Set up the CFE Basis Data Storage for Overlap Matrix
    std::shared_ptr<basis::FEBasisDataStorage<double, Host>> cfeBasisDataStorageGLL =
      std::make_shared<basis::CFEBasisDataStorageDealii<double, double,Host, dim>>
      (cfeBasisDofHandler, quadAttrGll, basisAttrMap);
  // evaluate basis data
  cfeBasisDataStorageGLL->evaluateBasisData(quadAttrGll, basisAttrMap);

    // Set up the CFE Basis Data Storage for Rhs
    std::shared_ptr<basis::FEBasisDataStorage<double, Host>> cfeBasisDataStorageAdaptive =
      std::make_shared<basis::CFEBasisDataStorageDealii<double, double,Host, dim>>
      (cfeBasisDofHandler, quadAttrAdaptive, basisAttrMap);
  // evaluate basis data
  cfeBasisDataStorageAdaptive->evaluateBasisData(quadAttrAdaptive, quadRuleContainerAdaptive, basisAttrMap);

    // Create the enrichmentClassicalInterface object for vtotal
    enrichClassIntfceTotalPot = std::make_shared<basis::EnrichmentClassicalInterfaceSpherical
                          <double, Host, dim>>
                          (cfeBasisDataStorageGLL,
                          cfeBasisDataStorageAdaptive,
                          atomSphericalDataContainer,
                          atomPartitionTolerance,
                          atomSymbolVec,
                          atomCoordinatesVec,
                          vTotal,
                          linAlgOpContext,
                          comm);

    // Create the enrichmentClassicalInterface object for wavefn
      std::shared_ptr<basis::EnrichmentClassicalInterfaceSpherical
                          <double, Host, dim>>
                          enrichClassIntfceOrbital = std::make_shared<basis::EnrichmentClassicalInterfaceSpherical
                          <double, Host, dim>>
                          (cfeBasisDataStorageGLL,
                          cfeBasisDataStorageAdaptive,
                          atomSphericalDataContainer,
                          atomPartitionTolerance,
                          atomSymbolVec,
                          atomCoordinatesVec,
                          orbital,
                          linAlgOpContext,
                          comm);                          

  // initialize the basis Manager

  std::shared_ptr<basis::FEBasisDofHandler<double, Host,dim>> basisDofHandlerTotalPot =  
    std::make_shared<basis::EFEBasisDofHandlerDealii<double, double,Host,dim>>(
      enrichClassIntfceTotalPot, feOrder, comm);

  std::shared_ptr<basis::FEBasisDofHandler<double, Host,dim>> basisDofHandlerOrbital =  
    std::make_shared<basis::EFEBasisDofHandlerDealii<double, double,Host,dim>>(
      enrichClassIntfceOrbital, feOrder, comm);

  std::map<global_size_type, utils::Point> dofCoords;
  basisDofHandlerTotalPot->getBasisCenters(dofCoords);

  std::cout << "Locally owned cells : " <<basisDofHandlerTotalPot->nLocallyOwnedCells() << "\n";
  std::cout << "Total Number of dofs : " << basisDofHandlerTotalPot->nGlobalNodes() << "\n";

  // Set up the quadrature rule

  basisAttrMap[basis::BasisStorageAttributes::StoreValues] = true;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradient] = true;
  basisAttrMap[basis::BasisStorageAttributes::StoreHessian] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreOverlap] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradNiGradNj] = true;
  basisAttrMap[basis::BasisStorageAttributes::StoreJxW] = true;

  // Set up Adaptive quadrature for EFE Basis Data Storage
  std::shared_ptr<basis::FEBasisDataStorage<double, Host>> efeBasisDataAdaptive =
  std::make_shared<basis::EFEBasisDataStorageDealii<double, double, Host,dim>>
  (basisDofHandlerTotalPot, quadAttrAdaptive, basisAttrMap);

  // evaluate basis data
  efeBasisDataAdaptive->evaluateBasisData(quadAttrAdaptive, quadRuleContainerAdaptive, basisAttrMap);

  // std::shared_ptr<basis::FEBasisDataStorage<double, Host>> efeBasisDataAdaptiveTotPot = efeBasisDataAdaptive;

  basisAttrMap[basis::BasisStorageAttributes::StoreValues] = true;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradient] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreHessian] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreOverlap] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradNiGradNj] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreJxW] = true;

  std::shared_ptr<basis::FEBasisDataStorage<double, Host>> efeBasisDataOrbitalAdaptive =
  std::make_shared<basis::EFEBasisDataStorageDealii<double, double, Host,dim>>
  (basisDofHandlerOrbital, quadAttrAdaptive, basisAttrMap);

  // evaluate basis data
  efeBasisDataOrbitalAdaptive->evaluateBasisData(quadAttrAdaptive, quadRuleContainerAdaptive, basisAttrMap);

  // std::shared_ptr<basis::FEBasisDataStorage<double, Host>> efeBasisDataAdaptiveHamiltonian = efeBasisDataAdaptive;

  std::shared_ptr<const quadrature::QuadratureRuleContainer> quadRuleContainer =  
                efeBasisDataAdaptive->getQuadratureRuleContainer();

    std::vector<double> atomChargesVecScaled(atomCoordinatesVec.size());
  for(size_type i = 0 ; i < atomCoordinatesVec.size() ; i++)
  {
    std::vector<double> chargeDensity(1, 0.0), mpiReducedChargeDensity(1, 0.0);

    const utils::SmearChargeDensityFunction smden(atomCoordinatesVec[i],
                                                  atomChargesVec[i],
                                                  smearedChargeRadiusVec[i]);

      double charge = 0;
      for(size_type i = 0 ; i < quadRuleContainer->nCells() ; i++)
      {
        std::vector<double> JxW = quadRuleContainer->getCellJxW(i);
        size_type quadId = 0;
        for (auto j : quadRuleContainer->getCellRealPoints(i))
        {
          charge += smden(j) * JxW[quadId];
          quadId = quadId + 1;
        }
      }
      chargeDensity[0] = charge;
    
    utils::mpi::MPIAllreduce<Host>(
          chargeDensity.data(),
          mpiReducedChargeDensity.data(),
          chargeDensity.size(),
          utils::mpi::MPIDouble,
          utils::mpi::MPISum,
          comm);

    atomChargesVecScaled[i] *= 1/std::abs(mpiReducedChargeDensity[0]);
  }

  quadRuleContainer =  
                efeBasisDataOrbitalAdaptive->getQuadratureRuleContainer();

   quadrature::QuadratureValuesContainer<double, Host> 
      electronChargeDensity(quadRuleContainer, 1, 0.0);

    std::shared_ptr<const utils::ScalarSpatialFunctionReal>
          zeroFunction = std::make_shared
            <utils::ScalarZeroFunctionReal>();

    std::shared_ptr<const basis::FEBasisManager
      <double, double, Host,dim>>
    basisManagerWaveFn = std::make_shared
      <basis::FEBasisManager<double, double, Host,dim>>
        (basisDofHandlerOrbital);

    std::shared_ptr<const utils::ScalarSpatialFunctionReal> smfunc =
      std::make_shared<const utils::SmearChargePotentialFunction>(
        atomCoordinatesVec,
        atomChargesVecScaled,
        smearedChargeRadiusVec);

    std::shared_ptr<const basis::FEBasisManager
      <double, double, Host,dim>>
    basisManagerTotalPot = std::make_shared
      <basis::FEBasisManager<double, double, Host,dim>>
        (basisDofHandlerTotalPot, smfunc);

  std::shared_ptr<ksdft::KineticFE<double,
                                    double,
                                    Host,
                                    dim>> 
                                  hamitonianKin =
    std::make_shared<ksdft::KineticFE<double,
                                      double,
                                      Host,
                                      dim>>
                                      (efeBasisDataAdaptive,
                                      linAlgOpContext,
                                      50);

  const utils::ScalarSpatialFunctionReal *externalPotentialFunction = new 
    utils::PointChargePotentialFunction(atomCoordinatesVec, atomChargesVec);

  std::shared_ptr<ksdft::ElectrostaticAllElectronFE<double,
                                                  double,
                                                  double,
                                                  Host,
                                                  dim>> 
                                            hamitonianElec =
    std::make_shared<ksdft::ElectrostaticAllElectronFE<double,
                                                  double,
                                                  double,
                                                  Host,
                                                  dim>>
                                                  (atomCoordinatesVec,
                                                  atomChargesVec,
                                                  smearedChargeRadiusVec,
                                                  electronChargeDensity,
                                                  basisManagerTotalPot,
                                                  efeBasisDataAdaptive,
                                                  efeBasisDataAdaptive,
                                                  efeBasisDataAdaptive,
                                                  *externalPotentialFunction,
                                                  linAlgOpContext,
                                                  50);
                                                  

    using HamiltonianPtrVariant =
      std::variant<ksdft::Hamiltonian<float, Host> *,
                    ksdft::Hamiltonian<double, Host> *,
                    ksdft::Hamiltonian<std::complex<float>, Host> *,
                    ksdft::Hamiltonian<std::complex<double>, Host> *>;

  std::vector<HamiltonianPtrVariant> hamiltonianComponentsVec{hamitonianKin.get(), hamitonianElec.get()};
  // form the kohn sham operator
  std::shared_ptr<linearAlgebra::OperatorContext<double,
                                                  double,
                                                  Host>> 
                                            hamitonianOperator =
    std::make_shared<ksdft::KohnShamOperatorContextFE<double,
                                                  double,
                                                  double,
                                                  Host,
                                                  dim>>
                                                  (*basisManagerWaveFn,
                                                  hamiltonianComponentsVec,
                                                  *linAlgOpContext,
                                                  50);

  // call the eigensolver

  linearAlgebra::MultiVector<double, Host> waveFunctionSubspaceGuess(basisManagerWaveFn->getMPIPatternP2P(),
                                                                     linAlgOpContext,
                                                                     numWantedEigenvalues,
                                                                     0.0,
                                                                     1.0);    

  linearAlgebra::Vector<double, Host> lanczosGuess(basisManagerWaveFn->getMPIPatternP2P(),
                                                    linAlgOpContext,
                                                    0.0,
                                                    1.0); 

  lanczosGuess.updateGhostValues();
  basisManagerWaveFn->getConstraints().distributeParentToChild(lanczosGuess, 1);

  waveFunctionSubspaceGuess.updateGhostValues();
  basisManagerWaveFn->getConstraints().distributeParentToChild(waveFunctionSubspaceGuess, numWantedEigenvalues);

  std::vector<double> kohnShamEnergies(numWantedEigenvalues, 0.0);
  linearAlgebra::MultiVector<double, Host> kohnShamWaveFunctions(waveFunctionSubspaceGuess, 0.0);

  // Create OperatorContext for Basisoverlap

  std::shared_ptr<const basis::OrthoEFEOverlapOperatorContext<double,
                                                double,
                                                Host,
                                                dim>> MContext =
  std::make_shared<basis::OrthoEFEOverlapOperatorContext<double,
                                                      double,
                                                      Host,
                                                      dim>>(
                                                      *basisManagerWaveFn,
                                                      *basisManagerWaveFn,
                                                      *cfeBasisDataStorageAdaptive,
                                                      *efeBasisDataOrbitalAdaptive,
                                                      *cfeBasisDataStorageGLL,
                                                      50); 

    std::shared_ptr<const basis::OrthoEFEOverlapOperatorContext<double,
                                                  double,
                                                  Host,
                                                  dim>> MContextForInv =
    std::make_shared<basis::OrthoEFEOverlapOperatorContext<double,
                                                        double,
                                                        Host,
                                                        dim>>(
                                                        *basisManagerWaveFn,
                                                        *basisManagerWaveFn,
                                                        *cfeBasisDataStorageGLL,
                                                        *efeBasisDataOrbitalAdaptive,
                                                        *cfeBasisDataStorageGLL,
                                                        50);                                                      

  std::shared_ptr<linearAlgebra::OperatorContext<double,
                                                   double,
                                                   Host>> MInvContext =
    std::make_shared<basis::OrthoEFEOverlapInverseOpContextGLL<double,
                                                   double,
                                                   Host,
                                                   dim>>
                                                   (*basisManagerWaveFn,
                                                    *cfeBasisDataStorageGLL,
                                                    *efeBasisDataOrbitalAdaptive,
                                                    linAlgOpContext);                                                  

  // form the kohn sham operator
  std::shared_ptr<ksdft::KohnShamEigenSolver<double,
                                              double,
                                              Host>> 
                                            ksEigSolve =
    std::make_shared<ksdft::KohnShamEigenSolver<double,
                                                double,
                                                Host>>
                                                (numElectrons,
                                                smearingTemperature,
                                                fermiEnergyTolerance,
                                                fracOccupancyTolerance,
                                                eigenSolveResidualTolerance,
                                                chebyshevPolynomialDegree,
                                                maxChebyshevFilterPass,
                                                waveFunctionSubspaceGuess,
                                                lanczosGuess,
                                                50,
                                                *MContextForInv,
                                                *MInvContext);

  linearAlgebra::EigenSolverError err = ksEigSolve->solve(*hamitonianOperator, 
                    kohnShamEnergies, 
                    kohnShamWaveFunctions,
                    true,
                    *MContext,
                    *MInvContext);       

  for(auto &i : kohnShamEnergies)
    std::cout <<  i <<", ";     
  std::cout << "\n";     

  std::cout << err.msg << "\n";      

  ksdft::DensityCalculator<double, double, Host, dim>
                              densCalc(efeBasisDataOrbitalAdaptive,
                              *basisManagerWaveFn,
                              linAlgOpContext,
                              50,
                              4);

  quadrature::QuadratureValuesContainer<double, Host>
                  rho(quadRuleContainer, 1);

  std::vector<double> occupation = ksEigSolve->getFractionalOccupancy();
  densCalc.computeRho(occupation, kohnShamWaveFunctions, rho);
  std::vector<double> intRho(2);
  for (size_type iCell = 0; iCell < rho.nCells(); iCell++)
    {
      const std::vector<double> JxW = quadRuleContainer->getCellJxW(iCell);
      std::vector<double> a(quadRuleContainer->nCellQuadraturePoints(iCell));
      const std::vector<dftefe::utils::Point> point = quadRuleContainer->getCellRealPoints(iCell);
      rho.getCellQuadValues<Host>(iCell, 0, a.data());
      for (size_type i = 0; i < a.size(); i++)
      {
        intRho[0] += std::pow((a[i] - rho1sOrbital(point[i], atomCoordinatesVec)),2) * JxW[i];
        intRho[1] += std::pow(rho1sOrbital(point[i], atomCoordinatesVec),2) * JxW[i];
      }
    }

  int mpierr = utils::mpi::MPIAllreduce<Host>(
    utils::mpi::MPIInPlace,
    intRho.data(),
    intRho.size(),
    utils::mpi::Types<double>::getMPIDatatype(),
    utils::mpi::MPISum,
    comm);

  std::cout << "Absolute L2 norm rho : "<< intRho[0] << "Relative L2 norm rho : "<<intRho[0]/intRho[1]<<"\n";

  hamitonianKin->evalEnergy(occupation, *basisManagerWaveFn, kohnShamWaveFunctions, 4);

  std::cout << "kin energy: "<<hamitonianKin->getEnergy() << "\n";

  hamitonianElec->evalEnergy(); 
  double elecEnergy = hamitonianElec->getEnergy();
  std::cout << "elec energy: " << elecEnergy << "\n";

  // calculate band energy
  double bandEnergy = 0;
  for(size_type i = 0 ; i < occupation.size(); i++)
  {
    bandEnergy += 2 * occupation[i] * kohnShamEnergies[i];
  }

  std::cout << "band energy: "<< bandEnergy << "\n";

  double totalEnergy = bandEnergy - elecEnergy;

  std::cout << "Total Energy: " << totalEnergy << "\n";       

  //gracefully end MPI

  int mpiFinalFlag = 0;
  utils::mpi::MPIFinalized(&mpiFinalFlag);
  if(!mpiFinalFlag)
  {
    utils::mpi::MPIFinalize();
  }
}
