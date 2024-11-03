#include <basis/TriangulationBase.h>
#include <basis/TriangulationDealiiParallel.h>
#include <basis/CellMappingBase.h>
#include <basis/LinearCellMappingDealii.h>
#include <basis/EFEBasisDofHandlerDealii.h>
#include <basis/EFEConstraintsLocalDealii.h>
#include <basis/EFEBasisDataStorageDealii.h>
#include <basis/CFEBDSOnTheFlyComputeDealii.h>
#include <basis/CFEBasisDataStorageDealii.h>
#include <basis/FEBasisOperations.h>
#include <basis/FEBasisManager.h>
#include <quadrature/QuadratureAttributes.h>
#include <quadrature/QuadratureRuleGauss.h>
#include <quadrature/QuadratureRuleAdaptive.h>
#include <atoms/AtomSevereFunction.h>
#include <utils/Point.h>
#include <utils/TypeConfig.h>
#include <utils/MemorySpaceType.h>
#include <utils/MemoryStorage.h>
#include <vector>
#include <fstream>
#include <memory>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h> 
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/dofs/dof_tools.h>
#include <iostream>
#include <basis/OrthoEFEOverlapInverseOpContextGLL.h>
using namespace dftefe;
int main()
{

  std::cout<<" Entering test data storage \n";

  //initialize MPI
  // NOTE : The test case only works for orthogonalized EFE basis

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
    <utils::MemorySpace::HOST>> blasQueuePtr = std::make_shared
      <linearAlgebra::blasLapack::BlasQueue
        <utils::MemorySpace::HOST>>(blasQueue);
  std::shared_ptr<linearAlgebra::blasLapack::LapackQueue
    <utils::MemorySpace::HOST>> lapackQueuePtr = std::make_shared
      <linearAlgebra::blasLapack::LapackQueue
        <utils::MemorySpace::HOST>>(lapackQueue);
  std::shared_ptr<linearAlgebra::LinAlgOpContext
    <utils::MemorySpace::HOST>> linAlgOpContext = 
    std::make_shared<linearAlgebra::LinAlgOpContext
    <utils::MemorySpace::HOST>>(blasQueuePtr, lapackQueuePtr);
    
  // Set up Triangulation
  const unsigned int dim = 3;
  std::shared_ptr<basis::TriangulationBase> triangulationBase =
  std::make_shared<basis::TriangulationDealiiParallel<dim>>(comm);
  std::vector<unsigned int>         subdivisions = {5, 5, 5};
  std::vector<bool>                 isPeriodicFlags(dim, false);
  std::vector<utils::Point> domainVectors(dim,
                                                  utils::Point(dim, 0.0));

  double xmax = 10.;
  double ymax = 10.;
  double zmax = 10.;
  unsigned int numComponents = 1;
  bool isAdaptiveGrid = false;

  domainVectors[0][0] = xmax;
  domainVectors[1][1] = ymax;
  domainVectors[2][2] = zmax;

  // initialize the triangulation
  triangulationBase->initializeTriangulationConstruction();
  triangulationBase->createUniformParallelepiped(subdivisions,
                                                  domainVectors,
                                                  isPeriodicFlags);
  triangulationBase->finalizeTriangulationConstruction();

  // Enrichment data file consisting of g(r,\theta,\phi) = f(r)*Y_lm(\theta, \phi)
  std::string sourceDir = "/home/avirup/dft-efe/test/basis/src/";
  std::string atomDataFile = "AtomData.in";
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

  std::vector<std::string> fieldNames{"vnuclear"};
  std::vector<std::string> metadataNames{ "symbol", "Z", "charge", "NR", "r" };
  std::shared_ptr<atoms::AtomSphericalDataContainer>  atomSphericalDataContainer = 
      std::make_shared<atoms::AtomSphericalDataContainer>(atomSymbolToFilename,
                                                      fieldNames,
                                                      metadataNames);

  std::string fieldName = "vnuclear";
  double atomPartitionTolerance = 1e-6;

  if(isAdaptiveGrid)
  {
    int flag = 1;
    int mpiReducedFlag = 1;
    bool refineFlag = true;
    while(mpiReducedFlag)
    {
      flag = 1;
      auto triaCellIter = triangulationBase->beginLocal();
      for( ; triaCellIter != triangulationBase->endLocal(); triaCellIter++)
      {
        refineFlag = false;
        utils::Point centerPoint(dim, 0.0); 
        (*triaCellIter)->center(centerPoint);
        double dist = (centerPoint[0] - 5)* (centerPoint[0] - 5);  
        dist += (centerPoint[1] - 5)* (centerPoint[1] - 5);
        dist += (centerPoint[2] - 5)* (centerPoint[2] - 5);
        dist = std::sqrt(dist); 
        if((dist < 1.0) || centerPoint[0] < 1.0)
          refineFlag = true;
        if ( refineFlag )
        {
          (*triaCellIter)->setRefineFlag();
          flag = 0;
        }
      }
      triangulationBase->executeCoarseningAndRefinement();
      triangulationBase->finalizeTriangulationConstruction();
      // Mpi_allreduce that all the flags are 1 (mpi_max)
      int err = utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
        &flag,
        &mpiReducedFlag,
        1,
        utils::mpi::MPIInt,
        utils::mpi::MPIMin,
        comm);
      std::pair<bool, std::string> mpiIsSuccessAndMsg =
        utils::mpi::MPIErrIsSuccessAndMsg(err);
      utils::throwException(mpiIsSuccessAndMsg.first,
                            "MPI Error:" + mpiIsSuccessAndMsg.second);
    }
  }
  // Make orthogonalized EFE basis

  unsigned int feOrder = 3;

    // Set up the vector of scalarSpatialRealFunctions for adaptive quadrature
    std::vector<std::shared_ptr<const utils::ScalarSpatialFunctionReal>> functionsVec(0);
    unsigned int numfun = 2;
    functionsVec.resize(numfun); // Enrichment Functions
    std::vector<double> tolerances(numfun);
    std::vector<double> integralThresholds(numfun);
    for ( unsigned int i=0 ;i < functionsVec.size() ; i++ )
    {
        functionsVec[i] = std::make_shared<atoms::AtomSevereFunction<dim>>(
            atomSphericalDataContainer,
            atomSymbolVec,
            atomCoordinatesVec,
            fieldName,
            i);
        tolerances[i] = 1e3;
        integralThresholds[i] = 1e3;
    }

    double smallestCellVolume = 1e-12;
    unsigned int maxRecursion = 1000;

    //Set up quadAttr for Rhs and OverlapMatrix

    quadrature::QuadratureRuleAttributes quadAttrAdaptive(quadrature::QuadratureFamily::ADAPTIVE,false);

    // Set up base quadrature rule for adaptive quadrature 

    unsigned int num1DGaussSize = feOrder + 1;
    std::shared_ptr<quadrature::QuadratureRule> baseQuadRule =
    std::make_shared<quadrature::QuadratureRuleGauss>(dim, num1DGaussSize);

    quadrature::QuadratureRuleAttributes quadAttrGauss(quadrature::QuadratureFamily::GAUSS,true,num1DGaussSize);

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
      tolerances,
      tolerances,
      integralThresholds,
      smallestCellVolume,
      maxRecursion);

    // Set the CFE basis manager and handler for bassiInterfaceCoeffcient distributed vector
  std::shared_ptr<const basis::FEBasisDofHandler<double, utils::MemorySpace::HOST,dim>> cfeBasisDofHandler =  
   std::make_shared<basis::CFEBasisDofHandlerDealii<double, utils::MemorySpace::HOST,dim>>(triangulationBase, feOrder, comm);

  basis::BasisStorageAttributesBoolMap basisAttrMap;
  basisAttrMap[basis::BasisStorageAttributes::StoreValues] = true;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradient] = true;
  basisAttrMap[basis::BasisStorageAttributes::StoreHessian] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreOverlap] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradNiGradNj] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreJxW] = false;

    // Set up the CFE Basis Data Storage for Rhs
    std::shared_ptr<basis::FEBasisDataStorage<double, utils::MemorySpace::HOST>> cfeBasisDataStorageAdaptUniformQuad =
      std::make_shared<basis::CFEBasisDataStorageDealii<double, double,utils::MemorySpace::HOST, dim>>
      (cfeBasisDofHandler, quadAttrAdaptive, basisAttrMap);
  // evaluate basis data
  cfeBasisDataStorageAdaptUniformQuad->evaluateBasisData(quadAttrAdaptive, quadRuleContainerAdaptive, basisAttrMap);
    
    unsigned int cellBlockSize = 10; 

    // Set up the CFE Basis Data Storage for Rhs
    std::shared_ptr<basis::FEBasisDataStorage<double, utils::MemorySpace::HOST>> cfeBasisDataStorageUniformQuad =
      std::make_shared<basis::CFEBDSOnTheFlyComputeDealii<double, double,utils::MemorySpace::HOST, dim>>
      (cfeBasisDofHandler, quadAttrGauss, basisAttrMap, cellBlockSize, *linAlgOpContext);
  // check basis data
  cfeBasisDataStorageUniformQuad->evaluateBasisData(quadAttrGauss, basisAttrMap);
    size_type numLocallyOwnedCells = cfeBasisDofHandler->nLocallyOwnedCells();
    size_type nDofs = cfeBasisDofHandler->nCellDofs(0);
    size_type nQuadPtInCell = cfeBasisDataStorageUniformQuad->getQuadratureRuleContainer()->nCellQuadraturePoints(0);
    for(int i = 0 ; i < numLocallyOwnedCells ; i += cellBlockSize)
    {
      size_type cellStartId = i;
      size_type cellEndId = std::min(i+cellBlockSize, numLocallyOwnedCells);
      utils::MemoryStorage<double,
          utils::MemorySpace::HOST> 
            basisDataStorageInCellRange((cellEndId - cellStartId)*nDofs*nQuadPtInCell);
      cfeBasisDataStorageUniformQuad->getBasisDataInCellRange(std::make_pair(cellStartId, cellEndId), basisDataStorageInCellRange);
      for(int j = 0 ; j < cellEndId - cellStartId ; j++)
      {
        auto basisDataStorageInCellAdaptUniform = cfeBasisDataStorageAdaptUniformQuad->getBasisDataInCell(j);
        for(int k = 0 ; k < basisDataStorageInCellAdaptUniform.size() ; k++)
        {
          double val1 = *(basisDataStorageInCellRange.data() + k);
          double val2 = *(basisDataStorageInCellAdaptUniform.data() + k);
          if(std::abs(val1 - val2) > 1e-12)
            std::cout << val1 << "\t" << val2 << "\n";
        }
      }
    }
  // check basis gradient data
    for(int i = 0 ; i < numLocallyOwnedCells ; i += cellBlockSize)
    {
      size_type cellStartId = i;
      size_type cellEndId = std::min(i+cellBlockSize, numLocallyOwnedCells);
      utils::MemoryStorage<double,
          utils::MemorySpace::HOST> 
            basisGradientDataStorageInCellRange((cellEndId - cellStartId)*nDofs*nQuadPtInCell*dim);
      cfeBasisDataStorageUniformQuad->getBasisGradientDataInCellRange(std::make_pair(cellStartId, cellEndId), basisGradientDataStorageInCellRange);
      for(int j = 0 ; j < cellEndId - cellStartId ; j++)
      {
        auto basisGradientDataStorageInCellAdaptUniform = cfeBasisDataStorageAdaptUniformQuad->getBasisGradientDataInCell(j);
        for(int k = 0 ; k < basisGradientDataStorageInCellAdaptUniform.size() ; k++)
        {
          double val1 = *(basisGradientDataStorageInCellRange.data() + k);
          double val2 = *(basisGradientDataStorageInCellAdaptUniform.data() + k);
          if(std::abs(val1 - val2) > 1e-12)
            std::cout << val1 << "\t" << val2 << "\n";
        }
      }
    }

    // Create the enrichmentClassicalInterface object
      std::shared_ptr<basis::EnrichmentClassicalInterfaceSpherical<double, utils::MemorySpace::HOST, dim>>
        enrichClassIntfce = std::make_shared<basis::EnrichmentClassicalInterfaceSpherical
                          <double, utils::MemorySpace::HOST, dim>>
                          (cfeBasisDataStorageUniformQuad,
                          cfeBasisDataStorageUniformQuad,
                          atomSphericalDataContainer,
                          atomPartitionTolerance,
                          atomSymbolVec,
                          atomCoordinatesVec,
                          fieldName,
                          linAlgOpContext,
                          comm);

  // initialize the basis 
  std::shared_ptr<basis::FEBasisDofHandler<double, utils::MemorySpace::HOST,dim>> basisDofHandler =  
    std::make_shared<basis::EFEBasisDofHandlerDealii<double, double,utils::MemorySpace::HOST,dim>>(
      enrichClassIntfce, comm);

  std::map<global_size_type, utils::Point> dofCoords;
  basisDofHandler->getBasisCenters(dofCoords);

  std::cout << "Locally owned cells : " <<basisDofHandler->nLocallyOwnedCells() << "\n";
  std::cout << "Total Number of dofs : " << basisDofHandler->nGlobalNodes() << "\n";

  basisAttrMap[basis::BasisStorageAttributes::StoreValues] = true;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradient] = true;
  basisAttrMap[basis::BasisStorageAttributes::StoreHessian] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreOverlap] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradNiGradNj] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreJxW] = false;

  // Set up Adaptive quadrature for EFE Basis Data Storage
  std::shared_ptr<basis::FEBasisDataStorage<double, utils::MemorySpace::HOST>> efeBasisDataStorageAdaptUniformQuad =
    std::make_shared<basis::EFEBasisDataStorageDealii<double, double, utils::MemorySpace::HOST,dim>>
    (basisDofHandler, quadAttrAdaptive, basisAttrMap);

  // evaluate basis data
  efeBasisDataStorageAdaptUniformQuad->evaluateBasisData(quadAttrAdaptive, quadRuleContainerAdaptive, basisAttrMap);

    // Set up the CFE Basis Data Storage for Rhs
    std::shared_ptr<basis::FEBasisDataStorage<double, utils::MemorySpace::HOST>> efeBasisDataStorageUniformQuad =
      std::make_shared<basis::EFEBDSOnTheFlyComputeDealii<double, double,utils::MemorySpace::HOST, dim>>
      (basisDofHandler, quadAttrGauss, basisAttrMap, cellBlockSize, *linAlgOpContext);
  // check basis data
  efeBasisDataStorageUniformQuad->evaluateBasisData(quadAttrGauss, basisAttrMap);
    numLocallyOwnedCells = basisDofHandler->nLocallyOwnedCells();
    nDofs = basisDofHandler->nCellDofs(0);
    nQuadPtInCell = efeBasisDataStorageUniformQuad->getQuadratureRuleContainer()->nCellQuadraturePoints(0);
    for(int i = 0 ; i < numLocallyOwnedCells ; i += cellBlockSize)
    {
      size_type cellStartId = i;
      size_type cellEndId = std::min(i+cellBlockSize, numLocallyOwnedCells);
      utils::MemoryStorage<double,
          utils::MemorySpace::HOST> 
            basisDataStorageInCellRange((cellEndId - cellStartId)*nDofs*nQuadPtInCell);
      efeBasisDataStorageUniformQuad->getBasisDataInCellRange(std::make_pair(cellStartId, cellEndId), basisDataStorageInCellRange);
      for(int j = 0 ; j < cellEndId - cellStartId ; j++)
      {
        auto basisDataStorageInCellAdaptUniform = efeBasisDataStorageAdaptUniformQuad->getBasisDataInCell(j);
        for(int k = 0 ; k < basisDataStorageInCellAdaptUniform.size() ; k++)
        {
          double val1 = *(basisDataStorageInCellRange.data() + k);
          double val2 = *(basisDataStorageInCellAdaptUniform.data() + k);
          if(std::abs(val1 - val2) > 1e-12)
            std::cout << val1 << "\t" << val2 << "\n";
        }
      }
    }
  // check basis gradient data
    for(int i = 0 ; i < numLocallyOwnedCells ; i += cellBlockSize)
    {
      size_type cellStartId = i;
      size_type cellEndId = std::min(i+cellBlockSize, numLocallyOwnedCells);
      utils::MemoryStorage<double,
          utils::MemorySpace::HOST> 
            basisGradientDataStorageInCellRange((cellEndId - cellStartId)*nDofs*nQuadPtInCell*dim);
      efeBasisDataStorageUniformQuad->getBasisGradientDataInCellRange(std::make_pair(cellStartId, cellEndId), basisGradientDataStorageInCellRange);
      for(int j = 0 ; j < cellEndId - cellStartId ; j++)
      {
        auto basisGradientDataStorageInCellAdaptUniform = efeBasisDataStorageAdaptUniformQuad->getBasisGradientDataInCell(j);
        for(int k = 0 ; k < basisGradientDataStorageInCellAdaptUniform.size() ; k++)
        {
          double val1 = *(basisGradientDataStorageInCellRange.data() + k);
          double val2 = *(basisGradientDataStorageInCellAdaptUniform.data() + k);
          if(std::abs(val1 - val2) > 1e-12)
            std::cout << val1 << "\t" << val2 << "\n";
        }
      }
    }

  utils::mpi::MPIBarrier(comm);

  //gracefully end MPI

  int mpiFinalFlag = 0;
  utils::mpi::MPIFinalized(&mpiFinalFlag);
  if(!mpiFinalFlag)
  {
    utils::mpi::MPIFinalize();
  }
}
