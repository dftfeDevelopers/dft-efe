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

int main()
{

  std::cout<<" Entering test data storage \n";

  //initialize MPI
  // NOTE : The test case only works for orthogonalized EFE basis

  int mpiInitFlag = 0;
  dftefe::utils::mpi::MPIInitialized(&mpiInitFlag);
  if(!mpiInitFlag)
  {
      dftefe::utils::mpi::MPIInit(NULL, NULL);
  }

  dftefe::utils::mpi::MPIComm comm = dftefe::utils::mpi::MPICommWorld;

  // Get the rank of the process
  int rank;
  dftefe::utils::mpi::MPICommRank(comm, &rank);

  // Get nProcs
  int numProcs;
  dftefe::utils::mpi::MPICommSize(comm, &numProcs);

  int blasQueue = 0;
  int lapackQueue = 0;
  std::shared_ptr<dftefe::linearAlgebra::blasLapack::BlasQueue
    <dftefe::utils::MemorySpace::HOST>> blasQueuePtr = std::make_shared
      <dftefe::linearAlgebra::blasLapack::BlasQueue
        <dftefe::utils::MemorySpace::HOST>>(blasQueue);
  std::shared_ptr<dftefe::linearAlgebra::blasLapack::LapackQueue
    <dftefe::utils::MemorySpace::HOST>> lapackQueuePtr = std::make_shared
      <dftefe::linearAlgebra::blasLapack::LapackQueue
        <dftefe::utils::MemorySpace::HOST>>(lapackQueue);
  std::shared_ptr<dftefe::linearAlgebra::LinAlgOpContext
    <dftefe::utils::MemorySpace::HOST>> linAlgOpContext = 
    std::make_shared<dftefe::linearAlgebra::LinAlgOpContext
    <dftefe::utils::MemorySpace::HOST>>(blasQueuePtr, lapackQueuePtr);
    
  // Set up Triangulation
  const unsigned int dim = 3;
  std::shared_ptr<dftefe::basis::TriangulationBase> triangulationBase =
  std::make_shared<dftefe::basis::TriangulationDealiiParallel<dim>>(comm);
  std::vector<unsigned int>         subdivisions = {10, 10, 10};
  std::vector<bool>                 isPeriodicFlags(dim, false);
  std::vector<dftefe::utils::Point> domainVectors(dim,
                                                  dftefe::utils::Point(dim, 0.0));

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
  std::vector<dftefe::utils::Point> atomCoordinatesVec;
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
  dftefe::utils::mpi::MPIBarrier(comm);
      
  std::map<std::string, std::string> atomSymbolToFilename;
  for (auto i:atomSymbolVec )
  {
      atomSymbolToFilename[i] = sourceDir + i + ".xml";
  }

  std::vector<std::string> fieldNames{"vnuclear"};
  std::vector<std::string> metadataNames{ "symbol", "Z", "charge", "NR", "r" };
  std::shared_ptr<dftefe::atoms::AtomSphericalDataContainer>  atomSphericalDataContainer = 
      std::make_shared<dftefe::atoms::AtomSphericalDataContainer>(atomSymbolToFilename,
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
        dftefe::utils::Point centerPoint(dim, 0.0); 
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
      int err = dftefe::utils::mpi::MPIAllreduce<dftefe::utils::MemorySpace::HOST>(
        &flag,
        &mpiReducedFlag,
        1,
        dftefe::utils::mpi::MPIInt,
        dftefe::utils::mpi::MPIMin,
        comm);
      std::pair<bool, std::string> mpiIsSuccessAndMsg =
        dftefe::utils::mpi::MPIErrIsSuccessAndMsg(err);
      dftefe::utils::throwException(mpiIsSuccessAndMsg.first,
                            "MPI Error:" + mpiIsSuccessAndMsg.second);
    }
  }
  // Make orthogonalized EFE basis

  unsigned int feOrder = 3;

    // Set up the vector of scalarSpatialRealFunctions for adaptive quadrature
    std::vector<std::shared_ptr<const dftefe::utils::ScalarSpatialFunctionReal>> functionsVec(0);
    unsigned int numfun = 2;
    functionsVec.resize(numfun); // Enrichment Functions
    std::vector<double> tolerances(numfun);
    std::vector<double> integralThresholds(numfun);
    for ( unsigned int i=0 ;i < functionsVec.size() ; i++ )
    {
        functionsVec[i] = std::make_shared<dftefe::atoms::AtomSevereFunction<dim>>(
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

    dftefe::quadrature::QuadratureRuleAttributes quadAttrAdaptive(dftefe::quadrature::QuadratureFamily::ADAPTIVE,false);

    // Set up base quadrature rule for adaptive quadrature 

    unsigned int num1DGaussSize = feOrder + 1;
    std::shared_ptr<dftefe::quadrature::QuadratureRule> baseQuadRule =
    std::make_shared<dftefe::quadrature::QuadratureRuleGauss>(dim, num1DGaussSize);

    dftefe::quadrature::QuadratureRuleAttributes quadAttrGauss(dftefe::quadrature::QuadratureFamily::GAUSS,true,num1DGaussSize);

    std::shared_ptr<dftefe::basis::CellMappingBase> cellMapping = std::make_shared<dftefe::basis::LinearCellMappingDealii<dim>>();
    std::shared_ptr<dftefe::basis::ParentToChildCellsManagerBase> parentToChildCellsManager = std::make_shared<dftefe::basis::ParentToChildCellsManagerDealii<dim>>();

    std::shared_ptr<dftefe::quadrature::QuadratureRuleContainer> quadRuleContainerAdaptive =
      std::make_shared<dftefe::quadrature::QuadratureRuleContainer>
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
  std::shared_ptr<const dftefe::basis::FEBasisDofHandler<double, dftefe::utils::MemorySpace::HOST,dim>> cfeBasisDofHandler =  
   std::make_shared<dftefe::basis::CFEBasisDofHandlerDealii<double, dftefe::utils::MemorySpace::HOST,dim>>(triangulationBase, feOrder, comm);

  dftefe::basis::BasisStorageAttributesBoolMap basisAttrMap;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreValues] = true;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreGradient] = true;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreHessian] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreOverlap] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreGradNiGradNj] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreJxW] = true;

    // Set up the CFE Basis Data Storage for Rhs
    std::shared_ptr<dftefe::basis::FEBasisDataStorage<double, dftefe::utils::MemorySpace::HOST>> cfeBasisDataStorageAdaptUniformQuad =
      std::make_shared<dftefe::basis::CFEBasisDataStorageDealii<double, double,dftefe::utils::MemorySpace::HOST, dim>>
      (cfeBasisDofHandler, quadAttrAdaptive, basisAttrMap);
  // evaluate basis data
  cfeBasisDataStorageAdaptUniformQuad->evaluateBasisData(quadAttrAdaptive, quadRuleContainerAdaptive, basisAttrMap);
    
    unsigned int cellBlockSize = 10; 

    // Set up the CFE Basis Data Storage for Rhs
    std::shared_ptr<dftefe::basis::FEBasisDataStorage<double, dftefe::utils::MemorySpace::HOST>> cfeBasisDataStorageUniformQuad =
      std::make_shared<dftefe::basis::CFEBDSOnTheFlyComputeDealii<double, double,dftefe::utils::MemorySpace::HOST, dim>>
      (cfeBasisDofHandler, quadAttrGauss, basisAttrMap, cellBlockSize ,*linAlgOpContext);
  // evaluate basis data
  cfeBasisDataStorageUniformQuad->evaluateBasisData(quadAttrGauss, basisAttrMap);
    size_type numLocallyOwnedCells = cfeBasisDofHandler->nLocallyOwnedCells();
    for(int i = 0 ; i < numLocallyOwnedCells ; i += cellBlockSize)
    {
        size_type cellStartId = i;
        size_type cellEndId = std::min(i+cellBlockSize, numLocallyOwnedCells);
        auto basisDataStorageInCellRange = 
            cfeBasisDataStorageUniformQuad->getBasisDataStorageInCellRange(std::make_pair(cellStartId, cellEndId));
        for(int j = 0 ; j < cellEndId - cellStartId ; j++)
        {
            auto basisDataStorageInCellAdaptUniform = cfeBasisDataStorageAdaptUniformQuad->getBasisDataInCell(j);
            for(int k = 0 ; k < basisDataStorageInCellAdaptUniform.size() ; k++)
            {
                std::cout << *(basisDataStorageInCellRange.data() + dofsPerCell * nQuadPerCell * j + k) << 
                    "\t" << *(basisDataStorageInCellAdaptUniform.data() + k) << "\n";
            }
        }
    }

//     // Create the enrichmentClassicalInterface object
//       std::shared_ptr<dftefe::basis::EnrichmentClassicalInterfaceSpherical
//         enrichClassIntfce = std::make_shared<dftefe::basis::EnrichmentClassicalInterfaceSpherical
//                           <double, dftefe::utils::MemorySpace::HOST, dim>>
//                           (cfeBasisDataStorageOverlapMatrix,
//                           cfeBasisDataStorageRhs,
//                           atomSphericalDataContainer,
//                           atomPartitionTolerance,
//                           atomSymbolVec,
//                           atomCoordinatesVec,
//                           fieldName,
//                           linAlgOpContext,
//                           comm);

//   // initialize the basis 
//   std::shared_ptr<dftefe::basis::FEBasisDofHandler<double, dftefe::utils::MemorySpace::HOST,dim>> basisDofHandler =  
//     std::make_shared<dftefe::basis::EFEBasisDofHandlerDealii<double, double,dftefe::utils::MemorySpace::HOST,dim>>(
//       enrichClassIntfce, feOrder, comm);

//   std::map<dftefe::global_size_type, dftefe::utils::Point> dofCoords;
//   basisDofHandler->getBasisCenters(dofCoords);

//   std::cout << "Locally owned cells : " <<basisDofHandler->nLocallyOwnedCells() << "\n";
//   std::cout << "Total Number of dofs : " << basisDofHandler->nGlobalNodes() << "\n";

//   basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreValues] = true;
//   basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreGradient] = false;
//   basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreHessian] = false;
//   basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreOverlap] = false;
//   basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreGradNiGradNj] = false;
//   basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreJxW] = false;

//   // Set up Adaptive quadrature for EFE Basis Data Storage
//   std::shared_ptr<dftefe::basis::FEBasisDataStorage<double, dftefe::utils::MemorySpace::HOST>> feBasisData =
//     std::make_shared<dftefe::basis::EFEBasisDataStorageDealii<double, double, dftefe::utils::MemorySpace::HOST,dim>>
//     (basisDofHandler, quadAttrAdaptive, basisAttrMap);

//   // evaluate basis data
//   feBasisData->evaluateBasisData(quadAttrAdaptive, quadRuleContainerAdaptive, basisAttrMap);


  dftefe::utils::mpi::MPIBarrier(comm);

  //gracefully end MPI

  int mpiFinalFlag = 0;
  dftefe::utils::mpi::MPIFinalized(&mpiFinalFlag);
  if(!mpiFinalFlag)
  {
    dftefe::utils::mpi::MPIFinalize();
  }
}
