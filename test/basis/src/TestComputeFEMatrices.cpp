#include <basis/TriangulationBase.h>
#include <basis/TriangulationDealiiParallel.h>
#include <basis/CellMappingBase.h>
#include <basis/LinearCellMappingDealii.h>
#include <basis/EFEBasisDofHandlerDealii.h>
#include <basis/BasisDofHandler.h>
#include <basis/ConstraintsLocal.h>
#include <basis/EFEBasisDataStorageDealii.h>
#include <basis/CFEBasisDataStorageDealii.h>
#include <basis/FEBasisOperations.h>
#include <quadrature/QuadratureAttributes.h>
#include <quadrature/QuadratureRuleGauss.h>
#include <utils/Point.h>
#include <atoms/AtomSevereFunction.h>
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

int main()
{

  std::cout<<" Entering test Compute FE Matrices\n";
  // Set up linAlgcontext

  // initialize the MPI environment
  dftefe::utils::mpi::MPIInit(NULL, NULL);

  dftefe::utils::mpi::MPIComm comm = dftefe::utils::mpi::MPICommWorld;

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
  dftefe::size_type numComponents = 1;
  std::shared_ptr<dftefe::basis::TriangulationBase> triangulationBase =
    std::make_shared<dftefe::basis::TriangulationDealiiParallel<dim>>(comm);
  std::vector<unsigned int>         subdivisions = {10, 10, 10};
  std::vector<bool>                 isPeriodicFlags(dim, false);
  std::vector<dftefe::utils::Point> domainVectors(dim,
                                                  dftefe::utils::Point(dim, 0.0));


  double xmax = 10.0;
  double ymax = 10.0;
  double zmax = 10.0;

  domainVectors[0][0] = xmax;
  domainVectors[1][1] = ymax;
  domainVectors[2][2] = zmax;

  // initialize the triangulation
  triangulationBase->initializeTriangulationConstruction();
  triangulationBase->createUniformParallelepiped(subdivisions,
                                                 domainVectors,
                                                 isPeriodicFlags);
  triangulationBase->finalizeTriangulationConstruction();
 
  triangulationBase->executeCoarseningAndRefinement();
  triangulationBase->finalizeTriangulationConstruction();

  // Enrichment data file consisting of g(r,\theta,\phi) = f(r)*Y_lm(\theta, \phi)
  std::string sourceDir = "/home/avirup/dft-efe/test/basis/src/";
  std::string atomDataFile = "AtomData.in";
  std::string inputFileName = sourceDir + atomDataFile;
  std::fstream fstream;

  fstream.open(inputFileName, std::fstream::in);
  
  // read the input file and create atomsymbol vector and atom coordinates vector.
  std::vector<dftefe::utils::Point> atomCoordinatesVec;
  std::vector<double> coordinates;
  coordinates.resize(dim,0.);
  std::vector<std::string> atomSymbol;
  std::string symbol;
  atomSymbol.resize(0);
  std::string line;
  while (std::getline(fstream, line)){
      std::stringstream ss(line);
      ss >> symbol; 
      for(unsigned int i=0 ; i<dim ; i++){
          ss >> coordinates[i]; 
      }
      atomCoordinatesVec.push_back(coordinates);
      atomSymbol.push_back(symbol);
  }
  dftefe::utils::mpi::MPIBarrier(comm);
      
  std::map<std::string, std::string> atomSymbolToFilename;
  for (auto i:atomSymbol )
  {
      atomSymbolToFilename[i] = sourceDir + i + ".xml";
  }

  std::vector<std::string> fieldNames{ "density", "vhartree", "vnuclear", "vtotal", "orbital" };
  std::vector<std::string> metadataNames{ "symbol", "Z", "charge", "NR", "r" };
  std::shared_ptr<dftefe::atoms::AtomSphericalDataContainer>  atomSphericalDataContainer = 
      std::make_shared<dftefe::atoms::AtomSphericalDataContainer>(atomSymbolToFilename,
                                                      fieldNames,
                                                      metadataNames);

  std::string fieldName = "vnuclear";
  double atomPartitionTolerance = 1e-6;

  // Make pristine EFE basis

  // 1. Make EnrichmentClassicalInterface object for Pristine enrichment
  // 2. Input to the EFEBasisDofHandler(eci, feOrder) 
  // 3. Make EFEBasisDataStorage with input as quadratureContainer.

  unsigned int feOrder = 3;
  std::shared_ptr<dftefe::basis::EnrichmentClassicalInterfaceSpherical
                          <double, dftefe::utils::MemorySpace::HOST, dim>>
                          enrichClassIntfce = std::make_shared<dftefe::basis::EnrichmentClassicalInterfaceSpherical
                          <double, dftefe::utils::MemorySpace::HOST, dim>>
                          (triangulationBase,
                          atomSphericalDataContainer,
                          atomPartitionTolerance,
                          atomSymbol,
                          atomCoordinatesVec,
                          fieldName,
                          comm);

    // Set up the vector of scalarSpatialRealFunctions for adaptive quadrature
    std::vector<std::shared_ptr<const dftefe::utils::ScalarSpatialFunctionReal>> functionsVec(0);
    unsigned int numfun = 2;
    functionsVec.resize(numfun); // Enrichment Functions
    std::vector<double> absoluteTolerances(numfun), relativeTolerances(numfun);
    std::vector<double> integralThresholds(numfun);
    for ( unsigned int i=0 ;i < functionsVec.size() ; i++ )
    {
        functionsVec[i] = std::make_shared<dftefe::atoms::AtomSevereFunction<dim>>(        
            enrichClassIntfce->getEnrichmentIdsPartition(),
            atomSphericalDataContainer,
            atomSymbol,
            atomCoordinatesVec,
            fieldName,
            i);
        absoluteTolerances[i] = 1e-3;
        relativeTolerances[i] = 1e-3;
        integralThresholds[i] = 1e-3;
    }

    double smallestCellVolume = 1e-12;
    unsigned int maxRecursion = 1000;

    //Set up quadAttr for Rhs and OverlapMatrix

    dftefe::quadrature::QuadratureRuleAttributes quadAttrAdaptive(dftefe::quadrature::QuadratureFamily::ADAPTIVE,false);

    // Set up base quadrature rule for adaptive quadrature 

    unsigned int num1DGaussSize =4;
    std::shared_ptr<dftefe::quadrature::QuadratureRule> baseQuadRule =
    std::make_shared<dftefe::quadrature::QuadratureRuleGauss>(dim, num1DGaussSize);

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
      absoluteTolerances,
      relativeTolerances,
      integralThresholds,
      smallestCellVolume,
      maxRecursion);

  // initialize the basis
  std::shared_ptr<dftefe::basis::FEBasisDofHandler<double, dftefe::utils::MemorySpace::HOST,dim>> basisDofHandler =  
    std::make_shared<dftefe::basis::EFEBasisDofHandlerDealii<double, double,dftefe::utils::MemorySpace::HOST,dim>>(
      enrichClassIntfce,
      feOrder,
      comm);
  std::map<dftefe::global_size_type, dftefe::utils::Point> dofCoords;
  basisDofHandler->getBasisCenters(dofCoords);

  std::cout << "Locally owned cells : " << basisDofHandler->nLocallyOwnedCells() << "\n";
  std::cout << "Total Number of dofs : " << basisDofHandler->nGlobalNodes() << "\n";

  dftefe::basis::BasisStorageAttributesBoolMap basisAttrMap;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreValues] = true;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreGradient] = true;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreHessian] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreOverlap] = true;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreGradNiGradNj] = true;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreJxW] = true;

  // Set up the FE Basis Data Storage
  std::shared_ptr<dftefe::basis::BasisDataStorage<double, dftefe::utils::MemorySpace::HOST>> feBasisData =
    std::make_shared<dftefe::basis::EFEBasisDataStorageDealii<double, double, dftefe::utils::MemorySpace::HOST,dim>>
    (basisDofHandler, quadAttrAdaptive, basisAttrMap);

  // evaluate basis data
  feBasisData->evaluateBasisData(quadAttrAdaptive, quadRuleContainerAdaptive, basisAttrMap);

  // Set up BasisManager
  std::shared_ptr<const dftefe::basis::BasisManager<double, dftefe::utils::MemorySpace::HOST>> basisManager =
    std::make_shared<dftefe::basis::FEBasisManager<double, double, dftefe::utils::MemorySpace::HOST,dim>>
    (basisDofHandler);

  // Set up basis Operations
  dftefe::basis::FEBasisOperations<double, double, dftefe::utils::MemorySpace::HOST,dim> feBasisOp(feBasisData,50);

  dftefe::quadrature::QuadratureValuesContainer<double, dftefe::utils::MemorySpace::HOST> f(quadRuleContainerAdaptive, numComponents);

  for(dftefe::size_type i = 0 ; i < f.nCells() ; i++)
  {
    dftefe::size_type quadId = 0;
    for (auto j : quadRuleContainerAdaptive->getCellRealPoints(i))
    {
      std::vector<double> a(numComponents, 3.0);
      double *b = a.data();
      f.setCellQuadValues<dftefe::utils::MemorySpace::HOST> (i, quadId, b);
      quadId = quadId + 1;
    }
  }

  dftefe::utils::MemoryStorage<double, dftefe::utils::MemorySpace::HOST>
    cellWiseFEData(0);

  feBasisOp.computeFEMatrices(dftefe::basis::realspace::LinearLocalOp::GRAD, 
    dftefe::basis::realspace::VectorMathOp::DOT, dftefe::basis::realspace::
      LinearLocalOp::GRAD, cellWiseFEData, *linAlgOpContext);

  for(dftefe::size_type i = 0 ; i < cellWiseFEData.size() ; i++)
  {
    if(std::abs (*(cellWiseFEData.data() + i ) - *(feBasisData->getBasisGradNiGradNjInAllCells().data() + i)) > 1e-12)
    std::cout << *(cellWiseFEData.data() + i ) << " , " << *(feBasisData->getBasisGradNiGradNjInAllCells().data() + i) <<"\n";
  }

  cellWiseFEData.resize(0);
  feBasisOp.computeFEMatrices
      (dftefe::basis::realspace::LinearLocalOp::IDENTITY, 
    dftefe::basis::realspace::VectorMathOp::MULT, dftefe::basis::realspace::
      VectorMathOp::MULT, dftefe::basis::realspace::LinearLocalOp::IDENTITY,
      f, cellWiseFEData, *linAlgOpContext);

  for(dftefe::size_type i = 0 ; i < cellWiseFEData.size() ; i++)
  {
    if(std::abs (*(cellWiseFEData.data() + i ) - 3 * *(feBasisData->getBasisOverlapInAllCells().data() + i)) > 1e-12)
    std::cout << *(cellWiseFEData.data() + i ) << " , " << 3 * *(feBasisData->getBasisOverlapInAllCells().data() + i) <<"\n";
  }

  dftefe::utils::mpi::MPIFinalize();

}
