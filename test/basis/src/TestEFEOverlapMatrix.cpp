#include <basis/TriangulationBase.h>
#include <basis/TriangulationDealiiParallel.h>
#include <basis/CellMappingBase.h>
#include <basis/LinearCellMappingDealii.h>
#include <basis/EFEBasisManagerDealii.h>
#include <basis/EFEConstraintsDealii.h>
#include <basis/EFEBasisDataStorageDealii.h>
#include <basis/FEBasisOperations.h>
#include <basis/EFEBasisHandlerDealii.h>
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
#include <basis/EFEOverlapInverseOperatorContext.h>

int main()
{

  std::cout<<" Entering test overlap matrix enrichment \n";

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
  dftefe::linearAlgebra::blasLapack::BlasQueue<dftefe::utils::MemorySpace::HOST> *blasQueuePtr = &blasQueue;

  std::shared_ptr<dftefe::linearAlgebra::LinAlgOpContext<dftefe::utils::MemorySpace::HOST>> linAlgOpContext = 
  std::make_shared<dftefe::linearAlgebra::LinAlgOpContext<dftefe::utils::MemorySpace::HOST>>(blasQueuePtr);

  // Set up Triangulation
  const unsigned int dim = 3;
  std::shared_ptr<dftefe::basis::TriangulationBase> triangulationBase =
  std::make_shared<dftefe::basis::TriangulationDealiiParallel<dim>>(comm);
  std::vector<unsigned int>         subdivisions = {5, 5, 5};
  std::vector<bool>                 isPeriodicFlags(dim, false);
  std::vector<dftefe::utils::Point> domainVectors(dim,
                                                  dftefe::utils::Point(dim, 0.0));

  double xmax = 5.0;
  double ymax = 5.0;
  double zmax = 5.0;
  unsigned int numComponents = 1;

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

  // Make orthogonalized EFE basis

  // 1. Make EnrichmentClassicalInterface object for Pristine enrichment
  // 2. Make FEbasisdatastorageDealii object for Rhs (ADAPTIVE with GAUSS and fns are N_i^2 - make quadrulecontainer), overlapmatrix (GAUSS)
  // 3. Make EnrichmentClassicalInterface object for Orthogonalized enrichment
  // 4. Input to the EFEBasisManager(eci, feOrder) 
  // 5. Make EFEBasisDataStorage with input as quadratureContainer.

  std::shared_ptr<dftefe::basis::EnrichmentClassicalInterfaceSpherical
                          <double, dftefe::utils::MemorySpace::HOST, dim>>
                          enrichClassIntfce = std::make_shared<dftefe::basis::EnrichmentClassicalInterfaceSpherical
                          <double, dftefe::utils::MemorySpace::HOST, dim>>(triangulationBase,
                          atomSphericalDataContainer,
                          atomPartitionTolerance,
                          atomSymbolVec,
                          atomCoordinatesVec,
                          fieldName,
                          comm);

    // Set up the vector of scalarSpatialRealFunctions for adaptive quadrature
    std::vector<std::shared_ptr<const dftefe::utils::ScalarSpatialFunctionReal>> functionsVec(0);
    unsigned int numfun = 2;
    functionsVec.resize(numfun); // Enrichment Functions
    std::vector<double> tolerances(numfun);
    std::vector<double> integralThresholds(numfun);
    for ( unsigned int i=0 ;i < functionsVec.size() ; i++ )
    {
        functionsVec[i] = std::make_shared<dftefe::atoms::AtomSevereFunction<dim>>(        
            enrichClassIntfce->getEnrichmentIdsPartition(),
            atomSphericalDataContainer,
            atomSymbolVec,
            atomCoordinatesVec,
            fieldName,
            i);
        tolerances[i] = 1e-6;
        integralThresholds[i] = 1e-14;
    }

    double smallestCellVolume = 1e-14;
    unsigned int maxRecursion = 1e3;

    //Set up quadAttr for Rhs and OverlapMatrix

    dftefe::quadrature::QuadratureRuleAttributes quadAttrAdaptive(dftefe::quadrature::QuadratureFamily::ADAPTIVE,false);

    unsigned int num1DGllSize =4;
    dftefe::quadrature::QuadratureRuleAttributes quadAttrGll(dftefe::quadrature::QuadratureFamily::GLL,true,num1DGllSize);

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
      tolerances,
      integralThresholds,
      smallestCellVolume,
      maxRecursion);

    unsigned int feOrder = 3;
    // Set the CFE basis manager and handler for bassiInterfaceCoeffcient distributed vector
    std::shared_ptr<dftefe::basis::FEBasisManager> cfeBasisManager =   std::make_shared<dftefe::basis::FEBasisManagerDealii<dim>>(triangulationBase, feOrder);

    std::string constraintName = "HangingWithHomogeneous";
    std::vector<std::shared_ptr<dftefe::basis::FEConstraintsBase<double, dftefe::utils::MemorySpace::HOST>>>
      constraintsVec;
    constraintsVec.resize(1);
    for ( unsigned int i=0 ;i < constraintsVec.size() ; i++ )
    constraintsVec[i] = std::make_shared<dftefe::basis::FEConstraintsDealii<double, dftefe::utils::MemorySpace::HOST, dim>>();
    
    constraintsVec[0]->clear();
    constraintsVec[0]->makeHangingNodeConstraint(cfeBasisManager);
    constraintsVec[0]->setHomogeneousDirichletBC();
    constraintsVec[0]->close();

    std::map<std::string,
            std::shared_ptr<const dftefe::basis::Constraints<double, dftefe::utils::MemorySpace::HOST>>> constraintsMap;

    constraintsMap[constraintName] = constraintsVec[0];

  dftefe::basis::BasisStorageAttributesBoolMap basisAttrMap;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreValues] = true;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreGradient] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreHessian] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreOverlap] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreGradNiGradNj] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreJxW] = true;

    // Set up the CFE Basis Data Storage for Overlap Matrix
    std::shared_ptr<dftefe::basis::FEBasisDataStorage<double, dftefe::utils::MemorySpace::HOST>> cfeBasisDataStorageOverlapMatrix =
      std::make_shared<dftefe::basis::FEBasisDataStorageDealii<double, dftefe::utils::MemorySpace::HOST, dim>>
      (cfeBasisManager, quadAttrAdaptive, basisAttrMap);
  // evaluate basis data
  cfeBasisDataStorageOverlapMatrix->evaluateBasisData(quadAttrGll, basisAttrMap);

    // Set up the CFE Basis Data Storage for Rhs
    std::shared_ptr<dftefe::basis::FEBasisDataStorage<double, dftefe::utils::MemorySpace::HOST>> cfeBasisDataStorageRhs =
      std::make_shared<dftefe::basis::FEBasisDataStorageDealii<double, dftefe::utils::MemorySpace::HOST, dim>>
      (cfeBasisManager, quadAttrGll, basisAttrMap);
  // evaluate basis data
  cfeBasisDataStorageRhs->evaluateBasisData(quadAttrAdaptive, quadRuleContainerAdaptive, basisAttrMap);

    // // Set up BasisHandler
    std::shared_ptr<dftefe::basis::FEBasisHandler<double, dftefe::utils::MemorySpace::HOST, dim>> cfeBasisHandler =
      std::make_shared<dftefe::basis::FEBasisHandlerDealii<double, dftefe::utils::MemorySpace::HOST, dim>>
      (cfeBasisManager, constraintsMap, comm);

    // Create OperatorContext for CFEBasisoverlap
        std::shared_ptr<const dftefe::basis::FEOverlapOperatorContext<double,
                                                      double,
                                                      dftefe::utils::MemorySpace::HOST,
                                                      dim>> cfeBasisOverlapOperator =
        std::make_shared<dftefe::basis::FEOverlapOperatorContext<double,
                                                           double,
                                                           dftefe::utils::MemorySpace::HOST,
                                                           dim>>(
                                                            *cfeBasisHandler,
                                                            *cfeBasisDataStorageOverlapMatrix,
                                                            constraintName,
                                                            constraintName,
                                                            50);

    // Create the enrichmentClassicalInterface object
    enrichClassIntfce = std::make_shared<dftefe::basis::EnrichmentClassicalInterfaceSpherical
                          <double, dftefe::utils::MemorySpace::HOST, dim>>
                          (cfeBasisOverlapOperator,
                          cfeBasisDataStorageRhs,
                          cfeBasisManager,
                          cfeBasisHandler,
                          atomSphericalDataContainer,
                          atomPartitionTolerance,
                          atomSymbolVec,
                          atomCoordinatesVec,
                          fieldName,
                          constraintName,
                          linAlgOpContext,
                          comm);

  // initialize the basis Manager
  std::shared_ptr<dftefe::basis::FEBasisManager> basisManager =   std::make_shared<dftefe::basis::EFEBasisManagerDealii<double,dftefe::utils::MemorySpace::HOST,dim>>(
      enrichClassIntfce,
      feOrder);
  std::map<dftefe::global_size_type, dftefe::utils::Point> dofCoords;
  basisManager->getBasisCenters(dofCoords);

  std::cout << "Locally owned cells : " << basisManager->nLocallyOwnedCells() << "\n";
  std::cout << "Total Number of dofs : " << basisManager->nGlobalNodes() << "\n";

  // Set the constraints

  std::string constraintHanging = "HangingNodeConstraint"; //give BC to rho

  constraintsVec.clear();
  constraintsVec.resize(1);
  for ( unsigned int i=0 ;i < constraintsVec.size() ; i++ )
  constraintsVec[i] = std::make_shared<dftefe::basis::EFEConstraintsDealii<double, double, dftefe::utils::MemorySpace::HOST, dim>>();

  constraintsVec[0]->clear();
  constraintsVec[0]->makeHangingNodeConstraint(basisManager);
  constraintsVec[0]->close();

  constraintsMap.clear();
  constraintsMap[constraintHanging] = constraintsVec[0];

  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreValues] = true;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreGradient] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreHessian] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreOverlap] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreGradNiGradNj] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreJxW] = false;

  // Set up Adaptive quadrature for EFE Basis Data Storage
  std::shared_ptr<dftefe::basis::FEBasisDataStorage<double, dftefe::utils::MemorySpace::HOST>> feBasisData =
    std::make_shared<dftefe::basis::EFEBasisDataStorageDealii<double, dftefe::utils::MemorySpace::HOST,dim>>
    (basisManager, quadAttr, basisAttrMap);

  // evaluate basis data
  feBasisData->evaluateBasisData(quadAttrAdaptive, quadRuleContainerAdaptive, basisAttrMap);

  // Set up BasisHandler
  std::shared_ptr<dftefe::basis::FEBasisHandler<double, dftefe::utils::MemorySpace::HOST, dim>> basisHandler =
    std::make_shared<dftefe::basis::EFEBasisHandlerDealii<double, double, dftefe::utils::MemorySpace::HOST,dim>>
    (basisManager, constraintsMap, comm);

  // Set up basis Operations
  dftefe::basis::FEBasisOperations<double, double, dftefe::utils::MemorySpace::HOST,dim> feBasisOp(feBasisData,50);

  // set up MPIPatternP2P for the constraints
  auto mpiPatternP2PHanging = basisHandler->getMPIPatternP2P(constraintHanging);

  std::shared_ptr<dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>
   X = std::make_shared<
    dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>(
      mpiPatternP2PHanging, linAlgOpContext, numComponents, double());

  std::shared_ptr<dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>
   Y = std::make_shared<
    dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>(
      mpiPatternP2PHanging, linAlgOpContext, numComponents, double());

  std::shared_ptr<dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>
   Z = std::make_shared<
    dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>(
      mpiPatternP2PHanging, linAlgOpContext, numComponents, double());

  //populate the value of the Potential at the nodes for the analytic expressions

  dftefe::size_type numLocallyOwnedCells  = basisManager->nLocallyOwnedCells();
  auto itField  = X->begin();
  dftefe::utils::Point nodeLoc(dim,0.0);
  dftefe::size_type nodeCount = 0; 
  for (dftefe::size_type iCell = 0; iCell < numLocallyOwnedCells ; iCell++)
    {
      // get cell dof global ids
      std::vector<dftefe::global_size_type> cellGlobalNodeIds;
      basisManager->getCellDofsGlobalIds(iCell, cellGlobalNodeIds);

      // loop over nodes of a cell
      for ( dftefe::size_type iNode = 0 ; iNode < cellGlobalNodeIds.size() ; iNode++)
        {
          // If node not constrained then get the local id and coordinates of the node
          dftefe::global_size_type globalId = cellGlobalNodeIds[iNode];
         if( !basisHandler->getConstraints(constraintHanging).isConstrained(globalId))
         {
            dftefe::size_type localId = basisHandler->globalToLocalIndex(globalId,constraintHanging) ;
            basisHandler->getBasisCenters(localId,constraintHanging,nodeLoc);
            *(itField + localId )  = 1 ; //((double) rand() / (RAND_MAX));
         }
        }
    }

    // Create OperatorContext for Basisoverlap CFE - GLL, EFE - Adaptive
    std::shared_ptr<const dftefe::basis::EFEOverlapOperatorContext<double,
                                                  double,
                                                  dftefe::utils::MemorySpace::HOST,
                                                  dim>> MContext =
    std::make_shared<dftefe::basis::EFEOverlapOperatorContext<double,
                                                        double,
                                                        dftefe::utils::MemorySpace::HOST,
                                                        dim>>(
                                                        *basisHandler,
                                                        *cfeBasisDataStorageOverlapMatrix,
                                                        *feBasisData,
                                                        constraintHanging,
                                                        constraintHanging,
                                                        50);


      MContext->apply(*X,*Y);
  //feBasisOp.interpolate( *dens, constraintHomwHan, *basisHandler, quadValuesContainer);

  std::shared_ptr<dftefe::linearAlgebra::OperatorContext<double,
                                                   double,
                                                   dftefe::utils::MemorySpace::HOST>> MInvContext =
    std::make_shared<dftefe::basis::EFEOverlapInverseGLLOperatorContext<double,
                                                   double,
                                                   dftefe::utils::MemorySpace::HOST,
                                                   dim>>
                                                   (*basisHandler,
                                                    *MContext,
                                                    constraintHanging,
                                                    linAlgOpContext);

     MInvContext->apply(*Y,*Z);

    std::cout << "\n" <<X->l2Norms()[0] << "," << Y->l2Norms()[0] <<"," << Z->l2Norms()[0] << "\n";

  std::shared_ptr<dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>
   error = std::make_shared<
    dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>(
      mpiPatternP2PHanging, linAlgOpContext, numComponents, double());

  std::vector<double> ones(0);
  ones.resize(numComponents, (double)1.0);
  std::vector<double> nOnes(0);
  nOnes.resize(numComponents, (double)-1.0);

  dftefe::linearAlgebra::add(ones, *X, nOnes, *Z, *error);

    std::cout<<"No of dofs: "<< basisManager->nGlobalNodes() <<", error norm: "<<error->l2Norms()[0]<<", relative error: "<<(error->l2Norms()[0]/Z->l2Norms()[0])<<"\n";

  //gracefully end MPI

  int mpiFinalFlag = 0;
  dftefe::utils::mpi::MPIFinalized(&mpiFinalFlag);
  if(!mpiFinalFlag)
  {
    dftefe::utils::mpi::MPIFinalize();
  }
}
