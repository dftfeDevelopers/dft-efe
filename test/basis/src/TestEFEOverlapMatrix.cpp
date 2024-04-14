#include <basis/TriangulationBase.h>
#include <basis/TriangulationDealiiParallel.h>
#include <basis/CellMappingBase.h>
#include <basis/LinearCellMappingDealii.h>
#include <basis/EFEBasisDofHandlerDealii.h>
#include <basis/EFEConstraintsLocalDealii.h>
#include <basis/EFEBasisDataStorageDealii.h>
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

  std::cout<<" Entering test overlap matrix enrichment \n";

  const long max_rand = 1000000L;

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
  // 2. Make CFEBasisDataStorageDealii object for Rhs (ADAPTIVE with GAUSS and fns are N_i^2 - make quadrulecontainer), overlapmatrix (GAUSS)
  // 3. Make EnrichmentClassicalInterface object for Orthogonalized enrichment
  // 4. Input to the EFEBasisDofHandler(eci, feOrder) 
  // 5. Make EFEBasisDataStorage with input as quadratureContainer.

  unsigned int feOrder = 3;
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
        tolerances[i] = 1e-3;
        integralThresholds[i] = 1e-3;
    }

    double smallestCellVolume = 1e-12;
    unsigned int maxRecursion = 1000;

    //Set up quadAttr for Rhs and OverlapMatrix

    dftefe::quadrature::QuadratureRuleAttributes quadAttrAdaptive(dftefe::quadrature::QuadratureFamily::ADAPTIVE,false);

    unsigned int num1DGllSize =4;
    dftefe::quadrature::QuadratureRuleAttributes quadAttrGll(dftefe::quadrature::QuadratureFamily::GLL,true,num1DGllSize);

    // Set up base quadrature rule for adaptive quadrature 

    unsigned int num1DGaussSize =4;
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
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreGradient] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreHessian] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreOverlap] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreGradNiGradNj] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreJxW] = true;

    // Set up the CFE Basis Data Storage for Overlap Matrix
    std::shared_ptr<dftefe::basis::FEBasisDataStorage<double, dftefe::utils::MemorySpace::HOST>> cfeBasisDataStorageOverlapMatrix =
      std::make_shared<dftefe::basis::CFEBasisDataStorageDealii<double, double, dftefe::utils::MemorySpace::HOST, dim>>
      (cfeBasisDofHandler, quadAttrGll, basisAttrMap);
  // evaluate basis data
  cfeBasisDataStorageOverlapMatrix->evaluateBasisData(quadAttrGll, basisAttrMap);

    // Set up the CFE Basis Data Storage for Overlap Matrix
    std::shared_ptr<dftefe::basis::FEBasisDataStorage<double, dftefe::utils::MemorySpace::HOST>> cfeBasisDataStorageOverlapMatrixGauss =
      std::make_shared<dftefe::basis::CFEBasisDataStorageDealii<double, double, dftefe::utils::MemorySpace::HOST, dim>>
      (cfeBasisDofHandler, quadAttrGauss, basisAttrMap);
  // evaluate basis data
  cfeBasisDataStorageOverlapMatrixGauss->evaluateBasisData(quadAttrGauss, basisAttrMap);

    // Set up the CFE Basis Data Storage for Rhs
    std::shared_ptr<dftefe::basis::FEBasisDataStorage<double, dftefe::utils::MemorySpace::HOST>> cfeBasisDataStorageRhs =
      std::make_shared<dftefe::basis::CFEBasisDataStorageDealii<double, double, dftefe::utils::MemorySpace::HOST, dim>>
      (cfeBasisDofHandler, quadAttrAdaptive, basisAttrMap);
  // evaluate basis data
  cfeBasisDataStorageRhs->evaluateBasisData(quadAttrAdaptive, quadRuleContainerAdaptive, basisAttrMap);

    // Create the enrichmentClassicalInterface object
    enrichClassIntfce = std::make_shared<dftefe::basis::EnrichmentClassicalInterfaceSpherical
                          <double, dftefe::utils::MemorySpace::HOST, dim>>
                          (cfeBasisDataStorageOverlapMatrix,
                          cfeBasisDataStorageRhs,
                          atomSphericalDataContainer,
                          atomPartitionTolerance,
                          atomSymbolVec,
                          atomCoordinatesVec,
                          fieldName,
                          linAlgOpContext,
                          comm);

  // initialize the basis 
  std::shared_ptr<dftefe::basis::FEBasisDofHandler<double, dftefe::utils::MemorySpace::HOST,dim>> basisDofHandler =  
    std::make_shared<dftefe::basis::EFEBasisDofHandlerDealii<double, double,dftefe::utils::MemorySpace::HOST,dim>>(
      enrichClassIntfce, feOrder, comm);

  std::map<dftefe::global_size_type, dftefe::utils::Point> dofCoords;
  basisDofHandler->getBasisCenters(dofCoords);

  std::cout << "Locally owned cells : " << basisDofHandler->nLocallyOwnedCells() << "\n";
  std::cout << "Total Number of dofs : " << basisDofHandler->nGlobalNodes() << "\n";

  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreValues] = true;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreGradient] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreHessian] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreOverlap] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreGradNiGradNj] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreJxW] = false;

  // Set up Adaptive quadrature for EFE Basis Data Storage
  std::shared_ptr<dftefe::basis::FEBasisDataStorage<double, dftefe::utils::MemorySpace::HOST>> feBasisData =
    std::make_shared<dftefe::basis::EFEBasisDataStorageDealii<double, double, dftefe::utils::MemorySpace::HOST,dim>>
    (basisDofHandler, quadAttrAdaptive, basisAttrMap);

  // evaluate basis data
  feBasisData->evaluateBasisData(quadAttrAdaptive, quadRuleContainerAdaptive, basisAttrMap);

  // Set up BasisManager
  std::shared_ptr<const dftefe::basis::FEBasisManager<double, double, dftefe::utils::MemorySpace::HOST, dim>> basisManager =
    std::make_shared<dftefe::basis::FEBasisManager<double, double, dftefe::utils::MemorySpace::HOST,dim>>
    (basisDofHandler);

  // Set up basis Operations
  dftefe::basis::FEBasisOperations<double, double, dftefe::utils::MemorySpace::HOST,dim> feBasisOp(feBasisData,50);

  // set up MPIPatternP2P for the constraints
  auto mpiPatternP2PHanging = basisManager->getMPIPatternP2P();

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

  dftefe::size_type numLocallyOwnedCells  = basisDofHandler->nLocallyOwnedCells();
  auto itField  = X->begin();
  dftefe::utils::Point nodeLoc(dim,0.0);
  dftefe::size_type nodeCount = 0; 
  for (dftefe::size_type iCell = 0; iCell < numLocallyOwnedCells ; iCell++)
    {
      // get cell dof global ids
      std::vector<dftefe::global_size_type> cellGlobalNodeIds;
      basisDofHandler->getCellDofsGlobalIds(iCell, cellGlobalNodeIds);

      // loop over nodes of a cell
      for ( dftefe::size_type iNode = 0 ; iNode < cellGlobalNodeIds.size() ; iNode++)
        {
          // If node not constrained then get the local id and coordinates of the node
          dftefe::global_size_type globalId = cellGlobalNodeIds[iNode];
         if( !basisManager->getConstraints().isConstrained(globalId))
         {
            dftefe::size_type localId = basisManager->globalToLocalIndex(globalId) ;
            basisManager->getBasisCenters(localId,nodeLoc);
            for(int comp = 0 ; comp < numComponents ; comp++)
            {
              double lower_bound = comp;
              double upper_bound = comp+1;
              *(itField + localId*numComponents + comp )  = comp+1 ;//lower_bound + (upper_bound - lower_bound) * (random() % max_rand)/ max_rand;
            }
              //((double) rand() / (RAND_MAX));
         }
        }
    }

    X->updateGhostValues();
    basisManager->getConstraints().distributeParentToChild(*X, X->getNumberComponents());


    // Create OperatorContext for Basisoverlap CFE - GLL, EFE - Adaptive
    std::shared_ptr<const dftefe::basis::EFEOverlapOperatorContext<double,
                                                  double,
                                                  dftefe::utils::MemorySpace::HOST,
                                                  dim>> MContext =
    std::make_shared<dftefe::basis::EFEOverlapOperatorContext<double,
                                                        double,
                                                        dftefe::utils::MemorySpace::HOST,
                                                        dim>>(
                                                        *basisManager,
                                                        *basisManager,
                                                        *cfeBasisDataStorageOverlapMatrix,
                                                        *feBasisData,
                                                        *cfeBasisDataStorageOverlapMatrix,
                                                        50);

  // feBasisOp.interpolate( *dens, constraintHomwHan, *basisManager, quadValuesContainer);

  std::shared_ptr<dftefe::linearAlgebra::OperatorContext<double,
                                                   double,
                                                   dftefe::utils::MemorySpace::HOST>> MInvContext =
    std::make_shared<dftefe::basis::OrthoEFEOverlapInverseOpContextGLL<double,
                                                   double,
                                                   dftefe::utils::MemorySpace::HOST,
                                                   dim>>
                                                   (*basisManager,
                                                    *cfeBasisDataStorageOverlapMatrix,
                                                    *feBasisData,
                                                    linAlgOpContext);

    MInvContext->apply(*X,*Y);
    MContext->apply(*Y,*Z);

    for (int comp = 0 ; comp < numComponents ; comp++)
      std::cout << "Component "<<comp << ":" << X->l2Norms()[comp] << "," << Y->l2Norms()[comp] <<"," << Z->l2Norms()[comp] << "\n";

  std::shared_ptr<dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>
   error = std::make_shared<
    dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>(
      mpiPatternP2PHanging, linAlgOpContext, numComponents, double());

  std::vector<double> ones(0);
  ones.resize(numComponents, (double)1.0);
  std::vector<double> nOnes(0);
  nOnes.resize(numComponents, (double)-1.0);

  dftefe::linearAlgebra::add(ones, *X, nOnes, *Z, *error);

  for (int comp = 0 ; comp < numComponents ; comp++)
        std::cout << "Component "<<comp << ":" << "Error norm: "<<error->l2Norms()[0]<<" Relative error: "<<(error->l2Norms()[comp]/X->l2Norms()[comp])<<"\n";

    // // Form the Overlap Matrix
    // dftefe::global_size_type totalDofs = basisDofHandler->nGlobalNodes();
    // std::shared_ptr<dftefe::utils::MemoryStorage<double, dftefe::utils::MemorySpace::HOST>>
    //   basisOverlapBlock = std::make_shared<dftefe::utils::MemoryStorage<double, dftefe::utils::MemorySpace::HOST>>(totalDofs * totalDofs);

    // std::vector<double> basisOverlapBlockSTL(totalDofs * totalDofs,0), 
    // basisOverlapBlockSTLTmp(totalDofs * totalDofs,0);
    // dftefe::size_type cumulativeBasisDataInCells = 0;
    // for (dftefe::size_type iCell = 0; iCell < basisDofHandler->nLocallyOwnedCells() ; iCell++)
    // {
    //   // get cell dof global ids
    //   std::vector<dftefe::global_size_type> cellGlobalNodeIds(0);
    //   basisDofHandler->getCellDofsGlobalIds(iCell, cellGlobalNodeIds);
    //   // loop over nodes of a cell
    //   for ( dftefe::size_type iNode = 0 ; iNode < cellGlobalNodeIds.size() ; iNode++)
    //   {
    //     for ( dftefe::size_type jNode = 0 ; jNode < cellGlobalNodeIds.size() ; jNode++)
    //     {
    //       *(basisOverlapBlockSTLTmp.data() + cellGlobalNodeIds[iNode]*totalDofs
    //           + cellGlobalNodeIds[jNode]) +=
    //         *(MContext->getBasisOverlapInAllCells().data() + cumulativeBasisDataInCells + 
    //           cellGlobalNodeIds.size()*iNode + jNode);
    //     }
    //   }
    //   cumulativeBasisDataInCells += dftefe::utils::mathFunctions::sizeTypePow((cellGlobalNodeIds.size()),2);
    // }

  // int err = dftefe::utils::mpi::MPIAllreduce<dftefe::utils::MemorySpace::HOST>(
  //   basisOverlapBlockSTLTmp.data(),
  //   basisOverlapBlockSTL.data(),
  //   basisOverlapBlockSTLTmp.size(),
  //   dftefe::utils::mpi::MPIDouble,
  //   dftefe::utils::mpi::MPISum,
  //   comm);
  // std::pair<bool, std::string> mpiIsSuccessAndMsg =
  //   dftefe::utils::mpi::MPIErrIsSuccessAndMsg(err);
  // dftefe::utils::throwException(mpiIsSuccessAndMsg.first,
  //                       "MPI Error:" + mpiIsSuccessAndMsg.second);

  // if(rank == 0)
  // {
  // std::cout << "Enrichment Block : \n";
  // int cc = 0;
  // for (int i = 0 ; i < totalDofs; i++)
  // {
  // for (int j = 0 ; j < totalDofs; j++)
  // {
  //   if(basisOverlapBlockSTL[i*totalDofs+j] > 1e-10 && i == totalDofs-1 || j == totalDofs-1)
  //     std::cout << i << " " << j << " " << basisOverlapBlockSTL[i*totalDofs+j] << "\n";
  // }
  // }
  // }

  dftefe::utils::mpi::MPIBarrier(comm);

  //gracefully end MPI

  int mpiFinalFlag = 0;
  dftefe::utils::mpi::MPIFinalized(&mpiFinalFlag);
  if(!mpiFinalFlag)
  {
    dftefe::utils::mpi::MPIFinalize();
  }
}
