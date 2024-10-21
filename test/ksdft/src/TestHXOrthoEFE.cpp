#include <basis/TriangulationBase.h>
#include <basis/TriangulationDealiiParallel.h>
#include <basis/CellMappingBase.h>
#include <basis/LinearCellMappingDealii.h>
#include <basis/EFEBasisDofHandlerDealii.h>
#include <basis/EFEBasisDataStorageDealii.h>
#include <basis/FEBasisOperations.h>
#include <basis/EFEConstraintsLocalDealii.h>
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
#include <ksdft/ElectrostaticLocalFE.h>
#include <ksdft/KineticFE.h>
#include <ksdft/ExchangeCorrelationFE.h>
#include <ksdft/KohnShamOperatorContextFE.h>
#include <ksdft/KohnShamEigenSolver.h>
#include <basis/OrthoEFEOverlapInverseOpContextGLL.h>
#include <utils/PointChargePotentialFunction.h>
#include <ksdft/DensityCalculator.h>
#include <ksdft/KohnShamDFT.h>
#include <basis/GenerateMesh.h>
#include <utils/ConditionalOStream.h>
#include <atoms/AtomSevereFunction.h>
#include <atoms/SphericalHarmonics.h>

#include <iostream>

using namespace dftefe;
const utils::MemorySpace Host = utils::MemorySpace::HOST;


  class RhoFunction : public utils::ScalarSpatialFunctionReal
  {
  private:
      std::shared_ptr<const atoms::AtomSphericalDataContainer>
                                d_atomSphericalDataContainer;
      std::vector<std::string>  d_atomSymbolVec;
      std::vector<utils::Point> d_atomCoordinatesVec;
      std::vector<double> d_atomChargesVec;

  public:
    RhoFunction(
      std::shared_ptr<const atoms::AtomSphericalDataContainer>
                                         atomSphericalDataContainer,
        const std::vector<std::string> & atomSymbol,
        const std::vector<double> &      atomCharges,
        const std::vector<utils::Point> &atomCoordinates)
      : d_atomSphericalDataContainer(atomSphericalDataContainer)
      , d_atomSymbolVec(atomSymbol)
      , d_atomCoordinatesVec(atomCoordinates)
      , d_atomChargesVec(atomCharges)
      {}

    double
    operator()(const utils::Point &point) const
    {
      double ylm00 = atoms::Clm(0, 0) * atoms::Dm(0) * atoms::Plm(0, 0, 1) * atoms::Qm(0, 0);
      double   retValue = 0;
      for (size_type atomId = 0 ; atomId < d_atomCoordinatesVec.size() ; atomId++)
        {
          utils::Point origin(d_atomCoordinatesVec[atomId]);
          for(auto &enrichmentObjId : 
            d_atomSphericalDataContainer->getSphericalData(d_atomSymbolVec[atomId], "density"))
          {
            retValue = retValue + std::abs(enrichmentObjId->getValue(point, origin) * (1/ylm00));
          }
        }
      return retValue;
    }
    std::vector<double>
    operator()(const std::vector<utils::Point> &points) const
    {
      std::vector<double> ret(0);
      ret.resize(points.size());
      for (unsigned int i = 0 ; i < points.size() ; i++)
      {
        ret[i] = (*this)(points[i]);
      }
      return ret;
    }
  };

int main()
{
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

  utils::ConditionalOStream rootCout(std::cout);
  rootCout.setCondition(rank == 0);
  rootCout<<" Entering test HX ortho enrichment\n";

    // Get nProcs
    int numProcs;
    utils::mpi::MPICommSize(comm, &numProcs);


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
  double xmax = 24.0;
  double ymax = 24.0;
  double zmax = 24.0;
  double rc = 0.6;
  double hMin = 1e6;
  size_type maxIter = 2e7;
  double absoluteTol = 1e-1;
  double relativeTol = 1e-1;
  double integralThreshold = 1e-1;
  double refineradius = 3*rc;
  unsigned int feOrder = 3;
  unsigned int num1DGaussSize = 4;
  unsigned int num1DGLLSize = 4;
  unsigned int numComponents = 1;
  
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
  
   // read the input file and create atomsymbol vector and atom coordinates vector.
  std::vector<utils::Point> atomCoordinatesVec(0,utils::Point(dim, 0.0));
    std::vector<double> coordinates;
  coordinates.resize(dim,0.);
  std::vector<std::string> atomSymbolVec;
  std::vector<double> atomChargesVec(0);
  std::string symbol;
  double atomicNumber;
  atomSymbolVec.resize(0);
  std::string line;
  while (std::getline(fstream, line)){
      std::stringstream ss(line);
      ss >> symbol; 
      ss >> atomicNumber; 
      for(unsigned int i=0 ; i<dim ; i++){
          ss >> coordinates[i]; 
      }
      atomCoordinatesVec.push_back(coordinates);
      atomSymbolVec.push_back(symbol);
      atomChargesVec.push_back((-1.0)*atomicNumber);
  }
  utils::mpi::MPIBarrier(comm);
  fstream.close();

  size_type numElectrons = 0;
  for(auto &i : atomChargesVec)
  {
    numElectrons += (size_type)(std::abs(i));
  }

  std::map<std::string, std::string> atomSymbolToFilename;
  for (auto i:atomSymbolVec )
  {
      atomSymbolToFilename[i] = sourceDir + i + ".xml";
      rootCout << "Reading xml file: "<<atomSymbolToFilename[i]<<std::endl; 
  }

    std::vector<std::string> fieldNames{"density","vtotal","orbital"};
    std::vector<std::string> metadataNames{ "symbol", "Z", "charge", "NR", "r" };
    std::shared_ptr<atoms::AtomSphericalDataContainer>  atomSphericalDataContainer = 
        std::make_shared<atoms::AtomSphericalDataContainer>(atomSymbolToFilename,
                                                        fieldNames,
                                                        metadataNames);

    double atomPartitionTolerance = 1e-6;

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
  // Compute Adaptive QuadratureRuleContainer for electrostaics

    // Set up the vector of scalarSpatialRealFunctions for adaptive quadrature
    std::vector<std::shared_ptr<const utils::ScalarSpatialFunctionReal>> functionsVec(0);
    unsigned int numfun = 4;
    functionsVec.resize(numfun); // Enrichment Functions
    std::vector<double> absoluteTolerances(numfun), relativeTolerances(numfun), integralThresholds(numfun);
    for ( unsigned int i=0 ;i < 2 ; i++ )
    {
      functionsVec[i] = std::make_shared<atoms::AtomSevereFunction<dim>>(        
          atomSphericalDataContainer,
          atomSymbolVec,
          atomCoordinatesVec,
          "vtotal",
          i);      
    }
    for ( unsigned int i=2 ;i < 4 ; i++ )
    {
      functionsVec[i] = std::make_shared<atoms::AtomSevereFunction<dim>>(        
          atomSphericalDataContainer,
          atomSymbolVec,
          atomCoordinatesVec,
          "orbital",
          i-2);      
    }
    for ( unsigned int i=0 ;i < numfun ; i++ )
    {
      absoluteTolerances[i] = absoluteTol;
      relativeTolerances[i] = relativeTol;
      integralThresholds[i] = integralThreshold;
    }
    //Set up quadAttr for Rhs and OverlapMatrix

    quadrature::QuadratureRuleAttributes quadAttrAdaptive(quadrature::QuadratureFamily::ADAPTIVE,false);

    quadrature::QuadratureRuleAttributes quadAttrGll(quadrature::QuadratureFamily::GLL,true,num1DGLLSize);

    // Set up base quadrature rule for adaptive quadrature 

    std::shared_ptr<quadrature::QuadratureRule> baseQuadRule =
      std::make_shared<quadrature::QuadratureRuleGauss>(dim, num1DGaussSize);

    std::shared_ptr<basis::ParentToChildCellsManagerBase> parentToChildCellsManager = std::make_shared<basis::ParentToChildCellsManagerDealii<dim>>();

   std::shared_ptr<basis::CellMappingBase> cellMapping = std::make_shared<basis::LinearCellMappingDealii<dim>>();

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
      1e-14,
      1000);

    unsigned int nQuad = quadRuleContainerAdaptive->nQuadraturePoints();
    int mpierr = utils::mpi::MPIAllreduce<Host>(
      utils::mpi::MPIInPlace,
      &nQuad,
      1,
      utils::mpi::Types<size_type>::getMPIDatatype(),
      utils::mpi::MPISum,
      comm);

  rootCout << "Number of quadrature points in electrostatics adaptive quadrature: "<< nQuad<<"\n";

    // Set up base quadrature rule for adaptive quadrature 

    nQuad = quadRuleContainerAdaptive->nQuadraturePoints();
    mpierr = utils::mpi::MPIAllreduce<Host>(
      utils::mpi::MPIInPlace,
      &nQuad,
      1,
       utils::mpi::Types<size_type>::getMPIDatatype(),
      utils::mpi::MPISum,
      comm);

  rootCout << "Number of quadrature points in wave function adaptive quadrature: "<<nQuad<<"\n";

  // Make orthogonalized EFE basis for all the fields

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
      std::shared_ptr<basis::EnrichmentClassicalInterfaceSpherical
                          <double, Host, dim>>
        enrichClassIntfceTotalPot = std::make_shared<basis::EnrichmentClassicalInterfaceSpherical
                          <double, Host, dim>>
                          (cfeBasisDataStorageGLL,
                          cfeBasisDataStorageAdaptive,
                          atomSphericalDataContainer,
                          atomPartitionTolerance,
                          atomSymbolVec,
                          atomCoordinatesVec,
                          "vtotal",
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
                          "orbital",
                          linAlgOpContext,
                          comm);

  // initialize the basis Manager

  std::shared_ptr<basis::FEBasisDofHandler<double, Host,dim>> basisDofHandlerTotalPot =  
    std::make_shared<basis::EFEBasisDofHandlerDealii<double, double,Host,dim>>(
      enrichClassIntfceTotalPot, feOrder, comm);

  std::shared_ptr<basis::FEBasisDofHandler<double, Host,dim>> basisDofHandlerWaveFn =  
    std::make_shared<basis::EFEBasisDofHandlerDealii<double, double,Host,dim>>(
      enrichClassIntfceOrbital, feOrder, comm);

  std::map<global_size_type, utils::Point> dofCoords;
  basisDofHandlerTotalPot->getBasisCenters(dofCoords);

  rootCout << "Total Number of dofs electrostatics: " << basisDofHandlerTotalPot->nGlobalNodes() << "\n";
  rootCout << "Total Number of dofs eigensolve: " << basisDofHandlerWaveFn->nGlobalNodes() << "\n";

  // Set up the quadrature rule

  basisAttrMap[basis::BasisStorageAttributes::StoreValues] = true;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradient] = true;
  basisAttrMap[basis::BasisStorageAttributes::StoreHessian] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreOverlap] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradNiGradNj] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreJxW] = true;

  // Set up Adaptive quadrature for EFE Basis Data Storage
  std::shared_ptr<basis::FEBasisDataStorage<double, Host>> efeBasisDataAdaptiveTotPot =
  std::make_shared<basis::EFEBasisDataStorageDealii<double, double, Host,dim>>
  (basisDofHandlerTotalPot, quadAttrAdaptive, basisAttrMap);

  efeBasisDataAdaptiveTotPot->evaluateBasisData(quadAttrAdaptive, quadRuleContainerAdaptive, basisAttrMap);

  basisAttrMap[basis::BasisStorageAttributes::StoreValues] = true;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradient] = true;
  basisAttrMap[basis::BasisStorageAttributes::StoreHessian] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreOverlap] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradNiGradNj] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreJxW] = true;

  std::shared_ptr<basis::FEBasisDataStorage<double, Host>> efeBasisDataAdaptiveOrbital =
  std::make_shared<basis::EFEBasisDataStorageDealii<double, double, Host,dim>>
  (basisDofHandlerWaveFn, quadAttrAdaptive, basisAttrMap);

  efeBasisDataAdaptiveOrbital->evaluateBasisData(quadAttrAdaptive, quadRuleContainerAdaptive, basisAttrMap);

  std::shared_ptr<const quadrature::QuadratureRuleContainer> quadRuleContainerRho = 
                efeBasisDataAdaptiveOrbital->getQuadratureRuleContainer();

   quadrature::QuadratureValuesContainer<double, Host> 
      electronChargeDensity(quadRuleContainerRho, 1, 0.0);

    std::shared_ptr<const utils::ScalarSpatialFunctionReal> rho = std::make_shared
                <RhoFunction>(atomSphericalDataContainer, atomSymbolVec, atomChargesVec, atomCoordinatesVec);
 
  for (size_type iCell = 0; iCell < electronChargeDensity.nCells(); iCell++)
    {
      for (size_type iComp = 0; iComp < 1; iComp++)
        {
          size_type             quadId = 0;
          std::vector<double> a(
            electronChargeDensity.nCellQuadraturePoints(iCell));
          a = (*rho)(quadRuleContainerRho->getCellRealPoints(iCell));
          double *b = a.data();
          electronChargeDensity.template 
            setCellQuadValues<Host>(iCell, iComp, b);
        }
    }

    std::shared_ptr<const utils::ScalarSpatialFunctionReal>
          zeroFunction = std::make_shared
            <utils::ScalarZeroFunctionReal>();
            
    std::shared_ptr<const basis::FEBasisManager
      <double, double, Host,dim>>
    basisManagerWaveFn = std::make_shared
      <basis::FEBasisManager<double, double, Host,dim>>
        (basisDofHandlerWaveFn);

    // std::shared_ptr<const utils::ScalarSpatialFunctionReal> smfunc =
    //   std::make_shared<const utils::SmearChargePotentialFunction>(
    //     atomCoordinatesVec,
    //     atomChargesVec,
    //     smearedChargeRadiusVec);

    std::shared_ptr<const basis::FEBasisManager
      <double, double, Host,dim>>
    basisManagerTotalPot = std::make_shared
      <basis::FEBasisManager<double, double, Host,dim>>
        (basisDofHandlerTotalPot, zeroFunction);

  const utils::ScalarSpatialFunctionReal *externalPotentialFunction = new 
    utils::PointChargePotentialFunction(atomCoordinatesVec, atomChargesVec);

  // set up MPIPatternP2P for the constraints
  auto mpiPatternP2PHanging = basisManagerWaveFn->getMPIPatternP2P();

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

  dftefe::size_type numLocallyOwnedCells  = basisDofHandlerWaveFn->nLocallyOwnedCells();
  auto itField  = X->begin();
  dftefe::utils::Point nodeLoc(dim,0.0);
  dftefe::size_type nodeCount = 0; 
  for (dftefe::size_type iCell = 0; iCell < numLocallyOwnedCells ; iCell++)
    {
      // get cell dof global ids
      std::vector<dftefe::global_size_type> cellGlobalNodeIds;
      basisDofHandlerWaveFn->getCellDofsGlobalIds(iCell, cellGlobalNodeIds);

      // loop over nodes of a cell
      for ( dftefe::size_type iNode = 0 ; iNode < cellGlobalNodeIds.size() ; iNode++)
        {
          // If node not constrained then get the local id and coordinates of the node
          dftefe::global_size_type globalId = cellGlobalNodeIds[iNode];
         if( !basisManagerWaveFn->getConstraints().isConstrained(globalId))
         {
            dftefe::size_type localId = basisManagerWaveFn->globalToLocalIndex(globalId) ;
            basisManagerWaveFn->getBasisCenters(localId,nodeLoc);
            for(int comp = 0 ; comp < numComponents ; comp++)
            {
              double lower_bound = comp;
              double upper_bound = comp+1;
              *(itField + localId*numComponents + comp )  = 1; //lower_bound + (upper_bound - lower_bound) * (random() % max_rand)/ max_rand;
            }
              //((double) rand() / (RAND_MAX));
         }
        }
    }

    X->updateGhostValues();
    basisManagerWaveFn->getConstraints().distributeParentToChild(*X, X->getNumberComponents());

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
                                                      *efeBasisDataAdaptiveOrbital,
                                                      *cfeBasisDataStorageGLL,
                                                      50); 

    // std::shared_ptr<const basis::OrthoEFEOverlapOperatorContext<double,
    //                                               double,
    //                                               Host,
    //                                               dim>> MContextForInv =
    // std::make_shared<basis::OrthoEFEOverlapOperatorContext<double,
    //                                                     double,
    //                                                     Host,
    //                                                     dim>>(
    //                                                     *basisManagerWaveFn,
    //                                                     *basisManagerWaveFn,
    //                                                     *cfeBasisDataStorageGLL,
    //                                                     *efeBasisDataAdaptiveOrbital,
    //                                                     *cfeBasisDataStorageGLL,
    //                                                     50);                                                      

  std::shared_ptr<linearAlgebra::OperatorContext<double,
                                                   double,
                                                   Host>> MInvContext =
    std::make_shared<basis::OrthoEFEOverlapInverseOpContextGLL<double,
                                                   double,
                                                   Host,
                                                   dim>>
                                                   (*basisManagerWaveFn,
                                                    *cfeBasisDataStorageGLL,
                                                    *efeBasisDataAdaptiveOrbital,
                                                    linAlgOpContext);                                                  

    MInvContext->apply(*X,*Y);
    MContext->apply(*Y,*Z);


    for (int comp = 0 ; comp < numComponents ; comp++)
      rootCout << "Component "<<comp << ":" << X->l2Norms()[comp] << "," << Y->l2Norms()[comp] <<"," << Z->l2Norms()[comp] << "\n";

  std::shared_ptr<dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>
   error = std::make_shared<
    dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>(
      mpiPatternP2PHanging, linAlgOpContext, numComponents, double());

  std::vector<double> ones(0);
  ones.resize(numComponents, (double)1.0);
  std::vector<double> nOnes(0);
  nOnes.resize(numComponents, (double)-1.0);

  basisManagerWaveFn->getConstraints().setConstrainedNodesToZero(*X,
                                                          1); 

  dftefe::linearAlgebra::add(ones, *X, nOnes, *Z, *error);

  for (int comp = 0 ; comp < numComponents ; comp++)
        rootCout << "Component "<<comp << ":" << "Error norm: "<<error->l2Norms()[0]<<" Relative error: "<<(error->l2Norms()[comp]/X->l2Norms()[comp])<<"\n";

  std::shared_ptr<const basis::FEBasisDataStorage<double, Host>> feBDTotalChargeStiffnessMatrix = efeBasisDataAdaptiveTotPot;
  std::shared_ptr<const basis::FEBasisDataStorage<double, Host>> feBDTotalChargeRhs = efeBasisDataAdaptiveTotPot;
  std::shared_ptr<const basis::FEBasisDataStorage<double, Host>> feBDElectrostaticsHamiltonian = efeBasisDataAdaptiveOrbital;
  std::shared_ptr<const basis::FEBasisDataStorage<double,Host>> feBDKineticHamiltonian =  efeBasisDataAdaptiveOrbital;
  std::shared_ptr<const basis::FEBasisDataStorage<double, Host>> feBDEXCHamiltonian = efeBasisDataAdaptiveOrbital;


    std::shared_ptr<ksdft::KineticFE<double,
                                      double,
                                      Host,
                                      dim>> hamitonianKin = std::make_shared<ksdft::KineticFE<double,
          double,
          Host,
          dim>>(feBDKineticHamiltonian,
                linAlgOpContext,
                50);

    std::shared_ptr<ksdft::ElectrostaticLocalFE<double,
                                        double,
                                        double,
                                        Host,
                                        dim>>  hamitonianElec =
    std::make_shared<ksdft::ElectrostaticLocalFE<double,
                                          double,
                                          double,
                                          Host,
                                          dim>>(
      atomCoordinatesVec,
      atomChargesVec,
      smearedChargeRadiusVec,
      electronChargeDensity,
      basisManagerTotalPot,
      feBDTotalChargeStiffnessMatrix,
      feBDTotalChargeRhs,
      feBDElectrostaticsHamiltonian,
      *externalPotentialFunction,
      linAlgOpContext,
      50);

    std::shared_ptr<ksdft::ExchangeCorrelationFE<double,
                                        double,
                                        Host,
                                        dim>> hamitonianXC =
    std::make_shared<ksdft::ExchangeCorrelationFE<double,
                                            double,
                                            Host,
                                            dim>>(electronChargeDensity,
                                                  feBDEXCHamiltonian,
                                                  linAlgOpContext,
                                                  50);

  using HamiltonianPtrVariant =
    std::variant<ksdft::Hamiltonian<float, Host> *,
                  ksdft::Hamiltonian<double, Host> *,
                  ksdft::Hamiltonian<std::complex<float>, Host> *,
                  ksdft::Hamiltonian<std::complex<double>, Host> *>;
                                                                    
  std::vector<HamiltonianPtrVariant> hamiltonianComponentsVec{
    /*hamitonianKin.get(),*/ hamitonianElec.get()/*, hamitonianXC.get()*/};
  // form the kohn sham operator
  
    std::shared_ptr<ksdft::KohnShamOperatorContextFE<double,
                                            double,
                                            double,
                                            Host,
                                            dim>>
                                                hamitonianOperator =
    std::make_shared<ksdft::KohnShamOperatorContextFE<double,
                                                double,
                                                double,
                                                Host,
                                                dim>>(
      *basisManagerWaveFn,
      hamiltonianComponentsVec,
      *linAlgOpContext,
      50);

  hamitonianElec->evalEnergy();
  double elecEnergy = hamitonianElec->getEnergy();
  rootCout << "Electrostatic energy with guess density: " << elecEnergy
              << "\n";                                   

    X->setValue(1.0);
    Y->setValue(0.0);
    hamitonianOperator->apply(*X,*Y);

    for (int comp = 0 ; comp < numComponents ; comp++)
      rootCout << "Component "<<comp << ":" << X->l2Norms()[comp] << "," << Y->l2Norms()[comp] << "\n";

  //gracefully end MPI

  int mpiFinalFlag = 0;
  utils::mpi::MPIFinalized(&mpiFinalFlag);
  if(!mpiFinalFlag)
  {
    utils::mpi::MPIFinalize();
  }
}
