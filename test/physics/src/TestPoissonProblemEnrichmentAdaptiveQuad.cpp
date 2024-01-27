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
#include <linearAlgebra/LinearSolverFunction.h>
#include <physics/PoissonLinearSolverFunctionFE.h>
#include <linearAlgebra/LinearAlgebraProfiler.h>
#include <linearAlgebra/CGLinearSolver.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/dofs/dof_tools.h>

#include <iostream>

double rho(dftefe::utils::Point &point, std::vector<dftefe::utils::Point> &origin, double rc)
{
double ret = 0;
// The function should have homogeneous dirichlet BC
for (unsigned int i = 0 ; i < origin.size() ; i++ )
{
double r = 0;
for (unsigned int j = 0 ; j < point.size() ; j++ )
{
    r += std::pow((point[j]-origin[i][j]),2);
}
r = std::sqrt(r);
if( r > rc )
    ret += 0;
else
    ret += -21*std::pow((r-rc),3)*(6*r*r + 3*r*rc + rc*rc)/(5*M_PI*std::pow(rc,8))*4*M_PI;
}
return ret;
}

double potential(dftefe::utils::Point &point, std::vector<dftefe::utils::Point> &origin, double rc)
{
    double ret = 0;
    // The function should have homogeneous dirichlet BC
    for (unsigned int i = 0 ; i < origin.size() ; i++ )
    {
    double r = 0;
    for (unsigned int j = 0 ; j < point.size() ; j++ )
    {
        r += std::pow((point[j]-origin[i][j]),2);
    }
    r = std::sqrt(r);
    if( r > rc )
        ret += 1/r;
    else
        ret += (9*std::pow(r,7)-30*std::pow(r,6)*rc
        +28*std::pow(r,5)*std::pow(rc,2)-14*std::pow(r,2)*std::pow(rc,5)
        +12*std::pow(rc,7))/(5*std::pow(rc,8));
    }
    return ret;
} 

int main()
{

    std::cout<<" Entering test poisson problem enrichement\n";

    //initialize MPI

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
    std::vector<unsigned int>         subdivisions = {10, 10, 10};
    std::vector<bool>                 isPeriodicFlags(dim, false);
    std::vector<dftefe::utils::Point> domainVectors(dim,
                                                    dftefe::utils::Point(dim, 0.0));

    double xmax = 20.0;
    double ymax = 20.0;
    double zmax = 20.0;
    double rc = 0.5;
    unsigned int numComponents = 1;
    double hMin = 1e6;
    dftefe::size_type maxIter = 2e7;
    double absoluteTol = 1e-10;
    double relativeTol = 1e-12;
    double divergenceTol = 1e10;
    double refineradius = 3*rc;

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
    char* dftefe_path = getenv("DFTEFE_PATH");
    std::string sourceDir;
    // if executes if a non null value is returned
    // otherwise else executes
    if (dftefe_path != NULL) 
    {
      sourceDir = (std::string)dftefe_path + "/test/physics/src/";
    }
    else
    {
      dftefe::utils::throwException(false,
                            "dftefe_path does not exist!");
    }

    std::string atomDataFile = "SingleAtomData.in";
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
        dftefe::utils::Point centerPoint(dim, 0.0); 
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
    int err = dftefe::utils::mpi::MPIAllreduce<dftefe::utils::MemorySpace::HOST>(
        &flag,
        &mpiReducedFlag,
        1,
        dftefe::utils::mpi::MPIInt,
        dftefe::utils::mpi::MPIMax,
        comm);
    std::pair<bool, std::string> mpiIsSuccessAndMsg =
        dftefe::utils::mpi::MPIErrIsSuccessAndMsg(err);
    dftefe::utils::throwException(mpiIsSuccessAndMsg.first,
                            "MPI Error:" + mpiIsSuccessAndMsg.second);
    }

  // Make pristine EFE basis

  // 1. Make EnrichmentClassicalInterface object for Pristine enrichment
  // 2. Input to the EFEBasisManager(eci, feOrder) 
  // 3. Make EFEBasisDataStorage with input as quadratureContainer.

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
    std::vector<double> absoluteTolerances(numfun), relativeTolerances(numfun);
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
        absoluteTolerances[i] = 1e-4;
        relativeTolerances[i] = 1e-4;
        integralThresholds[i] = 1e-10;
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
    std::string constraintHomwHan = "HomogeneousWithHanging"; // use this to solve the laplace equation

    std::vector<std::shared_ptr<dftefe::basis::FEConstraintsBase<double, dftefe::utils::MemorySpace::HOST>>>
      constraintsVec;
    constraintsVec.resize(2);
    for ( unsigned int i=0 ;i < constraintsVec.size() ; i++ )
    constraintsVec[i] = std::make_shared<dftefe::basis::EFEConstraintsDealii<double, double, dftefe::utils::MemorySpace::HOST, dim>>();

    constraintsVec[0]->clear();
    constraintsVec[0]->makeHangingNodeConstraint(basisManager);
    constraintsVec[0]->close();

    constraintsVec[1]->clear();
    constraintsVec[1]->makeHangingNodeConstraint(basisManager);
    constraintsVec[1]->setHomogeneousDirichletBC();
    constraintsVec[1]->close();

    std::map<std::string,
            std::shared_ptr<const dftefe::basis::Constraints<double, dftefe::utils::MemorySpace::HOST>>> constraintsMap;

    constraintsMap[constraintHanging] = constraintsVec[0];
    constraintsMap[constraintHomwHan] = constraintsVec[1];

    dftefe::basis::BasisStorageAttributesBoolMap basisAttrMap;
    basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreValues] = true;
    basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreGradient] = false;
    basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreHessian] = false;
    basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreOverlap] = false;
    basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreGradNiGradNj] = true;
    basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreJxW] = true;

    // Set up Adaptive quadrature for EFE Basis Data Storage
    std::shared_ptr<dftefe::basis::FEBasisDataStorage<double, dftefe::utils::MemorySpace::HOST>> feBasisData =
    std::make_shared<dftefe::basis::EFEBasisDataStorageDealii<double, dftefe::utils::MemorySpace::HOST,dim>>
    (basisManager, quadAttrAdaptive, basisAttrMap);

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
    auto mpiPatternP2PHomwHan = basisHandler->getMPIPatternP2P(constraintHomwHan);

    // set up different multivectors - vh with inhomogeneous BC, vh

    std::shared_ptr<dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>
    vhNHDB = std::make_shared<
    dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>(
        mpiPatternP2PHanging, linAlgOpContext, numComponents, double());

    std::shared_ptr<dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>
    solution = std::make_shared<
    dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>(
        mpiPatternP2PHomwHan, linAlgOpContext, numComponents, double());

    //populate the value of the Density at the nodes for interpolating to quad points
    auto numLocallyOwnedCells  = basisManager->nLocallyOwnedCells();
    dftefe::utils::Point nodeLoc(dim,0.0);
    // vector for lhs
    auto itField  = vhNHDB->begin();
    const unsigned int dofs_per_cell =
        basisManager->nCellDofs(0);
    const unsigned int faces_per_cell =
        dealii::GeometryInfo<dim>::faces_per_cell;
    const unsigned int dofs_per_face =
        std::pow((basisManager->getFEOrder(0)+1),2);
    std::vector<dftefe::global_size_type> cellGlobalDofIndices(dofs_per_cell);
    std::vector<dftefe::global_size_type> iFaceGlobalDofIndices(dofs_per_face);
    std::vector<bool> dofs_touched(basisManager->nGlobalNodes(), false);
    auto icell = basisManager->beginLocallyOwnedCells();
    for (; icell != basisManager->endLocallyOwnedCells(); ++icell)
    {
        (*icell)->cellNodeIdtoGlobalNodeId(cellGlobalDofIndices);
        for (unsigned int iFace = 0; iFace < faces_per_cell; ++iFace)
        {
            (*icell)->getFaceDoFGlobalIndices(iFace, iFaceGlobalDofIndices);
            const dftefe::size_type boundaryId = (*icell)->getFaceBoundaryId(iFace);
            if (boundaryId == 0)
            {
                for (unsigned int iFaceDof = 0; iFaceDof < dofs_per_face;
                    ++iFaceDof)
                {
                    const dftefe::global_size_type globalId =
                    iFaceGlobalDofIndices[iFaceDof];
                    if (dofs_touched[globalId])
                    continue;
                    dofs_touched[globalId] = true;
                    if (!basisHandler->getConstraints(constraintHanging).isConstrained(globalId))
                    {
                        dftefe::size_type localId = basisHandler->globalToLocalIndex(globalId,constraintHanging) ;
                        basisHandler->getBasisCenters(localId,constraintHanging,nodeLoc);
                        *(itField + localId )  = potential(nodeLoc, atomCoordinatesVec, rc);  
                    } // non-hanging node check
                }     // Face dof loop
            }
        } // Face loop
    }     // cell locally owned

    // create the quadrature Value Container

    dftefe::basis::LinearCellMappingDealii<dim> linearCellMappingDealii;
    std::shared_ptr<const dftefe::quadrature::QuadratureRuleContainer> quadRuleContainer =  
                feBasisData->getQuadratureRuleContainer();

    dftefe::quadrature::QuadratureValuesContainer<double, dftefe::utils::MemorySpace::HOST> quadValuesContainer(quadRuleContainer, numComponents);
    dftefe::quadrature::QuadratureValuesContainer<double, dftefe::utils::MemorySpace::HOST> quadValuesContainerAnalytical(quadRuleContainer, numComponents);
    dftefe::quadrature::QuadratureValuesContainer<double, dftefe::utils::MemorySpace::HOST> quadValuesContainerNumerical(quadRuleContainer, numComponents);


    for(dftefe::size_type i = 0 ; i < quadValuesContainer.nCells() ; i++)
    {
        dftefe::size_type quadId = 0;
        for (auto j : quadRuleContainer->getCellRealPoints(i))
        {
            double a = rho( j, atomCoordinatesVec, rc);
            double *b = &a;
            quadValuesContainer.setCellQuadValues<dftefe::utils::MemorySpace::HOST> (i, quadId, b);
            quadId = quadId + 1;
        }
    }


    std::shared_ptr<dftefe::linearAlgebra::LinearSolverFunction<double,
                                                    double,
                                                    dftefe::utils::MemorySpace::HOST>> linearSolverFunction =
    std::make_shared<dftefe::physics::PoissonLinearSolverFunctionFE<double,
                                                    double,
                                                    dftefe::utils::MemorySpace::HOST,
                                                    dim>>
                                                    (basisHandler,
                                                    feBasisData,
                                                    feBasisData,
                                                    quadValuesContainer,
                                                    constraintHanging,
                                                    constraintHomwHan,
                                                    *vhNHDB,
                                                    dftefe::linearAlgebra::PreconditionerType::JACOBI,
                                                    linAlgOpContext,
                                                    50);

    dftefe::linearAlgebra::LinearAlgebraProfiler profiler;

    std::shared_ptr<dftefe::linearAlgebra::LinearSolverImpl<double,
                                                    double,
                                                    dftefe::utils::MemorySpace::HOST>> CGSolve =
    std::make_shared<dftefe::linearAlgebra::CGLinearSolver<double,
                                                    double,
                                                    dftefe::utils::MemorySpace::HOST>>
                                                    ( maxIter,
                                                    absoluteTol,
                                                    relativeTol,
                                                    divergenceTol,
                                                    profiler);

    CGSolve->solve(*linearSolverFunction);

    linearSolverFunction->getSolution(*solution);

    for(dftefe::size_type i = 0 ; i < quadValuesContainerAnalytical.nCells() ; i++)
    {
        dftefe::size_type quadId = 0;
        for (auto j : quadRuleContainer->getCellRealPoints(i))
        {
            double a = potential( j, atomCoordinatesVec, rc);
            double *b = &a;
            quadValuesContainerAnalytical.setCellQuadValues<dftefe::utils::MemorySpace::HOST> (i, quadId, b);
            quadId = quadId + 1;
        }
    }

    feBasisOp.interpolate( *solution, constraintHanging, *basisHandler, quadValuesContainerNumerical);

    auto iterPotAnalytic = quadValuesContainerAnalytical.begin();
    auto iterPotNumeric = quadValuesContainerNumerical.begin();
    auto iterRho = quadValuesContainer.begin();
    dftefe::size_type numQuadraturePoints = quadRuleContainer->nQuadraturePoints(), mpinumQuadraturePoints=0;
    const std::vector<double> JxW = quadRuleContainer->getJxW();
    std::vector<double> integral(3, 0.0), mpiReducedIntegral(integral.size(), 0.0);
    const std::vector<dftefe::utils::Point> & locQuadPoints = quadRuleContainer->getRealPoints();
    int count = 0;

    for (unsigned int i = 0 ; i < numQuadraturePoints ; i++ )
    {
        integral[0] += std::pow((*(i+iterPotAnalytic) - *(i+iterPotNumeric)),2) * JxW[i];
        if(std::abs(*(i+iterPotAnalytic) - *(i+iterPotNumeric)) > 1e-2)
        {
            count = count + 1;
        }
        integral[1] += *(i+iterRho) * *(i+iterPotNumeric) * JxW[i] * 0.5/(4*M_PI);
        integral[2] += *(i+iterRho) * JxW[i]/(4*M_PI);
    }

    dftefe::utils::mpi::MPIAllreduce<dftefe::utils::MemorySpace::HOST>(
        &numQuadraturePoints,
        &mpinumQuadraturePoints,
        1,
        dftefe::utils::mpi::MPIUnsigned,
        dftefe::utils::mpi::MPISum,
        comm);

    std::cout << "No. of quad points: "<< mpinumQuadraturePoints<<"\n";

    dftefe::utils::mpi::MPIAllreduce<dftefe::utils::MemorySpace::HOST>(
        integral.data(),
        mpiReducedIntegral.data(),
        integral.size(),
        dftefe::utils::mpi::MPIDouble,
        dftefe::utils::mpi::MPISum,
        comm);

    std::cout << "Integral of b over volume: "<< mpiReducedIntegral[2]<<"\n";

    std::cout << "The integral L2 norm of potential: " << std::sqrt(mpiReducedIntegral[0]) << "\n";

    double Ig = 10976./(17875*rc);
    double vg0 = potential(atomCoordinatesVec[0], atomCoordinatesVec, rc);
    double analyticalSelfPotantial = 0.5 * (Ig - vg0);
    
    std::cout << "\n The self energy: "<< analyticalSelfPotantial << " Difference in self energy wrt analytical: " << (mpiReducedIntegral[1] + analyticalSelfPotantial) << "\n";

    int mpiFinalFlag = 0;
    dftefe::utils::mpi::MPIFinalized(&mpiFinalFlag);
    if(!mpiFinalFlag)
    {
        dftefe::utils::mpi::MPIFinalize();
    }
}
