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
    #include <linearAlgebra/LinearSolverFunction.h>
    #include <electrostatics/PoissonLinearSolverFunctionFE.h>
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

template<typename T>
T readParameter(std::string ParamFile, std::string param)
{
  T t(0);
  std::string line;
  std::fstream fstream;
  fstream.open(ParamFile, std::fstream::in);
  int count = 0;
  while (std::getline(fstream, line))
  {
    for (int i = 0; i < line.length(); i++)
    {
        if (line[i] == ' ')
        {
            line.erase(line.begin() + i);
            i--;
        }
    }
    std::istringstream iss(line);
    std::string type;
    std::getline(iss, type, '=');
    if (type.compare(param) == 0)
    {
      iss >> t;
      count = 1;
      break;
    }
  }
  if(count == 0)
  {
    dftefe::utils::throwException(false, "The parameter is not found: "+ param);
  }
  fstream.close();
  return t;
}


    double rho(const dftefe::utils::Point &point, const std::vector<dftefe::utils::Point> &origin, double rc)
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

    double potential(const dftefe::utils::Point &point, const std::vector<dftefe::utils::Point> &origin, double rc)
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

    class ScalarSpatialPotentialFunctionReal : public dftefe::utils::ScalarSpatialFunctionReal
  {
    public:
    ScalarSpatialPotentialFunctionReal(std::vector<dftefe::utils::Point> &origin, double rc)
    :d_rc(rc), d_origin(origin)
    {}

    double
    operator()(const dftefe::utils::Point &point) const
    {
      return potential(point, d_origin, d_rc);
    }

    std::vector<double>
    operator()(const std::vector<dftefe::utils::Point> &points) const
    {
      std::vector<double> ret(0);
      ret.resize(points.size());
      for (unsigned int i = 0 ; i < points.size() ; i++)
      {
        ret[i] = potential(points[i], d_origin, d_rc);
      }
      return ret;
    }

    private:
    std::vector<dftefe::utils::Point> d_origin; 
    double d_rc;
  };

int main(int argc, char** argv)
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

        // Read the parameter files and atom coordinate files
        char* dftefe_path = getenv("DFTEFE_PATH");
        std::string sourceDir;
        // if executes if a non null value is returned
        // otherwise else executes
        if (dftefe_path != NULL) 
        {
        sourceDir = (std::string)dftefe_path + "/analysis/classicalEnrichmentComparison/";
        }
        else
        {
        dftefe::utils::throwException(false,
                                "dftefe_path does not exist!");
        }
        std::string atomDataFile = "SingleSmearedCharge.in";
        std::string enrichmentDataFile = "SmearedCharge3e-5Uniform_rc0.6_cutoff3_sm0.6.xml";
        std::string paramDataFile = argv[1];
        std::string inputFileName = sourceDir + atomDataFile;
        std::string parameterInputFileName = sourceDir + paramDataFile;

        double xmax = readParameter<double>(parameterInputFileName, "xmax");
        double ymax = readParameter<double>(parameterInputFileName, "ymax");
        double zmax = readParameter<double>(parameterInputFileName, "zmax");
        unsigned int subdivisionx = readParameter<unsigned int>(parameterInputFileName, "subdivisionx");
        unsigned int subdivisiony = readParameter<unsigned int>(parameterInputFileName, "subdivisiony");
        unsigned int subdivisionz = readParameter<unsigned int>(parameterInputFileName, "subdivisionz");
        double rc = readParameter<double>(parameterInputFileName, "rc");
        double hMin = readParameter<double>(parameterInputFileName, "hMin");
        unsigned int maxIter = readParameter<unsigned int>(parameterInputFileName, "maxIter");
        double absoluteTol = readParameter<double>(parameterInputFileName, "absoluteTol");
        double relativeTol = readParameter<double>(parameterInputFileName, "relativeTol");
        double divergenceTol = readParameter<double>(parameterInputFileName, "divergenceTol");
        double refineradius = readParameter<double>(parameterInputFileName, "refineradius");
        unsigned int num1DGaussSize = readParameter<unsigned int>(parameterInputFileName, "num1DQuadratureSize");
        unsigned int feOrder = readParameter<unsigned int>(parameterInputFileName, "feOrder");
        double atomPartitionTolerance = readParameter<double>(parameterInputFileName, "atomPartitionTolerance");
        double smallestCellVolume = readParameter<double>(parameterInputFileName, "smallestCellVolume");
        unsigned int maxRecursion = readParameter<unsigned int>(parameterInputFileName, "maxRecursion");
        double adaptiveQuadAbsTolerance = readParameter<double>(parameterInputFileName, "adaptiveQuadAbsTolerance");
        double adaptiveQuadRelTolerance = readParameter<double>(parameterInputFileName, "adaptiveQuadRelTolerance");
        double integralThreshold = readParameter<double>(parameterInputFileName, "integralThreshold");
        unsigned int numComponents = 1;

        // Set up Triangulation
        const unsigned int dim = 3;
        std::shared_ptr<dftefe::basis::TriangulationBase> triangulationBase =
        std::make_shared<dftefe::basis::TriangulationDealiiParallel<dim>>(comm);
        std::vector<unsigned int>         subdivisions = {subdivisionx, subdivisiony ,subdivisionz};
        std::vector<bool>                 isPeriodicFlags(dim, false);
        std::vector<dftefe::utils::Point> domainVectors(dim,
                                                        dftefe::utils::Point(dim, 0.0));

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
        triangulationBase->shiftTriangulation(dftefe::utils::Point(origin));
        triangulationBase->finalizeTriangulationConstruction();

        // Enrichment data file consisting of g(r,\theta,\phi) = f(r)*Y_lm(\theta, \phi)
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
            atomSymbolToFilename[i] = sourceDir + enrichmentDataFile;
        }

        std::vector<std::string> fieldNames{"vnuclear"};
        std::vector<std::string> metadataNames{ "symbol", "Z", "charge", "NR", "r" };
        std::shared_ptr<dftefe::atoms::AtomSphericalDataContainer>  atomSphericalDataContainer = 
            std::make_shared<dftefe::atoms::AtomSphericalDataContainer>(atomSymbolToFilename,
                                                            fieldNames,
                                                            metadataNames);

        std::string fieldName = "vnuclear";

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
        absoluteTolerances[i] = adaptiveQuadAbsTolerance;
        relativeTolerances[i] = adaptiveQuadRelTolerance;
        integralThresholds[i] = integralThreshold;
        }


        //Set up quadAttr for Rhs and OverlapMatrix

        dftefe::quadrature::QuadratureRuleAttributes quadAttrAdaptive(dftefe::quadrature::QuadratureFamily::ADAPTIVE,false);

        // Set up base quadrature rule for adaptive quadrature 

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
    std::shared_ptr<dftefe::basis::FEBasisDofHandler<double, dftefe::utils::MemorySpace::HOST,dim>> basisDofHandler =  
    std::make_shared<dftefe::basis::EFEBasisDofHandlerDealii<double, double,dftefe::utils::MemorySpace::HOST,dim>>(
        enrichClassIntfce, feOrder, comm);

    std::map<dftefe::global_size_type, dftefe::utils::Point> dofCoords;
    basisDofHandler->getBasisCenters(dofCoords);

    std::cout << "Locally owned cells : " <<basisDofHandler->nLocallyOwnedCells() << "\n";
    std::cout << "Total Number of dofs : " << basisDofHandler->nGlobalNodes() << "\n";

        dftefe::basis::BasisStorageAttributesBoolMap basisAttrMap;
        basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreValues] = true;
        basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreGradient] = false;
        basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreHessian] = false;
        basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreOverlap] = false;
        basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreGradNiGradNj] = true;
        basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreJxW] = true;

        // Set up Adaptive quadrature for EFE Basis Data Storage
        std::shared_ptr<dftefe::basis::FEBasisDataStorage<double, dftefe::utils::MemorySpace::HOST>> feBasisData =
        std::make_shared<dftefe::basis::EFEBasisDataStorageDealii<double, double, dftefe::utils::MemorySpace::HOST,dim>>
        (basisDofHandler, quadAttrAdaptive, basisAttrMap);

        // evaluate basis data
        feBasisData->evaluateBasisData(quadAttrAdaptive, quadRuleContainerAdaptive, basisAttrMap);


        std::shared_ptr<const dftefe::utils::ScalarSpatialFunctionReal>
    potentialFunction = std::make_shared<ScalarSpatialPotentialFunctionReal>(atomCoordinatesVec, rc);

  // // Set up BasisManager
        std::shared_ptr<const dftefe::basis::FEBasisManager<double, double, dftefe::utils::MemorySpace::HOST,dim>> basisManager =
            std::make_shared<dftefe::basis::FEBasisManager<double, double, dftefe::utils::MemorySpace::HOST,dim>>
            (basisDofHandler, potentialFunction);

        // Set up basis Operations
        dftefe::basis::FEBasisOperations<double, double, dftefe::utils::MemorySpace::HOST,dim> feBasisOp(feBasisData,50);

        // set up MPIPatternP2P for the constraints
        auto mpiPatternP2PPotential = basisManager->getMPIPatternP2P();

        std::shared_ptr<dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>
        solution = std::make_shared<
        dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>(
            mpiPatternP2PPotential, linAlgOpContext, numComponents, double());

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
        std::make_shared<dftefe::electrostatics::PoissonLinearSolverFunctionFE<double,
                                                        double,
                                                        dftefe::utils::MemorySpace::HOST,
                                                        dim>>
                                                        (basisManager,
                                                        feBasisData,
                                                        feBasisData,
                                                        quadValuesContainer,
                                                        dftefe::linearAlgebra::PreconditionerType::JACOBI ,
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

        feBasisOp.interpolate( *solution, *basisManager, quadValuesContainerNumerical);

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
        
        std::cout << "\n The self energy: "<< analyticalSelfPotantial << " Error in self energy: " << (mpiReducedIntegral[1] + analyticalSelfPotantial) << "\n";

        if(rank == 0)
        {
        std::ofstream myfile;
        std::stringstream ss;
        ss << "EFE"<<"domain_"<<xmax<<"x"<<ymax<<"x"<<zmax<<
        "subdiv_"<<subdivisionx<<"x"<<subdivisiony<<"x"<<subdivisionz<<
        "feOrder_"<<feOrder<<"hMin_"<<hMin<<"nQuad_"<<num1DGaussSize<<
        "adapAbsTol_"<<adaptiveQuadAbsTolerance<<"adapRelTol_"<<adaptiveQuadRelTolerance<<
        "threshold"<<integralThreshold<<".out";
        std::string outputFile = ss.str();
        myfile.open (outputFile, std::ios::out | std::ios::trunc);
          myfile << "Total Number of dofs : " << basisDofHandler->nGlobalNodes() << "\n";
          myfile << "No. of quad points: "<< mpinumQuadraturePoints << "\n";
          myfile << "Integral of b over volume: "<< mpiReducedIntegral[2] << "\n";
          myfile << "The L2 potential norm: " << std::sqrt(mpiReducedIntegral[0]) << "\n";
          myfile << "The self energy: "<< analyticalSelfPotantial << " Error in self energy: "
            << (mpiReducedIntegral[1] + analyticalSelfPotantial) << ", Relative Error: "
            << (mpiReducedIntegral[1] + analyticalSelfPotantial)/analyticalSelfPotantial << "\n";
        myfile.close();
        }

        int mpiFinalFlag = 0;
        dftefe::utils::mpi::MPIFinalized(&mpiFinalFlag);
        if(!mpiFinalFlag)
        {
            dftefe::utils::mpi::MPIFinalize();
        }
    }
