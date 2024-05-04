    #include <basis/TriangulationBase.h>
    #include <basis/TriangulationDealiiParallel.h>
    #include <basis/CellMappingBase.h>
    #include <basis/LinearCellMappingDealii.h>
    #include <basis/CFEBasisDofHandlerDealii.h>
    #include <basis/CFEBasisDataStorageDealii.h>
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
    #include <fstream>
    #include <memory>
    #include <cmath>
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

// operator - nabla^2 in weak form
// operand - V_H
// memoryspace - HOST

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
      ret += -21*std::pow((r-rc),3)*(6*r*r + 3*r*rc + rc*rc)/(5*M_PI*std::pow(rc,8));
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
  // argv[1] = "PoissonProblemClassicalTwoBodyForce/param.in"

  std::cout<<" Entering test two body interaction \n";

  // Required to solve : \nabla^2 V_H = g(r,r_c) Solve using CG in linearAlgebra
  // In the weak form the eqn is:
  // (N_i,N_j)*V_H = (N_i, g(r,r_c))
  // Input to CG are : linearSolverFnction. Reqd to create a derived class of the base.
  // For the nabla : LaplaceOperatorContextFE to get \nabla^2(A)*x = y

  //initialize MPI

  int mpiInitFlag = 0;
  dftefe::utils::mpi::MPIInitialized(&mpiInitFlag);
  if(!mpiInitFlag)
  {
    dftefe::utils::mpi::MPIInit(NULL, NULL);
  }

  dftefe::utils::mpi::MPIComm comm = dftefe::utils::mpi::MPICommWorld;

  int rank;
  dftefe::utils::mpi::MPICommRank(comm, &rank);

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
  std::string atomDataFile = "TwoSmearedCharge_dist1.5.in";
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
  double gridSizeFD = readParameter<double>(parameterInputFileName, "gridSizeFD");
  unsigned int numDimPerturbed = readParameter<unsigned int>(parameterInputFileName, "numDimPerturbed");
  unsigned int numAtomPerturbed = readParameter<unsigned int>(parameterInputFileName, "numAtomPerturbed");

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

  std::fstream fstream;
  fstream.open(inputFileName, std::fstream::in);
  
  // read the input file and create atomsymbol vector and atom coordinates vector.
  std::vector<dftefe::utils::Point> atomCoordinatesVec(0,dftefe::utils::Point(dim, 0.0));
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

  const unsigned int nAtoms = atomCoordinatesVec.size(); 
const unsigned int numComponents = 1;
  double distBetweenAtomsBase = 0;
  for (unsigned int j = 0 ; j < dim ; j++ )
  {
    distBetweenAtomsBase += std::pow((atomCoordinatesVec[0][j]-atomCoordinatesVec[1][j]),2);
  }
  distBetweenAtomsBase = std::sqrt(distBetweenAtomsBase);

  std::vector<std::vector<dftefe::utils::Point>> 
    atomCoordinatesVecInGrid(0, std::vector<dftefe::utils::Point>(0,dftefe::utils::Point(dim, 0.0)));
  
  atomCoordinatesVecInGrid.push_back(atomCoordinatesVec);
  for(unsigned int perturbDim = 0 ; perturbDim < numDimPerturbed ; perturbDim ++)
  {
    for(unsigned int perturbAtomId = 0 ; perturbAtomId < numAtomPerturbed ; perturbAtomId++ )
    {
      for(int gridPt = -2 ; gridPt <= 2 ; gridPt++)
      {
        std::vector<dftefe::utils::Point> coordinatesVec(nAtoms,dftefe::utils::Point(dim, 0.0));
        for (unsigned int atomId = 0 ; atomId < nAtoms ; atomId++)
        {
          for (unsigned int iDim = 0 ; iDim < dim ; iDim ++)
          {
            if(atomId == perturbAtomId && iDim == perturbDim)
              coordinatesVec[atomId][iDim] = atomCoordinatesVec[atomId][iDim] + gridPt*gridSizeFD;
            else
              coordinatesVec[atomId][iDim] = atomCoordinatesVec[atomId][iDim];
          }
        }
        if(gridPt!=0)
        atomCoordinatesVecInGrid.push_back(coordinatesVec);
      }
    }
  }

  std::vector<dftefe::utils::Point> atomCoordinatesVecBase = atomCoordinatesVecInGrid[0];


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
      for ( unsigned int i=0 ; i<atomCoordinatesVecBase.size() ; i++)
      {
        double dist = 0;
        for (unsigned int j = 0 ; j < dim ; j++ )
        {
          dist += std::pow((centerPoint[j]-atomCoordinatesVecBase[i][j]),2);
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

  std::vector<double> electrostaticEnergyAnalytical(atomCoordinatesVecInGrid.size(),0.0);
  std::vector<double> electrostaticEnergyNumerical(atomCoordinatesVecInGrid.size(),0.0);

  if(rank == 0)
  {
    std::ofstream myfile;
    std::stringstream ss;
        ss << "CFE"<<"domain_"<<xmax<<"x"<<ymax<<"x"<<zmax<<
        "subdiv_"<<subdivisionx<<"x"<<subdivisiony<<"x"<<subdivisionz<<
        "feOrder_"<<feOrder<<"nQuad_"<<num1DGaussSize<<"hMin_"<<hMin<<".out";
    std::string outputFile = ss.str();
    myfile.open (outputFile, std::ios::out | std::ios::trunc);
    myfile.close();
  }



  std::shared_ptr<const dftefe::basis::FEBasisDofHandler<double, dftefe::utils::MemorySpace::HOST,dim>> basisDofHandler =  
   std::make_shared<dftefe::basis::CFEBasisDofHandlerDealii<double, dftefe::utils::MemorySpace::HOST,dim>>(triangulationBase, feOrder, comm);
  
    std::map<dftefe::global_size_type, dftefe::utils::Point> dofCoords;
  basisDofHandler->getBasisCenters(dofCoords);

  std::cout << "Locally owned cells : " <<basisDofHandler->nLocallyOwnedCells() << "\n";
  std::cout << "Total Number of dofs : " << basisDofHandler->nGlobalNodes() << "\n";

  // Set up the quadrature rule

  dftefe::quadrature::QuadratureRuleAttributes quadAttr(dftefe::quadrature::QuadratureFamily::GAUSS,true,num1DGaussSize);

  dftefe::basis::BasisStorageAttributesBoolMap basisAttrMap;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreValues] = true;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreGradient] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreHessian] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreOverlap] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreGradNiGradNj] = true;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreJxW] = true;

    // Set up the FE Basis Data Storage
        std::shared_ptr<dftefe::basis::FEBasisDataStorage<double, dftefe::utils::MemorySpace::HOST>> feBasisData =
        std::make_shared<dftefe::basis::CFEBasisDataStorageDealii<double, double, dftefe::utils::MemorySpace::HOST,dim>>
    (basisDofHandler, quadAttr, basisAttrMap);
        
        // evaluate basis data
    feBasisData->evaluateBasisData(quadAttr, basisAttrMap);

  // create the quadrature Rule Container

  std::shared_ptr<dftefe::quadrature::QuadratureRule> quadRule =
    std::make_shared<dftefe::quadrature::QuadratureRuleGauss>(dim, num1DGaussSize);

    dftefe::basis::LinearCellMappingDealii<dim> linearCellMappingDealii;
    std::shared_ptr<const dftefe::quadrature::QuadratureRuleContainer> quadRuleContainer =  
                  feBasisData->getQuadratureRuleContainer();

    dftefe::size_type numQuadraturePoints = quadRuleContainer->nQuadraturePoints(), mpinumQuadraturePoints=0;

    dftefe::utils::mpi::MPIAllreduce<dftefe::utils::MemorySpace::HOST>(
        &numQuadraturePoints,
        &mpinumQuadraturePoints,
        1,
        dftefe::utils::mpi::MPIUnsigned,
        dftefe::utils::mpi::MPISum,
        comm);

  // Set up basis Operations
    dftefe::basis::FEBasisOperations<double, double, dftefe::utils::MemorySpace::HOST,dim> feBasisOp(feBasisData,50);

  if(rank == 0)
  {
    std::ofstream myfile;
    std::stringstream ss;
        ss << "CFE"<<"domain_"<<xmax<<"x"<<ymax<<"x"<<zmax<<
        "subdiv_"<<subdivisionx<<"x"<<subdivisiony<<"x"<<subdivisionz<<
        "feOrder_"<<feOrder<<"nQuad_"<<num1DGaussSize<<"hMin_"<<hMin<<".out";
    std::string outputFile = ss.str();
    myfile.open (outputFile, std::ios::out | std::ios::app);
    myfile << "\nTotal Number of dofs : " << basisDofHandler->nGlobalNodes() << "\n";
    myfile << "No. of quad points: "<< mpinumQuadraturePoints << "\n";
    myfile.close();
  }

  // Perturbation loop
  for(unsigned int perturbId = 1 ; perturbId < atomCoordinatesVecInGrid.size(); perturbId++ )
  {
std::vector<std::vector<dftefe::utils::Point>> atomsVecInDomain(0);
    for (unsigned int i = 0 ; i < nAtoms ; i++)
    {
      std::vector<dftefe::utils::Point> coord{atomCoordinatesVecInGrid[perturbId][i]};
      atomsVecInDomain.push_back(coord);
    }
    atomsVecInDomain.push_back(atomCoordinatesVecInGrid[perturbId]);

    double distBetweenAtoms = 0;
    for (unsigned int j = 0 ; j < dim ; j++ )
    {
      distBetweenAtoms += std::pow((atomCoordinatesVecInGrid[perturbId][0][j]-atomCoordinatesVecInGrid[perturbId][1][j]),2);
    }
    distBetweenAtoms = std::sqrt(distBetweenAtoms);
    std::vector<double> chargeDensity(nAtoms+1, 0.0), mpiReducedChargeDensity(chargeDensity.size(), 0.0);
    // Poisson Problem loop
    for(unsigned int iProb = 0 ; iProb < atomsVecInDomain.size() ; iProb++)
    {
      double charge = 0;
      for(dftefe::size_type i = 0 ; i < quadRuleContainer->nCells() ; i++)
    {
      std::vector<double> JxW = quadRuleContainer->getCellJxW(i);
      dftefe::size_type quadId = 0;
      for (auto j : quadRuleContainer->getCellRealPoints(i))
      {
        charge += rho( j, atomsVecInDomain[iProb], rc) * JxW[quadId];
                quadId = quadId + 1;
      }
    }
      chargeDensity[iProb] = charge;
    }

    dftefe::utils::mpi::MPIAllreduce<dftefe::utils::MemorySpace::HOST>(
          chargeDensity.data(),
          mpiReducedChargeDensity.data(),
          chargeDensity.size(),
          dftefe::utils::mpi::MPIDouble,
          dftefe::utils::mpi::MPISum,
          comm);

std::cout << rank << "," << mpiReducedChargeDensity[0] << 
    "," << mpiReducedChargeDensity[1] << "," << mpiReducedChargeDensity[2] << std::endl;

    std::vector<double> energy(nAtoms+1, 0.0), mpiReducedEnergy(energy.size(), 0.0);
  
    for( unsigned int iProb = 0 ; iProb < atomsVecInDomain.size() ; iProb++)
    {
        std::shared_ptr<const dftefe::utils::ScalarSpatialFunctionReal>
          potentialFunction = std::make_shared<ScalarSpatialPotentialFunctionReal>(atomsVecInDomain[iProb], rc);

        // Set up BasisManager for all poisson problems
        std::shared_ptr<const dftefe::basis::FEBasisManager
          <double, double, dftefe::utils::MemorySpace::HOST,dim>>
        basisManager = std::make_shared
          <dftefe::basis::FEBasisManager<double, double, dftefe::utils::MemorySpace::HOST,dim>>
            (basisDofHandler, potentialFunction);

        // set up MPIPatternP2P for the constraints
        auto mpiPatternP2PPotential = basisManager->getMPIPatternP2P();

        std::shared_ptr<dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>
        solution = std::make_shared<
          dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>(
            mpiPatternP2PPotential, linAlgOpContext, numComponents, double());

        solution->setValue(0);

    // Store the charge density at the quadrature points for the poisson problem

dftefe::quadrature::QuadratureValuesContainer<double, dftefe::utils::MemorySpace::HOST> 
                quadValuesContainer(quadRuleContainer, numComponents);
        dftefe::quadrature::QuadratureValuesContainer<double, dftefe::utils::MemorySpace::HOST> 
          quadValuesContainerNumerical(quadRuleContainer, numComponents);

    for(dftefe::size_type i = 0 ; i < quadValuesContainer.nCells() ; i++)
    {
    for(dftefe::size_type iComp = 0 ; iComp < numComponents ; iComp ++)
    {
      dftefe::size_type quadId = 0;
      std::vector<double> a(quadRuleContainer->nCellQuadraturePoints(i));
      for (auto j : quadRuleContainer->getCellRealPoints(i))
      {
        a[quadId] = rho( j, atomsVecInDomain[iProb], rc) * (4*M_PI) * (1.0*atomsVecInDomain[iProb].size()/mpiReducedChargeDensity[iProb]);
        quadId = quadId + 1;
      }
      double *b = a.data();
      quadValuesContainer.setCellQuadValues<dftefe::utils::MemorySpace::HOST> (i, iComp, b);
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

    // perform integral rho vh 

    feBasisOp.interpolate( *solution, *basisManager, quadValuesContainerNumerical);

    auto iter1 = quadValuesContainer.begin();
    auto iter2 = quadValuesContainerNumerical.begin();
        const std::vector<double> JxW = quadRuleContainer->getJxW();
    double e = 0;
    for (unsigned int i = 0 ; i < numQuadraturePoints ; i++ )
    {
      for (unsigned int j = 0 ; j < numComponents ; j++ )
      {
        e += *(i*numComponents+j+iter1) * *(i*numComponents+j+iter2) * JxW[i] * 0.5/(4*M_PI);
      }
    }
      energy[iProb] = e;
    }

    dftefe::utils::mpi::MPIAllreduce<dftefe::utils::MemorySpace::HOST>(
          energy.data(),
          mpiReducedEnergy.data(),
          energy.size(),
          dftefe::utils::mpi::MPIDouble,
          dftefe::utils::mpi::MPISum,
          comm);

    double Ig = 10976./(17875*rc);
    double analyticalSelfEnergy = 0, numericalSelfEnergy = 0;
    for (unsigned int i = 0 ; i < nAtoms ; i++)
    {
      std::vector<dftefe::utils::Point> coord{atomCoordinatesVec[i]};
      analyticalSelfEnergy += 0.5 * (Ig - potential(atomCoordinatesVec[i], coord, rc));
      numericalSelfEnergy += mpiReducedEnergy[i];
    }

    electrostaticEnergyAnalytical[perturbId] = (mpiReducedEnergy[nAtoms] + analyticalSelfEnergy);
    electrostaticEnergyNumerical[perturbId] = (mpiReducedEnergy[nAtoms] - numericalSelfEnergy);
      
    std::cout << "The error in electrostatic energy: " << 
      electrostaticEnergyAnalytical[perturbId] - 1.0/distBetweenAtoms << "\n";

    if(rank == 0)
    {
      std::ofstream myfile;
      std::stringstream ss;
        ss << "CFE"<<"domain_"<<xmax<<"x"<<ymax<<"x"<<zmax<<
        "subdiv_"<<subdivisionx<<"x"<<subdivisiony<<"x"<<subdivisionz<<
        "feOrder_"<<feOrder<<"nQuad_"<<num1DGaussSize<<"hMin_"<<hMin<<".out";
      std::string outputFile = ss.str();
      myfile.open (outputFile, std::ios::out | std::ios::app);
      myfile << "\nAtom Location: ";
      for(auto j : atomCoordinatesVecInGrid[perturbId])
      {
        myfile << j[0] << "," << j[1] << "," << j[2];
        myfile << "\t";
      }
      myfile << "\n";
      myfile << std::fixed << std::setprecision(15) << std::endl;
      myfile << "Integral of b smear over volume: "<< mpiReducedChargeDensity[nAtoms] << "\n";
      myfile << "The electrostatic energy (analy/num) : "<< electrostaticEnergyAnalytical[perturbId] <<
        "," << electrostaticEnergyNumerical[perturbId]  << "\n";
      myfile << "The error in electrostatic energy from analytical self energy: " <<
        electrostaticEnergyAnalytical[perturbId] - 1.0/distBetweenAtoms << 
          " Relative error: "<<(electrostaticEnergyAnalytical[perturbId] 
            - 1.0/distBetweenAtoms)*distBetweenAtoms<<"\n";
      myfile << "The error in electrostatic energy from numerical self energy : " << 
        electrostaticEnergyNumerical[perturbId] - 1.0/distBetweenAtoms << 
          " Relative error: "<<(electrostaticEnergyNumerical[perturbId] 
            - 1.0/distBetweenAtoms)*distBetweenAtoms<<"\n";
      myfile.close();
    }
  }

  std::vector<double> forceAnalytical(dim*nAtoms,0);
  std::vector<double> forceNumerical(dim*nAtoms,0);
  unsigned int count = 0;
  for(unsigned int perturbDim = 0 ; perturbDim < numDimPerturbed ; perturbDim ++)
  {
    for(unsigned int perturbAtomId = 0 ; perturbAtomId < numAtomPerturbed ; perturbAtomId++ )
    {
      unsigned int index = (nAtoms*perturbDim + perturbAtomId)*4;
      forceAnalytical[count] = (-electrostaticEnergyAnalytical[index+1] + 8*electrostaticEnergyAnalytical[index+2] - 
        8*electrostaticEnergyAnalytical[index+3] + electrostaticEnergyAnalytical[index+4])/(12*gridSizeFD);
      forceNumerical[count] = (-electrostaticEnergyNumerical[index+1] + 8*electrostaticEnergyNumerical[index+2] - 
        8*electrostaticEnergyNumerical[index+3] + electrostaticEnergyNumerical[index+4])/(12*gridSizeFD);
      count+=1;
    }
  }

  std::vector<double> forceExpression(nAtoms,0);
  forceExpression[0] = -1/(distBetweenAtomsBase*distBetweenAtomsBase);
  forceExpression[1] = 1/(distBetweenAtomsBase*distBetweenAtomsBase);
  if(rank == 0)
  {
    std::ofstream myfile;
    std::stringstream ss;
        ss << "CFE"<<"domain_"<<xmax<<"x"<<ymax<<"x"<<zmax<<
        "subdiv_"<<subdivisionx<<"x"<<subdivisiony<<"x"<<subdivisionz<<
        "feOrder_"<<feOrder<<"nQuad_"<<num1DGaussSize<<"hMin_"<<hMin<<".out";
    std::string outputFile = ss.str();
    myfile.open (outputFile, std::ios::out | std::ios::app);
    myfile << "\n";
    myfile << std::fixed << std::setprecision(15) << std::endl;
    for(unsigned int perturbDim = 0 ; perturbDim < numDimPerturbed ; perturbDim ++)
    {
      for(unsigned int perturbAtomId = 0 ; perturbAtomId < numAtomPerturbed ; perturbAtomId++ )
      {
        if(perturbDim == 0)
        {
          unsigned int index = (nAtoms*perturbDim + perturbAtomId);
          myfile << "The force by analytical energy cancellation for perturbation to atom "<< perturbAtomId <<
          " along Dim " << perturbDim <<" is: "<< forceAnalytical[nAtoms*perturbDim + perturbAtomId]<<" Relative error: "<< 
            (forceAnalytical[nAtoms*perturbDim + perturbAtomId]-forceExpression[perturbAtomId])/forceExpression[perturbAtomId]<<"\n";
          myfile << "The force by numerical energy cancellation for perturbation to atom "<< perturbAtomId <<
          " along Dim " << perturbDim <<" is: "<< forceNumerical[nAtoms*perturbDim + perturbAtomId]<<" Relative error: "<< 
            (forceNumerical[nAtoms*perturbDim + perturbAtomId]-forceExpression[perturbAtomId])/forceExpression[perturbAtomId]<<"\n";
        }
        else
        {
          myfile << "The force by analytical energy cancellation for perturbation to atom "<< perturbAtomId <<
          " along Dim " << perturbDim <<" is: "<< forceAnalytical[nAtoms*perturbDim + perturbAtomId]<<"\n";
          myfile << "The force by numerical energy cancellation for perturbation to atom "<< perturbAtomId <<
          " along Dim " << perturbDim <<" is: "<< forceNumerical[nAtoms*perturbDim + perturbAtomId]<<"\n";
        }
      }
    }
    myfile.close();
  }

  //gracefully end MPI

  int mpiFinalFlag = 0;
  dftefe::utils::mpi::MPIFinalized(&mpiFinalFlag);
  if(!mpiFinalFlag)
  {
    dftefe::utils::mpi::MPIFinalize();
  }
}
