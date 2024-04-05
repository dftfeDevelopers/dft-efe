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

int main()
{

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
  unsigned int feDegree = 3;
  unsigned int num1DGaussSize = 4;
  
  // Set up Triangulation
    std::shared_ptr<dftefe::basis::TriangulationBase> triangulationBase =
        std::make_shared<dftefe::basis::TriangulationDealiiParallel<dim>>(comm);
  std::vector<unsigned int>         subdivisions = {20, 20 ,20};
  std::vector<bool>                 isPeriodicFlags(dim, false);
  std::vector<dftefe::utils::Point> domainVectors(dim,
                                                  dftefe::utils::Point(dim, 0.0));

  domainVectors[0][0] = xmax;
  domainVectors[1][1] = ymax;
  domainVectors[2][2] = zmax;

  // initialize the triangulation
  triangulationBase->initializeTriangulationConstruction();
  triangulationBase->createUniformParallelepiped(subdivisions,
                                                 domainVectors,
                                                 isPeriodicFlags);
    triangulationBase->finalizeTriangulationConstruction();

    char* dftefe_path = getenv("DFTEFE_PATH");
    std::string sourceDir;
    // if executes if a non null value is returned
    // otherwise else executes
    if (dftefe_path != NULL) 
    {
      sourceDir = (std::string)dftefe_path + "/test/electrostatics/src/";
    }
    else
    {
      dftefe::utils::throwException(false,
                            "dftefe_path does not exist!");
    }
    std::string atomDataFile = "TwoAtomData1_5.in";
    std::string inputFileName = sourceDir + atomDataFile;

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

  // initialize the basis Manager

  std::shared_ptr<const dftefe::basis::FEBasisDofHandler<double, dftefe::utils::MemorySpace::HOST,dim>> basisDofHandler =  
   std::make_shared<dftefe::basis::CFEBasisDofHandlerDealii<double, dftefe::utils::MemorySpace::HOST,dim>>(triangulationBase, feDegree, comm);

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

  // Set up basis Operations
  dftefe::basis::FEBasisOperations<double, double, dftefe::utils::MemorySpace::HOST,dim> feBasisOp(feBasisData,50);

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

  std::vector<std::vector<dftefe::utils::Point>> atomsVecInDomain(0);
  for (unsigned int i = 0 ; i < nAtoms ; i++)
  {
    std::vector<dftefe::utils::Point> coord{atomCoordinatesVec[i]};
    atomsVecInDomain.push_back(coord);
  }
  atomsVecInDomain.push_back(atomCoordinatesVec);

  std::vector<double> chargeDensity(nAtoms+1, 0.0), mpiReducedChargeDensity(chargeDensity.size(), 0.0);
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

    // set solution

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
      dftefe::size_type quadId = 0;
      for (auto j : quadRuleContainer->getCellRealPoints(i))
      {
        std::vector<double> a(numComponents, 0);
        for (unsigned int k = 0 ; k < numComponents ; k++)
          a[k] = rho( j, atomsVecInDomain[iProb], rc) * (4*M_PI) * (1.0*atomsVecInDomain[iProb].size()/mpiReducedChargeDensity[iProb]);
        double *b = a.data();
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

    double dist = 0;
    for (unsigned int j = 0 ; j < dim ; j++ )
    {
      dist += std::pow((atomCoordinatesVec[0][j]-atomCoordinatesVec[1][j]),2);
    }
    dist = std::sqrt(dist);

    if(rank == 0)
    {
      std::cout << "Integral of b smear over volume: "<< mpiReducedChargeDensity[nAtoms] << "\n";
      std::cout << "The electrostatic energy (analy/num) : "<< (mpiReducedEnergy[nAtoms] + analyticalSelfEnergy) << "," << (mpiReducedEnergy[nAtoms] - 
        numericalSelfEnergy)  << "\n";
      std::cout << "The difference in electrostatic energy from analytical self potential: " << (mpiReducedEnergy[nAtoms] + analyticalSelfEnergy) - 1.0/dist 
        << " Relative difference: "<<((mpiReducedEnergy[nAtoms] + analyticalSelfEnergy) - 1.0/dist)*dist<<"\n";
      std::cout << "The difference in electrostatic energy from numerical self potntial : " << (mpiReducedEnergy[nAtoms] - numericalSelfEnergy) - 1.0/dist 
        << " Relative difference: "<<((mpiReducedEnergy[nAtoms] - numericalSelfEnergy) - 1.0/dist)*dist<<"\n";
    }

  dftefe::utils::mpi::MPIBarrier(comm);

/*
  dftefe::quadrature::QuadratureValuesContainer<double, dftefe::utils::MemorySpace::HOST> potentialQuadGradients(quadRuleContainer, numComponents*dim);

  dftefe::basis::FEBasisOperations<double,
                    double,
                    dftefe::utils::MemorySpace::HOST,
                    dim> feBasisOpStiffnessMatrix(feBasisData, 50);

  feBasisOpStiffnessMatrix.interpolateWithBasisGradient(
    *solution,
    *basisManager,
    potentialQuadGradients);

  auto iter4 = potentialQuadGradients.begin();
  const std::vector<double> JxW1 = quadRuleContainer->getJxW();
  numQuadraturePoints = quadRuleContainer->nQuadraturePoints();
  std::vector<double> gradPotentialSqby8PI(numComponents,0), mpiReducedGradPotentialSqby8PI(gradPotentialSqby8PI.size(),0);
  for (unsigned int i = 0 ; i < numQuadraturePoints ; i++ )
  {
    for (unsigned int j = 0 ; j < dim ; j++ )
    {
      for (unsigned int k = 0 ; k < numComponents ; k++ )
      { 
        gradPotentialSqby8PI[k] += *(i*dim*numComponents+j*numComponents+k+iter4) * *(i*dim*numComponents+j*numComponents+k+iter4) * JxW1[i] * 0.5 * (1/(4*M_PI));
      }
    }
  }

  dftefe::utils::mpi::MPIAllreduce<dftefe::utils::MemorySpace::HOST>(
      &gradPotentialSqby8PI,
      &mpiReducedGradPotentialSqby8PI,
      numComponents,
      dftefe::utils::mpi::MPIDouble,
      dftefe::utils::mpi::MPISum,
      comm);

  for (int k = 0 ; k < numComponents ; k++)
  {
    std::cout << "mpiReducedGradPotentialSqby8PI: " << mpiReducedGradPotentialSqby8PI[k] << std::endl;
    std::cout << "mpiReducedSelfEnergy: " << mpiReducedEnergy[k] << std::endl;
    std::cout << "Difference: " << mpiReducedGradPotentialSqby8PI[k]-mpiReducedEnergy[k] << std::endl;
  }
  */
  //gracefully end MPI

  int mpiFinalFlag = 0;
  dftefe::utils::mpi::MPIFinalized(&mpiFinalFlag);
  if(!mpiFinalFlag)
  {
    dftefe::utils::mpi::MPIFinalize();
  }
}
