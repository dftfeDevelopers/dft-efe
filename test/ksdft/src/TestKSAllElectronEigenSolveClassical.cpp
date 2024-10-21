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
#include <ksdft/ElectrostaticLocalFE.h>
#include <ksdft/KineticFE.h>
#include <ksdft/ExchangeCorrelationFE.h>
#include <ksdft/KohnShamOperatorContextFE.h>
#include <ksdft/KohnShamEigenSolver.h>
#include <basis/CFEOverlapInverseOpContextGLL.h>
#include <utils/PointChargePotentialFunction.h>
#include <ksdft/DensityCalculator.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/dofs/dof_tools.h>

#include <iostream>

using namespace dftefe;
const utils::MemorySpace Host = utils::MemorySpace::HOST;

// e- charge density
double rho1sOrbital(const dftefe::utils::Point &point, const std::vector<dftefe::utils::Point> &origin)
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
    ret += (1/M_PI)*exp(-2*r);
  }
  return ret;
}

// operator - nabla^2 in weak form
// operand - V_H
// memoryspace - HOST
int main()
{
  std::cout<<" Entering test kohn sham eigensolve classical \n";
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

    // Get nProcs
    int numProcs;
    utils::mpi::MPICommSize(comm, &numProcs);

  int blasQueue = 0;
  int lapackQueue = 0;
  std::shared_ptr<linearAlgebra::blasLapack::BlasQueue
    <Host>> blasQueuePtr = std::make_shared
      <linearAlgebra::blasLapack::BlasQueue
        <Host>>(blasQueue);
  std::shared_ptr<linearAlgebra::blasLapack::LapackQueue
    <Host>> lapackQueuePtr = std::make_shared
      <linearAlgebra::blasLapack::LapackQueue
        <Host>>(lapackQueue);
  std::shared_ptr<linearAlgebra::LinAlgOpContext
    <Host>> linAlgOpContext = 
    std::make_shared<linearAlgebra::LinAlgOpContext
    <Host>>(blasQueuePtr, lapackQueuePtr);

  // Set up Triangulation
  const unsigned int dim = 3;
  double xmax = 24.0;
  double ymax = 24.0;
  double zmax = 24.0;
  double rc = 0.6;
  double hMin = 1e6;
  size_type maxIter = 2e7;
  double absoluteTol = 1e-10;
  double relativeTol = 1e-12;
  double divergenceTol = 1e10;
  double refineradius = 3*rc;
  unsigned int feDegree = 3;
  unsigned int num1DGaussSize = 4;
  unsigned int num1DGLLSize = 4;

  double    smearingTemperature = 10.0;
  double    fermiEnergyTolerance = 1e-10;
  double    fracOccupancyTolerance = 1e-3;
  double    eigenSolveResidualTolerance = 1e-4;
  size_type chebyshevPolynomialDegree = 50;
  size_type maxChebyshevFilterPass = 100;
  size_type numWantedEigenvalues = 15;
  size_type numElectrons = 1;
  double nuclearCharge = -1.0;
  
  // Set up Triangulation
    std::shared_ptr<basis::TriangulationBase> triangulationBase =
        std::make_shared<basis::TriangulationDealiiParallel<dim>>(comm);
  std::vector<unsigned int>         subdivisions = {20, 20, 20};
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
  triangulationBase->shiftTriangulation(dftefe::utils::Point(origin));
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

  std::vector<double> atomChargesVec(atomCoordinatesVec.size(), nuclearCharge);
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

  // initialize the basis Manager

  std::shared_ptr<const basis::FEBasisDofHandler<double, Host,dim>> basisDofHandler =  
   std::make_shared<basis::CFEBasisDofHandlerDealii<double, Host,dim>>(triangulationBase, feDegree, comm);

  std::map<global_size_type, utils::Point> dofCoords;
  basisDofHandler->getBasisCenters(dofCoords);

  std::cout << "Locally owned cells : " <<basisDofHandler->nLocallyOwnedCells() << "\n";
  std::cout << "Total Number of dofs : " << basisDofHandler->nGlobalNodes() << "\n";

  // Set up the quadrature rule

  quadrature::QuadratureRuleAttributes quadAttr(quadrature::QuadratureFamily::GAUSS,true,num1DGaussSize);

  basis::BasisStorageAttributesBoolMap basisAttrMap;
  basisAttrMap[basis::BasisStorageAttributes::StoreValues] = true;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradient] = true;
  basisAttrMap[basis::BasisStorageAttributes::StoreHessian] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreOverlap] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradNiGradNj] = true;
  basisAttrMap[basis::BasisStorageAttributes::StoreJxW] = true;

  // Set up the FE Basis Data Storage
  std::shared_ptr<basis::FEBasisDataStorage<double, Host>> feBasisData =
    std::make_shared<basis::CFEBasisDataStorageDealii<double, double, Host,dim>>
    (basisDofHandler, quadAttr, basisAttrMap);

  // evaluate basis data
  feBasisData->evaluateBasisData(quadAttr, basisAttrMap);

  std::shared_ptr<const quadrature::QuadratureRuleContainer> quadRuleContainer =  
                feBasisData->getQuadratureRuleContainer();

    std::vector<double> atomChargesVecScaled(atomCoordinatesVec.size());
  for(size_type i = 0 ; i < atomCoordinatesVec.size() ; i++)
  {
    std::vector<double> chargeDensity(1, 0.0), mpiReducedChargeDensity(1, 0.0);

    const utils::SmearChargeDensityFunction smden(atomCoordinatesVec[i],
                                                  atomChargesVec[i],
                                                  smearedChargeRadiusVec[i]);

      double charge = 0;
      for(size_type i = 0 ; i < quadRuleContainer->nCells() ; i++)
      {
        std::vector<double> JxW = quadRuleContainer->getCellJxW(i);
        size_type quadId = 0;
        for (auto j : quadRuleContainer->getCellRealPoints(i))
        {
          charge += smden(j) * JxW[quadId];
          quadId = quadId + 1;
        }
      }
      chargeDensity[0] = charge;
    
    utils::mpi::MPIAllreduce<Host>(
          chargeDensity.data(),
          mpiReducedChargeDensity.data(),
          chargeDensity.size(),
          utils::mpi::MPIDouble,
          utils::mpi::MPISum,
          comm);

    atomChargesVecScaled[i] *= 1/std::abs(mpiReducedChargeDensity[0]);
  }

   quadrature::QuadratureValuesContainer<double, Host> 
      electronChargeDensity(quadRuleContainer, 1, 0.0);

    std::shared_ptr<const utils::ScalarSpatialFunctionReal>
          zeroFunction = std::make_shared
            <utils::ScalarZeroFunctionReal>();
            
    std::shared_ptr<const basis::FEBasisManager
      <double, double, Host,dim>>
    basisManagerWaveFn = std::make_shared
      <basis::FEBasisManager<double, double, Host,dim>>
        (basisDofHandler);

    std::shared_ptr<const utils::ScalarSpatialFunctionReal> smfunc =
      std::make_shared<const utils::SmearChargePotentialFunction>(
        atomCoordinatesVec,
        atomChargesVecScaled,
        smearedChargeRadiusVec);

    std::shared_ptr<const basis::FEBasisManager
      <double, double, Host,dim>>
    basisManagerTotalPot = std::make_shared
      <basis::FEBasisManager<double, double, Host,dim>>
        (basisDofHandler, smfunc);

  std::shared_ptr<ksdft::KineticFE<double,
                                    double,
                                    Host,
                                    dim>> 
                                  hamitonianKin =
    std::make_shared<ksdft::KineticFE<double,
                                      double,
                                      Host,
                                      dim>>
                                      (feBasisData,
                                      linAlgOpContext,
                                      50);

  const utils::ScalarSpatialFunctionReal *externalPotentialFunction = new 
    utils::PointChargePotentialFunction(atomCoordinatesVec, atomChargesVec);

  std::shared_ptr<ksdft::ElectrostaticLocalFE<double,
                                                  double,
                                                  double,
                                                  Host,
                                                  dim>> 
                                            hamitonianElec =
    std::make_shared<ksdft::ElectrostaticLocalFE<double,
                                                  double,
                                                  double,
                                                  Host,
                                                  dim>>
                                                  (atomCoordinatesVec,
                                                  atomChargesVec,
                                                  smearedChargeRadiusVec,
                                                  electronChargeDensity,
                                                  basisManagerTotalPot,
                                                  feBasisData,
                                                  feBasisData,   
                                                  feBasisData,
                                                  feBasisData, 
                                                  feBasisData,                                                                                          
                                                  *externalPotentialFunction,
                                                  linAlgOpContext,
                                                  50);
                                                  

    using HamiltonianPtrVariant =
      std::variant<ksdft::Hamiltonian<float, Host> *,
                    ksdft::Hamiltonian<double, Host> *,
                    ksdft::Hamiltonian<std::complex<float>, Host> *,
                    ksdft::Hamiltonian<std::complex<double>, Host> *>;

  std::vector<HamiltonianPtrVariant> hamiltonianComponentsVec{hamitonianKin.get(), hamitonianElec.get()};
  // form the kohn sham operator
  std::shared_ptr<linearAlgebra::OperatorContext<double,
                                                  double,
                                                  Host>> 
                                            hamitonianOperator =
    std::make_shared<ksdft::KohnShamOperatorContextFE<double,
                                                  double,
                                                  double,
                                                  Host,
                                                  dim>>
                                                  (*basisManagerWaveFn,
                                                  hamiltonianComponentsVec,
                                                  *linAlgOpContext,
                                                  50);

  // call the eigensolver

  linearAlgebra::MultiVector<double, Host> waveFunctionSubspaceGuess(basisManagerWaveFn->getMPIPatternP2P(),
                                                                     linAlgOpContext,
                                                                     numWantedEigenvalues,
                                                                     0.0,
                                                                     1.0);    

  linearAlgebra::Vector<double, Host> lanczosGuess(basisManagerWaveFn->getMPIPatternP2P(),
                                                    linAlgOpContext,
                                                    0.0,
                                                    1.0); 

  lanczosGuess.updateGhostValues();
  basisManagerWaveFn->getConstraints().distributeParentToChild(lanczosGuess, 1);

  waveFunctionSubspaceGuess.updateGhostValues();
  basisManagerWaveFn->getConstraints().distributeParentToChild(waveFunctionSubspaceGuess, numWantedEigenvalues);

  std::vector<double> kohnShamEnergies(numWantedEigenvalues, 0.0);
  linearAlgebra::MultiVector<double, Host> kohnShamWaveFunctions(waveFunctionSubspaceGuess, 0.0);

  // Create OperatorContext for Basisoverlap
  std::shared_ptr<const basis::CFEOverlapOperatorContext<double,
                                                double,
                                                Host,
                                                dim>> MContext =
  std::make_shared<basis::CFEOverlapOperatorContext<double,
                                                      double,
                                                      Host,
                                                      dim>>(
                                                      *basisManagerWaveFn,
                                                      *basisManagerWaveFn,
                                                      *feBasisData,
                                                      50);

  // Set up the quadrature rule

  dftefe::quadrature::QuadratureRuleAttributes quadAttrGLL(dftefe::quadrature::QuadratureFamily::GLL,true,num1DGLLSize);

  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreValues] = true;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreGradient] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreHessian] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreOverlap] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreGradNiGradNj] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreJxW] = false;

  // Set up the FE Basis Data Storage
  std::shared_ptr<dftefe::basis::FEBasisDataStorage<double, dftefe::utils::MemorySpace::HOST>> feBasisDataGLL =
    std::make_shared<dftefe::basis::CFEBasisDataStorageDealii<double, double, dftefe::utils::MemorySpace::HOST,dim>>
    (basisDofHandler, quadAttrGLL, basisAttrMap);

  // evaluate basis data
  feBasisDataGLL->evaluateBasisData(quadAttrGLL, basisAttrMap);

  // Create OperatorContext for Basisoverlap
  std::shared_ptr<const basis::CFEOverlapOperatorContext<double,
                                                double,
                                                Host,
                                                dim>> MContextForInv =
  std::make_shared<basis::CFEOverlapOperatorContext<double,
                                                      double,
                                                      Host,
                                                      dim>>(
                                                      *basisManagerWaveFn,
                                                      *basisManagerWaveFn,
                                                      *feBasisDataGLL,
                                                      50);

std::shared_ptr<linearAlgebra::OperatorContext<double,
                                                  double,
                                                  Host>> MInvContext =
  std::make_shared<basis::CFEOverlapInverseOpContextGLL<double,
                                                  double,
                                                  Host,
                                                  dim>>
                                                  (*basisManagerWaveFn,
                                                  *feBasisDataGLL,
                                                  linAlgOpContext);

  // form the kohn sham operator
  std::shared_ptr<ksdft::KohnShamEigenSolver<double,
                                              double,
                                              Host>> 
                                            ksEigSolve =
    std::make_shared<ksdft::KohnShamEigenSolver<double,
                                                double,
                                                Host>>
                                                (numElectrons,
                                                smearingTemperature,
                                                fermiEnergyTolerance,
                                                fracOccupancyTolerance,
                                                eigenSolveResidualTolerance,
                                                chebyshevPolynomialDegree,
                                                maxChebyshevFilterPass,
                                                waveFunctionSubspaceGuess,
                                                lanczosGuess,
                                                50,
                                                *MContextForInv,
                                                *MInvContext);

  linearAlgebra::EigenSolverError err = ksEigSolve->solve(*hamitonianOperator, 
                    kohnShamEnergies, 
                    kohnShamWaveFunctions,
                    true,
                    *MContext,
                    *MInvContext);       

  for(auto &i : kohnShamEnergies)
    std::cout <<  i <<", ";     
  std::cout << "\n";     

  std::cout << err.msg << "\n";   

  ksdft::DensityCalculator<double, double, Host, dim>
                              densCalc(feBasisData,
                              *basisManagerWaveFn,
                              linAlgOpContext,
                              50,
                              4);

  quadrature::QuadratureValuesContainer<double, Host>
                  rho(quadRuleContainer, 1);

  std::vector<double> occupation = ksEigSolve->getFractionalOccupancy();
  densCalc.computeRho(occupation, kohnShamWaveFunctions, rho);
  std::vector<double> intRho(2);
  for (size_type iCell = 0; iCell < rho.nCells(); iCell++)
    {
      const std::vector<double> JxW = quadRuleContainer->getCellJxW(iCell);
      std::vector<double> a(quadRuleContainer->nCellQuadraturePoints(iCell));
      const std::vector<dftefe::utils::Point> point = quadRuleContainer->getCellRealPoints(iCell);
      rho.getCellQuadValues<Host>(iCell, 0, a.data());
      for (size_type i = 0; i < a.size(); i++)
      {
        intRho[0] += std::pow((a[i] - rho1sOrbital(point[i], atomCoordinatesVec)),2) * JxW[i];
        intRho[1] += std::pow(rho1sOrbital(point[i], atomCoordinatesVec),2) * JxW[i];
      }
    }

  int mpierr = utils::mpi::MPIAllreduce<Host>(
    utils::mpi::MPIInPlace,
    intRho.data(),
    intRho.size(),
    utils::mpi::Types<double>::getMPIDatatype(),
    utils::mpi::MPISum,
    comm);

  std::cout << "Absolute L2 norm rho : "<< intRho[0] << "Relative L2 norm rho : "<<intRho[0]/intRho[1]<<"\n";

  hamitonianKin->evalEnergy(occupation, *basisManagerWaveFn, kohnShamWaveFunctions, 4);

  std::cout << "kin energy: "<<hamitonianKin->getEnergy() << "\n";

  hamitonianElec->evalEnergy(); 
  double elecEnergy = hamitonianElec->getEnergy();
  std::cout << "elec energy: " << elecEnergy << "\n";

  // calculate band energy
  double bandEnergy = 0;
  for(size_type i = 0 ; i < occupation.size(); i++)
  {
    bandEnergy += 2 * occupation[i] * kohnShamEnergies[i];
  }

  std::cout << "band energy: "<< bandEnergy << "\n";

  double totalEnergy = bandEnergy - elecEnergy;

  std::cout << "Total Energy: " << totalEnergy << "\n";

  //gracefully end MPI

  int mpiFinalFlag = 0;
  utils::mpi::MPIFinalized(&mpiFinalFlag);
  if(!mpiFinalFlag)
  {
    utils::mpi::MPIFinalize();
  }
}
