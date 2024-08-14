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
#include <ksdft/KohnShamDFT.h>
#include <basis/GenerateMesh.h>
#include <utils/ConditionalOStream.h>
#include <atoms/AtomSphericalDataContainer.h>
#include <atoms/SphericalHarmonics.h>

#include <iostream>

using namespace dftefe;
const utils::MemorySpace Host = utils::MemorySpace::HOST;

template<typename T>
T readParameter(std::string ParamFile, std::string param, utils::ConditionalOStream &rootCout)
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
    utils::throwException(false, "The parameter is not found: "+ param);
  }
  fstream.close();
  rootCout << "Reading parameter -- " << param << " = "<<t<<std::endl;
  return t;
}
/*
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
*/

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

// operand - V_H
// memoryspace - HOST
int main(int argc, char** argv)
{
  // argv[1] = "H_Atom.in"
  // argv[2] = "KSDFTClassical/param.in"
  //initialize MPI

  // freopen(argv[3],"w",stdout);
  
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

  rootCout<<" Entering test kohn sham dft classical \n";

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
    utils::throwException(false,
                          "dftefe_path does not exist!");
  }
  std::string atomDataFile = argv[1];
  std::string inputFileName = sourceDir + atomDataFile;
  std::string paramDataFile = argv[2];
  std::string parameterInputFileName = sourceDir + paramDataFile;

  rootCout << "Reading input file: "<<inputFileName<<std::endl;
  rootCout << "Reading parameter file: "<<parameterInputFileName<<std::endl;  

  // Read parameters
  double xmax = readParameter<double>(parameterInputFileName, "xmax", rootCout);
  double ymax = readParameter<double>(parameterInputFileName, "ymax", rootCout);
  double zmax = readParameter<double>(parameterInputFileName, "zmax", rootCout);
  double radiusAtAtom = readParameter<double>(parameterInputFileName, "radiusAtAtom", rootCout);
  double meshSizeAtAtom = readParameter<double>(parameterInputFileName, "meshSizeAtAtom", rootCout);
  double radiusAroundAtom = readParameter<double>(parameterInputFileName, "radiusAroundAtom", rootCout);
  double meshSizeAroundAtom = readParameter<double>(parameterInputFileName, "meshSizeAroundAtom", rootCout);
  double rc = readParameter<double>(parameterInputFileName, "rc", rootCout);
  unsigned int feOrderElec = readParameter<unsigned int>(parameterInputFileName, "feOrderElectrostatics", rootCout);
  unsigned int feOrderEigen = readParameter<unsigned int>(parameterInputFileName, "feOrderEigenSolve", rootCout); 
  double    smearingTemperature = readParameter<double>(parameterInputFileName, "smearingTemperature", rootCout);
  double    fermiEnergyTolerance = readParameter<double>(parameterInputFileName, "fermiEnergyTolerance", rootCout);
  double    fracOccupancyTolerance = readParameter<double>(parameterInputFileName, "fracOccupancyTolerance", rootCout);
  double    eigenSolveResidualTolerance = readParameter<double>(parameterInputFileName, "eigenSolveResidualTolerance", rootCout);
  size_type chebyshevPolynomialDegree = readParameter<size_type>(parameterInputFileName, "chebyshevPolynomialDegree", rootCout);
  size_type maxChebyshevFilterPass = readParameter<size_type>(parameterInputFileName, "maxChebyshevFilterPass", rootCout);
  size_type numWantedEigenvalues = readParameter<size_type>(parameterInputFileName, "numWantedEigenvalues", rootCout);
  double scfDensityResidualNormTolerance = readParameter<double>(parameterInputFileName, "scfDensityResidualNormTolerance", rootCout);
  size_type maxSCFIter = readParameter<size_type>(parameterInputFileName, "maxSCFIter", rootCout);
  size_type mixingHistory = readParameter<size_type>(parameterInputFileName, "mixingHistory", rootCout);
  double mixingParameter = readParameter<double>(parameterInputFileName, "mixingParameter", rootCout);
  bool isAdaptiveAndersonMixingParameter = readParameter<bool>(parameterInputFileName, "isAdaptiveAndersonMixingParameter", rootCout);
  bool evaluateEnergyEverySCF = readParameter<bool>(parameterInputFileName, "evaluateEnergyEverySCF", rootCout);
  const size_type dim = 3;

  unsigned int num1DGaussSizeVCorrecPlusPhi = readParameter<unsigned int>(parameterInputFileName, "num1DGaussSizeVCorrecPlusPhi", rootCout);
  bool isNumericalNuclearSolve = readParameter<bool>(parameterInputFileName, "isNumericalNuclearSolve", rootCout);

  // Set up Triangulation
    std::shared_ptr<basis::TriangulationBase> triangulationBase =
        std::make_shared<basis::TriangulationDealiiParallel<dim>>(comm);
  std::vector<bool>                 isPeriodicFlags(dim, false);
  std::vector<utils::Point> domainVectors(dim, utils::Point(dim, 0.0));

  domainVectors[0][0] = xmax;
  domainVectors[1][1] = ymax;
  domainVectors[2][2] = zmax;

  /*
  // Uniform mesh creation
  std::vector<unsigned int>         subdivisions = {15, 15, 15};
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
  */

  std::fstream fstream;
  fstream.open(inputFileName, std::fstream::in);
  
  // read the input file and create atomsymbol vector and atom coordinates vector.
  std::vector<utils::Point> atomCoordinatesVec(0,utils::Point(dim, 0.0));
    std::vector<double> coordinates;
  coordinates.resize(dim,0.);
  std::vector<std::string> atomSymbolVec(0);
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
  // Generate mesh
   std::shared_ptr<basis::CellMappingBase> cellMapping = std::make_shared<basis::LinearCellMappingDealii<dim>>();

  basis::GenerateMesh adaptiveMesh(atomCoordinatesVec, 
                            domainVectors,
                            radiusAtAtom,
                            meshSizeAtAtom,
                            radiusAroundAtom,
                            meshSizeAroundAtom,
                            isPeriodicFlags,
                            *cellMapping,
                            comm);

  adaptiveMesh.createMesh(*triangulationBase); 

  std::vector<double> smearedChargeRadiusVec(atomCoordinatesVec.size(),rc);

  // initialize the basis Manager

  std::shared_ptr<const basis::FEBasisDofHandler<double, Host,dim>> basisDofHandlerTotalPot =  
   std::make_shared<basis::CFEBasisDofHandlerDealii<double, Host,dim>>(triangulationBase, feOrderElec, comm);

  std::shared_ptr<basis::FEBasisDofHandler<double, Host,dim>> basisDofHandlerWaveFn =  
   std::make_shared<basis::CFEBasisDofHandlerDealii<double, Host,dim>>(triangulationBase, feOrderEigen, comm);

  rootCout << "Total Number of dofs electrostatics: " << basisDofHandlerTotalPot->nGlobalNodes() << "\n";
  rootCout << "Total Number of dofs eigensolve: " << basisDofHandlerWaveFn->nGlobalNodes() << "\n";

  // Set up the quadrature rule

  quadrature::QuadratureRuleAttributes quadAttrElec(quadrature::QuadratureFamily::GAUSS,true,feOrderElec+1);

  basis::BasisStorageAttributesBoolMap basisAttrMap;
  basisAttrMap[basis::BasisStorageAttributes::StoreValues] = true;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradient] = true;
  basisAttrMap[basis::BasisStorageAttributes::StoreHessian] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreOverlap] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradNiGradNj] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreJxW] = true;

  // Set up the FE Basis Data Storage
  std::shared_ptr<basis::FEBasisDataStorage<double, Host>> feBasisDataElec =
    std::make_shared<basis::CFEBasisDataStorageDealii<double, double, Host,dim>>
    (basisDofHandlerTotalPot, quadAttrElec, basisAttrMap);

  // evaluate basis data
  feBasisDataElec->evaluateBasisData(quadAttrElec, basisAttrMap);

  std::map<std::string, std::string> atomSymbolToFilename;
  for (auto i:atomSymbolVec )
  {
      atomSymbolToFilename[i] = sourceDir + i + ".xml";
  }

  std::vector<std::string> fieldNames{"density"};
  std::vector<std::string> metadataNames{ "symbol", "Z", "charge", "NR", "r" };
  std::shared_ptr<atoms::AtomSphericalDataContainer>  atomSphericalDataContainer = 
      std::make_shared<atoms::AtomSphericalDataContainer>(atomSymbolToFilename,
                                                      fieldNames,
                                                      metadataNames);

    std::shared_ptr<const utils::ScalarSpatialFunctionReal> rho = std::make_shared
                <RhoFunction>(atomSphericalDataContainer, atomSymbolVec, atomChargesVec, atomCoordinatesVec);
                
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

  // // Create OperatorContext for Basisoverlap
  // std::shared_ptr<const basis::CFEOverlapOperatorContext<double,
  //                                               double,
  //                                               Host,
  //                                               dim>> MContext =
  // std::make_shared<basis::CFEOverlapOperatorContext<double,
  //                                                     double,
  //                                                     Host,
  //                                                     dim>>(
  //                                                     *basisManagerWaveFn,
  //                                                     *feBasisData,
  //                                                     50);

  // Set up the quadrature rule

  quadrature::QuadratureRuleAttributes quadAttrGLLEigen(quadrature::QuadratureFamily::GLL,true,feOrderEigen + 1);

  basisAttrMap[basis::BasisStorageAttributes::StoreValues] = true;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradient] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreHessian] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreOverlap] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradNiGradNj] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreJxW] = false;

  // Set up the FE Basis Data Storage
  std::shared_ptr<basis::FEBasisDataStorage<double, Host>> feBasisDataGLLEigen =
    std::make_shared<basis::CFEBasisDataStorageDealii<double, double, Host,dim>>
    (basisDofHandlerWaveFn, quadAttrGLLEigen, basisAttrMap);

  // evaluate basis data
  feBasisDataGLLEigen->evaluateBasisData(quadAttrGLLEigen, basisAttrMap);

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
                                                      *feBasisDataGLLEigen,
                                                      linAlgOpContext);

std::shared_ptr<linearAlgebra::OperatorContext<double,
                                                  double,
                                                  Host>> MInvContext =
  std::make_shared<basis::CFEOverlapInverseOpContextGLL<double,
                                                  double,
                                                  Host,
                                                  dim>>
                                                  (*basisManagerWaveFn,
                                                  *feBasisDataGLLEigen,
                                                  linAlgOpContext);
  rootCout << "Entering KohnSham DFT Class....\n\n";

  std::shared_ptr<const basis::FEBasisDataStorage<double, Host>> feBDTotalChargeStiffnessMatrix = feBasisDataElec;
  
  quadrature::QuadratureRuleAttributes quadAttrVCorrecPlusPhi(quadrature::QuadratureFamily::GAUSS,true,num1DGaussSizeVCorrecPlusPhi);

  basisAttrMap[basis::BasisStorageAttributes::StoreValues] = true;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradient] = true;
  basisAttrMap[basis::BasisStorageAttributes::StoreHessian] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreOverlap] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradNiGradNj] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreJxW] = true;

  std::shared_ptr<basis::FEBasisDataStorage<double, Host>> feBDElectrostaticsHamiltonian = 
    std::make_shared<basis::CFEBasisDataStorageDealii<double, double, Host,dim>>
      (basisDofHandlerWaveFn, quadAttrVCorrecPlusPhi, basisAttrMap);
  feBDElectrostaticsHamiltonian->evaluateBasisData(quadAttrVCorrecPlusPhi, basisAttrMap);

  std::shared_ptr<const quadrature::QuadratureRuleContainer> quadRuleContainerRho =  
                feBDElectrostaticsHamiltonian->getQuadratureRuleContainer();

  // scale the electronic charges
   quadrature::QuadratureValuesContainer<double, Host> 
      electronChargeDensity(quadRuleContainerRho, 1, 0.0);

  std::shared_ptr<const basis::FEBasisDataStorage<double,Host>> feBDKineticHamiltonian =  feBDElectrostaticsHamiltonian;
  std::shared_ptr<const basis::FEBasisDataStorage<double, Host>> feBDEXCHamiltonian = feBDElectrostaticsHamiltonian;

  std::shared_ptr<basis::FEBasisDataStorage<double, Host>> feBDTotalChargeRhs = 
    std::make_shared<basis::CFEBasisDataStorageDealii<double, double, Host,dim>>
      (basisDofHandlerTotalPot, quadAttrVCorrecPlusPhi, basisAttrMap);
  feBDTotalChargeRhs->evaluateBasisData(quadAttrVCorrecPlusPhi, basisAttrMap);

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

  if(isNumericalNuclearSolve)
  {

    unsigned int num1DGaussSizeSmearNucl = readParameter<unsigned int>(parameterInputFileName, "num1DGaussSizeSmearNucl", rootCout);
    quadrature::QuadratureRuleAttributes quadAttrSmearNucl(quadrature::QuadratureFamily::GAUSS,true,num1DGaussSizeSmearNucl);

    basisAttrMap[basis::BasisStorageAttributes::StoreValues] = true;
    basisAttrMap[basis::BasisStorageAttributes::StoreGradient] = false;
    basisAttrMap[basis::BasisStorageAttributes::StoreHessian] = false;
    basisAttrMap[basis::BasisStorageAttributes::StoreOverlap] = false;
    basisAttrMap[basis::BasisStorageAttributes::StoreGradNiGradNj] = false;
    basisAttrMap[basis::BasisStorageAttributes::StoreJxW] = true;

    // Set up the FE Basis Data Storage
    std::shared_ptr<basis::FEBasisDataStorage<double, Host>> feBDNuclearChargeRhs =
      std::make_shared<basis::CFEBasisDataStorageDealii<double, double, Host,dim>>
        (basisDofHandlerTotalPot, quadAttrSmearNucl, basisAttrMap);

    // evaluate basis data
    feBDNuclearChargeRhs->evaluateBasisData(quadAttrSmearNucl, basisAttrMap);

    std::shared_ptr<const basis::FEBasisDataStorage<double,Host>> feBDNuclearChargeStiffnessMatrix = feBasisDataElec;

    std::shared_ptr<ksdft::KohnShamDFT<double,
                                        double,
                                        double,
                                        double,
                                        Host,
                                        dim>> dftefeSolve =
    std::make_shared<ksdft::KohnShamDFT<double,
                                          double,
                                          double,
                                          double,
                                          Host,
                                          dim>>(
                                          atomCoordinatesVec,
                                          atomChargesVec,
                                          smearedChargeRadiusVec,
                                          numElectrons,
                                          numWantedEigenvalues,
                                          smearingTemperature,
                                          fermiEnergyTolerance,
                                          fracOccupancyTolerance,
                                          eigenSolveResidualTolerance,
                                          scfDensityResidualNormTolerance,
                                          chebyshevPolynomialDegree,
                                          maxChebyshevFilterPass,
                                          maxSCFIter,
                                          evaluateEnergyEverySCF,
                                          mixingHistory,
                                          mixingParameter,
                                          isAdaptiveAndersonMixingParameter,
                                          electronChargeDensity,
                                          basisManagerTotalPot,
                                          basisManagerWaveFn,
                                          feBDTotalChargeStiffnessMatrix,
                                          feBDTotalChargeRhs,   
                                          feBDNuclearChargeStiffnessMatrix,
                                          feBDNuclearChargeRhs, 
                                          feBDKineticHamiltonian,     
                                          feBDElectrostaticsHamiltonian, 
                                          feBDEXCHamiltonian,                                                                                
                                          *externalPotentialFunction,
                                          linAlgOpContext,
                                          50,
                                          50,
                                          *MContextForInv,
                                          *MContextForInv,
                                          *MInvContext);

    dftefeSolve->solve();                                            
  }
  else
  {
    std::shared_ptr<ksdft::KohnShamDFT<double,
                                        double,
                                        double,
                                        double,
                                        Host,
                                        dim>> dftefeSolve =
    std::make_shared<ksdft::KohnShamDFT<double,
                                          double,
                                          double,
                                          double,
                                          Host,
                                          dim>>(
                                          atomCoordinatesVec,
                                          atomChargesVec,
                                          smearedChargeRadiusVec,
                                          numElectrons,
                                          numWantedEigenvalues,
                                          smearingTemperature,
                                          fermiEnergyTolerance,
                                          fracOccupancyTolerance,
                                          eigenSolveResidualTolerance,
                                          scfDensityResidualNormTolerance,
                                          chebyshevPolynomialDegree,
                                          maxChebyshevFilterPass,
                                          maxSCFIter,
                                          evaluateEnergyEverySCF,
                                          mixingHistory,
                                          mixingParameter,
                                          isAdaptiveAndersonMixingParameter,
                                          electronChargeDensity,
                                          basisManagerTotalPot,
                                          basisManagerWaveFn,
                                          feBDTotalChargeStiffnessMatrix,
                                          feBDTotalChargeRhs,
                                          feBDKineticHamiltonian,     
                                          feBDElectrostaticsHamiltonian, 
                                          feBDEXCHamiltonian,                                                                                
                                          *externalPotentialFunction,
                                          linAlgOpContext,
                                          50,
                                          50,
                                          *MContextForInv,
                                          *MContextForInv,
                                          *MInvContext);

    dftefeSolve->solve();                                           
  }

  //gracefully end MPI

  int mpiFinalFlag = 0;
  utils::mpi::MPIFinalized(&mpiFinalFlag);
  if(!mpiFinalFlag)
  {
    utils::mpi::MPIFinalize();
  }
}
