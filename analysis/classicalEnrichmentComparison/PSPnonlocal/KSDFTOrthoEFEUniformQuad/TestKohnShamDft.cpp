#include <basis/TriangulationBase.h>
#include <basis/TriangulationDealiiParallel.h>
#include <basis/CellMappingBase.h>
#include <basis/LinearCellMappingDealii.h>
#include <basis/EFEBasisDofHandlerDealii.h>
#include <basis/EFEBDSOnTheFlyComputeDealii.h>
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
#include <basis/OEFEAtomBlockOverlapInvOpContextGLL.h>
#include <utils/PointChargePotentialFunction.h>
#include <ksdft/DensityCalculator.h>
#include <ksdft/KohnShamDFT.h>
#include <basis/GenerateMesh.h>
#include <utils/ConditionalOStream.h>
#include <atoms/AtomSevereFunction.h>
#include <atoms/SphericalHarmonics.h>

#include <iostream>

#include <chrono>
using namespace std::chrono;

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

class RhoFunction : public utils::ScalarSpatialFunctionReal
{
private:
    std::shared_ptr<const atoms::AtomSphericalDataContainer>
                              d_atomSphericalDataContainer;
    std::vector<std::string>  d_atomSymbolVec;
    std::vector<utils::Point> d_atomCoordinatesVec;
    std::vector<double> d_atomChargesVec;
    double d_ylm00;

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
    , d_ylm00(atoms::Clm(0, 0) * atoms::Dm(0) * atoms::Qm(0, 0))
    {}

  double
  operator()(const utils::Point &point) const
  {
    double   retValue = 0;
    for (size_type atomId = 0 ; atomId < d_atomCoordinatesVec.size() ; atomId++)
      {
        utils::Point origin(d_atomCoordinatesVec[atomId]);
        for(auto &enrichmentObjId : 
          d_atomSphericalDataContainer->getSphericalData(d_atomSymbolVec[atomId], "density"))
        {
          retValue = retValue + std::abs(enrichmentObjId->getValue(point, origin) * (1/d_ylm00));
        }
      }
    return retValue;
  }
  std::vector<double>
  operator()(const std::vector<utils::Point> &points) const
  {
    std::vector<double> ret(0);
    ret.resize(points.size());
    for (size_type atomId = 0 ; atomId < d_atomCoordinatesVec.size() ; atomId++)
      {
        utils::Point origin(d_atomCoordinatesVec[atomId]);
        auto vec = d_atomSphericalDataContainer->getSphericalData(d_atomSymbolVec[atomId], "density");
        for(auto &enrichmentObjId : vec)
        for (unsigned int i = 0 ; i < points.size() ; i++)            
        {
          ret[i] = ret[i] + std::abs(enrichmentObjId->getValue(points[i], origin) * (1/d_ylm00));
        }
      }
    return ret;
  }
};

class AtomicTotalElectrostaticPotentialFunction : public utils::ScalarSpatialFunctionReal
{
private:
    std::shared_ptr<const atoms::AtomSphericalDataContainer>
                              d_atomSphericalDataContainer;
    std::vector<std::string>  d_atomSymbolVec;
    std::vector<utils::Point> d_atomCoordinatesVec;
    double d_ylm00;

public:
  AtomicTotalElectrostaticPotentialFunction(
    std::shared_ptr<const atoms::AtomSphericalDataContainer>
                                      atomSphericalDataContainer,
      const std::vector<std::string> & atomSymbol,
      const std::vector<utils::Point> &atomCoordinates)
    : d_atomSphericalDataContainer(atomSphericalDataContainer)
    , d_atomSymbolVec(atomSymbol)
    , d_atomCoordinatesVec(atomCoordinates)
    , d_ylm00(atoms::Clm(0, 0) * atoms::Dm(0) * atoms::Qm(0, 0))
    {}

  double
  operator()(const utils::Point &point) const
  {
    double   retValue = 0;
    for (size_type atomId = 0 ; atomId < d_atomCoordinatesVec.size() ; atomId++)
      {
        utils::Point origin(d_atomCoordinatesVec[atomId]);
        for(auto &enrichmentObjId : 
          d_atomSphericalDataContainer->getSphericalData(d_atomSymbolVec[atomId], "vtotal"))
        {
          retValue = retValue + enrichmentObjId->getValue(point, origin) * (1/d_ylm00);
        }
      }
    return retValue;
  }
  std::vector<double>
  operator()(const std::vector<utils::Point> &points) const
  {
    std::vector<double> ret(0);
    ret.resize(points.size());
    for (size_type atomId = 0 ; atomId < d_atomCoordinatesVec.size() ; atomId++)
      {
        utils::Point origin(d_atomCoordinatesVec[atomId]);
        auto vec = d_atomSphericalDataContainer->getSphericalData(d_atomSymbolVec[atomId], "vtotal");
        for(auto &enrichmentObjId : vec)
        for (unsigned int i = 0 ; i < points.size() ; i++)            
        {
          ret[i] = ret[i] + enrichmentObjId->getValue(points[i], origin) * (1/d_ylm00);
        }
      }
    return ret;
  }
};

  template <typename ValueTypeBasisData,
            utils::MemorySpace memorySpace,
            size_type          dim>
  size_type getNumClassicalDofsInSystemExcludingVacuum(const std::vector<utils::Point> &atomCoordinates,
                                                      const basis::FEBasisDofHandler<ValueTypeBasisData,
                                                                                      memorySpace,
                                                                                      dim> &basisDofHandler,
                                                      utils::mpi::MPIComm comm)
  {
    const std::vector<std::pair<global_size_type, global_size_type>> &numLocallyOwnedRanges  = basisDofHandler.getLocallyOwnedRanges();
    size_type dofs = 0;
    double domainSizeExcludingVacuum = 0;

    std::vector<double> maxAtomCoordinates(dim, 0);
    std::vector<double> minAtomCoordinates(dim, 0);

    for (int j = 0; j < dim; j++)
      {
        for (int i = 0; i < atomCoordinates.size(); i++)
          {
            if (maxAtomCoordinates[j] < atomCoordinates[i][j])
              maxAtomCoordinates[j] = atomCoordinates[i][j];
            if (minAtomCoordinates[j] > atomCoordinates[i][j])
              minAtomCoordinates[j] = atomCoordinates[i][j];
          }
      }

    for (int i = 0; i < dim; i++)
      {
        double axesLen = std::max(std::abs(maxAtomCoordinates[i]),
                                  std::abs(minAtomCoordinates[i])) + 8.0;
        if (domainSizeExcludingVacuum < axesLen)
          domainSizeExcludingVacuum = axesLen;
      }

    std::map<global_size_type, utils::Point> dofCoords;
    basisDofHandler.getBasisCenters(dofCoords);
    dftefe::utils::Point nodeLoc(dim,0.0);
    for (dftefe::global_size_type iDof = numLocallyOwnedRanges[0].first; iDof < numLocallyOwnedRanges[0].second ; iDof++)
      {
        nodeLoc = dofCoords.find(iDof)->second;
        double dist = 0;
        for( int j = 0 ; j < dim ; j++)
        {
          dist += nodeLoc[j]* nodeLoc[j];
        }
        dist = std::sqrt(dist);
        if(dist <= domainSizeExcludingVacuum)
          dofs += 1;
      }
      int mpierr = utils::mpi::MPIAllreduce<Host>(
        utils::mpi::MPIInPlace,
        &dofs,
        1,
        utils::mpi::Types<size_type>::getMPIDatatype(),
        utils::mpi::MPISum,
        comm);
      return dofs;
  }

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

  utils::Profiler pTot(comm, "Total Statistics");
  utils::Profiler p(comm, "Initilization Breakdown Statistics");
  pTot.registerStart("Initilization");

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

  rootCout<<" Entering test kohn sham dft ortho enrichment \n";
  rootCout << "Number of processes: "<<numProcs<<"\n";

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
  double radiusAtAtom = readParameter<double>(parameterInputFileName, "radiusAtAtom", rootCout);;
  double meshSizeAtAtom = readParameter<double>(parameterInputFileName, "meshSizeAtAtom", rootCout);;
  double radiusAroundAtom = readParameter<double>(parameterInputFileName, "radiusAroundAtom", rootCout);;
  double meshSizeAroundAtom = readParameter<double>(parameterInputFileName, "meshSizeAroundAtom", rootCout);;
  double rc = readParameter<double>(parameterInputFileName, "rc", rootCout);
  unsigned int feOrderElec = readParameter<unsigned int>(parameterInputFileName, "feOrderElectrostatics", rootCout);
  unsigned int feOrderEigen = readParameter<unsigned int>(parameterInputFileName, "feOrderEigenSolve", rootCout); 
  double    smearingTemperature = readParameter<double>(parameterInputFileName, "smearingTemperature", rootCout);
  double    fermiEnergyTolerance = readParameter<double>(parameterInputFileName, "fermiEnergyTolerance", rootCout);
  double    fracOccupancyTolerance = readParameter<double>(parameterInputFileName, "fracOccupancyTolerance", rootCout);
  double    eigenSolveResidualTolerance = readParameter<double>(parameterInputFileName, "eigenSolveResidualTolerance", rootCout);
  size_type maxChebyshevFilterPass = readParameter<size_type>(parameterInputFileName, "maxChebyshevFilterPass", rootCout);
  size_type numWantedEigenvalues = readParameter<size_type>(parameterInputFileName, "numWantedEigenvalues", rootCout);
  double scfDensityResidualNormTolerance = readParameter<double>(parameterInputFileName, "scfDensityResidualNormTolerance", rootCout);
  size_type maxSCFIter = readParameter<size_type>(parameterInputFileName, "maxSCFIter", rootCout);
  size_type mixingHistory = readParameter<size_type>(parameterInputFileName, "mixingHistory", rootCout);
  double mixingParameter = readParameter<double>(parameterInputFileName, "mixingParameter", rootCout);
  bool isAdaptiveAndersonMixingParameter = readParameter<bool>(parameterInputFileName, "isAdaptiveAndersonMixingParameter", rootCout);
  bool evaluateEnergyEverySCF = readParameter<bool>(parameterInputFileName, "evaluateEnergyEverySCF", rootCout);
  const size_type dim = 3;

  double atomPartitionTolerance = readParameter<double>(parameterInputFileName, "atomPartitionTolerance", rootCout);
  unsigned int num1DGaussSubdividedSizeElec = readParameter<unsigned int>(parameterInputFileName, "num1DGaussSubdividedSizeElec", rootCout);
  unsigned int gaussSubdividedCopiesElec = readParameter<unsigned int>(parameterInputFileName, "gaussSubdividedCopiesElec", rootCout);
  
  unsigned int num1DGaussSubdividedSizeEigen = readParameter<unsigned int>(parameterInputFileName, "num1DGaussSubdividedSizeEigen", rootCout);
  unsigned int gaussSubdividedCopiesEigen = readParameter<unsigned int>(parameterInputFileName, "gaussSubdividedCopiesEigen", rootCout);
  
  unsigned int num1DGaussSubdividedSizeGrad = readParameter<unsigned int>(parameterInputFileName, "num1DGaussSubdividedSizeGrad", rootCout);
  unsigned int gaussSubdividedCopiesGrad = readParameter<unsigned int>(parameterInputFileName, "gaussSubdividedCopiesGrad", rootCout);
  
  bool isNumericalNuclearSolve = readParameter<bool>(parameterInputFileName, "isNumericalNuclearSolve", rootCout);
  bool isDeltaRhoPoissonSolve = readParameter<bool>(parameterInputFileName, "isDeltaRhoPoissonSolve", rootCout);

  unsigned int num1DGaussSubdividedSizeNonLocOperator = 14;
  unsigned int gaussSubdividedCopiesNonLocOperator = 1;

  // Set up Triangulation
    std::shared_ptr<basis::TriangulationBase> triangulationBase =
        std::make_shared<basis::TriangulationDealiiParallel<dim>>(comm);
  std::vector<bool>                 isPeriodicFlags(dim, false);
  std::vector<utils::Point> domainVectors(dim, utils::Point(dim, 0.0));

  domainVectors[0][0] = xmax;
  domainVectors[1][1] = ymax;
  domainVectors[2][2] = zmax;

  // //Uniform mesh creation
  // std::vector<unsigned int>         subdivisions = {10, 10, 10};
  // std::vector<double> origin(0);
  // origin.resize(dim);
  // for(unsigned int i = 0 ; i < dim ; i++)
  //   origin[i] = -domainVectors[i][i]*0.5;

  // // initialize the triangulation
  // triangulationBase->initializeTriangulationConstruction();
  // triangulationBase->createUniformParallelepiped(subdivisions,
  //                                                domainVectors,
  //                                                isPeriodicFlags);
  // triangulationBase->shiftTriangulation(utils::Point(origin));
  // triangulationBase->finalizeTriangulationConstruction();

  std::fstream fstream;
  fstream.open(inputFileName, std::fstream::in);
  
  // read the input file and create atomsymbol vector and atom coordinates vector.
  std::vector<utils::Point> atomCoordinatesVec(0,utils::Point(dim, 0.0));
    std::vector<double> coordinates;
        std::vector<std::string> pspFilePathVec(0);
  coordinates.resize(dim,0.);
  std::vector<std::string> atomSymbolVec(0);
  std::vector<double> atomChargesVec(0);
  std::string symbol;
  double valanceNumber;
  atomSymbolVec.resize(0);
  std::string line;
  std::string pspFilePath;
  while (std::getline(fstream, line)){
      std::stringstream ss(line);
      ss >> symbol; 
      ss >> valanceNumber; 
      ss >> pspFilePath;
      for(unsigned int i=0 ; i<dim ; i++){
          ss >> coordinates[i]; 
      }
      pspFilePathVec.push_back(pspFilePath);
      atomCoordinatesVec.push_back(coordinates);
      atomSymbolVec.push_back(symbol);
      atomChargesVec.push_back((-1.0)*valanceNumber);
  }
  utils::mpi::MPIBarrier(comm);
  fstream.close();

  std::map<std::string, std::string> atomSymbolToPSPFilename;
  for (int i = 0 ; i < atomSymbolVec.size() ; i++)
  {
      atomSymbolToPSPFilename[atomSymbolVec[i]] = sourceDir + pspFilePathVec[i];
  }

  size_type numElectrons = 0;
  for(auto &i : atomChargesVec)
  {
    numElectrons += (size_type)(std::abs(i));
  }

  std::map<std::string, std::string> atomSymbolToFilename;
  for (auto i:atomSymbolVec )
  {
      atomSymbolToFilename[i] = sourceDir + i + ".xml";
  }

  std::vector<std::string> fieldNames{"orbital","vtotal","density"};
  std::vector<std::string> metadataNames{ "symbol", "Z", "charge", "NR", "r" };
  std::shared_ptr<atoms::AtomSphericalDataContainer>  atomSphericalDataContainer = 
      std::make_shared<atoms::AtomSphericalDataContainer>(
                                                      atoms::AtomSphericalDataType::ENRICHMENT,
                                                      atomSymbolToFilename,
                                                      fieldNames,
                                                      metadataNames);

  for (auto i:atomSymbolVec )
  {
    rootCout << "Reading xml file: "<<atomSymbolToFilename[i]<<std::endl; 
    rootCout << "Cutoff and smoothness for "<<i<<std::endl; 
    for(auto j:fieldNames)
    {
      rootCout << " for "<<j<<" : "; 
      for(auto &enrichmentObjId : 
        atomSphericalDataContainer->getSphericalData(i, j))
      {
        rootCout << enrichmentObjId->getCutoff() << ","<<enrichmentObjId->getSmoothness()<<"\t";
      }
      rootCout << std::endl;
    }
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

  // Make orthogonalized EFE basis for all the fields

  // 1. Make CFEBDSOnTheFlyComputeDealii object for Rhs (ADAPTIVE with GAUSS and fns are N_i^2 - make quadrulecontainer), overlapmatrix (GLL).
  // 2. Make EnrichmentClassicalInterface object for Orthogonalized enrichment.
  // 3. Input to the EFEBasisDofHandler(eci, feOrder).
  // 4. Make EFEBasisDataStorage with input as quadratureContainer.

    // Set the CFE basis manager and handler for bassiInterfaceCoeffcient distributed vector
  std::shared_ptr<basis::FEBasisDofHandler<double, Host,dim>> cfeBasisDofHandlerElec =  
   std::make_shared<basis::CFEBasisDofHandlerDealii<double, Host,dim>>(triangulationBase, feOrderElec, comm);

  std::shared_ptr<basis::FEBasisDofHandler<double, Host,dim>> cfeBasisDofHandlerEigen =  
   std::make_shared<basis::CFEBasisDofHandlerDealii<double, Host,dim>>(triangulationBase, feOrderEigen, comm);

  rootCout << "Total Number of classical dofs electrostatics: " << cfeBasisDofHandlerElec->nGlobalNodes() << "\n";
  rootCout << "Total Number of classical dofs eigensolve: " << cfeBasisDofHandlerEigen->nGlobalNodes() << "\n";

  rootCout << "The Number of classical dofs electrostatics excluding Vacuum: " << 
    getNumClassicalDofsInSystemExcludingVacuum<double, Host, dim>(atomCoordinatesVec,
      *cfeBasisDofHandlerElec,
      comm) << "\n";

  rootCout << "The Number of classical dofs eigenSolve excluding Vacuum: " << 
    getNumClassicalDofsInSystemExcludingVacuum<double, Host, dim>(atomCoordinatesVec,
      *cfeBasisDofHandlerEigen,
      comm) << "\n";

    //quadrature::QuadratureRuleAttributes quadAttrAdaptive(quadrature::QuadratureFamily::ADAPTIVE,false);

    quadrature::QuadratureRuleAttributes quadAttrGllElec(quadrature::QuadratureFamily::GLL,true,feOrderElec + 1);

    p.registerStart("Quadrature Rule Creation");
  
    std::shared_ptr<quadrature::QuadratureRule> gaussSubdivQuadRuleElec =
      std::make_shared<quadrature::QuadratureRuleGaussIterated>(dim, num1DGaussSubdividedSizeElec, gaussSubdividedCopiesElec);

    quadrature::QuadratureRuleAttributes quadAttrGaussSubdivided(quadrature::QuadratureFamily::GAUSS_SUBDIVIDED,true);

    std::shared_ptr<quadrature::QuadratureRuleContainer> quadRuleContainerGaussSubdividedElec =
      std::make_shared<quadrature::QuadratureRuleContainer>
      (quadAttrGaussSubdivided, 
      gaussSubdivQuadRuleElec, 
      triangulationBase, 
      *cellMapping); 

    unsigned int nQuad = quadRuleContainerGaussSubdividedElec->nQuadraturePoints();
    unsigned int nQuadMax = nQuad;
    auto mpierr = utils::mpi::MPIAllreduce<Host>(
      utils::mpi::MPIInPlace,
      &nQuad,
      1,
      utils::mpi::Types<size_type>::getMPIDatatype(),
      utils::mpi::MPISum,
      comm);

    mpierr = utils::mpi::MPIAllreduce<Host>(
      utils::mpi::MPIInPlace,
      &nQuadMax,
      1,
      utils::mpi::Types<size_type>::getMPIDatatype(),
      utils::mpi::MPIMax,
      comm);
    rootCout << "Maximum Number of quadrature points in a processor: "<< nQuadMax<<"\n";
    rootCout << "Number of quadrature points in gauss subdivided quadrature: "<< nQuad<<"\n";

  //Set up quadAttr for Rhs and OverlapMatrix
  
  quadrature::QuadratureRuleAttributes quadAttrGllEigen(quadrature::QuadratureFamily::GLL,true,feOrderEigen + 1);

  // Set up base quadrature rule for adaptive quadrature 

  std::shared_ptr<quadrature::QuadratureRule> baseQuadRuleEigen = std::make_shared<quadrature::QuadratureRuleGauss>(dim, feOrderEigen + 1);
    // feOrderEigen > feOrderElec ? std::make_shared<quadrature::QuadratureRuleGauss>(dim, feOrderEigen + 1) : 
    //   std::make_shared<quadrature::QuadratureRuleGauss>(dim, feOrderElec + 1);

  std::shared_ptr<quadrature::QuadratureRule> gaussSubdivQuadRuleEigen =
    std::make_shared<quadrature::QuadratureRuleGaussIterated>(dim, num1DGaussSubdividedSizeEigen, gaussSubdividedCopiesEigen);

  std::shared_ptr<quadrature::QuadratureRuleContainer> quadRuleContainerAdaptiveOrbital =  /*quadRuleContainerGaussSubdividedElec;*/
    std::make_shared<quadrature::QuadratureRuleContainer>
    (quadAttrGaussSubdivided, 
    gaussSubdivQuadRuleEigen, 
    triangulationBase, 
    *cellMapping); 

  std::shared_ptr<quadrature::QuadratureRule> gaussSubdivQuadRuleNonLocOperator =
    std::make_shared<quadrature::QuadratureRuleGaussIterated>(dim, num1DGaussSubdividedSizeNonLocOperator, 
      gaussSubdividedCopiesNonLocOperator);

  std::shared_ptr<quadrature::QuadratureRuleContainer>  quadRuleContainerAdaptiveAtomCenterNonLocalOperator = 
    std::make_shared<quadrature::QuadratureRuleContainer>
    (quadAttrGaussSubdivided, 
    gaussSubdivQuadRuleNonLocOperator, 
    triangulationBase, 
    *cellMapping); 

  std::shared_ptr<quadrature::QuadratureRule> gaussSubdivQuadRuleGrad =
    std::make_shared<quadrature::QuadratureRuleGaussIterated>(dim, num1DGaussSubdividedSizeGrad, gaussSubdividedCopiesGrad);

  std::shared_ptr<quadrature::QuadratureRuleContainer> quadRuleContainerAdaptiveGrad =  
    std::make_shared<quadrature::QuadratureRuleContainer>
    (quadAttrGaussSubdivided, 
    gaussSubdivQuadRuleGrad, 
    triangulationBase, 
    *cellMapping); 

  p.registerEnd("Quadrature Rule Creation");

  nQuad = quadRuleContainerAdaptiveOrbital->nQuadraturePoints();
  mpierr = utils::mpi::MPIAllreduce<Host>(
    utils::mpi::MPIInPlace,
    &nQuad,
    1,
      utils::mpi::Types<size_type>::getMPIDatatype(),
    utils::mpi::MPISum,
    comm);

  rootCout << "Number of quadrature points in wave function adaptive quadrature: "<<nQuad<<"\n";

  p.registerStart("Ortho EFE basis manager creation");
        
  basis::BasisStorageAttributesBoolMap basisAttrMap;
  basisAttrMap[basis::BasisStorageAttributes::StoreValues] = true;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradient] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreHessian] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreOverlap] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradNiGradNj] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreJxW] = true;

    // Set up the CFE Basis Data Storage for Overlap Matrix
    std::shared_ptr<basis::FEBasisDataStorage<double, Host>> cfeBasisDataStorageGLLElec =
      std::make_shared<basis::CFEBDSOnTheFlyComputeDealii<double, double,Host, dim>>
      (cfeBasisDofHandlerElec, quadAttrGllElec, basisAttrMap, ksdft::KSDFTDefaults::CELL_BATCH_SIZE_GRAD_EVAL, *linAlgOpContext);

    std::shared_ptr<basis::FEBasisDataStorage<double, Host>> cfeBasisDataStorageGLLEigen =
      std::make_shared<basis::CFEBDSOnTheFlyComputeDealii<double, double,Host, dim>>
      (cfeBasisDofHandlerEigen, quadAttrGllEigen, basisAttrMap, ksdft::KSDFTDefaults::CELL_BATCH_SIZE_GRAD_EVAL, *linAlgOpContext);

  // evaluate basis data
  cfeBasisDataStorageGLLElec->evaluateBasisData(quadAttrGllElec, basisAttrMap);
  cfeBasisDataStorageGLLEigen->evaluateBasisData(quadAttrGllEigen, basisAttrMap);

    // Set up the CFE Basis Data Storage for Rhs
    std::shared_ptr<basis::FEBasisDataStorage<double, Host>> cfeBasisDataStorageGaussSubdividedElec =
      std::make_shared<basis::CFEBDSOnTheFlyComputeDealii<double, double,Host, dim>>
      (cfeBasisDofHandlerElec, quadAttrGaussSubdivided, basisAttrMap, ksdft::KSDFTDefaults::CELL_BATCH_SIZE_GRAD_EVAL, *linAlgOpContext);
  // evaluate basis data
  cfeBasisDataStorageGaussSubdividedElec->evaluateBasisData(quadAttrGaussSubdivided, quadRuleContainerGaussSubdividedElec, basisAttrMap);

    // Set the CFE basis manager and handler for bassiInterfaceCoeffcient distributed vector

  basisAttrMap[basis::BasisStorageAttributes::StoreValues] = true;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradient] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreHessian] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreOverlap] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradNiGradNj] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreJxW] = true;

    // Set up the CFE Basis Data Storage for Rhs
    std::shared_ptr<basis::FEBasisDataStorage<double, Host>> cfeBasisDataStorageAdaptiveOrbital =
      std::make_shared<basis::CFEBDSOnTheFlyComputeDealii<double, double,Host, dim>>
      (cfeBasisDofHandlerEigen, quadAttrGaussSubdivided, basisAttrMap, ksdft::KSDFTDefaults::CELL_BATCH_SIZE_GRAD_EVAL, *linAlgOpContext);
  // evaluate basis data
  cfeBasisDataStorageAdaptiveOrbital->evaluateBasisData(quadAttrGaussSubdivided, quadRuleContainerAdaptiveOrbital, basisAttrMap);

  std::shared_ptr<basis::EnrichmentClassicalInterfaceSpherical
                          <double, Host, dim>>
        enrichClassIntfceTotalPot = nullptr;
  if (!isDeltaRhoPoissonSolve)
    // Create the enrichmentClassicalInterface object for vtotal
    enrichClassIntfceTotalPot = std::make_shared<basis::EnrichmentClassicalInterfaceSpherical
                        <double, Host, dim>>
                        (cfeBasisDataStorageGLLElec,
                        cfeBasisDataStorageGaussSubdividedElec,
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
                          (cfeBasisDataStorageGLLEigen,
                          cfeBasisDataStorageAdaptiveOrbital,
                          atomSphericalDataContainer,
                          atomPartitionTolerance,
                          atomSymbolVec,
                          atomCoordinatesVec,
                          "orbital",
                          linAlgOpContext,
                          comm);

  // initialize the basis Manager

  std::shared_ptr<basis::FEBasisDofHandler<double, Host,dim>> basisDofHandlerTotalPot = nullptr;
  if (!isDeltaRhoPoissonSolve)
     basisDofHandlerTotalPot =  
      std::make_shared<basis::EFEBasisDofHandlerDealii<double, double,Host,dim>>(
        enrichClassIntfceTotalPot, comm);
  else
    basisDofHandlerTotalPot = cfeBasisDofHandlerElec;

  std::shared_ptr<basis::FEBasisDofHandler<double, Host,dim>> basisDofHandlerWaveFn =  
    std::make_shared<basis::EFEBasisDofHandlerDealii<double, double,Host,dim>>(
      enrichClassIntfceOrbital, comm);

  /*
  utils::ConditionalOStream allCout(std::cout);
  for(int iProc = 0 ; iProc < numProcs ; iProc++)
  {
    if(rank == iProc)
    {
      allCout << rank << "\tNumber of Cells: " << 
        basisDofHandlerTotalPot->nLocallyOwnedCells() 
          << "\tElec Dofs: " << basisDofHandlerTotalPot->nLocalNodes() 
            << "\tEigen Dofs: " << basisDofHandlerWaveFn->nLocalNodes()
              << std::flush << std::endl;
    }
    utils::mpi::MPIBarrier(comm);
  }
  utils::mpi::MPIBarrier(comm);
  */

  p.registerEnd("Ortho EFE basis manager creation");

  std::map<global_size_type, utils::Point> dofCoords;
  basisDofHandlerTotalPot->getBasisCenters(dofCoords);

  rootCout << "Total Number of dofs electrostatics: " << basisDofHandlerTotalPot->nGlobalNodes() << "\n";
  rootCout << "Total Number of dofs eigensolve: " << basisDofHandlerWaveFn->nGlobalNodes() << "\n";

  // Set up the quadrature rule

  basisAttrMap[basis::BasisStorageAttributes::StoreValues] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradient] = true;
  basisAttrMap[basis::BasisStorageAttributes::StoreHessian] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreOverlap] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradNiGradNj] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreJxW] = true;

   quadrature::QuadratureRuleAttributes quadAttrGaussElectro(quadrature::QuadratureFamily::GAUSS,true,feOrderElec + 1);

  // Set up Adaptive quadrature for EFE Basis Data Storage
  std::shared_ptr<basis::FEBasisDataStorage<double, Host>> feBDTotalChargeStiffnessMatrix = nullptr;
  if (!isDeltaRhoPoissonSolve)
    feBDTotalChargeStiffnessMatrix =
    std::make_shared<basis::EFEBDSOnTheFlyComputeDealii<double, double, Host,dim>>
    (basisDofHandlerTotalPot, quadAttrGaussSubdivided, basisAttrMap, ksdft::KSDFTDefaults::CELL_BATCH_SIZE_GRAD_EVAL, *linAlgOpContext);
  else
    feBDTotalChargeStiffnessMatrix =
    std::make_shared<basis::CFEBDSOnTheFlyComputeDealii<double, double, Host,dim>>
    (basisDofHandlerTotalPot, quadAttrGaussElectro, basisAttrMap, ksdft::KSDFTDefaults::CELL_BATCH_SIZE_GRAD_EVAL, *linAlgOpContext);

  p.registerStart("Electrostatics basis grad datastorage eval");

  if (!isDeltaRhoPoissonSolve)
    feBDTotalChargeStiffnessMatrix->evaluateBasisData(quadAttrGaussSubdivided, quadRuleContainerAdaptiveGrad, basisAttrMap);
  else 
    feBDTotalChargeStiffnessMatrix->evaluateBasisData(quadAttrGaussElectro, basisAttrMap);

  p.registerEnd("Electrostatics basis grad datastorage eval");
  p.registerStart("Electrostatics basis bsmear datastorage eval");

  basisAttrMap[basis::BasisStorageAttributes::StoreValues] = true;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradient] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreHessian] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreOverlap] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradNiGradNj] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreJxW] = true;

  std::shared_ptr<basis::FEBasisDataStorage<double, Host>> feBDNucChargeRhs = nullptr;
  if (!isDeltaRhoPoissonSolve)
    feBDNucChargeRhs =   
      std::make_shared<basis::EFEBDSOnTheFlyComputeDealii<double, double, Host,dim>>
      (basisDofHandlerTotalPot, quadAttrGaussSubdivided, basisAttrMap, ksdft::KSDFTDefaults::CELL_BATCH_SIZE_GRAD_EVAL, *linAlgOpContext);
  else
    feBDNucChargeRhs =   
      std::make_shared<basis::CFEBDSOnTheFlyComputeDealii<double, double, Host,dim>>
      (basisDofHandlerTotalPot, quadAttrGaussSubdivided, basisAttrMap, ksdft::KSDFTDefaults::CELL_BATCH_SIZE_GRAD_EVAL, *linAlgOpContext);
  
  feBDNucChargeRhs->evaluateBasisData(quadAttrGaussSubdivided, quadRuleContainerGaussSubdividedElec, basisAttrMap);

  p.registerEnd("Electrostatics basis bsmear datastorage eval");
  p.registerStart("Electrostatics basis rho datastorage eval");

  std::shared_ptr<basis::FEBasisDataStorage<double, Host>> feBDElecChargeRhs = nullptr;
  if (!isDeltaRhoPoissonSolve)
    feBDElecChargeRhs =   
      std::make_shared<basis::EFEBDSOnTheFlyComputeDealii<double, double, Host,dim>>
      (basisDofHandlerTotalPot, quadAttrGaussSubdivided, basisAttrMap, ksdft::KSDFTDefaults::CELL_BATCH_SIZE_GRAD_EVAL, *linAlgOpContext);
  else
    feBDElecChargeRhs =   
      std::make_shared<basis::CFEBDSOnTheFlyComputeDealii<double, double, Host,dim>>
      (basisDofHandlerTotalPot, quadAttrGaussSubdivided, basisAttrMap, ksdft::KSDFTDefaults::CELL_BATCH_SIZE_GRAD_EVAL, *linAlgOpContext);

  feBDElecChargeRhs->evaluateBasisData(quadAttrGaussSubdivided, quadRuleContainerAdaptiveOrbital, basisAttrMap);

  p.registerEnd("Electrostatics basis rho datastorage eval");

  p.registerStart("Orbital basis datastorage eval");

  basisAttrMap[basis::BasisStorageAttributes::StoreValues] = true;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradient] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreHessian] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreOverlap] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradNiGradNj] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreJxW] = true;

  std::shared_ptr<basis::FEBasisDataStorage<double, Host>> efeBasisDataAdaptiveOrbital =
    std::make_shared<basis::EFEBDSOnTheFlyComputeDealii<double, double, Host,dim>>
      (basisDofHandlerWaveFn, quadAttrGaussSubdivided, basisAttrMap, ksdft::KSDFTDefaults::CELL_BATCH_SIZE_GRAD_EVAL, *linAlgOpContext);

  efeBasisDataAdaptiveOrbital->evaluateBasisData(quadAttrGaussSubdivided, quadRuleContainerAdaptiveOrbital, basisAttrMap);

  basisAttrMap[basis::BasisStorageAttributes::StoreValues] = true;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradient] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreHessian] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreOverlap] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradNiGradNj] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreJxW] = true;

  std::shared_ptr<basis::FEBasisDataStorage<double, Host>> feBDAtomCenterNonLocalOperator =
    std::make_shared<basis::EFEBDSOnTheFlyComputeDealii<double, double, Host,dim>>
      (basisDofHandlerWaveFn, quadAttrGaussSubdivided, basisAttrMap, ksdft::KSDFTDefaults::CELL_BATCH_SIZE_GRAD_EVAL, *linAlgOpContext);

  feBDAtomCenterNonLocalOperator->evaluateBasisData(quadAttrGaussSubdivided, quadRuleContainerAdaptiveAtomCenterNonLocalOperator, basisAttrMap);

  basisAttrMap[basis::BasisStorageAttributes::StoreValues] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradient] = true;
  basisAttrMap[basis::BasisStorageAttributes::StoreHessian] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreOverlap] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradNiGradNj] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreJxW] = true;

  std::shared_ptr<basis::FEBasisDataStorage<double, Host>> efeBasisDataAdaptiveGrad =
    std::make_shared<basis::EFEBDSOnTheFlyComputeDealii<double, double, Host,dim>>
      (basisDofHandlerWaveFn, quadAttrGaussSubdivided, basisAttrMap, ksdft::KSDFTDefaults::CELL_BATCH_SIZE_GRAD_EVAL, *linAlgOpContext);

  p.registerEnd("Orbital basis datastorage eval");
  p.registerStart("Grad basis datastorage eval");

   efeBasisDataAdaptiveGrad->evaluateBasisData(quadAttrGaussSubdivided, quadRuleContainerAdaptiveGrad, basisAttrMap);

    std::shared_ptr<const basis::FEBasisDataStorage<double, Host>> feBDElectrostaticsHamiltonian = efeBasisDataAdaptiveOrbital;
    std::shared_ptr<const basis::FEBasisDataStorage<double,Host>> feBDKineticHamiltonian =  efeBasisDataAdaptiveGrad;
    std::shared_ptr<const basis::FEBasisDataStorage<double, Host>> feBDEXCHamiltonian = efeBasisDataAdaptiveOrbital;

    p.registerEnd("Grad basis datastorage eval");
    p.print();

    std::shared_ptr<const utils::ScalarSpatialFunctionReal>
          zeroFunction = std::make_shared
            <utils::ScalarZeroFunctionReal>();

    std::shared_ptr<const basis::FEBasisManager
      <double, double, Host,dim>>
    basisManagerWaveFn = std::make_shared
      <basis::FEBasisManager<double, double, Host,dim>>
        (basisDofHandlerWaveFn);

    std::shared_ptr<const basis::FEBasisManager
      <double, double, Host,dim>>
    basisManagerTotalPot = std::make_shared
      <basis::FEBasisManager<double, double, Host,dim>>
        (basisDofHandlerTotalPot, zeroFunction);

    p.registerStart("Hamiltonian Basis overlap eval");

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
                                                      *cfeBasisDataStorageAdaptiveOrbital,
                                                      *efeBasisDataAdaptiveOrbital,
                                                      *cfeBasisDataStorageAdaptiveOrbital,
                                                      ksdft::KSDFTDefaults::CELL_BATCH_SIZE,
                                                      numWantedEigenvalues,
                                                      linAlgOpContext,
                                                      true); 

  //   quadrature::QuadratureRuleAttributes quadAttrGaussEigen(quadrature::QuadratureFamily::GAUSS,true,feOrderEigen + 1);

  //   std::shared_ptr<basis::FEBasisDataStorage<double, Host>> cfeBasisDataStorageGaussEigen =
  //     std::make_shared<basis::CFEBDSOnTheFlyComputeDealii<double, double,Host, dim>>
  //     (cfeBasisDofHandlerEigen, quadAttrGaussEigen, basisAttrMap, ksdft::KSDFTDefaults::CELL_BATCH_SIZE_GRAD_EVAL, *linAlgOpContext);

  // cfeBasisDataStorageGaussEigen->evaluateBasisData(quadAttrGaussEigen , basisAttrMap);

  //   std::shared_ptr<const basis::OrthoEFEOverlapOperatorContext<double,
  //                                                 double,
  //                                                 Host,
  //                                                 dim>> MContextTestGauss =
  //   std::make_shared<basis::OrthoEFEOverlapOperatorContext<double,
  //                                                       double,
  //                                                       Host,
  //                                                       dim>>(
  //                                                       *basisManagerWaveFn,
  //                                                       *cfeBasisDataStorageGaussEigen,
  //                                                       *efeBasisDataAdaptiveOrbital,
  //                                                       /**cfeBasisDataStorageGLLEigen,*/
  //                                                       numWantedEigenvalues * ksdft::KSDFTDefaults::CELL_BATCH_SIZE,);  

    std::shared_ptr<const basis::OrthoEFEOverlapOperatorContext<double,
                                                  double,
                                                  Host,
                                                  dim>> MContextForInv =
    std::make_shared<basis::OrthoEFEOverlapOperatorContext<double,
                                                        double,
                                                        Host,
                                                        dim>>(
                                                        *basisManagerWaveFn,
                                                        *cfeBasisDataStorageGLLEigen,
                                                        *efeBasisDataAdaptiveOrbital,
                                                        *cfeBasisDataStorageGLLEigen,
                                                        linAlgOpContext,
                                                        true);  

    p.registerEnd("Hamiltonian Basis overlap eval");
    p.registerStart("Hamiltonian Basis overlap inverse eval");

  std::shared_ptr<linearAlgebra::OperatorContext<double,
                                                   double,
                                                   Host>> MInvContext =
    std::make_shared<basis::/*OrthoEFEOverlapInverseOpContextGLL*/OEFEAtomBlockOverlapInvOpContextGLL<double,
                                                   double,
                                                   Host,
                                                   dim>>
                                                   (*basisManagerWaveFn,
                                                    /**MContext,*/
                                                    *cfeBasisDataStorageGLLEigen,
                                                    *efeBasisDataAdaptiveOrbital,
                                                    *cfeBasisDataStorageGLLEigen,
                                                    linAlgOpContext);    

  p.registerEnd("Hamiltonian Basis overlap inverse eval");
  p.registerStart("Kohn Sham DFT Class Init");
  std::shared_ptr<ksdft::KohnShamDFT<double,
                                        double,
                                        double,
                                        double,
                                        Host,
                                        dim>> dftefeSolve = nullptr;

  if(isNumericalNuclearSolve && !isDeltaRhoPoissonSolve)
  {
    utils::throwException(false, "Option not there for KohnShamDFT class creation.");                        
  }
  else if (!isNumericalNuclearSolve && !isDeltaRhoPoissonSolve)
  {
    dftefeSolve =
    std::make_shared<ksdft::KohnShamDFT<double,
                                          double,
                                          double,
                                          double,
                                          Host,
                                          dim>>(
                                          atomCoordinatesVec,
                                          atomChargesVec,
                                          atomSymbolVec,
                                          smearedChargeRadiusVec,
                                          numElectrons,
                                          numWantedEigenvalues,
                                          smearingTemperature,
                                          fermiEnergyTolerance,
                                          fracOccupancyTolerance,
                                          eigenSolveResidualTolerance,
                                          scfDensityResidualNormTolerance,
                                          maxChebyshevFilterPass,
                                          maxSCFIter,
                                          evaluateEnergyEverySCF,
                                          mixingHistory,
                                          mixingParameter,
                                          isAdaptiveAndersonMixingParameter,
                                          basisManagerTotalPot,
                                          basisManagerWaveFn,
                                          feBDTotalChargeStiffnessMatrix,
                                          feBDNucChargeRhs, 
                                          feBDElecChargeRhs,  
                                          feBDKineticHamiltonian,     
                                          feBDElectrostaticsHamiltonian, 
                                          feBDEXCHamiltonian,      
                                          feBDAtomCenterNonLocalOperator,                                                                          
                                          atomSymbolToPSPFilename,
                                          linAlgOpContext,
                                          *MContextForInv,
                                          *MContext,
                                          /**MContextForInv,*/
                                          *MInvContext);
  }
  else if (!isNumericalNuclearSolve && isDeltaRhoPoissonSolve)
  {
    std::shared_ptr<utils::ScalarSpatialFunctionReal> smfuncAtTotPot = 
      std::make_shared<AtomicTotalElectrostaticPotentialFunction>(atomSphericalDataContainer,
                      atomSymbolVec,
                      atomCoordinatesVec);

  std::shared_ptr<utils::ScalarSpatialFunctionReal> elecChargeDens = 
    std::make_shared<RhoFunction>(atomSphericalDataContainer,
                    atomSymbolVec,
                    atomChargesVec,
                    atomCoordinatesVec);                      

    dftefeSolve =
    std::make_shared<ksdft::KohnShamDFT<double,
                                          double,
                                          double,
                                          double,
                                          Host,
                                          dim>>(
                                          atomCoordinatesVec,
                                          atomChargesVec,
                                          atomSymbolVec,
                                          smearedChargeRadiusVec,
                                          numElectrons,
                                          numWantedEigenvalues,
                                          smearingTemperature,
                                          fermiEnergyTolerance,
                                          fracOccupancyTolerance,
                                          eigenSolveResidualTolerance,
                                          scfDensityResidualNormTolerance,
                                          maxChebyshevFilterPass,
                                          maxSCFIter,
                                          evaluateEnergyEverySCF,
                                          mixingHistory,
                                          mixingParameter,
                                          isAdaptiveAndersonMixingParameter,
                                          *smfuncAtTotPot,
                                          *elecChargeDens,
                                          basisManagerTotalPot,
                                          basisManagerWaveFn,
                                          feBDTotalChargeStiffnessMatrix,
                                          feBDNucChargeRhs, 
                                          feBDElecChargeRhs,  
                                          feBDKineticHamiltonian,     
                                          feBDElectrostaticsHamiltonian, 
                                          feBDEXCHamiltonian,  
                                          feBDAtomCenterNonLocalOperator,                                                                              
                                          atomSymbolToPSPFilename,
                                          linAlgOpContext,
                                          *MContextForInv,
                                          /**MContextForInv,*/
                                          *MContext,
                                          *MInvContext);
  }
  else
  {
    utils::throwException(false, "Option not there for KohnShamDFT class creation.");
  }
  p.registerEnd("Kohn Sham DFT Class Init"); 
  p.print();

  pTot.registerEnd("Initilization");   
  pTot.registerStart("Kohn Sham DFT Solve");

  dftefeSolve->solve(); 

  pTot.registerEnd("Kohn Sham DFT Solve");
  pTot.print();

  //gracefully end MPI

  int mpiFinalFlag = 0;
  utils::mpi::MPIFinalized(&mpiFinalFlag);
  if(!mpiFinalFlag)
  {
    utils::mpi::MPIFinalize();
  }
}
