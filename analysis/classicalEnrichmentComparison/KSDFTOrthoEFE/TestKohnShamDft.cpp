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

  class BPlusRhoTimesVTotalFunction : public utils::ScalarSpatialFunctionReal
  {
  private:
      std::shared_ptr<const atoms::AtomSphericalDataContainer>
                                d_atomSphericalDataContainer;
      std::vector<std::string>  d_atomSymbolVec;
      std::vector<utils::Point> d_atomCoordinatesVec;
      std::shared_ptr<const utils::ScalarSpatialFunctionReal> d_b, d_rho;

  public:
    BPlusRhoTimesVTotalFunction(
      std::shared_ptr<const atoms::AtomSphericalDataContainer>
                                         atomSphericalDataContainer,
        const std::vector<std::string> & atomSymbol,
        const std::vector<double> &      atomCharges,
        const std::vector<double> &      smearedChargeRadius,
        const std::vector<utils::Point> &atomCoordinates)
      : d_atomSphericalDataContainer(atomSphericalDataContainer)
      , d_atomSymbolVec(atomSymbol)
      , d_atomCoordinatesVec(atomCoordinates)
      {
        d_b = std::make_shared
                <utils::SmearChargeDensityFunction>(atomCoordinates, atomCharges, smearedChargeRadius);
        d_rho = std::make_shared
                <RhoFunction>(atomSphericalDataContainer, atomSymbol, atomCharges, atomCoordinates);
      }

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
            retValue = retValue + std::abs(enrichmentObjId->getValue(point, origin) *
                                    ((*d_b)(point) + (*d_rho)(point)));
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

  class BTimesVNuclearFunction : public utils::ScalarSpatialFunctionReal
  {
  private:
      std::shared_ptr<const atoms::AtomSphericalDataContainer>
                                d_atomSphericalDataContainer;
      std::vector<std::string>  d_atomSymbolVec;
      std::vector<utils::Point> d_atomCoordinatesVec;
      std::shared_ptr<const utils::ScalarSpatialFunctionReal> d_b;

  public:
    BTimesVNuclearFunction(
      std::shared_ptr<const atoms::AtomSphericalDataContainer>
                                         atomSphericalDataContainer,
        const std::vector<std::string> & atomSymbol,
        const std::vector<double> &      atomCharges,
        const std::vector<double> &      smearedChargeRadius,
        const std::vector<utils::Point> &atomCoordinates)
      : d_atomSphericalDataContainer(atomSphericalDataContainer)
      , d_atomSymbolVec(atomSymbol)
      , d_atomCoordinatesVec(atomCoordinates)
      {
        d_b = std::make_shared
                <utils::SmearChargeDensityFunction>(atomCoordinates, atomCharges, smearedChargeRadius);
      }

    double
    operator()(const utils::Point &point) const
    {
      double   retValue = 0;
      for (size_type atomId = 0 ; atomId < d_atomCoordinatesVec.size() ; atomId++)
        {
          utils::Point origin(d_atomCoordinatesVec[atomId]);
          for(auto &enrichmentObjId : 
            d_atomSphericalDataContainer->getSphericalData(d_atomSymbolVec[atomId], "vnuclear"))
          {
            retValue = retValue + std::abs(enrichmentObjId->getValue(point, origin) *
                                    ((*d_b)(point)));
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

  class VExternalTimesOrbitalSqFunction : public utils::ScalarSpatialFunctionReal
  {
  private:
      std::shared_ptr<const atoms::AtomSphericalDataContainer>
                                d_atomSphericalDataContainer;
      std::vector<std::string>  d_atomSymbolVec;
      std::vector<utils::Point> d_atomCoordinatesVec;
      std::shared_ptr<const utils::ScalarSpatialFunctionReal> d_vext;

  public:
    VExternalTimesOrbitalSqFunction(
      std::shared_ptr<const atoms::AtomSphericalDataContainer>
                                         atomSphericalDataContainer,
        const std::vector<std::string> & atomSymbol,
        const std::vector<double> &      atomCharges,
        const std::vector<utils::Point> &atomCoordinates)
      : d_atomSphericalDataContainer(atomSphericalDataContainer)
      , d_atomSymbolVec(atomSymbol)
      , d_atomCoordinatesVec(atomCoordinates)
      {
        d_vext = std::make_shared
                <utils::PointChargePotentialFunction>(atomCoordinates, atomCharges);
      }

    double
    operator()(const utils::Point &point) const
    {
      double   retValue = 0;
      for (size_type atomId = 0 ; atomId < d_atomCoordinatesVec.size() ; atomId++)
        {
          utils::Point origin(d_atomCoordinatesVec[atomId]);
          for(auto &enrichmentObjId : 
            d_atomSphericalDataContainer->getSphericalData(d_atomSymbolVec[atomId], "orbital"))
          {
            retValue = retValue + std::abs(enrichmentObjId->getValue(point, origin) * enrichmentObjId->getValue(point, origin) *
                                    (*d_vext)(point));
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

    utils::mpi::MPIBarrier(comm);
    auto startTotal = std::chrono::high_resolution_clock::now();

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
  unsigned int num1DGaussSize = readParameter<unsigned int>(parameterInputFileName, "num1DGaussSize", rootCout);
  unsigned int num1DGLLSize = readParameter<unsigned int>(parameterInputFileName, "num1DGLLSize", rootCout);
  unsigned int feOrder = readParameter<unsigned int>(parameterInputFileName, "feOrder", rootCout);
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

  double atomPartitionTolerance = readParameter<double>(parameterInputFileName, "atomPartitionTolerance", rootCout);
  double smallestCellVolume = readParameter<double>(parameterInputFileName, "smallestCellVolume", rootCout);
  unsigned int maxRecursion = readParameter<unsigned int>(parameterInputFileName, "maxRecursion", rootCout);
  double adaptiveQuadAbsTolerance = readParameter<double>(parameterInputFileName, "adaptiveQuadAbsTolerance", rootCout);
  double adaptiveQuadRelTolerance = readParameter<double>(parameterInputFileName, "adaptiveQuadRelTolerance", rootCout);
  double integralThreshold = readParameter<double>(parameterInputFileName, "integralThreshold", rootCout);

  bool isNumericalNuclearSolve = readParameter<bool>(parameterInputFileName, "isNumericalNuclearSolve", rootCout);

  // Set up Triangulation
    std::shared_ptr<basis::TriangulationBase> triangulationBase =
        std::make_shared<basis::TriangulationDealiiParallel<dim>>(comm);
  std::vector<bool>                 isPeriodicFlags(dim, false);
  std::vector<utils::Point> domainVectors(dim, utils::Point(dim, 0.0));

  domainVectors[0][0] = xmax;
  domainVectors[1][1] = ymax;
  domainVectors[2][2] = zmax;

  
  // // Uniform mesh creation
  // std::vector<unsigned int>         subdivisions = {30, 30, 30};
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
  //   triangulationBase->finalizeTriangulationConstruction();
  

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

  std::vector<std::string> fieldNames{"density","vtotal","orbital","vnuclear"};
  std::vector<std::string> metadataNames{ "symbol", "Z", "charge", "NR", "r" };
  std::shared_ptr<atoms::AtomSphericalDataContainer>  atomSphericalDataContainer = 
      std::make_shared<atoms::AtomSphericalDataContainer>(atomSymbolToFilename,
                                                      fieldNames,
                                                      metadataNames);

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

  // Compute Adaptive QuadratureRuleContainer for electrostaics

    // Set up the vector of scalarSpatialRealFunctions for adaptive quadrature
    std::vector<std::shared_ptr<const utils::ScalarSpatialFunctionReal>> functionsVec(0);
    unsigned int numfun = 0;
    if(!isNumericalNuclearSolve)
      numfun = 6;
    else
      numfun = 9;
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
      functionsVec[4] = std::make_shared<BPlusRhoTimesVTotalFunction>(
        atomSphericalDataContainer,
        atomSymbolVec,
        atomChargesVec,
        smearedChargeRadiusVec,
        atomCoordinatesVec);
      functionsVec[5] = std::make_shared<VExternalTimesOrbitalSqFunction>(
        atomSphericalDataContainer,
        atomSymbolVec,
        atomChargesVec,
        atomCoordinatesVec);
    if(isNumericalNuclearSolve)
    {
      for ( unsigned int i=0 ;i < 3 ; i++ )
      {
        if( i < 2)
          functionsVec[i+6] = std::make_shared<atoms::AtomSevereFunction<dim>>(        
            atomSphericalDataContainer,
            atomSymbolVec,
            atomCoordinatesVec,
            "vnuclear",
            i);
        else
            functionsVec[i+6] = std::make_shared<BTimesVNuclearFunction>(
            atomSphericalDataContainer,
            atomSymbolVec,
            atomChargesVec,
            smearedChargeRadiusVec,
            atomCoordinatesVec);
      }
    }
    for ( unsigned int i=0 ;i < numfun ; i++ )
    {
      absoluteTolerances[i] = adaptiveQuadAbsTolerance;
      relativeTolerances[i] = adaptiveQuadRelTolerance;
      integralThresholds[i] = integralThreshold;
    }
    //Set up quadAttr for Rhs and OverlapMatrix

    quadrature::QuadratureRuleAttributes quadAttrAdaptive(quadrature::QuadratureFamily::ADAPTIVE,false);

    quadrature::QuadratureRuleAttributes quadAttrGll(quadrature::QuadratureFamily::GLL,true,num1DGLLSize);

    // Set up base quadrature rule for adaptive quadrature 

    std::shared_ptr<quadrature::QuadratureRule> baseQuadRule =
      std::make_shared<quadrature::QuadratureRuleGauss>(dim, num1DGaussSize);

    std::shared_ptr<basis::ParentToChildCellsManagerBase> parentToChildCellsManager = std::make_shared<basis::ParentToChildCellsManagerDealii<dim>>();

    // add device synchronize for gpu
    utils::mpi::MPIBarrier(comm);
    auto start = std::chrono::high_resolution_clock::now();

    std::shared_ptr<quadrature::QuadratureRuleContainer> quadRuleContainerAdaptiveElec =
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
      smallestCellVolume,
      maxRecursion);

    // add device synchronize for gpu
      utils::mpi::MPIBarrier(comm);
      auto stop = std::chrono::high_resolution_clock::now();

      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    unsigned int nQuad = quadRuleContainerAdaptiveElec->nQuadraturePoints();
    int mpierr = utils::mpi::MPIAllreduce<Host>(
      utils::mpi::MPIInPlace,
      &nQuad,
      1,
      utils::mpi::Types<size_type>::getMPIDatatype(),
      utils::mpi::MPISum,
      comm);

  rootCout << "Number of quadrature points in electrostatics adaptive quadrature: "<< nQuad<<"\n";

  rootCout << "Time for adaptive quadrature creation is(in secs) : " << duration.count()/1e6 << std::endl;

  // Compute Adaptive QuadratureRuleContainer for wavefn
/*
    // Set up the vector of scalarSpatialRealFunctions for adaptive quadrature
    numfun = 3;
    functionsVec.resize(numfun); // Enrichment Functions
    absoluteTolerances.resize(numfun);
    relativeTolerances.resize(numfun);
    integralThresholds.resize(numfun);
    for ( unsigned int i=0 ;i < functionsVec.size() ; i++ )
    {
      if( i < 2)
        functionsVec[i] = std::make_shared<atoms::AtomSevereFunction<dim>>(        
            atomSphericalDataContainer,
            atomSymbolVec,
            atomCoordinatesVec,
            "orbital",
            i);
        else
          functionsVec[i] = std::make_shared<VExternalTimesOrbitalSqFunction>(
            atomSphericalDataContainer,
            atomSymbolVec,
            atomChargesVec,
            atomCoordinatesVec);
        absoluteTolerances[i] = adaptiveQuadAbsTolerance;
        relativeTolerances[i] = adaptiveQuadRelTolerance;
        integralThresholds[i] = integralThreshold;
    }
*/
    //Set up quadAttr for Rhs and OverlapMatrix

    // Set up base quadrature rule for adaptive quadrature 
    std::shared_ptr<quadrature::QuadratureRuleContainer> quadRuleContainerAdaptiveOrbital = quadRuleContainerAdaptiveElec;

    nQuad = quadRuleContainerAdaptiveOrbital->nQuadraturePoints();
    mpierr = utils::mpi::MPIAllreduce<Host>(
      utils::mpi::MPIInPlace,
      &nQuad,
      1,
       utils::mpi::Types<size_type>::getMPIDatatype(),
      utils::mpi::MPISum,
      comm);

  rootCout << "Number of quadrature points in wave function adaptive quadrature: "<<nQuad<<"\n";

    // add device synchronize for gpu
    utils::mpi::MPIBarrier(comm);
    start = std::chrono::high_resolution_clock::now();

  // Make orthogonalized EFE basis for all the fields

  // 1. Make CFEBasisDataStorageDealii object for Rhs (ADAPTIVE with GAUSS and fns are N_i^2 - make quadrulecontainer), overlapmatrix (GLL)
  // 2. Make EnrichmentClassicalInterface object for Orthogonalized enrichment
  // 3. Input to the EFEBasisDofHandler(eci, feOrder) 
  // 4. Make EFEBasisDataStorage with input as quadratureContainer.

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
    std::shared_ptr<basis::FEBasisDataStorage<double, Host>> cfeBasisDataStorageAdaptiveElec =
      std::make_shared<basis::CFEBasisDataStorageDealii<double, double,Host, dim>>
      (cfeBasisDofHandler, quadAttrAdaptive, basisAttrMap);
  // evaluate basis data
  cfeBasisDataStorageAdaptiveElec->evaluateBasisData(quadAttrAdaptive, quadRuleContainerAdaptiveElec, basisAttrMap);

    // Set the CFE basis manager and handler for bassiInterfaceCoeffcient distributed vector

  basisAttrMap[basis::BasisStorageAttributes::StoreValues] = true;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradient] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreHessian] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreOverlap] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradNiGradNj] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreJxW] = true;

    // Set up the CFE Basis Data Storage for Rhs
    std::shared_ptr<basis::FEBasisDataStorage<double, Host>> cfeBasisDataStorageAdaptiveOrbital =
      std::make_shared<basis::CFEBasisDataStorageDealii<double, double,Host, dim>>
      (cfeBasisDofHandler, quadAttrAdaptive, basisAttrMap);
  // evaluate basis data
  cfeBasisDataStorageAdaptiveOrbital->evaluateBasisData(quadAttrAdaptive, quadRuleContainerAdaptiveOrbital, basisAttrMap);

    // Create the enrichmentClassicalInterface object for vtotal
      std::shared_ptr<basis::EnrichmentClassicalInterfaceSpherical
                          <double, Host, dim>>
        enrichClassIntfceTotalPot = std::make_shared<basis::EnrichmentClassicalInterfaceSpherical
                          <double, Host, dim>>
                          (cfeBasisDataStorageGLL,
                          cfeBasisDataStorageAdaptiveElec,
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
                          cfeBasisDataStorageAdaptiveOrbital,
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

    // add device synchronize for gpu
      utils::mpi::MPIBarrier(comm);
      stop = std::chrono::high_resolution_clock::now();

      duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  rootCout << "Time for Ortho EFE basis manager creation is(in secs) : " << duration.count()/1e6 << std::endl;

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

  rootCout << "Number of quadrature points in wave function adaptive quadrature: "<<nQuad<<"\n";

    // add device synchronize for gpu
    utils::mpi::MPIBarrier(comm);
    start = std::chrono::high_resolution_clock::now();

  efeBasisDataAdaptiveTotPot->evaluateBasisData(quadAttrAdaptive, quadRuleContainerAdaptiveOrbital, basisAttrMap);

    // add device synchronize for gpu
      utils::mpi::MPIBarrier(comm);
      stop = std::chrono::high_resolution_clock::now();

      duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  rootCout << "Time for electrostatics basis datastorage evaluation is(in secs) : " << duration.count()/1e6 << std::endl;

  // std::ofstream fout("filename.txt");
  // utils::ConditionalOStream allCout(fout);
  // auto basisDataAdapElec = efeBasisDataAdaptiveTotPot->getBasisDataInAllCells();
  // for(int iProc = 0 ; iProc < numProcs ; iProc++)
  // {
  //   if(rank == iProc)
  //   {
  //     int quadId = 0;
  //     for (size_type iCell = 0; iCell < quadRuleContainerAdaptiveSolPot->nCells(); iCell++)
  //       {
  //         for(auto &i : quadRuleContainerAdaptiveSolPot->getCellRealPoints(iCell))
  //         {
  //           if(std::abs(i[2]) <= 1e-12)
  //             allCout << rank << " " << i[0] << " " << i[1] << " " << *(basisDataAdapElec.data() + quadId) << std::flush << std::endl;
  //           quadId += 1;
  //         }
  //       }
  //   }
  //   utils::mpi::MPIBarrier(comm);
  // }
  // utils::mpi::MPIBarrier(comm);
  // fout.close();

  basisAttrMap[basis::BasisStorageAttributes::StoreValues] = true;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradient] = true;
  basisAttrMap[basis::BasisStorageAttributes::StoreHessian] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreOverlap] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradNiGradNj] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreJxW] = true;

  std::shared_ptr<basis::FEBasisDataStorage<double, Host>> efeBasisDataAdaptiveOrbital =
  std::make_shared<basis::EFEBasisDataStorageDealii<double, double, Host,dim>>
  (basisDofHandlerWaveFn, quadAttrAdaptive, basisAttrMap);

    // add device synchronize for gpu
    utils::mpi::MPIBarrier(comm);
    start = std::chrono::high_resolution_clock::now();

  efeBasisDataAdaptiveOrbital->evaluateBasisData(quadAttrAdaptive, quadRuleContainerAdaptiveOrbital, basisAttrMap);

    // add device synchronize for gpu
      utils::mpi::MPIBarrier(comm);
      stop = std::chrono::high_resolution_clock::now();

      duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  rootCout << "Time for orbital basis datastorage evaluation is(in secs) : " << duration.count()/1e6 << std::endl;

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

  rootCout << "Number of quadrature points in wave function adaptive quadrature: "<<nQuad<<"\n";

    // add device synchronize for gpu
    utils::mpi::MPIBarrier(comm);
    start = std::chrono::high_resolution_clock::now();

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
                                                      *cfeBasisDataStorageAdaptiveOrbital,
                                                      *efeBasisDataAdaptiveOrbital,
                                                      *cfeBasisDataStorageGLL,
                                                      50); 

    // add device synchronize for gpu
      utils::mpi::MPIBarrier(comm);
      stop = std::chrono::high_resolution_clock::now();

      duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  rootCout << "Time for creation of MContext is(in secs) : " << duration.count()/1e6 << std::endl;                                                                                                  

    // add device synchronize for gpu
    utils::mpi::MPIBarrier(comm);
    start = std::chrono::high_resolution_clock::now();

    std::shared_ptr<const basis::OrthoEFEOverlapOperatorContext<double,
                                                  double,
                                                  Host,
                                                  dim>> MContextForInv =
    std::make_shared<basis::OrthoEFEOverlapOperatorContext<double,
                                                        double,
                                                        Host,
                                                        dim>>(
                                                        *basisManagerWaveFn,
                                                        *basisManagerWaveFn,
                                                        *cfeBasisDataStorageGLL,
                                                        *efeBasisDataAdaptiveOrbital,
                                                        *cfeBasisDataStorageGLL,
                                                        50);  

    // add device synchronize for gpu
      utils::mpi::MPIBarrier(comm);
      stop = std::chrono::high_resolution_clock::now();

      duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);                                                                                                            

  rootCout << "Time for creation of MContextForInv is(in secs) : " << duration.count()/1e6 << std::endl;                                                                                                  

    // add device synchronize for gpu
    utils::mpi::MPIBarrier(comm);
    start = std::chrono::high_resolution_clock::now();

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

    // add device synchronize for gpu
      utils::mpi::MPIBarrier(comm);
      stop = std::chrono::high_resolution_clock::now();

      duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  rootCout << "Time for creation of MInv is(in secs) : " << duration.count()/1e6 << std::endl;                                                                                                  

  rootCout << "Entering KohnSham DFT Class....\n\n";

  std::shared_ptr<const basis::FEBasisDataStorage<double, Host>> feBDTotalChargeStiffnessMatrix = efeBasisDataAdaptiveTotPot;
  std::shared_ptr<const basis::FEBasisDataStorage<double, Host>> feBDTotalChargeRhs = efeBasisDataAdaptiveTotPot;
  std::shared_ptr<const basis::FEBasisDataStorage<double, Host>> feBDElectrostaticsHamiltonian = efeBasisDataAdaptiveOrbital;
  std::shared_ptr<const basis::FEBasisDataStorage<double,Host>> feBDKineticHamiltonian =  efeBasisDataAdaptiveOrbital;
  std::shared_ptr<const basis::FEBasisDataStorage<double, Host>> feBDEXCHamiltonian = efeBasisDataAdaptiveOrbital;

  if(isNumericalNuclearSolve)
  {

    // Create the enrichmentClassicalInterface object for vnuclear
        std::shared_ptr<basis::EnrichmentClassicalInterfaceSpherical
                          <double, Host, dim>>
          enrichClassIntfceNucPot = std::make_shared<basis::EnrichmentClassicalInterfaceSpherical
                          <double, Host, dim>>
                          (cfeBasisDataStorageGLL,
                          cfeBasisDataStorageAdaptiveElec,
                          atomSphericalDataContainer,
                          atomPartitionTolerance,
                          atomSymbolVec,
                          atomCoordinatesVec,
                          "vnuclear",
                          linAlgOpContext,
                          comm);

    std::shared_ptr<basis::FEBasisDofHandler<double, Host,dim>> basisDofHandlerNucl =  
    std::make_shared<basis::EFEBasisDofHandlerDealii<double, double,Host,dim>>(
      enrichClassIntfceNucPot, feOrder, comm);

    rootCout << "Total Number of dofs electrostatics: " << basisDofHandlerNucl->nGlobalNodes() << "\n";

  basisAttrMap[basis::BasisStorageAttributes::StoreValues] = true;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradient] = true;
  basisAttrMap[basis::BasisStorageAttributes::StoreHessian] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreOverlap] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradNiGradNj] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreJxW] = true;

  std::shared_ptr<basis::FEBasisDataStorage<double, Host>> efeBasisDataAdaptiveNucl =
  std::make_shared<basis::EFEBasisDataStorageDealii<double, double, Host,dim>>
  (basisDofHandlerNucl, quadAttrAdaptive, basisAttrMap);

  efeBasisDataAdaptiveNucl->evaluateBasisData(quadAttrAdaptive, quadRuleContainerAdaptiveElec, basisAttrMap);

  std::shared_ptr<const basis::FEBasisDataStorage<double,Host>> feBDNuclearChargeStiffnessMatrix = efeBasisDataAdaptiveNucl;
  std::shared_ptr<const basis::FEBasisDataStorage<double,Host>> feBDNuclearChargeRhs = efeBasisDataAdaptiveNucl;

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
                                          *MContext,
                                          *MInvContext);

    // add device synchronize for gpu
    utils::mpi::MPIBarrier(comm);
    start = std::chrono::high_resolution_clock::now();

    dftefeSolve->solve();                 

    // add device synchronize for gpu
      utils::mpi::MPIBarrier(comm);
      stop = std::chrono::high_resolution_clock::now();

      duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    rootCout << "Time for scf iterations is(in secs) : " << duration.count()/1e6 << std::endl;    
                             
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
                                          *MContext,
                                          *MInvContext);

    // add device synchronize for gpu
    utils::mpi::MPIBarrier(comm);
    start = std::chrono::high_resolution_clock::now();

    dftefeSolve->solve();                

    // add device synchronize for gpu
      utils::mpi::MPIBarrier(comm);
      stop = std::chrono::high_resolution_clock::now();

      duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    rootCout << "Time for scf iterations is(in secs) : " << duration.count()/1e6 << std::endl;    

  }                          

  // add device synchronize for gpu
    utils::mpi::MPIBarrier(comm);
    auto stopTotal = std::chrono::high_resolution_clock::now();

    auto durationTotal = std::chrono::duration_cast<std::chrono::microseconds>(stopTotal - startTotal);

    rootCout << "Total wall time(in secs) : " << durationTotal.count()/1e6 << std::endl;

  //gracefully end MPI

  int mpiFinalFlag = 0;
  utils::mpi::MPIFinalized(&mpiFinalFlag);
  if(!mpiFinalFlag)
  {
    utils::mpi::MPIFinalize();
  }
}
