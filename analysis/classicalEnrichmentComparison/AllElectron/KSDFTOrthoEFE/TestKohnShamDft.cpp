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
#include <basis/OEFEAtomBlockOverlapInvOpContextGLL.h>
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

template<typename T>
T readParameter(const std::string &ParamFile,
                const std::string &param,
                utils::ConditionalOStream &rootCout,
                bool throwIfEmpty   = true,
                bool throwIfMissing = true,
                T defaultValue      = T{})
{
  T t = defaultValue;
  std::string line;
  std::fstream fstream(ParamFile, std::fstream::in);
  bool found = false;

  while (std::getline(fstream, line))
  {
    auto pos = line.find('=');
    if (pos == std::string::npos) continue;

    std::string key = line.substr(0, pos);
    // trim spaces
    key.erase(std::remove_if(key.begin(), key.end(), ::isspace), key.end());

    if (key == param)
    {
      found = true;
      std::string value = line.substr(pos + 1);
      // trim leading spaces
      value.erase(value.begin(),
                  std::find_if(value.begin(), value.end(),
                               [](unsigned char ch){ return !std::isspace(ch); }));

      if (value.empty()) {
        if (throwIfEmpty) {
          utils::throwException(false, "Parameter found but empty: " + param);
        }
        t = defaultValue;
      } else {
        if constexpr (std::is_same<T, std::string>::value) {
          t = value;
        } else {
          std::istringstream iss(value);
          iss >> t;
        }
      }
      break;
    }
  }

  if (!found) {
    if (throwIfMissing) {
      utils::throwException(false, "The parameter is not found: " + param);
    }
    t = defaultValue;
  }

  fstream.close();
  rootCout << "Reading parameter -- " << param << " = " << t << std::endl;
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
        const double &      smearedChargeRadius,
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
      for (size_type atomId = 0 ; atomId < d_atomCoordinatesVec.size() ; atomId++)
        {
          utils::Point origin(d_atomCoordinatesVec[atomId]);
          auto vec = d_atomSphericalDataContainer->getSphericalData(d_atomSymbolVec[atomId], "vtotal");
          for (unsigned int i = 0 ; i < points.size() ; i++)
          {
            for(auto &enrichmentObjId : vec)
            {
              ret[i] = ret[i] + std::abs(enrichmentObjId->getValue(points[i], origin) *
                                      ((*d_b)(points[i]) + (*d_rho)(points[i])));
            }
          }  
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
        const double &      smearedChargeRadius,
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
      for (size_type atomId = 0 ; atomId < d_atomCoordinatesVec.size() ; atomId++)
        {
          utils::Point origin(d_atomCoordinatesVec[atomId]);
          auto vec = d_atomSphericalDataContainer->getSphericalData(d_atomSymbolVec[atomId], "vnuclear");
          for (unsigned int i = 0 ; i < points.size() ; i++)
          {
            for(auto &enrichmentObjId : vec)
            {
              ret[i] = ret[i] + std::abs(enrichmentObjId->getValue(points[i], origin) *
                                    ((*d_b)(points[i])));
            }
          }  
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
      for (size_type atomId = 0 ; atomId < d_atomCoordinatesVec.size() ; atomId++)
        {
          utils::Point origin(d_atomCoordinatesVec[atomId]);
          auto vec = d_atomSphericalDataContainer->getSphericalData(d_atomSymbolVec[atomId], "orbital");
          for (unsigned int i = 0 ; i < points.size() ; i++)
          {
            for(auto &enrichmentObjId : vec)
            {
              double val = enrichmentObjId->getValue(points[i], origin);              
              ret[i] = ret[i] + std::abs(val * val * (*d_vext)(points[i]));
            }
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

  p.registerStart("Reading Parameter file data");
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
  double smallestCellVolume = readParameter<double>(parameterInputFileName, "smallestCellVolume", rootCout);
  unsigned int maxRecursion = readParameter<unsigned int>(parameterInputFileName, "maxRecursion", rootCout);
  double adaptiveQuadAbsTolerance = readParameter<double>(parameterInputFileName, "adaptiveQuadAbsTolerance", rootCout);
  double adaptiveQuadRelTolerance = readParameter<double>(parameterInputFileName, "adaptiveQuadRelTolerance", rootCout);
  double integralThreshold = readParameter<double>(parameterInputFileName, "integralThreshold", rootCout);

  bool isNumericalNuclearSolve = readParameter<bool>(parameterInputFileName, "isNumericalNuclearSolve", rootCout);
  bool isDeltaRhoPoissonSolve = readParameter<bool>(parameterInputFileName, "isDeltaRhoPoissonSolve", rootCout);

 std::string tciaFolder = readParameter<std::string>(parameterInputFileName, "tciaFolder", rootCout , false , false);
  std::string tciaOutFilePrefix = readParameter<std::string>(parameterInputFileName, "tciaOutFilePrefix", rootCout , false , false);

  const atoms::TCIADataParams  tciaparams{tciaFolder , tciaOutFilePrefix};

  // Set up Triangulation
    std::shared_ptr<basis::TriangulationBase> triangulationBase =
        std::make_shared<basis::TriangulationDealiiParallel<dim>>(comm);
  std::vector<bool>                 isPeriodicFlags(dim, false);
  std::vector<utils::Point> domainVectors(dim, utils::Point(dim, 0.0));

  domainVectors[0][0] = xmax;
  domainVectors[1][1] = ymax;
  domainVectors[2][2] = zmax;

  p.registerEnd("Reading Parameter file data");
  p.registerStart("Reading XML data");
  std::fstream fstream;
  fstream.open(inputFileName, std::fstream::in);
  
  // read the input file and create atomsymbol vector and atom coordinates vector.
  std::vector<utils::Point> atomCoordinatesVec(0,utils::Point(dim, 0.0));
    std::vector<double> coordinates;
        std::vector<std::string> xmlFileNameVec(0);
  coordinates.resize(dim,0.);
  std::vector<std::string> atomSymbolVec(0);
  std::vector<double> atomChargesVec(0);
  std::string symbol , xmlFileName;
  double atomicNumber;
  atomSymbolVec.resize(0);
  std::string line;
  while (std::getline(fstream, line)){
      std::stringstream ss(line);
      ss >> symbol; 
      ss >> xmlFileName; 
      ss >> atomicNumber; 
      for(unsigned int i=0 ; i<dim ; i++){
          ss >> coordinates[i]; 
      }
      atomCoordinatesVec.push_back(coordinates);
      atomSymbolVec.push_back(symbol);
      atomChargesVec.push_back((-1.0)*atomicNumber);
      xmlFileNameVec.push_back(xmlFileName);
  }
  fstream.close();

  size_type numElectrons = 0;
  for(auto &i : atomChargesVec)
  {
    numElectrons += (size_type)(std::abs(i));
  }

  std::map<std::string, std::string> atomSymbolToFilename;
  for (int i = 0 ; i < atomSymbolVec.size() ; i++)
  {
      atomSymbolToFilename[atomSymbolVec[i]] = sourceDir + xmlFileNameVec[i];
  }

  std::vector<std::string> fieldNames{"density","vtotal","orbital","vnuclear"};
  std::vector<std::string> metadataNames{ "symbol", "Z", "charge", "NR", "r" };
  std::shared_ptr<atoms::AtomSphericalDataContainer>  atomSphericalDataContainer = 
      std::make_shared<atoms::AtomSphericalDataContainer>(
                                                      atoms::AtomSphericalDataType::ENRICHMENT,
                                                      atomSymbolToFilename,
                                                      fieldNames,
                                                      metadataNames);

  for (auto i:atomSymbolVec )
  {
    rootCout << "For atom symbol: "<<i<<std::endl;
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
  p.registerEnd("Reading XML data");

  // Generate mesh
   std::shared_ptr<basis::CellMappingBase> cellMapping = std::make_shared<basis::LinearCellMappingDealii<dim>>();

  p.registerStart("Create Mesh");
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
    p.registerEnd("Create Mesh");

  utils::printCurrentMemoryUsage(comm, "Create Mesh");

    p.registerStart("Quadrature Rule Creation");

  // Compute Adaptive QuadratureRuleContainer for electrostaics

    // Set up the vector of scalarSpatialRealFunctions for adaptive quadrature
    std::vector<std::shared_ptr<const utils::ScalarSpatialFunctionReal>> functionsVec(0);
    std::vector<double> absoluteTolerances(0), relativeTolerances(0), integralThresholds(0);

    for ( unsigned int i=0 ;i < 2 ; i++ )
    {
      functionsVec.push_back(std::make_shared<atoms::AtomSevereFunction<dim>>(        
          atomSphericalDataContainer,
          atomSymbolVec,
          atomCoordinatesVec,
          "orbital",
          i));      
    }
      functionsVec.push_back(std::make_shared<VExternalTimesOrbitalSqFunction>(
        atomSphericalDataContainer,
        atomSymbolVec,
        atomChargesVec,
        atomCoordinatesVec));

    if(!isDeltaRhoPoissonSolve)
    {
    for ( unsigned int i=0 ;i < 2 ; i++ )
    {
      functionsVec.push_back(std::make_shared<atoms::AtomSevereFunction<dim>>(        
          atomSphericalDataContainer,
          atomSymbolVec,
          atomCoordinatesVec,
          "vtotal",
          i));      
    }
      functionsVec.push_back(std::make_shared<BPlusRhoTimesVTotalFunction>(
        atomSphericalDataContainer,
        atomSymbolVec,
        atomChargesVec,
        rc,
        atomCoordinatesVec));
    }
    if(isNumericalNuclearSolve)
    {
      for ( unsigned int i=0 ;i < 3 ; i++ )
      {
        if( i < 2)
          functionsVec.push_back(std::make_shared<atoms::AtomSevereFunction<dim>>(        
            atomSphericalDataContainer,
            atomSymbolVec,
            atomCoordinatesVec,
            "vnuclear",
            i));
        else
            functionsVec.push_back(std::make_shared<BTimesVNuclearFunction>(
            atomSphericalDataContainer,
            atomSymbolVec,
            atomChargesVec,
            rc,
            atomCoordinatesVec));
      }
    }
    for ( unsigned int i=0 ;i < functionsVec.size() ; i++ )
    {
      absoluteTolerances.push_back(adaptiveQuadAbsTolerance);
      relativeTolerances.push_back(adaptiveQuadRelTolerance);
      integralThresholds.push_back(integralThreshold);
    }
    //Set up quadAttr for Rhs and OverlapMatrix

    quadrature::QuadratureRuleAttributes quadAttrAdaptive(quadrature::QuadratureFamily::ADAPTIVE,false);

    quadrature::QuadratureRuleAttributes quadAttrGllElec(quadrature::QuadratureFamily::GLL,true,feOrderElec + 1);

    // Set up base quadrature rule for adaptive quadrature 

       rootCout << "Creating Adaptive Quad for Eigensolve"<<"\n";
    quadrature::QuadratureRuleAttributes quadAttrGaussSubdivided(quadrature::QuadratureFamily::GAUSS_SUBDIVIDED,true);

    std::shared_ptr<basis::ParentToChildCellsManagerBase> parentToChildCellsManager = std::make_shared<basis::ParentToChildCellsManagerDealii<dim>>();


    std::shared_ptr<quadrature::QuadratureRuleContainer> quadRuleContainerAdaptiveElec = nullptr;
   if (!isDeltaRhoPoissonSolve)
    {
      rootCout << "Creating Adaptive Quad for Electrostatics"<<"\n";
    std::shared_ptr<quadrature::QuadratureRule> baseQuadRuleElec =
      std::make_shared<quadrature::QuadratureRuleGauss>(dim, feOrderElec + 1);

    quadRuleContainerAdaptiveElec =
      std::make_shared<quadrature::QuadratureRuleContainer>
      (quadAttrAdaptive, 
      baseQuadRuleElec, 
      triangulationBase, 
      *cellMapping, 
      *parentToChildCellsManager,
      functionsVec,
      absoluteTolerances,
      relativeTolerances,
      integralThresholds,
      smallestCellVolume,
      maxRecursion);

    unsigned int nQuad = quadRuleContainerAdaptiveElec->nQuadraturePoints();

    auto mpierr = utils::mpi::MPIAllreduce<Host>(
      utils::mpi::MPIInPlace,
      &nQuad,
      1,
      utils::mpi::Types<size_type>::getMPIDatatype(),
      utils::mpi::MPISum,
      comm);
  rootCout << "Number of quadrature points in adaptive quadrature: "<< nQuad<<"\n";
    }

    //Set up quadAttr for Rhs and OverlapMatrix
    
    quadrature::QuadratureRuleAttributes quadAttrGllEigen(quadrature::QuadratureFamily::GLL,true,feOrderEigen + 1);

    // Set up base quadrature rule for adaptive quadrature 

    std::shared_ptr<quadrature::QuadratureRule> baseQuadRuleEigen = std::make_shared<quadrature::QuadratureRuleGauss>(dim, feOrderEigen + 1);
      // feOrderEigen > feOrderElec ? std::make_shared<quadrature::QuadratureRuleGauss>(dim, feOrderEigen + 1) : 
      //   std::make_shared<quadrature::QuadratureRuleGauss>(dim, feOrderElec + 1);

    std::shared_ptr<quadrature::QuadratureRuleContainer> quadRuleContainerAdaptiveOrbital = (quadRuleContainerAdaptiveElec != nullptr) ? quadRuleContainerAdaptiveElec :
      std::make_shared<quadrature::QuadratureRuleContainer>
      (quadAttrAdaptive, 
      baseQuadRuleEigen, 
      triangulationBase, 
      *cellMapping, 
      *parentToChildCellsManager,
      functionsVec,
      absoluteTolerances,
      relativeTolerances,
      integralThresholds,
      smallestCellVolume,
      maxRecursion);

    unsigned int nQuad = quadRuleContainerAdaptiveOrbital->nQuadraturePoints();
    int mpierr = utils::mpi::MPIAllreduce<Host>(
      utils::mpi::MPIInPlace,
      &nQuad,
      1,
      utils::mpi::Types<size_type>::getMPIDatatype(),
      utils::mpi::MPISum,
      comm);

  rootCout << "Number of quadrature points in wave function adaptive quadrature: "<<nQuad<<"\n";

  p.registerEnd("Quadrature Rule Creation");
    utils::printCurrentMemoryUsage(comm, "Quadrature Rule Creation");
  p.registerStart("Ortho EFE basis manager creation");

  // Make orthogonalized EFE basis for all the fields

  // 1. Make CFEBasisDataStorageDealii object for Rhs (ADAPTIVE with GAUSS and fns are N_i^2 - make quadrulecontainer), overlapmatrix (GLL)
  // 2. Make EnrichmentClassicalInterface object for Orthogonalized enrichment
  // 3. Input to the EFEBasisDofHandler(eci, feOrder) 
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
        
  basis::BasisStorageAttributesBoolMap basisAttrMap;
  basisAttrMap[basis::BasisStorageAttributes::StoreValues] = true;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradient] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreHessian] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreOverlap] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradNiGradNj] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreJxW] = true;

    // Set up the CFE Basis Data Storage for Overlap Matrix
    std::shared_ptr<basis::FEBasisDataStorage<double, Host>> cfeBasisDataStorageGLLElec =
      std::make_shared<basis::CFEBasisDataStorageDealii<double, double,Host, dim>>
      (cfeBasisDofHandlerElec, quadAttrGllElec, basisAttrMap);

    std::shared_ptr<basis::FEBasisDataStorage<double, Host>> cfeBasisDataStorageGLLEigen =
      std::make_shared<basis::CFEBasisDataStorageDealii<double, double,Host, dim>>
      (cfeBasisDofHandlerEigen, quadAttrGllEigen, basisAttrMap);

  // evaluate basis data
  cfeBasisDataStorageGLLElec->evaluateBasisData(quadAttrGllElec, basisAttrMap);
  cfeBasisDataStorageGLLEigen->evaluateBasisData(quadAttrGllEigen, basisAttrMap);

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
      (cfeBasisDofHandlerEigen, quadAttrAdaptive, basisAttrMap);
  // evaluate basis data
  cfeBasisDataStorageAdaptiveOrbital->evaluateBasisData(quadAttrAdaptive, quadRuleContainerAdaptiveOrbital, basisAttrMap);

      std::shared_ptr<basis::EnrichmentClassicalInterfaceSpherical
                          <double, Host, dim>>
        enrichClassIntfceTotalPot = nullptr;

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
  {

    basisAttrMap[basis::BasisStorageAttributes::StoreValues] = true;
    basisAttrMap[basis::BasisStorageAttributes::StoreGradient] = false;
    basisAttrMap[basis::BasisStorageAttributes::StoreHessian] = false;
    basisAttrMap[basis::BasisStorageAttributes::StoreOverlap] = false;
    basisAttrMap[basis::BasisStorageAttributes::StoreGradNiGradNj] = false;
    basisAttrMap[basis::BasisStorageAttributes::StoreJxW] = true;

    // Set up the CFE Basis Data Storage for Rhs
    std::shared_ptr<basis::FEBasisDataStorage<double, Host>> cfeBasisDataStorageAdaptiveElec =
      std::make_shared<basis::CFEBasisDataStorageDealii<double, double,Host, dim>>
      (cfeBasisDofHandlerElec, quadAttrAdaptive, basisAttrMap);
    // evaluate basis data
    cfeBasisDataStorageAdaptiveElec->evaluateBasisData(quadAttrAdaptive, quadRuleContainerAdaptiveElec, basisAttrMap);

    enrichClassIntfceTotalPot = std::make_shared<basis::EnrichmentClassicalInterfaceSpherical
                        <double, Host, dim>>
                        (cfeBasisDataStorageGLLElec,
                        cfeBasisDataStorageAdaptiveElec,
                        atomSphericalDataContainer,
                        atomPartitionTolerance,
                        atomSymbolVec,
                        atomCoordinatesVec,
                        "vtotal",
                        linAlgOpContext,
                        comm);
     basisDofHandlerTotalPot =  
    std::make_shared<basis::EFEBasisDofHandlerDealii<double, double,Host,dim>>(
      enrichClassIntfceTotalPot, comm);
  }
  else
    basisDofHandlerTotalPot = cfeBasisDofHandlerElec;

  std::shared_ptr<basis::FEBasisDofHandler<double, Host,dim>> basisDofHandlerWaveFn =  
    std::make_shared<basis::EFEBasisDofHandlerDealii<double, double,Host,dim>>(
      enrichClassIntfceOrbital, comm);

  p.registerEnd("Ortho EFE basis manager creation");
  utils::printCurrentMemoryUsage(comm, "Ortho EFE basis manager creation");

  rootCout << "Total Number of dofs electrostatics: " << basisDofHandlerTotalPot->nGlobalNodes() << "\n";
  rootCout << "Total Number of dofs eigensolve: " << basisDofHandlerWaveFn->nGlobalNodes() << "\n";

  basisAttrMap[basis::BasisStorageAttributes::StoreValues] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradient] = true;
  basisAttrMap[basis::BasisStorageAttributes::StoreHessian] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreOverlap] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradNiGradNj] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreJxW] = true;

  // Set up Adaptive quadrature for EFE Basis Data Storage
  p.registerStart("Electrostatics basis datastorage eval");
  quadrature::QuadratureRuleAttributes quadAttrGaussElectro(quadrature::QuadratureFamily::GAUSS,true,feOrderElec + 1);

  std::shared_ptr<basis::FEBasisDataStorage<double, Host>> feBDTotalChargeStiffnessMatrix = nullptr;
  if (!isDeltaRhoPoissonSolve)
    feBDTotalChargeStiffnessMatrix =
      std::make_shared<basis::EFEBasisDataStorageDealii<double, double, Host,dim>>
        (basisDofHandlerTotalPot, quadAttrAdaptive, basisAttrMap);
  else
    feBDTotalChargeStiffnessMatrix =
    std::make_shared<basis::CFEBDSOnTheFlyComputeDealii<double, double, Host,dim>>
    (basisDofHandlerTotalPot, quadAttrGaussElectro, basisAttrMap, ksdft::KSDFTDefaults::CELL_BATCH_SIZE_GRAD_EVAL, *linAlgOpContext);

  if (!isDeltaRhoPoissonSolve)
    feBDTotalChargeStiffnessMatrix->evaluateBasisData(quadAttrAdaptive, quadRuleContainerAdaptiveElec, basisAttrMap);
  else 
    feBDTotalChargeStiffnessMatrix->evaluateBasisData(quadAttrGaussElectro, basisAttrMap);

    basisAttrMap[basis::BasisStorageAttributes::StoreValues] = true;
    basisAttrMap[basis::BasisStorageAttributes::StoreGradient] = false;
    basisAttrMap[basis::BasisStorageAttributes::StoreHessian] = false;
    basisAttrMap[basis::BasisStorageAttributes::StoreOverlap] = false;
    basisAttrMap[basis::BasisStorageAttributes::StoreGradNiGradNj] = false;
    basisAttrMap[basis::BasisStorageAttributes::StoreJxW] = true;

  std::shared_ptr<basis::FEBasisDataStorage<double, Host>> feBDNucChargeRhs = nullptr;
  if (!isDeltaRhoPoissonSolve)
    feBDNucChargeRhs =   std::make_shared<basis::EFEBasisDataStorageDealii<double, double, Host,dim>>
      (basisDofHandlerTotalPot, quadAttrAdaptive, basisAttrMap);
  else
    feBDNucChargeRhs =   
      std::make_shared<basis::CFEBDSOnTheFlyComputeDealii<double, double, Host,dim>>
      (basisDofHandlerTotalPot, quadAttrGaussSubdivided, basisAttrMap, ksdft::KSDFTDefaults::CELL_BATCH_SIZE_GRAD_EVAL, *linAlgOpContext);

  if (!isDeltaRhoPoissonSolve)
    feBDNucChargeRhs->evaluateBasisData(quadAttrAdaptive, quadRuleContainerAdaptiveElec, basisAttrMap);
  else
  {
    size_type num1DGaussSubdividedSizeBSmear = 2;
    size_type gaussSubdividedCopiesBSmear = 10;
    std::shared_ptr<quadrature::QuadratureRule> gaussSubdivQuadRuleElec =
      std::make_shared<quadrature::QuadratureRuleGaussIterated>(dim, num1DGaussSubdividedSizeBSmear, gaussSubdividedCopiesBSmear);

    std::shared_ptr<quadrature::QuadratureRuleContainer> quadRuleContainerGaussSubdividedBSmear =
      std::make_shared<quadrature::QuadratureRuleContainer>
      (quadAttrGaussSubdivided, 
      gaussSubdivQuadRuleElec, 
      triangulationBase, 
      *cellMapping); 

    feBDNucChargeRhs->evaluateBasisData(quadAttrGaussSubdivided, quadRuleContainerGaussSubdividedBSmear, basisAttrMap);
  }

  std::shared_ptr<basis::FEBasisDataStorage<double, Host>> feBDElecChargeRhs = nullptr;
  if (!isDeltaRhoPoissonSolve)
  {
    feBDElecChargeRhs = std::make_shared<basis::EFEBasisDataStorageDealii<double, double, Host,dim>>
        (basisDofHandlerTotalPot, quadAttrAdaptive, basisAttrMap);
  }
  else
  {
    feBDElecChargeRhs = std::make_shared<basis::CFEBasisDataStorageDealii<double, double, Host,dim>>
        (basisDofHandlerTotalPot, quadAttrAdaptive, basisAttrMap);
  }
  feBDElecChargeRhs->evaluateBasisData(quadAttrAdaptive, quadRuleContainerAdaptiveOrbital, basisAttrMap);

  p.registerEnd("Electrostatics basis datastorage eval");
  utils::printCurrentMemoryUsage(comm, "Electrostatics basis rho datastorage eval");

  p.registerStart("Orbital basis datastorage eval");
  basisAttrMap[basis::BasisStorageAttributes::StoreValues] = true;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradient] = true;
  basisAttrMap[basis::BasisStorageAttributes::StoreHessian] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreOverlap] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradNiGradNj] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreJxW] = true;

  std::shared_ptr<basis::FEBasisDataStorage<double, Host>> efeBasisDataAdaptiveOrbital =
  std::make_shared<basis::EFEBasisDataStorageDealii<double, double, Host,dim>>
  (basisDofHandlerWaveFn, quadAttrAdaptive, basisAttrMap);

  efeBasisDataAdaptiveOrbital->evaluateBasisData(quadAttrAdaptive, quadRuleContainerAdaptiveOrbital, basisAttrMap);

    std::shared_ptr<const basis::FEBasisDataStorage<double, Host>> feBDElectrostaticsHamiltonian = efeBasisDataAdaptiveOrbital;
    std::shared_ptr<const basis::FEBasisDataStorage<double,Host>> feBDKineticHamiltonian =  efeBasisDataAdaptiveOrbital;
    std::shared_ptr<const basis::FEBasisDataStorage<double, Host>> feBDEXCHamiltonian = efeBasisDataAdaptiveOrbital;
  
    p.registerEnd("Orbital basis datastorage eval");

  std::shared_ptr<const quadrature::QuadratureRuleContainer> quadRuleContainerRho = 
                efeBasisDataAdaptiveOrbital->getQuadratureRuleContainer();

   quadrature::QuadratureValuesContainer<double, Host> 
      electronChargeDensity(quadRuleContainerRho, 1, 0.0);

    std::shared_ptr<const utils::ScalarSpatialFunctionReal> rho = std::make_shared
                <RhoFunction>(atomSphericalDataContainer, atomSymbolVec, atomChargesVec, atomCoordinatesVec);
 
  for (size_type iCell = 0; iCell < electronChargeDensity.nCells(); iCell++)
    {
          size_type             quadId = 0;
          std::vector<double> a(
            electronChargeDensity.nCellQuadraturePoints(iCell));
          a = (*rho)(quadRuleContainerRho->getCellRealPoints(iCell));
          double *b = a.data();
          electronChargeDensity.template 
            setCellValues<Host>(iCell, b);
    }

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
  utils::printCurrentMemoryUsage(comm, "Hamiltonian Basis overlap and inv");

  const utils::ScalarSpatialFunctionReal *externalPotentialFunction = new 
    utils::PointChargePotentialFunction(atomCoordinatesVec, atomChargesVec);
    
  p.registerStart("Kohn Sham DFT Class Init");
  ksdft::KohnShamDFT<double,
                    double,
                    double,
                    double,
                    Host,
                    dim>* dftefeSolve = nullptr;

  if (!isNumericalNuclearSolve && !isDeltaRhoPoissonSolve)
  {
    dftefeSolve =
     new ksdft::KohnShamDFT<double,
                            double,
                            double,
                            double,
                            Host,
                            dim>(
                                  atomCoordinatesVec,
                                  atomChargesVec,
                                  rc,
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
                                  electronChargeDensity,
                                  basisManagerTotalPot,
                                  basisManagerWaveFn,
                                  feBDTotalChargeStiffnessMatrix,
                                  feBDNucChargeRhs, 
                                  feBDElecChargeRhs,  
                                  feBDKineticHamiltonian,     
                                  feBDElectrostaticsHamiltonian, 
                                  feBDEXCHamiltonian,                                                                      
                                  *externalPotentialFunction,
                                  linAlgOpContext,
                                  *MContextForInv,
                                  *MContext,
                                  *MInvContext);
  }
  else if(isNumericalNuclearSolve && !isDeltaRhoPoissonSolve)
  {
    // Set up the FE Basis Data Storage
    std::shared_ptr<basis::FEBasisDataStorage<double, Host>> feBDNuclearChargeRhs = feBDNucChargeRhs;
    std::shared_ptr<const basis::FEBasisDataStorage<double,Host>> feBDNuclearChargeStiffnessMatrix = feBDTotalChargeStiffnessMatrix;

    dftefeSolve =
     new ksdft::KohnShamDFT<double,
                            double,
                            double,
                            double,
                            Host,
                            dim>(
                                  atomCoordinatesVec,
                                  atomChargesVec,
                                  rc,
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
                                  electronChargeDensity,
                                  basisManagerTotalPot,
                                  basisManagerWaveFn,
                                  feBDTotalChargeStiffnessMatrix,
                                  feBDNucChargeRhs, 
                                  feBDElecChargeRhs, 
                                  feBDNuclearChargeStiffnessMatrix,
                                  feBDNuclearChargeRhs, 
                                  feBDKineticHamiltonian,     
                                  feBDElectrostaticsHamiltonian, 
                                  feBDEXCHamiltonian,                                                                                
                                  *externalPotentialFunction,
                                  linAlgOpContext,
                                  *MContextForInv,
                                  *MContext,
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
     new ksdft::KohnShamDFT<double,
                            double,
                            double,
                            double,
                            Host,
                            dim>(
                                  atomCoordinatesVec,
                                  atomChargesVec,
                                  atomSymbolVec,
                                  rc,
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
                                  *externalPotentialFunction,
                                  linAlgOpContext,
                                  *MContextForInv,
                                  *MContext,
                                  *MInvContext,
                                  true,
                                  tciaparams);                                     
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

  delete dftefeSolve;

  //gracefully end MPI

  int mpiFinalFlag = 0;
  utils::mpi::MPIFinalized(&mpiFinalFlag);
  if(!mpiFinalFlag)
  {
    utils::mpi::MPIFinalize();
  }
}
