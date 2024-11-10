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
      , d_ylm00(atoms::Clm(0, 0) * atoms::Dm(0) * atoms::Plm(0, 0, 1) * atoms::Qm(0, 0))
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
      std::shared_ptr<const utils::SmearChargePotentialFunction> d_vsmear;

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
        d_vsmear = std::make_shared
                <utils::SmearChargePotentialFunction>(atomCoordinates, atomCharges, smearedChargeRadius);
      }

    double
    operator()(const utils::Point &point) const
    {
      return std::abs(((*d_vsmear)(point)) * ((*d_b)(point)));
    }
    std::vector<double>
    operator()(const std::vector<utils::Point> &points) const
    {
      std::vector<double> ret(0);
      ret.resize(points.size());
      for (unsigned int i = 0 ; i < points.size() ; i++)
      {
          ret[i] = std::abs(((*d_vsmear)(points[i])) * ((*d_b)(points[i])));
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
      const utils::ScalarSpatialFunctionReal *d_vext;

  public:
    VExternalTimesOrbitalSqFunction(
      std::shared_ptr<const atoms::AtomSphericalDataContainer>
                                         atomSphericalDataContainer,
      const utils::ScalarSpatialFunctionReal &vext,
        const std::vector<std::string> & atomSymbol,
        const std::vector<utils::Point> &atomCoordinates)
      : d_atomSphericalDataContainer(atomSphericalDataContainer)
      , d_atomSymbolVec(atomSymbol)
      , d_atomCoordinatesVec(atomCoordinates)
      , d_vext(&vext)
      {}

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
            double val = enrichmentObjId->getValue(point, origin);  
            retValue = retValue + std::abs(val * val * (*d_vext)(point));
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

  void
  readFile(const unsigned int                numColumns,
            std::vector<std::vector<double>> &data,
            const std::string &               fileName)
  {
    std::vector<std::vector<double>> dat(0);
    std::vector<double> rowData(numColumns, 0.0);
    std::ifstream       readFile(fileName.c_str());
    if (readFile.fail())
      {
        std::cerr << "Error opening file: " << fileName.c_str() << std::endl;
        exit(-1);
      }
    //
    // String to store line and word
    //
    std::string readLine;
    std::string word;
    //
    // column index
    //
    int columnCount;
    if (readFile.is_open())
      {
        while (std::getline(readFile, readLine))
          {
            std::istringstream iss(readLine);

            columnCount = 0;

            while (iss >> word && columnCount < numColumns)
              rowData[columnCount++] = atof(word.c_str());

            dat.push_back(rowData);
          }
      }
    readFile.close();

    data.resize(dat[0].size());
    for (int i = 0; i < dat[0].size(); i++) 
        for (int j = 0; j < dat.size(); j++) {
            data[i].push_back(dat[j][i]);
        }
  }

void getVLoc(
  const std::string &pspFilePath,
  std::vector<double> &radialValuesSTL,
  std::vector<double> &potentialValuesSTL)
 {
  std::string locPotFileName = std::string(pspFilePath) + "/locPot.dat";
  std::vector<std::vector<double>> data(0);
  readFile(2, data, locPotFileName);
  radialValuesSTL = data[0];
  potentialValuesSTL = data[1];
}


  class LocalPSPPotentialFunction : public utils::ScalarSpatialFunctionReal
  {
  private:
      std::vector<utils::Spline> d_atomTolocPSPSplineMap;
      std::vector<utils::Point> d_atomCoordinatesVec;
      std::vector<double> d_atomChargesVec;
      std::vector<double> d_radialLastValVec;

  public:
    LocalPSPPotentialFunction(
        const std::vector<utils::Point> &atomCoordinates,
        const std::vector<double> &      atomCharges,
        const std::vector<std::string> &pspFilePathVec)
      : d_atomTolocPSPSplineMap(0)
        , d_atomCoordinatesVec(atomCoordinates)
        , d_radialLastValVec(0)
        , d_atomChargesVec(atomCharges)
      {
        for( int i = 0 ; i < atomCoordinates.size() ; i++)
        {
          std::vector<double> radialValuesSTL(0);
          std::vector<double> potentialValuesLocSTL(0);

          getVLoc(
            pspFilePathVec[i],
            radialValuesSTL,
            potentialValuesLocSTL);

          double val = (-1.0)*potentialValuesLocSTL.back()/radialValuesSTL.back();
          d_radialLastValVec.push_back(radialValuesSTL.back());
          d_atomTolocPSPSplineMap.push_back(
              utils::Spline(radialValuesSTL,
                    potentialValuesLocSTL,
                    utils::Spline::spline_type::cspline,
                    false,
                    utils::Spline::bd_type::first_deriv,
                    0.0,
                    utils::Spline::bd_type::first_deriv,
                    val));
        }
      }

    double
    operator()(const utils::Point &point) const
    {
      double   retValue = 0;
      for (size_type atomId = 0 ; atomId < d_atomCoordinatesVec.size() ; atomId++)
        {
          double r = 0;
          for(int i = 0 ; i < point.size() ; i++)
            r += std::pow((point[i] - d_atomCoordinatesVec[atomId][i]),2);
          r = std::sqrt(r);
          retValue += (r <= d_radialLastValVec[atomId]) ? 
            d_atomTolocPSPSplineMap[atomId](r) : (-1.0)*(d_atomChargesVec[atomId]/r);
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
          for (unsigned int i = 0 ; i < points.size() ; i++)
          {
            double r = 0;
            for(int j = 0 ; j < points[i].size() ; j++)
              r += std::pow((points[i][j] - d_atomCoordinatesVec[atomId][j]),2);
            r = std::sqrt(r);
            ret[i] = ret[i] +  (r <= d_radialLastValVec[atomId]) ? 
              d_atomTolocPSPSplineMap[atomId](r) : (-1.0)*(d_atomChargesVec[atomId]/r);
          }
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

  unsigned int num1DGaussSubdividedSize = readParameter<unsigned int>(parameterInputFileName, "num1DGaussSubdividedSize", rootCout);
  unsigned int gaussSubdividedCopies = readParameter<unsigned int>(parameterInputFileName, "gaussSubdividedCopies", rootCout);
  
  bool isNumericalNuclearSolve = readParameter<bool>(parameterInputFileName, "isNumericalNuclearSolve", rootCout);

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

  std::vector<std::string> fieldNames{"density","vtotal","orbital","vnuclear"};
  std::vector<std::string> metadataNames{ "symbol", "Z", "charge", "NR", "r" };
  std::shared_ptr<atoms::AtomSphericalDataContainer>  atomSphericalDataContainer = 
      std::make_shared<atoms::AtomSphericalDataContainer>(atomSymbolToFilename,
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

  // 1. Make CFEBDSOnTheFlyComputeDealii object for Rhs (ADAPTIVE with GAUSS and fns are N_i^2 - make quadrulecontainer), overlapmatrix (GLL)
  // 2. Make EnrichmentClassicalInterface object for Orthogonalized enrichment
  // 3. Input to the EFEBasisDofHandler(eci, feOrder) 
  // 4. Make EFEBasisDataStorage with input as quadratureContainer.

    // Set the CFE basis manager and handler for bassiInterfaceCoeffcient distributed vector
  std::shared_ptr<const basis::FEBasisDofHandler<double, Host,dim>> cfeBasisDofHandlerElec =  
   std::make_shared<basis::CFEBasisDofHandlerDealii<double, Host,dim>>(triangulationBase, feOrderElec, comm);

  std::shared_ptr<const basis::FEBasisDofHandler<double, Host,dim>> cfeBasisDofHandlerEigen =  
   std::make_shared<basis::CFEBasisDofHandlerDealii<double, Host,dim>>(triangulationBase, feOrderEigen, comm);

  rootCout << "Total Number of classical dofs electrostatics: " << cfeBasisDofHandlerElec->nGlobalNodes() << "\n";
  rootCout << "Total Number of classical dofs eigensolve: " << cfeBasisDofHandlerEigen->nGlobalNodes() << "\n";

  const utils::ScalarSpatialFunctionReal *externalPotentialFunction = new 
    LocalPSPPotentialFunction(atomCoordinatesVec, atomChargesVec, pspFilePathVec);
  utils::mpi::MPIBarrier(comm);

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
        *externalPotentialFunction,
        atomSymbolVec,
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

    quadrature::QuadratureRuleAttributes quadAttrGllElec(quadrature::QuadratureFamily::GLL,true,feOrderElec + 1);

    // // Set up base quadrature rule for adaptive quadrature 

    // std::shared_ptr<quadrature::QuadratureRule> baseQuadRuleElec =
    //   std::make_shared<quadrature::QuadratureRuleGauss>(dim, feOrderElec + 1);

    // std::shared_ptr<basis::ParentToChildCellsManagerBase> parentToChildCellsManager = std::make_shared<basis::ParentToChildCellsManagerDealii<dim>>();

  //   // add device synchronize for gpu
  //   utils::mpi::MPIBarrier(comm);
     auto start = std::chrono::high_resolution_clock::now();
  //   std::shared_ptr<quadrature::QuadratureRuleContainer> quadRuleContainerAdaptiveElec =
  //     std::make_shared<quadrature::QuadratureRuleContainer>
  //     (quadAttrAdaptive, 
  //     baseQuadRuleElec, 
  //     triangulationBase, 
  //     *cellMapping, 
  //     *parentToChildCellsManager,
  //     functionsVec,
  //     absoluteTolerances,
  //     relativeTolerances,
  //     integralThresholds,
  //     smallestCellVolume,
  //     maxRecursion);

  //   unsigned int nQuad = quadRuleContainerAdaptiveElec->nQuadraturePoints();
  //   unsigned int nQuadMax = nQuad;
  //   int mpierr = utils::mpi::MPIAllreduce<Host>(
  //     utils::mpi::MPIInPlace,
  //     &nQuad,
  //     1,
  //     utils::mpi::Types<size_type>::getMPIDatatype(),
  //     utils::mpi::MPISum,
  //     comm);

  //   mpierr = utils::mpi::MPIAllreduce<Host>(
  //     utils::mpi::MPIInPlace,
  //     &nQuadMax,
  //     1,
  //     utils::mpi::Types<size_type>::getMPIDatatype(),
  //     utils::mpi::MPIMax,
  //     comm);
  //   rootCout << "Maximum Number of quadrature points in a processor: "<< nQuadMax<<"\n";
  // rootCout << "Number of quadrature points in adaptive quadrature: "<< nQuad<<"\n";

  //   // add device synchronize for gpu
  //     utils::mpi::MPIBarrier(comm);
      auto stop = std::chrono::high_resolution_clock::now();

      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  // rootCout << "Time for adaptive quadrature creation is(in secs) : " << duration.count()/1e6 << std::endl;

  //   quadrature::QuadratureRuleAttributes quadAttrGaussSubdivided(quadrature::QuadratureFamily::GAUSS_SUBDIVIDED,true);

  //   for ( unsigned int i=0 ;i < numfun ; i++ )
  //   {
  //     absoluteTolerances[i] = adaptiveQuadAbsTolerance * 1e1;
  //     relativeTolerances[i] = adaptiveQuadRelTolerance * 1e1;
  //   }

  //  start = std::chrono::high_resolution_clock::now();
  //   std::shared_ptr<quadrature::QuadratureRuleContainer> quadRuleContainerGaussSubdividedElec =
  //     std::make_shared<quadrature::QuadratureRuleContainer>
  //     (quadAttrGaussSubdivided, 
  //      feOrderElec+1,
  //      (feOrderElec+1)*2,
  //      3,
  //     triangulationBase, 
  //     *cellMapping,
  //     functionsVec,
  //     absoluteTolerances,
  //     relativeTolerances,
  //     *quadRuleContainerAdaptiveElec,
  //     comm);

    std::shared_ptr<quadrature::QuadratureRule> gaussSubdivQuadRuleElec =
      std::make_shared<quadrature::QuadratureRuleGaussIterated>(dim, num1DGaussSubdividedSize, gaussSubdividedCopies);

    quadrature::QuadratureRuleAttributes quadAttrGaussSubdivided(quadrature::QuadratureFamily::GAUSS_SUBDIVIDED,true);

    std::shared_ptr<quadrature::QuadratureRuleContainer> quadRuleContainerGaussSubdividedElec =
      std::make_shared<quadrature::QuadratureRuleContainer>
      (quadAttrGaussSubdivided, 
      gaussSubdivQuadRuleElec, 
      triangulationBase, 
      *cellMapping); 

    // add device synchronize for gpu
    utils::mpi::MPIBarrier(comm);
    stop = std::chrono::high_resolution_clock::now();

    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

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

  rootCout << "Time for gauss subdivided quadrature creation is(in secs) : " << duration.count()/1e6 << std::endl;

    //Set up quadAttr for Rhs and OverlapMatrix
    
    quadrature::QuadratureRuleAttributes quadAttrGllEigen(quadrature::QuadratureFamily::GLL,true,feOrderEigen + 1);

    // Set up base quadrature rule for adaptive quadrature 

    std::shared_ptr<quadrature::QuadratureRule> baseQuadRuleEigen = std::make_shared<quadrature::QuadratureRuleGauss>(dim, feOrderEigen + 1);
      // feOrderEigen > feOrderElec ? std::make_shared<quadrature::QuadratureRuleGauss>(dim, feOrderEigen + 1) : 
      //   std::make_shared<quadrature::QuadratureRuleGauss>(dim, feOrderElec + 1);

    // add device synchronize for gpu
    utils::mpi::MPIBarrier(comm);
    start = std::chrono::high_resolution_clock::now();

    std::shared_ptr<quadrature::QuadratureRuleContainer> quadRuleContainerAdaptiveOrbital = quadRuleContainerGaussSubdividedElec;
      // std::make_shared<quadrature::QuadratureRuleContainer>
      // (quadAttrAdaptive, 
      // baseQuadRuleEigen, 
      // triangulationBase, 
      // *cellMapping, 
      // *parentToChildCellsManager,
      // functionsVec,
      // absoluteTolerances,
      // relativeTolerances,
      // integralThresholds,
      // smallestCellVolume,
      // maxRecursion);

    // add device synchronize for gpu
      utils::mpi::MPIBarrier(comm);
      stop = std::chrono::high_resolution_clock::now();

      duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    nQuad = quadRuleContainerAdaptiveOrbital->nQuadraturePoints();
    mpierr = utils::mpi::MPIAllreduce<Host>(
      utils::mpi::MPIInPlace,
      &nQuad,
      1,
       utils::mpi::Types<size_type>::getMPIDatatype(),
      utils::mpi::MPISum,
      comm);

  rootCout << "Number of quadrature points in wave function adaptive quadrature: "<<nQuad<<"\n";

  rootCout << "Time for adaptive quadrature creation is(in secs) : " << duration.count()/1e6 << std::endl;

    // add device synchronize for gpu
    utils::mpi::MPIBarrier(comm);
    start = std::chrono::high_resolution_clock::now();
        
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

    // Create the enrichmentClassicalInterface object for vtotal
      std::shared_ptr<basis::EnrichmentClassicalInterfaceSpherical
                          <double, Host, dim>>
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

  std::shared_ptr<basis::FEBasisDofHandler<double, Host,dim>> basisDofHandlerTotalPot =  
    std::make_shared<basis::EFEBasisDofHandlerDealii<double, double,Host,dim>>(
      enrichClassIntfceTotalPot, comm);

  std::shared_ptr<basis::FEBasisDofHandler<double, Host,dim>> basisDofHandlerWaveFn =  
    std::make_shared<basis::EFEBasisDofHandlerDealii<double, double,Host,dim>>(
      enrichClassIntfceOrbital, comm);

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

  basisAttrMap[basis::BasisStorageAttributes::StoreValues] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradient] = true;
  basisAttrMap[basis::BasisStorageAttributes::StoreHessian] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreOverlap] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradNiGradNj] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreJxW] = true;

  // Set up Adaptive quadrature for EFE Basis Data Storage
  std::shared_ptr<basis::FEBasisDataStorage<double, Host>> efeBasisDataAdaptiveTotPot =
  std::make_shared<basis::EFEBDSOnTheFlyComputeDealii<double, double, Host,dim>>
  (basisDofHandlerTotalPot, quadAttrGaussSubdivided, basisAttrMap, ksdft::KSDFTDefaults::CELL_BATCH_SIZE_GRAD_EVAL, *linAlgOpContext);

    // add device synchronize for gpu
    utils::mpi::MPIBarrier(comm);
    start = std::chrono::high_resolution_clock::now();

    efeBasisDataAdaptiveTotPot->evaluateBasisData(quadAttrGaussSubdivided, quadRuleContainerGaussSubdividedElec, basisAttrMap);

    std::shared_ptr<const basis::FEBasisDataStorage<double, Host>> feBDTotalChargeStiffnessMatrix = efeBasisDataAdaptiveTotPot;

    basisAttrMap[basis::BasisStorageAttributes::StoreValues] = true;
    basisAttrMap[basis::BasisStorageAttributes::StoreGradient] = false;
    basisAttrMap[basis::BasisStorageAttributes::StoreHessian] = false;
    basisAttrMap[basis::BasisStorageAttributes::StoreOverlap] = false;
    basisAttrMap[basis::BasisStorageAttributes::StoreGradNiGradNj] = false;
    basisAttrMap[basis::BasisStorageAttributes::StoreJxW] = true;

    std::shared_ptr<basis::FEBasisDataStorage<double, Host>> feBDTotalChargeRhs =   
      std::make_shared<basis::EFEBDSOnTheFlyComputeDealii<double, double, Host,dim>>
      (basisDofHandlerTotalPot, quadAttrGaussSubdivided, basisAttrMap, ksdft::KSDFTDefaults::CELL_BATCH_SIZE_GRAD_EVAL, *linAlgOpContext);
    feBDTotalChargeRhs->evaluateBasisData(quadAttrGaussSubdivided, quadRuleContainerAdaptiveOrbital, basisAttrMap);

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
    std::make_shared<basis::EFEBDSOnTheFlyComputeDealii<double, double, Host,dim>>
      (basisDofHandlerWaveFn, quadAttrGaussSubdivided, basisAttrMap, ksdft::KSDFTDefaults::CELL_BATCH_SIZE_GRAD_EVAL, *linAlgOpContext);

    // add device synchronize for gpu
    utils::mpi::MPIBarrier(comm);
    start = std::chrono::high_resolution_clock::now();

  efeBasisDataAdaptiveOrbital->evaluateBasisData(quadAttrGaussSubdivided, quadRuleContainerAdaptiveOrbital, basisAttrMap);

    std::shared_ptr<const basis::FEBasisDataStorage<double, Host>> feBDElectrostaticsHamiltonian = efeBasisDataAdaptiveOrbital;
    std::shared_ptr<const basis::FEBasisDataStorage<double,Host>> feBDKineticHamiltonian =  efeBasisDataAdaptiveOrbital;
    std::shared_ptr<const basis::FEBasisDataStorage<double, Host>> feBDEXCHamiltonian = efeBasisDataAdaptiveOrbital;

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
                                                      *cfeBasisDataStorageAdaptiveOrbital,
                                                      *efeBasisDataAdaptiveOrbital,
                                                      *cfeBasisDataStorageAdaptiveOrbital,
                                                      ksdft::KSDFTDefaults::CELL_BATCH_SIZE,
                                                      numWantedEigenvalues,
                                                      linAlgOpContext,
                                                      true); 

    // add device synchronize for gpu
      utils::mpi::MPIBarrier(comm);
      stop = std::chrono::high_resolution_clock::now();

      duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  rootCout << "Time for creation of MContext is(in secs) : " << duration.count()/1e6 << std::endl;                                                                                                  

    // add device synchronize for gpu
    utils::mpi::MPIBarrier(comm);
    start = std::chrono::high_resolution_clock::now();

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
                                                        linAlgOpContext);  

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
                                                    /**MContext,*/
                                                    *cfeBasisDataStorageGLLEigen,
                                                    *efeBasisDataAdaptiveOrbital,
                                                    *cfeBasisDataStorageGLLEigen,
                                                    linAlgOpContext);    

    // add device synchronize for gpu
      utils::mpi::MPIBarrier(comm);
      stop = std::chrono::high_resolution_clock::now();

      duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  rootCout << "Time for creation of MInv is(in secs) : " << duration.count()/1e6 << std::endl;                                                                                                  

  rootCout << "Entering KohnSham DFT Class....\n\n";

  if(isNumericalNuclearSolve)
  {

    // Create the enrichmentClassicalInterface object for vnuclear
        std::shared_ptr<basis::EnrichmentClassicalInterfaceSpherical
                          <double, Host, dim>>
          enrichClassIntfceNucPot = std::make_shared<basis::EnrichmentClassicalInterfaceSpherical
                          <double, Host, dim>>
                          (cfeBasisDataStorageGLLElec,
                          cfeBasisDataStorageGaussSubdividedElec,
                          atomSphericalDataContainer,
                          atomPartitionTolerance,
                          atomSymbolVec,
                          atomCoordinatesVec,
                          "vnuclear",
                          linAlgOpContext,
                          comm);

    std::shared_ptr<basis::FEBasisDofHandler<double, Host,dim>> basisDofHandlerNucl =  
    std::make_shared<basis::EFEBasisDofHandlerDealii<double, double,Host,dim>>(
      enrichClassIntfceNucPot, comm);

    rootCout << "Total Number of dofs electrostatics: " << basisDofHandlerNucl->nGlobalNodes() << "\n";

  basisAttrMap[basis::BasisStorageAttributes::StoreValues] = true;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradient] = true;
  basisAttrMap[basis::BasisStorageAttributes::StoreHessian] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreOverlap] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradNiGradNj] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreJxW] = true;

  std::shared_ptr<basis::FEBasisDataStorage<double, Host>> efeBasisDataAdaptiveNucl =
  std::make_shared<basis::EFEBDSOnTheFlyComputeDealii<double, double, Host,dim>>
  (basisDofHandlerNucl, quadAttrGaussSubdivided, basisAttrMap, ksdft::KSDFTDefaults::CELL_BATCH_SIZE_GRAD_EVAL, *linAlgOpContext);

  efeBasisDataAdaptiveNucl->evaluateBasisData(quadAttrGaussSubdivided, quadRuleContainerGaussSubdividedElec, basisAttrMap);

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
                                          *MContextForInv,
                                          *MContext,
                                          /**MContextForInv,*/
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
                                          *MContextForInv,
                                          *MContext,
                                          /**MContextForInv,*/
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
