#include <basis/TriangulationBase.h>
#include <basis/TriangulationDealiiParallel.h>
#include <basis/CellMappingBase.h>
#include <basis/LinearCellMappingDealii.h>
#include <basis/CFEBasisDofHandlerDealii.h>
#include <basis/CFEBDSOnTheFlyComputeDealii.h>
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
            d_atomTolocPSPSplineMap[atomId](r) : (-1.0)*std::abs(d_atomChargesVec[atomId]/r);
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
              d_atomTolocPSPSplineMap[atomId](r) : (-1.0)*std::abs(d_atomChargesVec[atomId]/r);
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
  double gridSizeFD = readParameter<double>(parameterInputFileName, "gridSizeFD", rootCout);
  unsigned int dimIdPerturbed = readParameter<unsigned int>(parameterInputFileName, "dimIdPerturbed", rootCout);
  unsigned int atomIdPerturbed = readParameter<unsigned int>(parameterInputFileName, "atomIdPerturbed", rootCout);
  size_type numDimPerturbed = 1;
  size_type numAtomPerturbed = 1;
  unsigned int num1DGaussSizeElec = readParameter<unsigned int>(parameterInputFileName, "num1DGaussSizeElec", rootCout);
  unsigned int gaussSubdividedCopiesElec = readParameter<unsigned int>(parameterInputFileName, "gaussSubdividedCopiesElec", rootCout);
  unsigned int num1DGaussSizeEigen = readParameter<unsigned int>(parameterInputFileName, "num1DGaussSizeEigen", rootCout);
  unsigned int gaussSubdividedCopiesEigen = readParameter<unsigned int>(parameterInputFileName, "gaussSubdividedCopiesEigen", rootCout);
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
    std::vector<std::string> pspFilePathVec(0);
  coordinates.resize(dim,0.);
  std::vector<std::string> atomSymbolVec(0);
  std::vector<double> atomChargesVec(0);
  std::string symbol;
  std::string pspFilePath;
  double valanceNumber;
  atomSymbolVec.resize(0);
  std::string line;
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

  std::vector<std::vector<dftefe::utils::Point>> 
    atomCoordinatesVecInGrid(0, std::vector<dftefe::utils::Point>(0,dftefe::utils::Point(dim, 0.0)));

  atomCoordinatesVecInGrid.push_back(atomCoordinatesVec);
  for(unsigned int perturbDim = 0 ; perturbDim < numDimPerturbed ; perturbDim ++)
  {
    for(unsigned int perturbAtomId = 0 ; perturbAtomId < numAtomPerturbed ; perturbAtomId++ )
    {
      for(int gridPt = -2 ; gridPt <= 2 ; gridPt++)
      {
        std::vector<dftefe::utils::Point> coordinatesVec(atomCoordinatesVec.size(),dftefe::utils::Point(dim, 0.0));
        for (unsigned int atomId = 0 ; atomId < atomCoordinatesVec.size() ; atomId++)
        {
          for (unsigned int iDim = 0 ; iDim < dim ; iDim ++)
          {
            if(atomId == atomIdPerturbed && iDim == dimIdPerturbed)
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

  // initialize the basis Manager
  std::shared_ptr<basis::ParentToChildCellsManagerBase> parentToChildCellsManager = std::make_shared<basis::ParentToChildCellsManagerDealii<dim>>();

  std::vector<double> smearedChargeRadiusVec(atomCoordinatesVec.size(),rc);

  std::shared_ptr<const basis::FEBasisDofHandler<double, Host,dim>> basisDofHandlerTotalPot =  
   std::make_shared<basis::CFEBasisDofHandlerDealii<double, Host,dim>>(triangulationBase, feOrderElec, comm);

  std::shared_ptr<basis::FEBasisDofHandler<double, Host,dim>> basisDofHandlerWaveFn =  
   std::make_shared<basis::CFEBasisDofHandlerDealii<double, Host,dim>>(triangulationBase, feOrderEigen, comm);

  rootCout << "Total Number of dofs electrostatics: " << basisDofHandlerTotalPot->nGlobalNodes() << "\n";
  rootCout << "Total Number of dofs eigensolve: " << basisDofHandlerWaveFn->nGlobalNodes() << "\n";

  rootCout << "The Number of classical dofs electrostatics excluding Vacuum: " << 
    getNumClassicalDofsInSystemExcludingVacuum<double, Host, dim>(atomCoordinatesVec,
      *basisDofHandlerTotalPot,
      comm) << "\n";
  rootCout << "The Number of classical dofs eigenSolve excluding Vacuum: " << 
    getNumClassicalDofsInSystemExcludingVacuum<double, Host, dim>(atomCoordinatesVec,
      *basisDofHandlerWaveFn,
      comm) << "\n";
      
  // Set up the quadrature rule

  std::map<std::string, std::string> atomSymbolToFilename;
  for (auto i:atomSymbolVec )
  {
      atomSymbolToFilename[i] = sourceDir + i + ".xml";
  }

  std::vector<std::string> fieldNames{"density", "orbital", "vnuclear"};
  std::vector<std::string> metadataNames{ "symbol", "Z", "charge", "NR", "r" };
  std::shared_ptr<atoms::AtomSphericalDataContainer>  atomSphericalDataContainer = 
      std::make_shared<atoms::AtomSphericalDataContainer>(atomSymbolToFilename,
                                                      fieldNames,
                                                      metadataNames);

  // Compute Adaptive QuadratureRuleContainer for electrostaics

    // Set up the vector of scalarSpatialRealFunctions for adaptive quadrature
    // std::vector<std::shared_ptr<const utils::ScalarSpatialFunctionReal>> functionsVec(0);
    // unsigned int numfun = 0;
    // numfun = 2;
    // functionsVec.resize(numfun); // Enrichment Functions
    // std::vector<double> absoluteTolerances(numfun), relativeTolerances(numfun), integralThresholds(numfun);
    //   functionsVec[0] = std::make_shared<VExternalTimesOrbitalSqFunction>(
    //     atomSphericalDataContainer,
    //     *externalPotentialFunction,
    //     atomSymbolVec,
    //     atomCoordinatesVec);
    //   functionsVec[1] = std::make_shared<BTimesVNuclearFunction>(
    //   atomSphericalDataContainer,
    //   atomSymbolVec,
    //   atomChargesVec,
    //   smearedChargeRadiusVec,
    //   atomCoordinatesVec);
    // for ( unsigned int i=0 ;i < numfun ; i++ )
    // {
    //   absoluteTolerances[i] = adaptiveQuadAbsTolerance;
    //   relativeTolerances[i] = adaptiveQuadRelTolerance;
    //   integralThresholds[i] = integralThreshold;
    // }
    
  // Set up the quadrature rule

    // add device synchronize for gpu
    utils::mpi::MPIBarrier(comm);
    auto start = std::chrono::high_resolution_clock::now();

  quadrature::QuadratureRuleAttributes quadAttrElec(quadrature::QuadratureFamily::GAUSS,true,feOrderElec+1);

  basis::BasisStorageAttributesBoolMap basisAttrMap;
  basisAttrMap[basis::BasisStorageAttributes::StoreValues] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradient] = true;
  basisAttrMap[basis::BasisStorageAttributes::StoreHessian] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreOverlap] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradNiGradNj] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreJxW] = true;

  // Set up the FE Basis Data Storage
  std::shared_ptr<basis::FEBasisDataStorage<double, Host>> feBDTotalChargeStiffnessMatrix =
    std::make_shared<basis::CFEBDSOnTheFlyComputeDealii<double, double, Host,dim>>
    (basisDofHandlerTotalPot, quadAttrElec, basisAttrMap, ksdft::KSDFTDefaults::CELL_BATCH_SIZE_GRAD_EVAL, *linAlgOpContext);

  // evaluate basis data
  feBDTotalChargeStiffnessMatrix->evaluateBasisData(quadAttrElec, basisAttrMap);

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
    std::make_shared<basis::CFEBDSOnTheFlyComputeDealii<double, double, Host,dim>>
    (basisDofHandlerWaveFn, quadAttrGLLEigen, basisAttrMap, ksdft::KSDFTDefaults::CELL_BATCH_SIZE_GRAD_EVAL, *linAlgOpContext);

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

    std::shared_ptr<quadrature::QuadratureRule> gaussSubdivQuadRuleElec =
      std::make_shared<quadrature::QuadratureRuleGaussIterated>(dim, num1DGaussSizeElec, gaussSubdividedCopiesElec);

    std::shared_ptr<quadrature::QuadratureRule> gaussSubdivQuadRuleEigen =
      std::make_shared<quadrature::QuadratureRuleGaussIterated>(dim, num1DGaussSizeEigen, gaussSubdividedCopiesEigen);

    quadrature::QuadratureRuleAttributes quadAttrGaussSubdivided(quadrature::QuadratureFamily::GAUSS_SUBDIVIDED,true);

    std::shared_ptr<quadrature::QuadratureRuleContainer> quadRuleContainerGaussSubdividedElec =
      std::make_shared<quadrature::QuadratureRuleContainer>
      (quadAttrGaussSubdivided, 
      gaussSubdivQuadRuleElec, 
      triangulationBase, 
      *cellMapping); 

    unsigned int nQuad = quadRuleContainerGaussSubdividedElec->nQuadraturePoints();
    unsigned int nQuadMax = nQuad;
    int mpierr = utils::mpi::MPIAllreduce<Host>(
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
    rootCout << "Maximum Number of quadrature points in a processor elec: "<< nQuadMax<<"\n";
  rootCout << "Number of quadrature points in gauss subdivided quadrature elec: "<< nQuad<<"\n";

    std::shared_ptr<quadrature::QuadratureRuleContainer> quadRuleContainerGaussSubdividedEigen =
      std::make_shared<quadrature::QuadratureRuleContainer>
      (quadAttrGaussSubdivided, 
      gaussSubdivQuadRuleEigen, 
      triangulationBase, 
      *cellMapping); 

    nQuad = quadRuleContainerGaussSubdividedEigen->nQuadraturePoints();
    nQuadMax = nQuad;
    mpierr = utils::mpi::MPIAllreduce<Host>(
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
    rootCout << "Maximum Number of quadrature points in a processor eigen: "<< nQuadMax<<"\n";
  rootCout << "Number of quadrature points in gauss subdivided quadrature eigen: "<< nQuad<<"\n";

  basisAttrMap[basis::BasisStorageAttributes::StoreValues] = true;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradient] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreHessian] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreOverlap] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradNiGradNj] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreJxW] = true;

  std::shared_ptr<basis::FEBasisDataStorage<double, Host>> feBDElectrostaticsHamiltonian = 
    std::make_shared<basis::CFEBDSOnTheFlyComputeDealii<double, double, Host,dim>>
      (basisDofHandlerWaveFn, quadAttrGaussSubdivided, basisAttrMap, ksdft::KSDFTDefaults::CELL_BATCH_SIZE_GRAD_EVAL, *linAlgOpContext);
  feBDElectrostaticsHamiltonian->evaluateBasisData(quadAttrGaussSubdivided, quadRuleContainerGaussSubdividedEigen, basisAttrMap);

  std::shared_ptr<basis::FEBasisDataStorage<double, Host>> feBDElecChargeRhs = 
    std::make_shared<basis::CFEBDSOnTheFlyComputeDealii<double, double, Host,dim>>
      (basisDofHandlerTotalPot, quadAttrGaussSubdivided, basisAttrMap, ksdft::KSDFTDefaults::CELL_BATCH_SIZE_GRAD_EVAL, *linAlgOpContext);
  feBDElecChargeRhs->evaluateBasisData(quadAttrGaussSubdivided, quadRuleContainerGaussSubdividedEigen, basisAttrMap);
                
  basisAttrMap[basis::BasisStorageAttributes::StoreValues] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradient] = true;
  basisAttrMap[basis::BasisStorageAttributes::StoreHessian] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreOverlap] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradNiGradNj] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreJxW] = true;

  quadrature::QuadratureRuleAttributes quadAttrEigen(quadrature::QuadratureFamily::GAUSS,true,feOrderEigen+1);

  // Set up the FE Basis Data Storage
  std::shared_ptr<basis::FEBasisDataStorage<double, Host>> feBDHamStiffnessMatrix =
    std::make_shared<basis::CFEBDSOnTheFlyComputeDealii<double, double, Host,dim>>
    (basisDofHandlerWaveFn, quadAttrEigen, basisAttrMap, ksdft::KSDFTDefaults::CELL_BATCH_SIZE_GRAD_EVAL, *linAlgOpContext);

  // evaluate basis data
  feBDHamStiffnessMatrix->evaluateBasisData(quadAttrEigen, basisAttrMap);

  std::shared_ptr<const quadrature::QuadratureRuleContainer> quadRuleContainerRho =  
                feBDElectrostaticsHamiltonian->getQuadratureRuleContainer();

  // scale the electronic charges

  std::shared_ptr<const basis::FEBasisDataStorage<double,Host>> feBDKineticHamiltonian =  feBDHamStiffnessMatrix;
  std::shared_ptr<const basis::FEBasisDataStorage<double, Host>> feBDEXCHamiltonian = feBDElectrostaticsHamiltonian;

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
                                                      *feBDElectrostaticsHamiltonian,
                                                      ksdft::KSDFTDefaults::CELL_BATCH_SIZE,
                                                      numWantedEigenvalues,
                                                      linAlgOpContext);

  basisAttrMap[basis::BasisStorageAttributes::StoreValues] = true;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradient] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreHessian] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreOverlap] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradNiGradNj] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreJxW] = true;

  std::shared_ptr<basis::FEBasisDataStorage<double, Host>> feBDNucChargeRhs =
    std::make_shared<basis::CFEBDSOnTheFlyComputeDealii<double, double, Host,dim>>
      (basisDofHandlerTotalPot, quadAttrGaussSubdivided, basisAttrMap, ksdft::KSDFTDefaults::CELL_BATCH_SIZE_GRAD_EVAL, *linAlgOpContext);
  feBDNucChargeRhs->evaluateBasisData(quadAttrGaussSubdivided, quadRuleContainerGaussSubdividedElec, basisAttrMap);

  std::vector<double> energyInPerturbIds(atomCoordinatesVecInGrid.size(),0);
  for(unsigned int perturbId = 0 ; perturbId < atomCoordinatesVecInGrid.size(); perturbId++ )
  {
   quadrature::QuadratureValuesContainer<double, Host> 
      electronChargeDensity(quadRuleContainerRho, 1, 0.0);

    rootCout << "\nAtom Locations Displacement: \n";
    int count = 0;
    for(auto j : atomCoordinatesVecInGrid[perturbId])
    {
      rootCout << atomSymbolVec[count] << "\t" << j[0] - atomCoordinatesVec[count][0] << "\t" << j[1] - atomCoordinatesVec[count][1] << "\t" << j[2] - atomCoordinatesVec[count][2];
      rootCout << "\n";
      count ++;
    }

    std::shared_ptr<const utils::ScalarSpatialFunctionReal> rho = std::make_shared
                <RhoFunction>(atomSphericalDataContainer, atomSymbolVec, atomChargesVec, atomCoordinatesVecInGrid[perturbId]);

    const utils::ScalarSpatialFunctionReal *externalPotentialFunction = new 
      LocalPSPPotentialFunction(atomCoordinatesVecInGrid[perturbId], atomChargesVec, pspFilePathVec);

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

    // add device synchronize for gpu
    utils::mpi::MPIBarrier(comm);
    auto stop = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  rootCout << "Time for all basis storage evaluations including overlap operators(in secs) : " << duration.count()/1e6 << std::endl;

  rootCout << "Entering KohnSham DFT Class....\n\n";
  
    std::vector<double> smearedChargeRadiusVec(atomCoordinatesVecInGrid[perturbId].size(),rc);

    if(isNumericalNuclearSolve)
    {
      // Set up the FE Basis Data Storage
    std::shared_ptr<basis::FEBasisDataStorage<double, Host>> feBDNuclearChargeRhs = feBDNucChargeRhs;

    std::shared_ptr<const basis::FEBasisDataStorage<double,Host>> feBDNuclearChargeStiffnessMatrix = feBDTotalChargeStiffnessMatrix;

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
                                            atomCoordinatesVecInGrid[perturbId],
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
                                            /**MContextForInv,*/
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
                   
      energyInPerturbIds[perturbId] = dftefeSolve->getGroundStateEnergy();
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
                                            atomCoordinatesVecInGrid[perturbId],
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
                                            feBDNucChargeRhs, 
                                            feBDElecChargeRhs,  
                                            feBDKineticHamiltonian,     
                                            feBDElectrostaticsHamiltonian, 
                                            feBDEXCHamiltonian,                                                                                
                                            *externalPotentialFunction,
                                            linAlgOpContext,
                                            *MContextForInv,
                                            /**MContextForInv,*/
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

      energyInPerturbIds[perturbId] = dftefeSolve->getGroundStateEnergy();                                   
    }
  }

  std::vector<double> force(numDimPerturbed*numAtomPerturbed,0);
  unsigned int count = 0;
  for(unsigned int perturbDim = 0 ; perturbDim < numDimPerturbed ; perturbDim ++)
  {
    for(unsigned int perturbAtomId = 0 ; perturbAtomId < numAtomPerturbed ; perturbAtomId++ )
    {
      unsigned int index = (numAtomPerturbed*perturbDim + perturbAtomId)*4 + 1;
      rootCout << "The energies calculated by perturbation to atom "<< atomIdPerturbed <<
      " along Dim " << dimIdPerturbed <<" are: "<< energyInPerturbIds[index] << ", "<<energyInPerturbIds[index+1]
      <<", "<<energyInPerturbIds[index+2]<<", "<<energyInPerturbIds[index+3]<<"\n";
      force[count] = (-energyInPerturbIds[index] + 8*energyInPerturbIds[index+1] - 
        8*energyInPerturbIds[index+2] + energyInPerturbIds[index+3])/(12*gridSizeFD);
      count+=1;
    }
  }

  for(unsigned int perturbDim = 0 ; perturbDim < numDimPerturbed ; perturbDim ++)
  {
    for(unsigned int perturbAtomId = 0 ; perturbAtomId < numAtomPerturbed ; perturbAtomId++ )
    {
      unsigned int index = (numAtomPerturbed*perturbDim + perturbAtomId) + 1;
      rootCout << "The force calculated by perturbation to atom "<< atomIdPerturbed <<
      " along Dim " << dimIdPerturbed <<" is: "<< force[numAtomPerturbed*perturbDim + perturbAtomId]<<"\n";
    }
  }

  //gracefully end MPI

  int mpiFinalFlag = 0;
  utils::mpi::MPIFinalized(&mpiFinalFlag);
  if(!mpiFinalFlag)
  {
    utils::mpi::MPIFinalize();
  }
}
