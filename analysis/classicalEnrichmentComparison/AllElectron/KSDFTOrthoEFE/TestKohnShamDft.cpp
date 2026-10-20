#include <basis/TriangulationBase.h>
#include <basis/PeriodicImageAtomGenerator.h>
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
#include <algorithm>
#include <cmath>
#include <memory>
#include <linearAlgebra/LinearSolverFunction.h>
#include <electrostatics/PoissonLinearSolverFunctionFE.h>
#include <linearAlgebra/LinearAlgebraProfiler.h>
#include <linearAlgebra/CGLinearSolver.h>
#include <ksdft/ElectrostaticLocalFE.h>
#include <ksdft/KineticFE.h>
#include <ksdft/ExchangeCorrelationFE.h>
#include <utils/DeviceUtils.h>
#include <ksdft/KohnShamOperatorContextFE.h>
#include <ksdft/KohnShamEigenSolver.h>
#include <basis/OrthoEFEOverlapInverseOpContextGLL.h>
#include <basis/OEFEAtomBlockOverlapInvOpContextGLL.h>
#include <utils/PointChargePotentialFunction.h>
#include <ksdft/DensityCalculator.h>
#include <ksdft/KohnShamDFT.h>
#include <basis/ReciprocalLattice.h>
#include <basis/GenerateMesh.h>
#include <utils/ConditionalOStream.h>
#include <atoms/AtomSevereFunction.h>
#include <atoms/SphericalHarmonics.h>
#include <filesystem>

#include <iostream>

using namespace dftefe;
const utils::MemorySpace memorySpace = utils::MemorySpace::DEVICE;
const utils::MemorySpace Host = utils::MemorySpace::HOST;

// Set the wavefunction pair to std::complex<double> for a k-point or a
// Gamma-complex run; the electrostatics pair is real in every case.
using ValueTypeElectrostaticsCoeff = double;
using ValueTypeElectrostaticsBasis = double;
using ValueTypeWaveFunctionCoeff   = std::complex<double>;
using ValueTypeWaveFunctionBasis   = std::complex<double>;

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
  utils::throwException(fstream.is_open(),
                        "Could not open the parameter file '" +
                          ParamFile + "'.");
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
      value.erase(std::find_if(value.rbegin(), value.rend(),
                               [](unsigned char ch){ return !std::isspace(ch); })
                    .base(),
                  value.end());

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

// The enrichment functions have a cusp at the nucleus and their radial spline is
// tabulated only down to r ~ 5e-4 a.u. A quadrature point that lands on (or very
// near) an atom therefore gets an extrapolated radial value, while
// SphericalDataNumerical::getGradientValue clamps the gradient to zero for
// r < RADIUS_TOL (1e-10). The gradXSq integrands are then wrong exactly at that
// point, the parent-vs-children comparison in QuadratureRuleAdaptive never
// converges, and the recursion runs to maxRecursion around that atom.
//
// An exact hit is possible because the coarse mesh is a uniform parallelepiped
// re-centred on the origin (GenerateMesh::generateCoarseMesh): with an odd number
// of subdivisions the origin is a cell *centre*, and the middle point of an odd
// order Gauss rule then sits exactly on an atom placed there. Near-hits are
// possible for any atom position.
//
// Report the closest approach of any quadrature point to any nucleus so that the
// condition is detectable instead of only showing up as a quadrature blow-up.
double
reportMinQuadPointAtomDistance(
  const quadrature::QuadratureRuleContainer &quadRuleContainer,
  const std::vector<utils::Point> &          atomCoordinates,
  const std::string &                        tag,
  utils::mpi::MPIComm                        comm,
  utils::ConditionalOStream &                rootCout,
  const double                               tolerance = 1e-8)
{
  const std::vector<utils::Point> &realPoints =
    quadRuleContainer.getRealPoints();
  const size_type numAtoms = atomCoordinates.size();
  const size_type dim      = (numAtoms == 0) ? 0 : atomCoordinates[0].size();

  double minDistSq   = std::numeric_limits<double>::max();
  double numTooClose = 0.0;

  for (const auto &q : realPoints)
    {
      double pointMinDistSq = std::numeric_limits<double>::max();
      for (size_type iAtom = 0; iAtom < numAtoms; ++iAtom)
        {
          double distSq = 0.0;
          for (size_type j = 0; j < dim; ++j)
            {
              const double d = q[j] - atomCoordinates[iAtom][j];
              distSq += d * d;
            }
          pointMinDistSq = std::min(pointMinDistSq, distSq);
        }
      minDistSq = std::min(minDistSq, pointMinDistSq);
      if (pointMinDistSq < tolerance * tolerance)
        numTooClose += 1.0;
    }

  double minDist = std::sqrt(minDistSq);
  utils::mpi::MPIAllreduce<Host>(utils::mpi::MPIInPlace,
                                 &minDist,
                                 1,
                                 utils::mpi::Types<double>::getMPIDatatype(),
                                 utils::mpi::MPIMin,
                                 comm);
  utils::mpi::MPIAllreduce<Host>(utils::mpi::MPIInPlace,
                                 &numTooClose,
                                 1,
                                 utils::mpi::Types<double>::getMPIDatatype(),
                                 utils::mpi::MPISum,
                                 comm);

  rootCout << tag << ": min |quadPoint - atom| = " << minDist << "\n";
  if (numTooClose > 0.0)
    {
      rootCout
        << "  *** WARNING: " << (size_type)numTooClose
        << " quadrature point(s) are within " << tolerance
        << " a.u. of a nucleus. The enrichment cusp is not resolvable there "
           "(radial spline is extrapolated below its first grid point and the "
           "gradient is clamped to zero), so the adaptive refinement will "
           "recurse to maxRecursion around that atom. ***\n";
    }
  return minDist;
}

// AtomicTotalElectrostaticPotentialFunction replaced by atoms::AtomSevereFunction<memorySpace> with "vtotal" field

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

  utils::Profiler<memorySpace> pTot(comm, "Total Statistics");
  utils::Profiler<memorySpace> p(comm, "Initilization Breakdown Statistics");
  pTot.registerStart("Initilization");

    // Get the rank of the process
  int rank;
  utils::mpi::MPICommRank(comm, &rank);

  utils::ConditionalOStream rootCout(std::cout);
  rootCout.setCondition(rank == 0);

  if constexpr (memorySpace == dftefe::utils::MemorySpace::DEVICE)
  {
  #  ifdef DFTEFE_WITH_DEVICE
        utils::DeviceUtils::setupDevice(rank);
        rootCout << "\nDFTEFE with GPU support, " << std::flush;
  #    ifdef DFTEFE_WITH_DEVICE_LANG_CUDA
        rootCout << "using CUDA, "<< std::flush;
  #    elif DFTEFE_WITH_DEVICE_LANG_HIP
        rootCout << "using HIP, "<< std::flush;
  #    endif
  #    endif
  #    ifdef DFTEFE_WITH_DEVICE_AWARE_MPI
        rootCout << "DFTEFE with device-aware MPI support, \n"<< std::flush;
  #    endif
  }

    // Get nProcs
    int numProcs;
    utils::mpi::MPICommSize(comm, &numProcs);

  p.registerStart("Setting LinAlgOpContext");
  int blasQueue = 0;
  int lapackQueue = 0;
  std::shared_ptr<linearAlgebra::LinAlgOpContext
    <memorySpace>> linAlgOpContext = 
    std::make_shared<linearAlgebra::LinAlgOpContext
    <memorySpace>>(10);

  std::shared_ptr<linearAlgebra::LinAlgOpContext
    <Host>> linAlgOpContextHost = 
    linearAlgebra::LinAlgOpContextDefaults::LINALG_OP_CONTXT_HOST;
  p.registerEnd("Setting LinAlgOpContext");

  p.registerStart("Reading Parameter file data");
  rootCout<<" Entering test kohn sham dft ortho enrichment \n";
  rootCout << "Number of processes: "<<numProcs<<"\n";

  //char* dftefe_path = getenv("DFTEFE_PATH");
  std::string sourceDir;
  // if executes if a non null value is returned
  // otherwise else executes

  try {
      // Get the current working directory
      std::filesystem::path currentPath = std::filesystem::current_path();
      sourceDir = currentPath.string();
  } catch (std::filesystem::filesystem_error const& ex) {
      std::cout << "Error: " << ex.what() << std::endl;
  }

  // if (dftefe_path != NULL) 
  // {
  //   sourceDir = (std::string)dftefe_path + "/analysis/classicalEnrichmentComparison/";
  // }
  // else
  // {
  //   utils::throwException(false,
  //                         "dftefe_path does not exist!");
  // }
  std::string paramDataFile = argv[1];
  std::string parameterInputFileName = sourceDir + paramDataFile;

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
  dftefe::size_type feOrderElec = readParameter<dftefe::size_type>(parameterInputFileName, "feOrderElectrostatics", rootCout);
  dftefe::size_type feOrderEigen = readParameter<dftefe::size_type>(parameterInputFileName, "feOrderEigenSolve", rootCout); 
  double    smearingTemperature = readParameter<double>(parameterInputFileName, "smearingTemperature", rootCout);
  double    fermiEnergyTolerance = readParameter<double>(parameterInputFileName, "fermiEnergyTolerance", rootCout);
  double    fracOccupancyTolerance = readParameter<double>(parameterInputFileName, "fracOccupancyTolerance", rootCout);
  double    eigenSolveResidualTolerance = readParameter<double>(parameterInputFileName, "eigenSolveResidualTolerance", rootCout);
  size_type maxChebyshevFilterPass = readParameter<size_type>(parameterInputFileName, "maxChebyshevFilterPass", rootCout);
  // Optional: if not present in the parameter file (or 0), the Chebyshev
  // polynomial degree is computed each SCF iteration from the internal
  // CHEBY_ORDER_LOOKUP table (see ksdft::LinearEigenSolverDefaults) instead.
  size_type chebyshevPolynomialDegree = readParameter<size_type>(parameterInputFileName, "chebyshevPolynomialDegree", rootCout, false, false, size_type(0));
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
  dftefe::size_type maxRecursion = readParameter<dftefe::size_type>(parameterInputFileName, "maxRecursion", rootCout);
  double adaptiveQuadAbsTolerance = readParameter<double>(parameterInputFileName, "adaptiveQuadAbsTolerance", rootCout);
  double adaptiveQuadRelTolerance = readParameter<double>(parameterInputFileName, "adaptiveQuadRelTolerance", rootCout);
  double integralThreshold = readParameter<double>(parameterInputFileName, "integralThreshold", rootCout);

  bool isNumericalNuclearSolve = readParameter<bool>(parameterInputFileName, "isNumericalNuclearSolve", rootCout);
  bool isDeltaRhoPoissonSolve = readParameter<bool>(parameterInputFileName, "isDeltaRhoPoissonSolve", rootCout);

  std::string coordinatesDataFile = readParameter<std::string>(parameterInputFileName, "coordinatesDataFile", rootCout , false , false);
  std::string basisDataFile = readParameter<std::string>(parameterInputFileName, "basisDataFile", rootCout , false , false);
  // optional: read density and vtotal (rho_at, phi_at) from a different
  // atomic reference. Empty means take them from basisDataFile.
  std::string basisDataFileRhoVtot = readParameter<std::string>(parameterInputFileName, "basisDataFileRhoVtot", rootCout , false , false, std::string(""));

 std::string tciaFolder = readParameter<std::string>(parameterInputFileName, "tciaFolder", rootCout , false , false);
  std::string tciaOutFilePrefix = readParameter<std::string>(parameterInputFileName, "tciaOutFilePrefix", rootCout , false , false);

  const atoms::TCIADataParams  tciaparams{tciaFolder , tciaOutFilePrefix};

  std::string xcFunctional = readParameter<std::string>(parameterInputFileName, "xc", rootCout, false, false, std::string("GGA-PBE"));
  std::string spinModeStr = readParameter<std::string>(parameterInputFileName, "spintype", rootCout, false, false, std::string("Unpolarized"));
  ksdft::SpinMode spinMode = ksdft::SpinMode::Unpolarized;
  if (spinModeStr == "Collinear")
    spinMode = ksdft::SpinMode::Collinear;
  else if (spinModeStr == "NonCollinear")
    spinMode = ksdft::SpinMode::NonCollinear;

  bool isGHEP = readParameter<bool>(parameterInputFileName, "isGHEP", rootCout, false, false, true);
  std::string orthoTypeStr = readParameter<std::string>(parameterInputFileName, "orthoType", rootCout, false, false, std::string("CHOLESKY_GRAMSCHMIDT"));
  linearAlgebra::OrthogonalizationType orthoType = linearAlgebra::OrthogonalizationType::CHOLESKY_GRAMSCHMIDT;
  if (orthoTypeStr == "MULTIPASS_CGS")
    orthoType = linearAlgebra::OrthogonalizationType::MULTIPASS_CGS;
  else if (orthoTypeStr == "MULTIPASS_LOWDIN")
    orthoType = linearAlgebra::OrthogonalizationType::MULTIPASS_LOWDIN;
  else if (orthoTypeStr != "CHOLESKY_GRAMSCHMIDT")
    utils::throwException(false, "Unknown orthoType: " + orthoTypeStr);

  // Set up Triangulation
    std::shared_ptr<basis::TriangulationBase> triangulationBase =
        std::make_shared<basis::TriangulationDealiiParallel<dim>>(comm);
  // default 0: non periodic,
  const int periodicX = readParameter<int>(
    parameterInputFileName, "periodicX", rootCout, true, false, 0);
  const int periodicY = readParameter<int>(
    parameterInputFileName, "periodicY", rootCout, true, false, 0);
  const int periodicZ = readParameter<int>(
    parameterInputFileName, "periodicZ", rootCout, true, false, 0);

  std::vector<bool> isPeriodicFlags(dim, false);
  isPeriodicFlags[0] = (periodicX != 0);
  isPeriodicFlags[1] = (periodicY != 0);
  isPeriodicFlags[2] = (periodicZ != 0);

  std::vector<utils::Point> domainVectors(dim, utils::Point(dim, 0.0));

  domainVectors[0][0] = xmax;
  domainVectors[1][1] = ymax;
  domainVectors[2][2] = zmax;

  const bool anyDirectionPeriodic = std::any_of(isPeriodicFlags.begin(),
                                                isPeriodicFlags.end(),
                                                [](bool v) { return v; });

    const utils::Point &a1 = domainVectors[0];
    const utils::Point &a2 = domainVectors[1];
    const utils::Point &a3 = domainVectors[2];
    const double        handedness =
      (a1[1] * a2[2] - a1[2] * a2[1]) * a3[0] +
      (a1[2] * a2[0] - a1[0] * a2[2]) * a3[1] +
      (a1[0] * a2[1] - a1[1] * a2[0]) * a3[2];
    utils::throwException(
      handedness > 0.0,
      "The domain bounding vectors must form a right-handed system, i.e. "
      "(a1 x a2).a3 > 0.");

  // Brillouin zone sampling; every parameter is optional and the defaults
  // give the single Gamma point. A non-periodic direction gets k = 0 only.
  const int useGroupSymmetry = readParameter<int>(
    parameterInputFileName, "useGroupSymmetry", rootCout, true, false, 0);
  const int useTimeReversalSymmetry = readParameter<int>(
    parameterInputFileName, "useTimeReversalSymmetry", rootCout, true, false, 0);
  const std::string kPointRuleFile = readParameter<std::string>(
    parameterInputFileName, "kPointRuleFile", rootCout, false, false, std::string(""));
  std::vector<size_type> samplingPoints(dim, 1);
  samplingPoints[0] = readParameter<size_type>(
    parameterInputFileName, "samplingPointsX", rootCout, true, false, 1);
  samplingPoints[1] = readParameter<size_type>(
    parameterInputFileName, "samplingPointsY", rootCout, true, false, 1);
  samplingPoints[2] = readParameter<size_type>(
    parameterInputFileName, "samplingPointsZ", rootCout, true, false, 1);
  std::vector<bool> samplingShift(dim, false);
  samplingShift[0] = (readParameter<int>(
    parameterInputFileName, "samplingShiftX", rootCout, true, false, 0) != 0);
  samplingShift[1] = (readParameter<int>(
    parameterInputFileName, "samplingShiftY", rootCout, true, false, 0) != 0);
  samplingShift[2] = (readParameter<int>(
    parameterInputFileName, "samplingShiftZ", rootCout, true, false, 0) != 0);

  utils::throwException(
    useGroupSymmetry == 0,
    "useGroupSymmetry: reducing the k-points by the point group is not "
    "implemented.");
  utils::throwException(
    kPointRuleFile.empty(),
    "kPointRuleFile: reading the k-points from a file is not implemented; use "
    "the Monkhorst-Pack parameters samplingPoints* and samplingShift*.");

  std::map<basis::kpoint::KPointId, basis::kpoint::KPointData> kPoints;
  basis::ReciprocalLattice(domainVectors)
    .generateMonkhorstPackGrid(samplingPoints,
                               isPeriodicFlags,
                               samplingShift,
                               useTimeReversalSymmetry != 0,
                               kPoints);
  // a non-periodic direction has k = 0 only, whatever was asked for
  for (size_type iDim = 0; iDim < dim; iDim++)
    if (!isPeriodicFlags[iDim] &&
        (samplingPoints[iDim] > 1 || samplingShift[iDim]))
      rootCout << "Warning: direction " << iDim
               << " is not periodic, so its k-point sampling is ignored.\n";
  rootCout << "Number of k-points: " << kPoints.size() << "\n";
  for (const auto &kPoint : kPoints)
    rootCout << "k-point " << (size_type)kPoint.first << ": ("
             << kPoint.second.coord[0] << ", " << kPoint.second.coord[1]
             << ", " << kPoint.second.coord[2]
             << "), weight " << kPoint.second.weight << "\n";

  p.registerEnd("Reading Parameter file data");
  p.registerStart("Reading Other Input data");
  std::fstream fstream;
  
  // read the input file and create atomsymbol vector and atom coordinates vector.
  std::vector<utils::Point> atomCoordinatesVec(0,utils::Point(dim, 0.0));
    std::vector<double> coordinates;
  coordinates.resize(dim,0.);
  std::vector<std::string> atomSymbolVec(0);
  std::vector<double> atomChargesVec(0);
  std::vector<double> atomMagMomentsVec(0);
  std::string symbol , basisFilePath, pspFilePath;
  std::map<std::string, double> atomSymbolToChargeMap;
  double atomicNumber;
  atomSymbolVec.resize(0);
  std::string line;

  std::map<std::string, std::string> atomSymbolToBasisFileName;
  std::vector<std::string> matchString(0);
  fstream.open(basisDataFile, std::fstream::in);
  if (!fstream.is_open()) {
      utils::throwException(false,
                            "Parameter 'basisDataFile' names a file that could "
                            "not be opened: '" + basisDataFile + "'");
  }
  while (std::getline(fstream, line)){
      std::stringstream ss(line);
      ss >> symbol; 
      ss >> basisFilePath; 
      atomSymbolToBasisFileName[symbol] = basisFilePath;
      if(std::find(matchString.begin(), matchString.end(), symbol) == matchString.end())
        matchString.push_back(symbol);
      else
        utils::throwException(false, "The atom Symbols were repeated for PSP filenames. ");       
  }
  utils::mpi::MPIBarrier(comm);
  fstream.close();

  // same "symbol path" format as basisDataFile; stays empty when the field
  // data is taken from basisDataFile itself
  std::map<std::string, std::string> atomSymbolToBasisFileNameRhoVtot;
  if (!basisDataFileRhoVtot.empty())
  {
    fstream.open(basisDataFileRhoVtot, std::fstream::in);
    utils::throwException(fstream.is_open(),
                          "Parameter 'basisDataFileRhoVtot' names a file that "
                          "could not be opened: '" + basisDataFileRhoVtot + "'");
    while (std::getline(fstream, line))
    {
      std::stringstream ss(line);
      symbol.clear();
      basisFilePath.clear();
      ss >> symbol;
      ss >> basisFilePath;
      if (symbol.empty())
        continue;
      utils::throwException(!basisFilePath.empty(),
                            "Parameter 'basisDataFileRhoVtot' has no path for "
                            "atom symbol '" + symbol + "'.");
      atomSymbolToBasisFileNameRhoVtot[symbol] = basisFilePath;
    }
    utils::mpi::MPIBarrier(comm);
    fstream.close();

    for (auto &i : atomSymbolToBasisFileName)
      utils::throwException(
        atomSymbolToBasisFileNameRhoVtot.count(i.first) == 1,
        "The file named by 'basisDataFileRhoVtot' has no entry for atom "
        "symbol '" + i.first + "', which is needed for rho_at and phi_at.");
  }

  fstream.open(coordinatesDataFile, std::fstream::in);
  if (!fstream.is_open()) {
      utils::throwException(false,
                            "Parameter 'coordinatesDataFile' names a file that could "
                            "not be opened: '" + coordinatesDataFile + "'");
  }
  while (std::getline(fstream, line)){
      std::stringstream ss(line);
      ss >> symbol; 
      ss >> atomicNumber; 
      for(dftefe::size_type i=0 ; i<dim ; i++){
          ss >> coordinates[i];
      }
      double magMoment = 0.0;
      ss >> magMoment;
      atomMagMomentsVec.push_back(magMoment);

      // dftefe reads FRACTIONAL coordinates whenever any direction is periodic
      // and converts to centred Cartesian before meshing
      // A fully non periodic file is Cartesian already, so left alone.
      if (anyDirectionPeriodic)
        {
          utils::Point cartesian(dim, 0.0);
          for (dftefe::size_type c = 0; c < dim; c++)
            {
              double value = 0.0;
              for (dftefe::size_type j = 0; j < dim; j++)
                value += coordinates[j] * domainVectors[j][c] -
                         0.5 * domainVectors[j][c];
              cartesian[c] = value;
            }
          for (dftefe::size_type c = 0; c < dim; c++)
            coordinates[c] = cartesian[c];
        }

      atomCoordinatesVec.push_back(coordinates);
      atomSymbolVec.push_back(symbol);
      if(atomSymbolToBasisFileName.find(symbol) == atomSymbolToBasisFileName.end())
      {
        utils::throwException(false, "Basis filename does not have the same atom symbol as Coordinate filename."); 
      }
      atomChargesVec.push_back((-1.0)*atomicNumber);
      if(atomSymbolToChargeMap.find(symbol) == atomSymbolToChargeMap.end())
        atomSymbolToChargeMap[symbol] = atomicNumber;
  }
  utils::mpi::MPIBarrier(comm);
  fstream.close();

  std::vector<double> atomMagZFactors;
  {
    bool anyNonZero = false;
    for (const auto &m : atomMagMomentsVec)
      if (std::abs(m) > 1e-12)
        {
          anyNonZero = true;
          break;
        }
    if (anyNonZero)
      {
        atomMagZFactors.resize(atomMagMomentsVec.size());
        for (dftefe::size_type i = 0; i < atomMagMomentsVec.size(); ++i)
          atomMagZFactors[i] = atomMagMomentsVec[i] / (-atomChargesVec[i]);
      }
  }

  size_type numElectrons = 0;
  for(auto &i : atomChargesVec)
  {
    numElectrons += (size_type)(std::abs(i));
  }

  if (numWantedEigenvalues <= numElectrons / 2.0 ||
             numWantedEigenvalues == 0)
  {
    rootCout << " Warning: User has requested the number of Kohn-Sham wavefunctions to be less than or"
          "equal to half the number of electrons in the system. Setting the Kohn-Sham wavefunctions"
          "to half the number of electrons with a 20 percent buffer to avoid convergence issues in"
          "SCF iterations" << std::endl;
    numWantedEigenvalues = (numElectrons / 2.0) + std::max((0.2) * (numElectrons / 2.0), 20.0);

    // start with 17-20% buffer in GPUs to leave room for additional modifications
    // due to block size restrictions

    rootCout << " Setting the number of Kohn-Sham wave functions to be " << numWantedEigenvalues << std::endl;
  }

  // orbital always. vtotal and density are read here too unless they were
  // given their own file; vtotal stays regardless outside the delta-rho
  // route, where the total potential enrichment basis reads it from here.
  const bool rhoVtotFromOwnFile = !basisDataFileRhoVtot.empty();
  const bool vtotalFromThisFile = !rhoVtotFromOwnFile || !isDeltaRhoPoissonSolve;
  std::vector<std::string> fieldNames(1 + (vtotalFromThisFile ? 1 : 0) +
                                      (rhoVtotFromOwnFile ? 0 : 1));
  {
    dftefe::size_type iField = 0;
    fieldNames[iField++]     = "orbital";
    if (vtotalFromThisFile)
      fieldNames[iField++] = "vtotal";
    if (!rhoVtotFromOwnFile)
      fieldNames[iField++] = "density";
  }
  std::vector<std::string> metadataNames{ "symbol", "Z", "charge", "NR" };
  std::map<std::string, std::string> enrichmentAdditionalParams({{"rcsmear", std::to_string(rc)}, {"PSP/AE", "AE"}});
  // vtotal specific, so it belongs to whichever container carries vtotal; the
  // long tail is only needed by the delta-rho solve
  std::map<std::string, std::string> enrichmentAdditionalParamsVtot =
    enrichmentAdditionalParams;
  if (!isDeltaRhoPoissonSolve)
    enrichmentAdditionalParamsVtot["vTotCutOffSmoothness"] = "3.01,0.5";
  std::shared_ptr<atoms::AtomSphericalDataContainer>  atomSphericalDataContainer =
      std::make_shared<atoms::AtomSphericalDataContainer>(
                                                      atoms::AtomSphericalDataType::ENRICHMENT,
                                                      atomSymbolToBasisFileName,
                                                      fieldNames,
                                                      metadataNames,
                                                      vtotalFromThisFile ?
                                                        enrichmentAdditionalParamsVtot :
                                                        enrichmentAdditionalParams);

  // rho_at and phi_at may come from their own atomic reference. A high
  // temperature reference leaves fractional occupancy in diffuse states, so
  // its rho_at carries charge far past any usable enrichment cutoff.
  std::vector<std::string> fieldNamesRhoVtot{"vtotal", "density"};

  // every rho_at / phi_at consumer goes through this one handle; without the
  // optional file it is the container built from basisDataFile
  std::shared_ptr<atoms::AtomSphericalDataContainer>
    atomSphericalDataContainerRhoVtot = atomSphericalDataContainer;
  if (!atomSymbolToBasisFileNameRhoVtot.empty())
    atomSphericalDataContainerRhoVtot =
      std::make_shared<atoms::AtomSphericalDataContainer>(
        atoms::AtomSphericalDataType::ENRICHMENT,
        atomSymbolToBasisFileNameRhoVtot,
        fieldNamesRhoVtot,
        metadataNames,
        enrichmentAdditionalParamsVtot);

  for (auto i:atomSymbolToBasisFileName )
  {
    rootCout << "For atom symbol: "<<i.first<<std::endl;
    rootCout << "Reading basis file: "<<i.second<<std::endl;
    rootCout << "Cutoff and smoothness for "<<i.first<<std::endl;
    for(auto j:fieldNames)
    {
      rootCout << " for "<<j<<" : "; 
      for(auto &enrichmentObjId : 
        atomSphericalDataContainer->getSphericalData(i.first, j))
      {
        rootCout << enrichmentObjId->getCutoff() << ","<<enrichmentObjId->getSmoothness()<<"\t";
      }
      rootCout << std::endl;
    }
    if(std::abs(std::stod(atomSphericalDataContainer->getMetadata(i.first, "Z"))) - std::abs(atomSymbolToChargeMap[i.first]) > 1e-12)
    {
      utils::throwException(false, "The input basis file Z does not match with that given in input.");       
  }
    if (!atomSymbolToBasisFileNameRhoVtot.empty())
    {
      rootCout << "Reading rho_at/phi_at basis file: "
               << atomSymbolToBasisFileNameRhoVtot[i.first] << std::endl;
      for (auto j : fieldNamesRhoVtot)
      {
        rootCout << " for " << j << " : ";
        for (auto &enrichmentObjId :
             atomSphericalDataContainerRhoVtot->getSphericalData(i.first, j))
          rootCout << enrichmentObjId->getCutoff() << ","
                   << enrichmentObjId->getSmoothness() << "\t";
        rootCout << std::endl;
      }
      utils::throwException<utils::InvalidArgument>(
        std::abs(
          std::stod(
            atomSphericalDataContainerRhoVtot->getMetadata(i.first, "Z")) -
          std::stod(atomSphericalDataContainer->getMetadata(i.first, "Z"))) <
          1e-12,
        "The rho_at/phi_at basis file for '" + i.first + "' has a different Z "
        "than the orbital basis file; they must describe the same atom.");
    }
  }
  p.registerEnd("Reading Other Input data");

  // Two envelopes are enough, and the generator provides exactly two. The
  // truncated one stays at the default and serves the orbital enrichments;
  // the full one covers the longest reaching field of any kind.
  double imageCutOff      = basis::PeriodicImageAtomGeneratorDefaults::CUTOFF;
  double imageCutOffTrunc = basis::PeriodicImageAtomGeneratorDefaults::CUTOFF_TRUNC;
  double needOrbital = 0.0, needRhoVtot = 0.0, cellCentreToCorner = 0.0;
  if (anyDirectionPeriodic)
    {
      // an image is admitted on its centre distance, so a field of reach R
      // must be admitted out to R plus the cell centre to corner distance
      for (int sx = -1; sx <= 1; sx += 2)
        for (int sy = -1; sy <= 1; sy += 2)
          for (int sz = -1; sz <= 1; sz += 2)
            {
              double nrm = 0.0;
              for (dftefe::size_type c = 0; c < dim; c++)
                {
                  double v = 0.5 * (sx * domainVectors[0][c] +
                                    sy * domainVectors[1][c] +
                                    sz * domainVectors[2][c]);
                  nrm += v * v;
                }
              cellCentreToCorner = std::max(cellCentreToCorner, std::sqrt(nrm));
            }

      // an enrichment is exactly zero past cutoff + cutoff/smoothness
      auto reachOf = [](const auto &sd) {
        return sd->getCutoff() + sd->getCutoff() / sd->getSmoothness();
      };

      for (auto &i : atomSymbolToBasisFileName)
        {
          for (auto &enrichmentObjId :
               atomSphericalDataContainer->getSphericalData(i.first, "orbital"))
            needOrbital = std::max(needOrbital, reachOf(enrichmentObjId));

          for (auto &j : fieldNamesRhoVtot)
            for (auto &enrichmentObjId :
                 atomSphericalDataContainerRhoVtot->getSphericalData(i.first,
                                                                     j))
              needRhoVtot = std::max(needRhoVtot, reachOf(enrichmentObjId));
        }

      imageCutOff = std::max(imageCutOff,
                             std::max(needOrbital, needRhoVtot) +
                               cellCentreToCorner);

      rootCout << "Image cutoff: full " << imageCutOff << " (max field reach "
               << std::max(needOrbital, needRhoVtot) << " + corner "
               << cellCentreToCorner << ", floored at "
               << basis::PeriodicImageAtomGeneratorDefaults::CUTOFF
               << "), truncated " << imageCutOffTrunc
               << "; reaches: orbital " << needOrbital << ", rho_at/phi_at "
               << needRhoVtot << "\n";
    }

  std::shared_ptr<const basis::PeriodicImageAtomGenerator> imageAtomGenerator =
    nullptr;
  if (anyDirectionPeriodic)
    imageAtomGenerator =
      std::make_shared<const basis::PeriodicImageAtomGenerator>(
        atomCoordinatesVec,
        atomChargesVec,
        domainVectors,
        isPeriodicFlags,
        imageCutOff,
        imageCutOffTrunc);

  // AtomSuperpositionFunction is built here, not in the library, so its atom
  // list is this caller's responsibility. rho_at, phi_at and V_ext must use
  // this full list: ElectrostaticLocalFE builds its own from getExtendedAtoms
  // and V_ext - V_smear only cancels when both run over the same images.
  std::vector<utils::Point> extendedAtomCoordinatesVec = atomCoordinatesVec;
  std::vector<std::string>  extendedAtomSymbolVec      = atomSymbolVec;
  std::vector<double>       extendedAtomChargesVec     = atomChargesVec;
  // the truncated envelope, which EnrichmentIdsPartition also works from
  std::vector<utils::Point> extendedAtomCoordinatesTruncVec =
    atomCoordinatesVec;
  std::vector<std::string> extendedAtomSymbolTruncVec  = atomSymbolVec;
  std::vector<double>      extendedAtomChargesTruncVec = atomChargesVec;
  // The criteria only have to reach as far as an orbital, so they take the
  // smaller envelope that still covers that reach. Nothing downstream depends
  // on which one they got, which is why the choice is free here and not for
  // rho_at / phi_at.
  const bool orbitalFitsTrunc =
    (needOrbital + cellCentreToCorner) <= imageCutOffTrunc;
  const std::vector<utils::Point> &extendedAtomCoordinatesCriteriaVec =
    orbitalFitsTrunc ? extendedAtomCoordinatesTruncVec :
                       extendedAtomCoordinatesVec;
  const std::vector<std::string> &extendedAtomSymbolCriteriaVec =
    orbitalFitsTrunc ? extendedAtomSymbolTruncVec : extendedAtomSymbolVec;
  const std::vector<double> &extendedAtomChargesCriteriaVec =
    orbitalFitsTrunc ? extendedAtomChargesTruncVec : extendedAtomChargesVec;

  if (imageAtomGenerator != nullptr)
    {
      imageAtomGenerator->getExtendedAtoms(atomCoordinatesVec,
                                           extendedAtomCoordinatesVec,
                                           extendedAtomChargesVec);

      const std::vector<dftefe::size_type> &imageIds =
        imageAtomGenerator->getImageIds();
      extendedAtomSymbolVec.resize(atomSymbolVec.size() + imageIds.size());
      for (dftefe::size_type j = 0; j < imageIds.size(); j++)
        extendedAtomSymbolVec[atomSymbolVec.size() + j] =
          atomSymbolVec[imageIds[j]];

      imageAtomGenerator->getExtendedAtomsTrunc(
        atomCoordinatesVec,
        extendedAtomCoordinatesTruncVec,
        extendedAtomChargesTruncVec);

      const std::vector<dftefe::size_type> &imageIdsTrunc =
        imageAtomGenerator->getImageIdsTrunc();
      extendedAtomSymbolTruncVec.resize(atomSymbolVec.size() +
                                        imageIdsTrunc.size());
      for (dftefe::size_type j = 0; j < imageIdsTrunc.size(); j++)
        extendedAtomSymbolTruncVec[atomSymbolVec.size() + j] =
          atomSymbolVec[imageIdsTrunc[j]];

      rootCout << "Extended atoms: full " << extendedAtomCoordinatesVec.size()
               << ", truncated " << extendedAtomCoordinatesTruncVec.size()
               << "\n";

      // Which consumer reads which list, so the assignment is visible in the
      // output instead of having to be traced through the source.
      rootCout << "Image list per consumer:\n";
      rootCout << "  criteria list (" << (orbitalFitsTrunc ? "truncated, " :
                                                             "full, ")
               << extendedAtomCoordinatesCriteriaVec.size()
               << " atoms): every adaptive quadrature criterion -- orbitalSq, "
               << "gradOrbitalSq, vExtTimesOrbitalSq, bTimesVSmear";
      if (!isDeltaRhoPoissonSolve)
        rootCout << ", vTotalSq, gradVTotalSq";
      if (isDeltaRhoPoissonSolve && (tciaFolder == "" || tciaOutFilePrefix == ""))
        rootCout << ", bPlusRhoTimesVTotal";
      if (isNumericalNuclearSolve)
        rootCout << ", vSmearSq, gradVSmearSq";
      rootCout << "\n";

      rootCout << "  extendedAtom*TruncVec (cutOffTrunc = " << imageCutOffTrunc
               << ", " << extendedAtomCoordinatesTruncVec.size()
               << " atoms): EnrichmentIdsPartition, via "
               << "getImagePositionsTrunc\n";

      rootCout << "  extendedAtom*Vec (cutOff = " << imageCutOff << ", "
               << extendedAtomCoordinatesVec.size()
               << " atoms): elecChargeDens(rho_at)";
      if (isDeltaRhoPoissonSolve)
        rootCout << ", smfuncAtTotPot(phi_at)";
      rootCout << ", externalPotentialFunction(point charges)";
      rootCout << "; the electrostatics operator reads getExtendedAtoms, so b, "
               << "V_smear, V_ext, rho_at and the TCIA lattice sum all share "
               << "this envelope\n";
    }

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

  utils::printCurrentMemoryUsage<memorySpace>(comm, "Create Mesh");

    p.registerStart("Quadrature Rule Creation");

  // Compute Adaptive QuadratureRuleContainer for electrostaics

    // Set up the vector of scalarSpatialRealFunctions for adaptive quadrature
    std::vector<std::shared_ptr<const utils::ScalarSpatialFunctionReal>> functionsVec(0);
    std::vector<double> absoluteTolerances(0), relativeTolerances(0), integralThresholds(0);

    functionsVec.push_back(std::make_shared<atoms::AtomSevereFunction<memorySpace>>(
        atomSphericalDataContainer,
        extendedAtomSymbolCriteriaVec,
        extendedAtomCoordinatesCriteriaVec,
        extendedAtomChargesCriteriaVec,
        rc,
        atoms::AtomSevereFuncType::Atomic::orbitalSq,
        1.0,
        linAlgOpContext.get()));
    functionsVec.push_back(std::make_shared<atoms::AtomSevereFunction<memorySpace>>(
        atomSphericalDataContainer,
        extendedAtomSymbolCriteriaVec,
        extendedAtomCoordinatesCriteriaVec,
        extendedAtomChargesCriteriaVec,
        rc,
        atoms::AtomSevereFuncType::Atomic::gradOrbitalSq,
        1.0,
        linAlgOpContext.get()));
    functionsVec.push_back(std::make_shared<atoms::AtomSevereFunction<memorySpace>>(
        atomSphericalDataContainer,
        extendedAtomSymbolCriteriaVec,
        extendedAtomCoordinatesCriteriaVec,
        extendedAtomChargesCriteriaVec,
        rc,
        atoms::AtomSevereFuncType::Atomic::vExtTimesOrbitalSq));
    // drives the grid to resolve b * V_smear, the nuclear self energy
    // integrand; nothing else in this list targets it
    functionsVec.push_back(std::make_shared<atoms::AtomSevereFunction<memorySpace>>(
        extendedAtomCoordinatesTruncVec,
        extendedAtomChargesTruncVec,
        rc,
        atoms::AtomSevereFuncType::Atomic::bTimesVSmear));

    if(!isDeltaRhoPoissonSolve)
    {
      functionsVec.push_back(std::make_shared<atoms::AtomSevereFunction<memorySpace>>(
          atomSphericalDataContainerRhoVtot,
          extendedAtomSymbolCriteriaVec,
          extendedAtomCoordinatesCriteriaVec,
          extendedAtomChargesCriteriaVec,
          0.0,
          atoms::AtomSevereFuncType::Atomic::vTotalSq,
          1.0,
          linAlgOpContext.get()));
      functionsVec.push_back(std::make_shared<atoms::AtomSevereFunction<memorySpace>>(
          atomSphericalDataContainerRhoVtot,
          extendedAtomSymbolCriteriaVec,
          extendedAtomCoordinatesCriteriaVec,
          extendedAtomChargesCriteriaVec,
          0.0,
          atoms::AtomSevereFuncType::Atomic::gradVTotalSq,
          1.0,
          linAlgOpContext.get()));
    }
    if(isDeltaRhoPoissonSolve && (tciaFolder == "" || tciaOutFilePrefix == ""))
    {
      functionsVec.push_back(std::make_shared<atoms::AtomSevereFunction<memorySpace>>(
        atomSphericalDataContainerRhoVtot,
        extendedAtomSymbolCriteriaVec,
        extendedAtomCoordinatesCriteriaVec,
        extendedAtomChargesCriteriaVec,
        rc,
        atoms::AtomSevereFuncType::Atomic::bPlusRhoTimesVTotal,
        1.0,
        linAlgOpContext.get()));
      // functionsVec.push_back(std::make_shared<atoms::AtomSevereFunction<memorySpace>>(
      // atomSphericalDataContainer, atomSymbolVec, atomCoordinatesVec,
      // atomChargesVec, rc, atoms::AtomSevereFuncType::Atomic::bTimesVNuclear));
    }
    if(isNumericalNuclearSolve)
    {
      functionsVec.push_back(std::make_shared<atoms::AtomSevereFunction<memorySpace>>(
          extendedAtomCoordinatesCriteriaVec,
          extendedAtomChargesCriteriaVec,
          rc,
          atoms::AtomSevereFuncType::Atomic::vSmearSq,
          1.0,
          linAlgOpContext.get()));
      functionsVec.push_back(std::make_shared<atoms::AtomSevereFunction<memorySpace>>(
          extendedAtomCoordinatesCriteriaVec,
          extendedAtomChargesCriteriaVec,
          rc,
          atoms::AtomSevereFuncType::Atomic::gradVSmearSq,
          1.0,
          linAlgOpContext.get()));
    }
    for ( dftefe::size_type i=0 ;i < functionsVec.size() ; i++ )
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
      (quadrature::FuncEvalTag<memorySpace>{},
      quadAttrAdaptive,
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

    dftefe::size_type nQuad = quadRuleContainerAdaptiveElec->nQuadraturePoints();

    auto mpierr = utils::mpi::MPIAllreduce<Host>(
      utils::mpi::MPIInPlace,
      &nQuad,
      1,
      utils::mpi::Types<size_type>::getMPIDatatype(),
      utils::mpi::MPISum,
      comm);
  rootCout << "Number of quadrature points in adaptive quadrature: "<< nQuad<<"\n";

    {
      double localQuad = (double)quadRuleContainerAdaptiveElec->nQuadraturePoints();
      auto   procStats = utils::mpi::MPIAllreduceMinMaxAvg<double, Host>(localQuad, comm);
      rootCout << "Elec adaptive quad proc load: min=" << (size_type)procStats.min
               << " max=" << (size_type)procStats.max
               << " ratio(min/max)=" << procStats.min / procStats.max << "\n";

      dftefe::size_type nLocalCells = quadRuleContainerAdaptiveElec->nCells();
      double minCell = std::numeric_limits<double>::max(), maxCell = 0.0;
      for (dftefe::size_type iCell = 0; iCell < nLocalCells; ++iCell)
        {
          double cq = (double)quadRuleContainerAdaptiveElec->nCellQuadraturePoints(iCell);
          minCell = std::min(minCell, cq);
          maxCell = std::max(maxCell, cq);
        }
      auto   cellMinStats = utils::mpi::MPIAllreduceMinMaxAvg<double, Host>(minCell, comm);
      auto   cellMaxStats = utils::mpi::MPIAllreduceMinMaxAvg<double, Host>(maxCell, comm);
      double totalCells   = (double)nLocalCells;
      utils::mpi::MPIAllreduce<Host>(utils::mpi::MPIInPlace, &totalCells, 1,
        utils::mpi::Types<double>::getMPIDatatype(), utils::mpi::MPISum, comm);
      rootCout << "Elec adaptive quad per cell: min=" << (size_type)cellMinStats.min
               << " max=" << (size_type)cellMaxStats.max
               << " avg=" << (double)nQuad / totalCells << "\n";
    }

    reportMinQuadPointAtomDistance(*quadRuleContainerAdaptiveElec,
                                   atomCoordinatesVec,
                                   "Elec adaptive quad",
                                   comm,
                                   rootCout);
    }

    //Set up quadAttr for Rhs and OverlapMatrix
    
    quadrature::QuadratureRuleAttributes quadAttrGllEigen(quadrature::QuadratureFamily::GLL,true,feOrderEigen + 1);

    // Set up base quadrature rule for adaptive quadrature 

    std::shared_ptr<quadrature::QuadratureRule> baseQuadRuleEigen = std::make_shared<quadrature::QuadratureRuleGauss>(dim, feOrderEigen > feOrderElec ? feOrderEigen : feOrderElec + 1);
      // feOrderEigen > feOrderElec ? std::make_shared<quadrature::QuadratureRuleGauss>(dim, feOrderEigen + 1) : 
      //   std::make_shared<quadrature::QuadratureRuleGauss>(dim, feOrderElec + 1);

    std::shared_ptr<quadrature::QuadratureRuleContainer> quadRuleContainerAdaptiveOrbital = (quadRuleContainerAdaptiveElec != nullptr) ? quadRuleContainerAdaptiveElec :
      std::make_shared<quadrature::QuadratureRuleContainer>
      (quadrature::FuncEvalTag<memorySpace>{},
      quadAttrAdaptive,
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

    dftefe::size_type nQuad = quadRuleContainerAdaptiveOrbital->nQuadraturePoints();
    int mpierr = utils::mpi::MPIAllreduce<Host>(
      utils::mpi::MPIInPlace,
      &nQuad,
      1,
      utils::mpi::Types<size_type>::getMPIDatatype(),
      utils::mpi::MPISum,
      comm);

  rootCout << "Number of quadrature points in wave function adaptive quadrature: "<<nQuad<<"\n";

    {
      double localQuad = (double)quadRuleContainerAdaptiveOrbital->nQuadraturePoints();
      auto   procStats = utils::mpi::MPIAllreduceMinMaxAvg<double, Host>(localQuad, comm);
      rootCout << "Orbital adaptive quad proc load: min=" << (size_type)procStats.min
               << " max=" << (size_type)procStats.max
               << " ratio(min/max)=" << procStats.min / procStats.max << "\n";

      dftefe::size_type nLocalCells = quadRuleContainerAdaptiveOrbital->nCells();
      double minCell = std::numeric_limits<double>::max(), maxCell = 0.0;
      for (dftefe::size_type iCell = 0; iCell < nLocalCells; ++iCell)
        {
          double cq = (double)quadRuleContainerAdaptiveOrbital->nCellQuadraturePoints(iCell);
          minCell = std::min(minCell, cq);
          maxCell = std::max(maxCell, cq);
        }
      auto   cellMinStats = utils::mpi::MPIAllreduceMinMaxAvg<double, Host>(minCell, comm);
      auto   cellMaxStats = utils::mpi::MPIAllreduceMinMaxAvg<double, Host>(maxCell, comm);
      double totalCells   = (double)nLocalCells;
      utils::mpi::MPIAllreduce<Host>(utils::mpi::MPIInPlace, &totalCells, 1,
        utils::mpi::Types<double>::getMPIDatatype(), utils::mpi::MPISum, comm);
      rootCout << "Orbital adaptive quad per cell: min=" << (size_type)cellMinStats.min
               << " max=" << (size_type)cellMaxStats.max
               << " avg=" << (double)nQuad / totalCells << "\n";
    }

    reportMinQuadPointAtomDistance(*quadRuleContainerAdaptiveOrbital,
                                   atomCoordinatesVec,
                                   "Orbital adaptive quad",
                                   comm,
                                   rootCout);

  p.registerEnd("Quadrature Rule Creation");
    utils::printCurrentMemoryUsage<memorySpace>(comm, "Quadrature Rule Creation");
  p.registerStart("Ortho EFE basis manager creation");

  // Make orthogonalized EFE basis for all the fields

  // 1. Make CFEBasisDataStorageDealii object for Rhs (ADAPTIVE with GAUSS and fns are N_i^2 - make quadrulecontainer), overlapmatrix (GLL)
  // 2. Make EnrichmentClassicalInterface object for Orthogonalized enrichment
  // 3. Input to the EFEBasisDofHandler(eci, feOrder) 
  // 4. Make EFEBasisDataStorage with input as quadratureContainer.

    // Set the CFE basis manager and handler for bassiInterfaceCoeffcient distributed vector
  p.registerStart("CFE DofHandler creation");
  std::shared_ptr<basis::FEBasisDofHandler<ValueTypeElectrostaticsCoeff, Host,dim>> cfeBasisDofHandlerElec =
   std::make_shared<basis::CFEBasisDofHandlerDealii<ValueTypeElectrostaticsCoeff, Host,dim>>(triangulationBase, feOrderElec, comm);

  std::shared_ptr<basis::FEBasisDofHandler<ValueTypeWaveFunctionCoeff, memorySpace,dim>> cfeBasisDofHandlerEigen =
   std::make_shared<basis::CFEBasisDofHandlerDealii<ValueTypeWaveFunctionCoeff, memorySpace,dim>>(triangulationBase, feOrderEigen, comm);
  p.registerEnd("CFE DofHandler creation");

  rootCout << "Total Number of classical dofs electrostatics: " << cfeBasisDofHandlerElec->nGlobalNodes() << "\n";
  rootCout << "Total Number of classical dofs eigensolve: " << cfeBasisDofHandlerEigen->nGlobalNodes() << "\n";

  rootCout << "The Number of classical dofs electrostatics excluding Vacuum: " << 
    getNumClassicalDofsInSystemExcludingVacuum<ValueTypeElectrostaticsCoeff, Host, dim>(atomCoordinatesVec,
      *cfeBasisDofHandlerElec,
      comm) << "\n";

  rootCout << "The Number of classical dofs eigenSolve excluding Vacuum: " << 
    getNumClassicalDofsInSystemExcludingVacuum<ValueTypeWaveFunctionCoeff, memorySpace, dim>(atomCoordinatesVec,
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
  p.registerStart("CFE GLL basis data eval");
    std::shared_ptr<basis::FEBasisDataStorage<ValueTypeElectrostaticsBasis, Host>> cfeBasisDataStorageGLLElec =
      std::make_shared<basis::CFEBasisDataStorageDealii<ValueTypeElectrostaticsCoeff, ValueTypeElectrostaticsBasis,Host, dim>>
      (cfeBasisDofHandlerElec, quadAttrGllElec, basisAttrMap, *linAlgOpContextHost);

    std::shared_ptr<basis::FEBasisDataStorage<ValueTypeWaveFunctionBasis, memorySpace>> cfeBasisDataStorageGLLEigen =
      std::make_shared<basis::CFEBasisDataStorageDealii<ValueTypeWaveFunctionCoeff, ValueTypeWaveFunctionBasis,memorySpace, dim>>
      (cfeBasisDofHandlerEigen, quadAttrGllEigen, basisAttrMap, *linAlgOpContext);

  // evaluate basis data
  cfeBasisDataStorageGLLElec->evaluateBasisData(quadAttrGllElec, basisAttrMap);
  cfeBasisDataStorageGLLEigen->evaluateBasisData(quadAttrGllEigen, basisAttrMap);
  p.registerEnd("CFE GLL basis data eval");

    // Set the CFE basis manager and handler for bassiInterfaceCoeffcient distributed vector

  // the classical halves of the per k-point EFE storages below are taken from
  // this one, so it carries what they store, gradients included
  basisAttrMap[basis::BasisStorageAttributes::StoreValues] = true;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradient] = true;
  basisAttrMap[basis::BasisStorageAttributes::StoreHessian] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreOverlap] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradNiGradNj] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreJxW] = true;

    // Set up the CFE Basis Data Storage for Rhs
  p.registerStart("CFE Adaptive orbital basis data eval");
    std::shared_ptr<basis::FEBasisDataStorage<ValueTypeWaveFunctionBasis, memorySpace>> cfeBasisDataStorageAdaptiveOrbital =
      std::make_shared<basis::CFEBasisDataStorageDealii<ValueTypeWaveFunctionCoeff, ValueTypeWaveFunctionBasis,memorySpace, dim>>
      (cfeBasisDofHandlerEigen, quadAttrAdaptive, basisAttrMap, *linAlgOpContext);
  // evaluate basis data
  cfeBasisDataStorageAdaptiveOrbital->evaluateBasisData(quadAttrAdaptive, quadRuleContainerAdaptiveOrbital, basisAttrMap);
  p.registerEnd("CFE Adaptive orbital basis data eval");

      std::shared_ptr<basis::EnrichmentClassicalInterfaceSpherical
                          <ValueTypeElectrostaticsCoeff,
                                                                   ValueTypeElectrostaticsBasis,
                                                                   Host,
                                                                   dim>>
        enrichClassIntfceTotalPot = nullptr;

    // Create the enrichmentClassicalInterface object for wavefn, one per
    // k-point with that k's Bloch phase, all on one common dof layout
  p.registerStart("ECI orbital construction");
  std::map<basis::kpoint::KPointId,
           std::shared_ptr<const basis::EnrichmentClassicalInterfaceSpherical
                          <ValueTypeWaveFunctionCoeff,
                            ValueTypeWaveFunctionBasis,
                            memorySpace,
                            dim>>>
    enrichClassIntfceOrbital = basis::EnrichmentClassicalInterfaceSpherical
                          <ValueTypeWaveFunctionCoeff,
                          ValueTypeWaveFunctionBasis,
                          memorySpace,
                          dim>::createECIObjectsForKPoints
                          (cfeBasisDataStorageGLLEigen,
                          cfeBasisDataStorageAdaptiveOrbital,
                          atomSphericalDataContainer,
                          atomPartitionTolerance,
                          atomSymbolVec,
                          atomCoordinatesVec,
                          "orbital",
                          linAlgOpContext,
                          comm,
                          basis::ECIDefaults::ENRICHMENT_BATCH_SIZE,
                          basis::BasisDataStorageDefaults<memorySpace>::CELL_BATCH_SIZE,
                          imageAtomGenerator,
                          kPoints);
  p.registerEnd("ECI orbital construction");

  // initialize the basis Manager

  std::shared_ptr<basis::FEBasisDofHandler<ValueTypeElectrostaticsCoeff, Host,dim>> basisDofHandlerTotalPot = nullptr;
  if (!isDeltaRhoPoissonSolve)
  {

    basisAttrMap[basis::BasisStorageAttributes::StoreValues] = true;
    basisAttrMap[basis::BasisStorageAttributes::StoreGradient] = false;
    basisAttrMap[basis::BasisStorageAttributes::StoreHessian] = false;
    basisAttrMap[basis::BasisStorageAttributes::StoreOverlap] = false;
    basisAttrMap[basis::BasisStorageAttributes::StoreGradNiGradNj] = false;
    basisAttrMap[basis::BasisStorageAttributes::StoreJxW] = true;

    // Set up the CFE Basis Data Storage for Rhs
  p.registerStart("CFE Adaptive vtotal basis data eval");
    std::shared_ptr<basis::FEBasisDataStorage<ValueTypeElectrostaticsBasis, Host>> cfeBasisDataStorageAdaptiveElec =
      std::make_shared<basis::CFEBasisDataStorageDealii<ValueTypeElectrostaticsCoeff, ValueTypeElectrostaticsBasis,Host, dim>>
      (cfeBasisDofHandlerElec, quadAttrAdaptive, basisAttrMap, *linAlgOpContextHost);
    // evaluate basis data
    cfeBasisDataStorageAdaptiveElec->evaluateBasisData(quadAttrAdaptive, quadRuleContainerAdaptiveElec, basisAttrMap);
  p.registerEnd("CFE Adaptive vtotal basis data eval");

  p.registerStart("ECI vtotal construction");
    enrichClassIntfceTotalPot = std::make_shared<basis::EnrichmentClassicalInterfaceSpherical
                        <ValueTypeElectrostaticsCoeff,
                                                                                              ValueTypeElectrostaticsBasis,
                                                                                              Host,
                                                                                              dim>>
                        (cfeBasisDataStorageGLLElec,
                        cfeBasisDataStorageAdaptiveElec,
                        atomSphericalDataContainer,
                        atomPartitionTolerance,
                        atomSymbolVec,
                        atomCoordinatesVec,
                        "vtotal",
                        linAlgOpContextHost,
                        comm,
                        basis::ECIDefaults::ENRICHMENT_BATCH_SIZE,
                        basis::BasisDataStorageDefaults<Host>::CELL_BATCH_SIZE,
                        imageAtomGenerator);
  p.registerEnd("ECI vtotal construction");

  p.registerStart("EFE DofHandler totalPot creation");
     basisDofHandlerTotalPot =
    std::make_shared<basis::EFEBasisDofHandlerDealii<ValueTypeElectrostaticsCoeff, ValueTypeElectrostaticsBasis,Host,dim>>(
      enrichClassIntfceTotalPot, comm);
  p.registerEnd("EFE DofHandler totalPot creation");
  }
  else
    basisDofHandlerTotalPot = cfeBasisDofHandlerElec;

  p.registerStart("EFE DofHandler wavefn creation");
  std::map<basis::kpoint::KPointId,
           std::shared_ptr<basis::FEBasisDofHandler<ValueTypeWaveFunctionCoeff, memorySpace,dim>>>
    basisDofHandlerWaveFn;
  for (const auto &kPoint : kPoints)
    basisDofHandlerWaveFn[kPoint.first] =
      std::make_shared<basis::EFEBasisDofHandlerDealii<ValueTypeWaveFunctionCoeff, ValueTypeWaveFunctionBasis,memorySpace,dim>>(
        enrichClassIntfceOrbital.at(kPoint.first), comm);
  p.registerEnd("EFE DofHandler wavefn creation");

  p.registerEnd("Ortho EFE basis manager creation");
  utils::printCurrentMemoryUsage<memorySpace>(comm, "Ortho EFE basis manager creation");

  rootCout << "Total Number of dofs electrostatics: " << basisDofHandlerTotalPot->nGlobalNodes() << "\n";
  rootCout << "Total Number of dofs eigensolve: " << basisDofHandlerWaveFn.begin()->second->nGlobalNodes() << "\n";

  // Set up the quadrature rule

  p.registerStart("Electrostatics basis grad datastorage eval");

  // the mean value constraint for full PBC integrates the basis functions,
  // so the stiffness storage needs values as well as gradients
  basisAttrMap[basis::BasisStorageAttributes::StoreValues] = true;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradient] = true;
  basisAttrMap[basis::BasisStorageAttributes::StoreHessian] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreOverlap] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradNiGradNj] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreJxW] = true;

  quadrature::QuadratureRuleAttributes quadAttrGaussElectro(quadrature::QuadratureFamily::GAUSS,true,feOrderElec + 1);

  std::shared_ptr<basis::FEBasisDataStorage<ValueTypeElectrostaticsBasis, Host>> feBDTotalChargeStiffnessMatrix = nullptr;
  if (!isDeltaRhoPoissonSolve)
    feBDTotalChargeStiffnessMatrix =
      std::make_shared<basis::EFEBasisDataStorageDealii<ValueTypeElectrostaticsCoeff, ValueTypeElectrostaticsBasis, Host,dim>>
        (basisDofHandlerTotalPot, quadAttrAdaptive, basisAttrMap, *linAlgOpContextHost);
  else
    feBDTotalChargeStiffnessMatrix =
    std::make_shared<basis::CFEBDSOnTheFlyComputeDealii<ValueTypeElectrostaticsCoeff, ValueTypeElectrostaticsBasis, Host,dim>>
    (basisDofHandlerTotalPot, quadAttrGaussElectro, basisAttrMap, ksdft::KSDFTDefaults<Host>::CELL_BATCH_SIZE_GRAD_EVAL, *linAlgOpContextHost);

  if (!isDeltaRhoPoissonSolve)
    feBDTotalChargeStiffnessMatrix->evaluateBasisData(quadAttrAdaptive, quadRuleContainerAdaptiveElec, basisAttrMap);
  else 
    feBDTotalChargeStiffnessMatrix->evaluateBasisData(quadAttrGaussElectro, basisAttrMap);

  p.registerEnd("Electrostatics basis grad datastorage eval");
  utils::printCurrentMemoryUsage<memorySpace>(comm, "Electrostatics basis grad datastorage eval");
  p.registerStart("Electrostatics basis bsmear datastorage eval");

    basisAttrMap[basis::BasisStorageAttributes::StoreValues] = true;
    basisAttrMap[basis::BasisStorageAttributes::StoreGradient] = false;
    basisAttrMap[basis::BasisStorageAttributes::StoreHessian] = false;
    basisAttrMap[basis::BasisStorageAttributes::StoreOverlap] = false;
    basisAttrMap[basis::BasisStorageAttributes::StoreGradNiGradNj] = false;
    basisAttrMap[basis::BasisStorageAttributes::StoreJxW] = true;

  std::shared_ptr<basis::FEBasisDataStorage<ValueTypeElectrostaticsBasis, Host>> feBDNucChargeRhs = nullptr;
  if (!isDeltaRhoPoissonSolve)
    feBDNucChargeRhs =   std::make_shared<basis::EFEBasisDataStorageDealii<ValueTypeElectrostaticsCoeff, ValueTypeElectrostaticsBasis, Host,dim>>
      (basisDofHandlerTotalPot, quadAttrAdaptive, basisAttrMap, *linAlgOpContextHost);
  else
    feBDNucChargeRhs =   
      std::make_shared<basis::CFEBasisDataStorageDealii<ValueTypeElectrostaticsCoeff, ValueTypeElectrostaticsBasis, Host,dim>>
      (basisDofHandlerTotalPot, quadAttrAdaptive, basisAttrMap, *linAlgOpContextHost);

  //if (!isDeltaRhoPoissonSolve)
    feBDNucChargeRhs->evaluateBasisData(quadAttrAdaptive, quadRuleContainerAdaptiveOrbital, basisAttrMap);
  // else
  // {
  //   size_type num1DGaussSubdividedSizeBSmear = 3;
  //   size_type gaussSubdividedCopiesBSmear = 10;
  //   std::shared_ptr<quadrature::QuadratureRule> gaussSubdivQuadRuleElec =
  //     std::make_shared<quadrature::QuadratureRuleGaussIterated>(dim, num1DGaussSubdividedSizeBSmear, gaussSubdividedCopiesBSmear);

  //   std::shared_ptr<quadrature::QuadratureRuleContainer> quadRuleContainerGaussSubdividedBSmear =
  //     std::make_shared<quadrature::QuadratureRuleContainer>
  //     (quadAttrGaussSubdivided, 
  //     gaussSubdivQuadRuleElec, 
  //     triangulationBase, 
  //     *cellMapping); 

  //   feBDNucChargeRhs->evaluateBasisData(quadAttrGaussSubdivided, quadRuleContainerGaussSubdividedBSmear, basisAttrMap);
  // }
  p.registerEnd("Electrostatics basis bsmear datastorage eval");
  utils::printCurrentMemoryUsage<memorySpace>(comm, "Electrostatics basis bsmear datastorage eval");
  p.registerStart("Electrostatics basis rho datastorage eval");

  std::shared_ptr<basis::FEBasisDataStorage<ValueTypeElectrostaticsBasis, Host>> feBDElecChargeRhs = nullptr;
  if (!isDeltaRhoPoissonSolve)
  {
    feBDElecChargeRhs = std::make_shared<basis::EFEBasisDataStorageDealii<ValueTypeElectrostaticsCoeff, ValueTypeElectrostaticsBasis, Host,dim>>
        (basisDofHandlerTotalPot, quadAttrAdaptive, basisAttrMap, *linAlgOpContextHost);
  }
  else
  {
    feBDElecChargeRhs = std::make_shared<basis::CFEBasisDataStorageDealii<ValueTypeElectrostaticsCoeff, ValueTypeElectrostaticsBasis, Host,dim>>
        (basisDofHandlerTotalPot, quadAttrAdaptive, basisAttrMap, *linAlgOpContextHost);
  }
  feBDElecChargeRhs->evaluateBasisData(quadAttrAdaptive, quadRuleContainerAdaptiveOrbital, basisAttrMap);

  p.registerEnd("Electrostatics basis rho datastorage eval");
  utils::printCurrentMemoryUsage<memorySpace>(comm, "Electrostatics basis rho datastorage eval");
  p.registerStart("Orbital basis datastorage eval");

  basisAttrMap[basis::BasisStorageAttributes::StoreValues] = true;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradient] = true;
  basisAttrMap[basis::BasisStorageAttributes::StoreHessian] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreOverlap] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreGradNiGradNj] = false;
  basisAttrMap[basis::BasisStorageAttributes::StoreJxW] = true;

  // one set of cell-block work buffers for every per k-point EFE storage
  // and the FEBasisOperations built on them, which never run at once
  std::shared_ptr<basis::BasisScratch<ValueTypeWaveFunctionBasis, memorySpace>>
    basisScratchWaveFn = std::make_shared<
      basis::BasisScratch<ValueTypeWaveFunctionBasis, memorySpace>>();

  std::map<basis::kpoint::KPointId,
           std::shared_ptr<const basis::FEBasisDataStorage<ValueTypeWaveFunctionBasis, memorySpace>>>
    efeBasisDataAdaptiveOrbital;
  for (const auto &kPoint : kPoints)
    {
      std::shared_ptr<basis::FEBasisDataStorage<ValueTypeWaveFunctionBasis, memorySpace>> efeBasisData =
        std::make_shared<basis::EFEBasisDataStorageDealii<ValueTypeWaveFunctionCoeff, ValueTypeWaveFunctionBasis, memorySpace,dim>>
          (basisDofHandlerWaveFn.at(kPoint.first), quadAttrAdaptive, basisAttrMap, *linAlgOpContext,
           cfeBasisDataStorageAdaptiveOrbital, basisScratchWaveFn);
      efeBasisData->evaluateBasisData(quadAttrAdaptive, quadRuleContainerAdaptiveOrbital, basisAttrMap);
      efeBasisDataAdaptiveOrbital[kPoint.first] = efeBasisData;
    }

    std::map<basis::kpoint::KPointId, std::shared_ptr<const basis::FEBasisDataStorage<ValueTypeWaveFunctionBasis, memorySpace>>> feBDElectrostaticsHamiltonian = efeBasisDataAdaptiveOrbital;
    std::map<basis::kpoint::KPointId, std::shared_ptr<const basis::FEBasisDataStorage<ValueTypeWaveFunctionBasis,memorySpace>>> feBDKineticHamiltonian =  efeBasisDataAdaptiveOrbital;
    std::map<basis::kpoint::KPointId, std::shared_ptr<const basis::FEBasisDataStorage<ValueTypeWaveFunctionBasis, memorySpace>>> feBDEXCHamiltonian = efeBasisDataAdaptiveOrbital;
  
    p.registerEnd("Orbital basis datastorage eval");
    p.registerStart("FE Basis Manager Init");

  std::shared_ptr<atoms::AtomSuperpositionFunction<memorySpace>> elecChargeDens =
    std::make_shared<atoms::AtomSuperpositionFunction<memorySpace>>(
      atomSphericalDataContainerRhoVtot,
      extendedAtomSymbolVec,
      extendedAtomCoordinatesVec,
      "density",
      linAlgOpContext.get());
    std::shared_ptr<const utils::ScalarSpatialFunctionReal>
          zeroFunction = std::make_shared
            <utils::ScalarZeroFunctionReal>();
            
    std::map<basis::kpoint::KPointId,
             std::shared_ptr<const basis::FEBasisManager
      <ValueTypeWaveFunctionCoeff, ValueTypeWaveFunctionBasis, memorySpace,dim>>>
    basisManagerWaveFn;
    for (const auto &kPoint : kPoints)
      basisManagerWaveFn[kPoint.first] = std::make_shared
        <basis::FEBasisManager<ValueTypeWaveFunctionCoeff, ValueTypeWaveFunctionBasis, memorySpace,dim>>
          (basisDofHandlerWaveFn.at(kPoint.first));

    std::shared_ptr<const basis::FEBasisManager
      <ValueTypeElectrostaticsCoeff, ValueTypeElectrostaticsBasis, Host,dim>>
    basisManagerTotalPot = std::make_shared
      <basis::FEBasisManager<ValueTypeElectrostaticsCoeff, ValueTypeElectrostaticsBasis, Host,dim>>
        (basisDofHandlerTotalPot, zeroFunction);
    p.registerEnd("FE Basis Manager Init");

    p.registerStart("Hamiltonian Basis overlap eval");

  // Create OperatorContext for Basisoverlap

  std::map<basis::kpoint::KPointId,
           std::shared_ptr<const linearAlgebra::OperatorContext<ValueTypeWaveFunctionBasis,
                                                ValueTypeWaveFunctionCoeff,
                                                memorySpace>>> MContext;
  for (const auto &kPoint : kPoints)
    MContext[kPoint.first] =
  std::make_shared<basis::OrthoEFEOverlapOperatorContext<ValueTypeWaveFunctionBasis,
                                                      ValueTypeWaveFunctionCoeff,
                                                      memorySpace,
                                                      dim>>(
                                                      *basisManagerWaveFn.at(kPoint.first),
                                                      *cfeBasisDataStorageAdaptiveOrbital,
                                                      *efeBasisDataAdaptiveOrbital.at(kPoint.first),
                                                      *cfeBasisDataStorageAdaptiveOrbital,
                                                      ksdft::KSDFTDefaults<memorySpace>::CELL_BATCH_SIZE,
                                                      numWantedEigenvalues,
                                                      linAlgOpContext,
                                                      true);

  utils::printCurrentMemoryUsage<memorySpace>(comm, "Hamiltonian Basis overlap");

  //   quadrature::QuadratureRuleAttributes quadAttrGaussEigen(quadrature::QuadratureFamily::GAUSS,true,feOrderEigen + 1);

  //   std::shared_ptr<basis::FEBasisDataStorage<double, memorySpace>> cfeBasisDataStorageGaussEigen =
  //     std::make_shared<basis::CFEBDSOnTheFlyComputeDealii<double, double,memorySpace, dim>>
  //     (cfeBasisDofHandlerEigen, quadAttrGaussEigen, basisAttrMap, ksdft::KSDFTDefaults<memorySpace>::CELL_BATCH_SIZE_GRAD_EVAL, *linAlgOpContext);

  // cfeBasisDataStorageGaussEigen->evaluateBasisData(quadAttrGaussEigen , basisAttrMap);

  //   std::shared_ptr<const basis::OrthoEFEOverlapOperatorContext<double,
  //                                                 double,
  //                                                 memorySpace,
  //                                                 dim>> MContextTestGauss =
  //   std::make_shared<basis::OrthoEFEOverlapOperatorContext<double,
  //                                                       double,
  //                                                       memorySpace,
  //                                                       dim>>(
  //                                                       *basisManagerWaveFn,
  //                                                       *cfeBasisDataStorageGaussEigen,
  //                                                       *efeBasisDataAdaptiveOrbital,
  //                                                       /**cfeBasisDataStorageGLLEigen,*/
  //                                                       numWantedEigenvalues * ksdft::KSDFTDefaults<memorySpace>::CELL_BATCH_SIZE,);  

    std::map<basis::kpoint::KPointId,
             std::shared_ptr<const linearAlgebra::OperatorContext<ValueTypeWaveFunctionBasis,
                                                  ValueTypeWaveFunctionCoeff,
                                                  memorySpace>>> MContextForInv;
    for (const auto &kPoint : kPoints)
      MContextForInv[kPoint.first] =
    std::make_shared<basis::OrthoEFEOverlapOperatorContext<ValueTypeWaveFunctionBasis,
                                                        ValueTypeWaveFunctionCoeff,
                                                        memorySpace,
                                                        dim>>(
                                                        *basisManagerWaveFn.at(kPoint.first),
                                                        *cfeBasisDataStorageGLLEigen,
                                                        *efeBasisDataAdaptiveOrbital.at(kPoint.first),
                                                        *cfeBasisDataStorageGLLEigen,
                                                        linAlgOpContext,
                                                        true);

    p.registerEnd("Hamiltonian Basis overlap eval");
    utils::printCurrentMemoryUsage<memorySpace>(comm, "Hamiltonian Basis overlap , overlap for inv");

    p.registerStart("Hamiltonian Basis overlap inverse eval");

  std::map<basis::kpoint::KPointId,
           std::shared_ptr<const linearAlgebra::OperatorContext<ValueTypeWaveFunctionBasis,
                                                   ValueTypeWaveFunctionCoeff,
                                                   memorySpace>>> MInvContext;
  for (const auto &kPoint : kPoints)
    MInvContext[kPoint.first] =
    std::make_shared<basis::/*OrthoEFEOverlapInverseOpContextGLL*/OEFEAtomBlockOverlapInvOpContextGLL<ValueTypeWaveFunctionBasis,
                                                   ValueTypeWaveFunctionCoeff,
                                                   memorySpace,
                                                   dim>>
                                                   (*basisManagerWaveFn.at(kPoint.first),
                                                    /**MContext,*/
                                                    *cfeBasisDataStorageGLLEigen,
                                                    *efeBasisDataAdaptiveOrbital.at(kPoint.first),
                                                    *cfeBasisDataStorageGLLEigen,
                                                    linAlgOpContext);

  p.registerEnd("Hamiltonian Basis overlap inverse eval");
  utils::printCurrentMemoryUsage<memorySpace>(comm, "Hamiltonian Basis overlap and inv");

  // ElectrostaticLocalFE pairs this against a smeared charge potential built
  // from its own extended atom list, so both must span the same atom set:
  // their difference is zero outside rc only when the two sums run over the
  // same images. The operator builds that list with getExtendedAtoms, so this
  // must be the full list.
  const utils::ScalarSpatialFunctionReal *externalPotentialFunction = new 
    utils::PointChargePotentialFunction(extendedAtomCoordinatesVec,
                                        extendedAtomChargesVec);
    
  p.registerStart("Kohn Sham DFT Class Init");
  ksdft::KohnShamDFT<ValueTypeElectrostaticsCoeff,
                    ValueTypeElectrostaticsBasis,
                    ValueTypeWaveFunctionCoeff,
                    ValueTypeWaveFunctionBasis,
                    memorySpace,
                    dim>* dftefeSolve = nullptr;

  utils::printCurrentMemoryUsage<memorySpace>(comm, "Before Kohn Sham DFT Class Init");
                                      
  if(isNumericalNuclearSolve && !isDeltaRhoPoissonSolve)
  {
    // Set up the FE Basis Data Storage
    std::shared_ptr<basis::FEBasisDataStorage<ValueTypeElectrostaticsBasis, Host>> feBDNuclearChargeRhs = feBDNucChargeRhs;
    std::shared_ptr<const basis::FEBasisDataStorage<ValueTypeElectrostaticsBasis,Host>> feBDNuclearChargeStiffnessMatrix = feBDTotalChargeStiffnessMatrix;

    dftefeSolve =
     new ksdft::KohnShamDFT<ValueTypeElectrostaticsCoeff,
                            ValueTypeElectrostaticsBasis,
                            ValueTypeWaveFunctionCoeff,
                            ValueTypeWaveFunctionBasis,
                            memorySpace,
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
                                  *elecChargeDens,
                                  basisManagerTotalPot,
                                  basisManagerWaveFn,
                                  kPoints,
                                  feBDTotalChargeStiffnessMatrix,
                                  feBDNucChargeRhs,
                                  feBDElecChargeRhs,
                                  feBDNuclearChargeStiffnessMatrix,
                                  feBDNuclearChargeRhs,
                                  feBDKineticHamiltonian,
                                  feBDElectrostaticsHamiltonian,
                                  feBDEXCHamiltonian,
                                  *externalPotentialFunction,
                                  xcFunctional,
                                  linAlgOpContext,
                                  MContextForInv,
                                  MContext,
                                  MInvContext,
                                  true,
                                  atomMagZFactors,
                                  spinMode,
                                  isGHEP,
                                  orthoType,
                                  chebyshevPolynomialDegree,
                                  ksdft::MixingDefaults::SPIN_MIXING_ENHANCEMENT_FACTOR,
                                  imageAtomGenerator);
  }
  else if (!isNumericalNuclearSolve && !isDeltaRhoPoissonSolve)
  {
    dftefeSolve =
     new ksdft::KohnShamDFT<ValueTypeElectrostaticsCoeff,
                            ValueTypeElectrostaticsBasis,
                            ValueTypeWaveFunctionCoeff,
                            ValueTypeWaveFunctionBasis,
                            memorySpace,
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
                                  *elecChargeDens,
                                  basisManagerTotalPot,
                                  basisManagerWaveFn,
                                  kPoints,
                                  feBDTotalChargeStiffnessMatrix,
                                  feBDNucChargeRhs,
                                  feBDElecChargeRhs,
                                  feBDKineticHamiltonian,
                                  feBDElectrostaticsHamiltonian,
                                  feBDEXCHamiltonian,
                                  *externalPotentialFunction,
                                  xcFunctional,
                                  linAlgOpContext,
                                  MContextForInv,
                                  MContext,
                                  MInvContext,
                                  true,
                                  atomMagZFactors,
                                  spinMode,
                                  isGHEP,
                                  orthoType,
                                  chebyshevPolynomialDegree,
                                  ksdft::MixingDefaults::SPIN_MIXING_ENHANCEMENT_FACTOR,
                                  imageAtomGenerator);
  }
  else if (!isNumericalNuclearSolve && isDeltaRhoPoissonSolve)
  {
    std::shared_ptr<atoms::AtomSuperpositionFunction<memorySpace>> smfuncAtTotPot =
      std::make_shared<atoms::AtomSuperpositionFunction<memorySpace>>(
        atomSphericalDataContainerRhoVtot,
        extendedAtomSymbolVec,
        extendedAtomCoordinatesVec,
        "vtotal",
        linAlgOpContext.get());

    dftefeSolve =
     new ksdft::KohnShamDFT<ValueTypeElectrostaticsCoeff,
                            ValueTypeElectrostaticsBasis,
                            ValueTypeWaveFunctionCoeff,
                            ValueTypeWaveFunctionBasis,
                            memorySpace,
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
                                  kPoints,
                                  feBDTotalChargeStiffnessMatrix,
                                  feBDNucChargeRhs, 
                                  feBDElecChargeRhs,  
                                  feBDKineticHamiltonian,     
                                  feBDElectrostaticsHamiltonian, 
                                  feBDEXCHamiltonian,                                                                            
                                  *externalPotentialFunction,
                                  xcFunctional,
                                  linAlgOpContext,
                                  MContextForInv,
                                  MContext,
                                  MInvContext,
                                  true,
                                  tciaparams,
                                  atomMagZFactors,
                                  spinMode,
                                  isGHEP,
                                  orthoType,
                                  chebyshevPolynomialDegree,
                                  ksdft::MixingDefaults::SPIN_MIXING_ENHANCEMENT_FACTOR,
                                  // the electrostatics operator is this
                                  // generator's only consumer; b, V_smear,
                                  // V_ext and rho_at share one envelope
                                  imageAtomGenerator);
  }
  else
  {
    utils::throwException(false, "Option not there for KohnShamDFT class creation.");
  }
  p.registerEnd("Kohn Sham DFT Class Init"); 
  utils::printCurrentMemoryUsage<memorySpace>(comm, "After Kohn Sham DFT Class Init");
  p.print();

  pTot.registerEnd("Initilization");   
  pTot.registerStart("Kohn Sham DFT Solve");

  dftefeSolve->solve(); 

  pTot.registerEnd("Kohn Sham DFT Solve");
  pTot.print();

  dftefeSolve->printTotalInScopeTimings();
  delete dftefeSolve;

  // gracefully end MPI

  int mpiFinalFlag = 0;
  utils::mpi::MPIFinalized(&mpiFinalFlag);
  if(!mpiFinalFlag)
  {
    utils::mpi::MPIFinalize();
  }
}
