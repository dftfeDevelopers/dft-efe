#include <basis/TriangulationBase.h>
#include <basis/TriangulationDealiiParallel.h>
#include <basis/CellMappingBase.h>
#include <basis/LinearCellMappingDealii.h>
#include <basis/EFEBasisDofHandlerDealii.h>
#include <basis/EFEConstraintsLocalDealii.h>
#include <basis/EFEBasisDataStorageDealii.h>
#include <basis/FEBasisOperations.h>
#include <basis/FEBasisManager.h>
#include <quadrature/QuadratureAttributes.h>
#include <quadrature/QuadratureRuleGauss.h>
#include <quadrature/QuadratureRuleAdaptive.h>
#include <atoms/AtomSevereFunction.h>
#include <utils/Point.h>
#include <utils/TypeConfig.h>
#include <utils/MemorySpaceType.h>
#include <utils/MemoryStorage.h>
#include <vector>
#include <fstream>
#include <memory>
#include <linearAlgebra/LinearSolverFunction.h>
#include <physics/PoissonLinearSolverFunctionFE.h>
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
#include <iomanip>

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

std::vector<std::vector<double>> readRhoValues(std::string ParamFile)
{
  std::vector<std::vector<double>> ret(2);
  std::string line;
  std::fstream fstream;
  fstream.open(ParamFile, std::fstream::in);
  while (std::getline(fstream, line))
  {
    std::istringstream iss(line);
    std::string valueStr;
    unsigned int count = 0;
    while (iss >> valueStr) 
    {
      ret[count].push_back(std::stod(valueStr));
      count = count + 1;
    }
  }
  fstream.close();
  return ret;
}

// e- charge density
double rho(const dftefe::utils::Point &point, const std::vector<dftefe::utils::Point> &origin, const dftefe::utils::Spline & spline)
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
    ret += -(spline)(r);
  }
  return ret;
}

// e- charge density
double rho(const dftefe::utils::Point &point, const std::vector<dftefe::utils::Point> &origin)
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
    ret += - (1/M_PI)*exp(-2*r);
  }
  return ret;
}

//smeared charge density
double bSmear(const dftefe::utils::Point &point, const std::vector<dftefe::utils::Point> &origin, double rc)
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

// smeared charge potential
double vSmear(const dftefe::utils::Point &point, const std::vector<dftefe::utils::Point> &origin, double rc)
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
      ret += 1/r;
    else
      ret += (9*std::pow(r,7)-30*std::pow(r,6)*rc
        +28*std::pow(r,5)*std::pow(rc,2)-14*std::pow(r,2)*std::pow(rc,5)
        +12*std::pow(rc,7))/(5*std::pow(rc,8));
  }
  return ret;
}

// point charge potential
double vPoint(const dftefe::utils::Point &point, const std::vector<dftefe::utils::Point> &origin)
{
  double ret = 0;
  for (unsigned int i = 0 ; i < origin.size() ; i++ )
  {
    double r = 0;
    for (unsigned int j = 0 ; j < point.size() ; j++ )
    {
      r += std::pow((point[j]-origin[i][j]),2);
    }
    dftefe::utils::throwException(r != 0, "Point Charge potential calculated at the atom, infty value.");
    r = std::sqrt(r);
    ret += 1/r;
  }
  return ret;
}

class ScalarSpatialSmearedPotentialFunctionReal : public dftefe::utils::ScalarSpatialFunctionReal
  {
    public:
    ScalarSpatialSmearedPotentialFunctionReal(std::vector<dftefe::utils::Point> &origin, double rc)
    :d_rc(rc), d_origin(origin)
    {}

    double
    operator()(const dftefe::utils::Point &point) const
    {
      return vSmear(point, d_origin, d_rc);
    }

    std::vector<double>
    operator()(const std::vector<dftefe::utils::Point> &points) const
    {
      std::vector<double> ret(0);
      ret.resize(points.size());
      for (unsigned int i = 0 ; i < points.size() ; i++)
      {
        ret[i] = vSmear(points[i], d_origin, d_rc);
      }
      return ret;
    }

    private:
    std::vector<dftefe::utils::Point> d_origin; 
    double d_rc;
  };

class ScalarSpatialTotalPotentialFunctionReal : public dftefe::utils::ScalarSpatialFunctionReal
  {
    public:
    ScalarSpatialTotalPotentialFunctionReal(std::vector<dftefe::utils::Point> &origin, double rc, const dftefe::utils::Spline & spline)
    :d_rc(rc), d_origin(origin), d_spline(spline)
    {}

    double
    operator()(const dftefe::utils::Point &point) const
    {
      return vSmear(point, d_origin, d_rc) + rho(point, d_origin, d_spline);
    }

    std::vector<double>
    operator()(const std::vector<dftefe::utils::Point> &points) const
    {
      std::vector<double> ret(0);
      ret.resize(points.size());
      for (unsigned int i = 0 ; i < points.size() ; i++)
      {
        ret[i] = vSmear(points[i], d_origin, d_rc) + rho(points[i], d_origin, d_spline);
      }
      return ret;
    }

    private:
    std::vector<dftefe::utils::Point> d_origin; 
    double d_rc;
    const dftefe::utils::Spline d_spline;
  };

  class AtomEnergyFunction : public dftefe::utils::ScalarSpatialFunctionReal
  {
  private:
    std::vector<dftefe::utils::Point> d_atomCoordinatesVec;
    double d_rc;

  public:
    AtomEnergyFunction(
      const std::vector<dftefe::utils::Point> &atomCoordinatesVec,
      double rc)
      : d_atomCoordinatesVec(atomCoordinatesVec),
        d_rc(rc)
      {}

    double
    operator()(const dftefe::utils::Point &point) const
    {
      return bSmear(point, d_atomCoordinatesVec, d_rc) * vSmear(point, d_atomCoordinatesVec, d_rc);
    }

    std::vector<double>
    operator()(const std::vector<dftefe::utils::Point> &points) const
    {
      std::vector<double> ret(0);
      ret.resize(points.size());
      for (unsigned int i = 0 ; i < points.size() ; i++)
      {
        ret[i] = bSmear(points[i], d_atomCoordinatesVec, d_rc) * vSmear(points[i], d_atomCoordinatesVec, d_rc);
      }
      return ret;
    }
};

int main(int argc, char** argv)
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

  int rank;
  dftefe::utils::mpi::MPICommRank(comm, &rank);

  int blasQueue = 0;
  dftefe::linearAlgebra::blasLapack::BlasQueue<dftefe::utils::MemorySpace::HOST> *blasQueuePtr = &blasQueue;

  std::shared_ptr<dftefe::linearAlgebra::LinAlgOpContext<dftefe::utils::MemorySpace::HOST>> linAlgOpContext =
    std::make_shared<dftefe::linearAlgebra::LinAlgOpContext<dftefe::utils::MemorySpace::HOST>>(blasQueuePtr);


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
  std::string electronDensityDataFile = "ElectronDensity.in";
  std::string enrichmentDataFile = "H1e-3Uniform_rc0.6_cutoff3_sm0.6.xml";
  std::string paramDataFile = argv[1];
  std::string inputFileName = sourceDir + atomDataFile;
  std::string parameterInputFileName = sourceDir + paramDataFile;
  std::string electronDensityInputFileName = sourceDir + electronDensityDataFile;

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
  double atomPartitionTolerance = readParameter<double>(parameterInputFileName, "atomPartitionTolerance");
  double smallestCellVolume = readParameter<double>(parameterInputFileName, "smallestCellVolume");
  unsigned int maxRecursion = readParameter<unsigned int>(parameterInputFileName, "maxRecursion");
  double adaptiveQuadAbsTolerance = readParameter<double>(parameterInputFileName, "adaptiveQuadAbsTolerance");
  double adaptiveQuadRelTolerance = readParameter<double>(parameterInputFileName, "adaptiveQuadRelTolerance");
  double integralThreshold = readParameter<double>(parameterInputFileName, "integralThreshold");
  
  // Read the electron density data file and set the Spline interpolation 
  std::vector<std::vector<double>> rhoRadialValues = readRhoValues(electronDensityInputFileName);
  const dftefe::utils::Spline spline(rhoRadialValues[0],
                                      rhoRadialValues[1],
                                      dftefe::utils::Spline::spline_type::cspline,
                                      false,
                                      dftefe::utils::Spline::bd_type::first_deriv,
                                      0.0,
                                      dftefe::utils::Spline::bd_type::first_deriv,
                                      0.0);

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
  
  // read the input file and create atomSymbolVec vector and atom coordinates vector.
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

    std::map<std::string, std::string> atomSymbolToFilename;
  for (auto i:atomSymbolVec )
  {
      atomSymbolToFilename[i] = sourceDir + enrichmentDataFile;
  }

  std::vector<std::string> fieldNames{"vtotal"};
  std::vector<std::string> metadataNames{ "symbol", "Z", "charge", "NR", "r" };
  std::shared_ptr<dftefe::atoms::AtomSphericalDataContainer>  atomSphericalDataContainer = 
      std::make_shared<dftefe::atoms::AtomSphericalDataContainer>(atomSymbolToFilename,
                                                      fieldNames,
                                                      metadataNames);

  std::string fieldName = "vtotal";

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

// Make pristine EFE basis

  // 1. Make EnrichmentClassicalInterface object for Pristine enrichment
  // 2. Input to the EFEBasisManager(eci, feOrder) 
  // 3. Make EFEBasisDataStorage with input as quadratureContainer.

  std::shared_ptr<dftefe::basis::EnrichmentClassicalInterfaceSpherical
                          <double, dftefe::utils::MemorySpace::HOST, dim>>
                          enrichClassIntfce = std::make_shared<dftefe::basis::EnrichmentClassicalInterfaceSpherical
                          <double, dftefe::utils::MemorySpace::HOST, dim>>(triangulationBase,
                          atomSphericalDataContainer,
                          atomPartitionTolerance,
                          atomSymbolVec,
                          atomCoordinatesVec,
                          fieldName,
                          comm);

    // Set up the vector of scalarSpatialRealFunctions for adaptive quadrature
    std::vector<std::shared_ptr<const dftefe::utils::ScalarSpatialFunctionReal>> functionsVec(0);
    unsigned int numfun = 3;
    functionsVec.resize(numfun); // Enrichment Functions
    std::vector<double> absoluteTolerances(numfun), relativeTolerances(numfun);
    std::vector<double> integralThresholds(numfun);
    for ( unsigned int i=0 ;i < functionsVec.size()-1 ; i++ )
    {
        functionsVec[i] = std::make_shared<dftefe::atoms::AtomSevereFunction<dim>>(        
            enrichClassIntfce->getEnrichmentIdsPartition(),
            atomSphericalDataContainer,
            atomSymbolVec,
            atomCoordinatesVec,
            fieldName,
            i);
        absoluteTolerances[i] = adaptiveQuadAbsTolerance;
        relativeTolerances[i] = adaptiveQuadRelTolerance;
        integralThresholds[i] = integralThreshold;
    }
functionsVec[2] = std::make_shared<AtomEnergyFunction>(
        atomCoordinatesVec,
        rc);
    absoluteTolerances[2] = adaptiveQuadAbsTolerance;
    relativeTolerances[2] = adaptiveQuadRelTolerance;
    integralThresholds[2] = integralThreshold;

    //Set up quadAttr for Rhs and OverlapMatrix

    dftefe::quadrature::QuadratureRuleAttributes quadAttrAdaptive(dftefe::quadrature::QuadratureFamily::ADAPTIVE,false);

    // Set up base quadrature rule for adaptive quadrature 

    std::shared_ptr<dftefe::quadrature::QuadratureRule> baseQuadRule =
    std::make_shared<dftefe::quadrature::QuadratureRuleGauss>(dim, num1DGaussSize);

    std::shared_ptr<dftefe::basis::CellMappingBase> cellMapping = std::make_shared<dftefe::basis::LinearCellMappingDealii<dim>>();
    std::shared_ptr<dftefe::basis::ParentToChildCellsManagerBase> parentToChildCellsManager = std::make_shared<dftefe::basis::ParentToChildCellsManagerDealii<dim>>();

    std::shared_ptr<dftefe::quadrature::QuadratureRuleContainer> quadRuleContainerAdaptive =
      std::make_shared<dftefe::quadrature::QuadratureRuleContainer>
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

    dftefe::quadrature::QuadratureRuleAttributes quadAttrAdaptiveStiffnessMatrix(dftefe::quadrature::QuadratureFamily::ADAPTIVE,false);

    absoluteTolerances.clear(), relativeTolerances.clear();
    for ( unsigned int i=0 ;i < functionsVec.size() ; i++ )
    {
        absoluteTolerances[i] = adaptiveQuadAbsTolerance*1e3;
        relativeTolerances[i] = adaptiveQuadRelTolerance*1e3;
    }

    std::shared_ptr<dftefe::quadrature::QuadratureRuleContainer> quadRuleContainerAdaptiveStiffnessMatrix =
      std::make_shared<dftefe::quadrature::QuadratureRuleContainer>
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

  // initialize the basis 
  std::shared_ptr<dftefe::basis::FEBasisDofHandler<double, dftefe::utils::MemorySpace::HOST,dim>> basisDofHandler =  
    std::make_shared<dftefe::basis::EFEBasisDofHandlerDealii<double, double,dftefe::utils::MemorySpace::HOST,dim>>(
      enrichClassIntfce, feOrder, comm);

  std::map<dftefe::global_size_type, dftefe::utils::Point> dofCoords;
  basisDofHandler->getBasisCenters(dofCoords);

  std::cout << "Locally owned cells : " <<basisDofHandler->nLocallyOwnedCells() << "\n";
  std::cout << "Total Number of dofs : " << basisDofHandler->nGlobalNodes() << "\n";

  // Set up quad attribute
    dftefe::quadrature::QuadratureRuleAttributes quadAttr(dftefe::quadrature::QuadratureFamily::ADAPTIVE,false);

  dftefe::basis::BasisStorageAttributesBoolMap basisAttrMap;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreValues] = true;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreGradient] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreHessian] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreOverlap] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreGradNiGradNj] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreJxW] = true;

  // Set up Adaptive quadrature for EFE Basis Data Storage
  std::shared_ptr<dftefe::basis::FEBasisDataStorage<double, dftefe::utils::MemorySpace::HOST>> feBasisData =
    std::make_shared<dftefe::basis::EFEBasisDataStorageDealii<double, double, dftefe::utils::MemorySpace::HOST,dim>>
    (basisDofHandler, quadAttrAdaptive, basisAttrMap);

  // evaluate basis data
  feBasisData->evaluateBasisData(quadAttrAdaptive, quadRuleContainerAdaptive, basisAttrMap);

basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreValues] = false;
    basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreGradient] = false;
    basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreHessian] = false;
    basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreOverlap] = false;
    basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreGradNiGradNj] = true;
    basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreJxW] = false;

    // Set up Adaptive quadrature for EFE Basis Data Storage
    std::shared_ptr<dftefe::basis::FEBasisDataStorage<double, dftefe::utils::MemorySpace::HOST>> feBasisDataStiffnessMatrix =
    std::make_shared<dftefe::basis::EFEBasisDataStorageDealii<double, double, dftefe::utils::MemorySpace::HOST,dim>>
    (basisDofHandler, quadAttrAdaptive, basisAttrMap);

    // evaluate basis data
    feBasisDataStiffnessMatrix->evaluateBasisData(quadAttrAdaptiveStiffnessMatrix, quadRuleContainerAdaptiveStiffnessMatrix, basisAttrMap);

  // Set up basis Operations
  dftefe::basis::FEBasisOperations<double, double, dftefe::utils::MemorySpace::HOST,dim> feBasisOp(feBasisData,50);

  // create the quadrature Rule Container

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

  std::vector<double> smearedChargeDensity(nAtoms+1, 0.0), mpiReducedSmearedChargeDensity(smearedChargeDensity.size(), 0.0);

  
    double echarge = 0, mpiReducedTotalElectronicChargeDensity= 0;
    for(dftefe::size_type i = 0 ; i < quadRuleContainer->nCells() ; i++)
    {
      std::vector<double> JxW = quadRuleContainer->getCellJxW(i);
      dftefe::size_type quadId = 0;
      for (auto j : quadRuleContainer->getCellRealPoints(i))
      {
        for(unsigned int iProb = 0 ; iProb < atomsVecInDomain.size() ; iProb++)
          smearedChargeDensity[iProb] += bSmear( j, atomsVecInDomain[iProb], rc) * JxW[quadId];
        echarge += rho(j, atomCoordinatesVec, spline) * JxW[quadId];
        quadId = quadId + 1;
      }
    }

  
  dftefe::utils::mpi::MPIAllreduce<dftefe::utils::MemorySpace::HOST>(
        smearedChargeDensity.data(),
        mpiReducedSmearedChargeDensity.data(),
        smearedChargeDensity.size(),
        dftefe::utils::mpi::MPIDouble,
        dftefe::utils::mpi::MPISum,
        comm);

  dftefe::utils::mpi::MPIAllreduce<dftefe::utils::MemorySpace::HOST>(
        &echarge,
        &mpiReducedTotalElectronicChargeDensity,
        1,
        dftefe::utils::mpi::MPIDouble,
        dftefe::utils::mpi::MPISum,
        comm);

    std::cout << "Charge Density over volume (bSmear, rho): "<< mpiReducedSmearedChargeDensity[2] << "," << mpiReducedTotalElectronicChargeDensity << "\n";

    std::vector<double> energy(nAtoms+1, 0.0), mpiReducedEnergy(energy.size(), 0.0);
  
  for( unsigned int iProb = 0 ; iProb < atomsVecInDomain.size() ; iProb++)
  {
    std::shared_ptr<const dftefe::utils::ScalarSpatialFunctionReal>
          potentialFunction;
      if(atomsVecInDomain[iProb].size() == 1)
        potentialFunction = std::make_shared<ScalarSpatialSmearedPotentialFunctionReal>(atomsVecInDomain[iProb], rc);
      else
        potentialFunction = std::make_shared<ScalarSpatialTotalPotentialFunctionReal>(atomsVecInDomain[iProb], rc, spline);

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
      if(atomsVecInDomain[iProb].size() == 1)
        a[k] = bSmear( j, atomsVecInDomain[iProb], rc) * (4*M_PI) * (1.0/mpiReducedSmearedChargeDensity[iProb]);
      else
        a[k] = (bSmear( j, atomsVecInDomain[iProb], rc)* (1.0*atomsVecInDomain[iProb].size()/mpiReducedSmearedChargeDensity[iProb])
          + std::abs(1.0*atomsVecInDomain[iProb].size()/mpiReducedTotalElectronicChargeDensity)
          * rho(j, atomsVecInDomain[iProb], spline)) * (4*M_PI);
      double *b = a.data();
      quadValuesContainer.setCellQuadValues<dftefe::utils::MemorySpace::HOST> (i, quadId, b);
      quadId = quadId + 1;
    }
  }

  std::shared_ptr<dftefe::linearAlgebra::LinearSolverFunction<double,
                                                   double,
                                                   dftefe::utils::MemorySpace::HOST>> linearSolverFunction =
    std::make_shared<dftefe::physics::PoissonLinearSolverFunctionFE<double,
                                                   double,
                                                   dftefe::utils::MemorySpace::HOST,
                                                   dim>>
                                                   (basisManager,
                                                    feBasisDataStiffnessMatrix,
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

  // calculate different energies

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
    analyticalSelfEnergy += 0.5 * (Ig - vSmear(atomCoordinatesVec[i], coord, rc));
    numericalSelfEnergy += mpiReducedEnergy[i];
      }
    
    std::cout << "The electrostatic interaction energy from analytical Self Energy: " << (mpiReducedEnergy[nAtoms] + analyticalSelfEnergy) << "\n";

    if(rank == 0)
    {
      std::ofstream myfile;
      std::stringstream ss;
        ss << "EFE"<<"domain_"<<xmax<<"x"<<ymax<<"x"<<zmax<<
        "subdiv_"<<subdivisionx<<"x"<<subdivisiony<<"x"<<subdivisionz<<
        "feOrder_"<<feOrder<<"hMin_"<<hMin<<"nQuad_"<<num1DGaussSize<<
        "adapAbsTol_"<<adaptiveQuadAbsTolerance<<"adapRelTol_"<<adaptiveQuadRelTolerance<<
        "threshold"<<integralThreshold<<".out";
      std::string outputFile = ss.str();
      myfile.open (outputFile, std::ios::out | std::ios::trunc);
        myfile << "Total Number of dofs : " << basisDofHandler->nGlobalNodes() << "\n";
        myfile << "No. of quad points: "<< mpinumQuadraturePoints << "\n";
        myfile << std::fixed << std::setprecision(15) << std::endl;
        myfile << "Charge Density over volume (bSmear, rho): "<< mpiReducedSmearedChargeDensity[2] << "," << mpiReducedTotalElectronicChargeDensity << "\n";
        myfile << std::endl;
        myfile << "The electrostatic interaction energy from analytical Self Energy: "<< 
            (mpiReducedEnergy[nAtoms] + analyticalSelfEnergy) << "\n";
        myfile << "The electrostatic interaction energy from numerical Self Energy: " << 
            (mpiReducedEnergy[nAtoms] - numericalSelfEnergy)<<"\n";
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
