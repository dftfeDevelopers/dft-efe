#include <basis/TriangulationBase.h>
#include <basis/TriangulationDealiiParallel.h>
#include <basis/CellMappingBase.h>
#include <basis/LinearCellMappingDealii.h>
#include <basis/EFEBasisManagerDealii.h>
#include <basis/BasisManager.h>
#include <basis/FEConstraintsDealii.h>
#include <basis/EFEBasisDataStorageDealii.h>
#include <basis/FEBasisOperations.h>
#include <basis/EFEBasisHandlerDealii.h>
#include <quadrature/QuadratureAttributes.h>
#include <quadrature/QuadratureRuleGauss.h>
#include <utils/Point.h>
#include <utils/TypeConfig.h>
#include <utils/MemorySpaceType.h>
#include <utils/MemoryStorage.h>
#include <vector>
#include <fstream>
#include <memory>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/dofs/dof_tools.h>

#include <iostream>

int main()
{

  std::cout<<" Entering test enrichment basis manager\n";
  // Set up linAlgcontext

  // initialize the MPI environment
  dftefe::utils::mpi::MPIInit(NULL, NULL);

  dftefe::utils::mpi::MPIComm comm = dftefe::utils::mpi::MPICommWorld;

  int blasQueue = 0;
  dftefe::linearAlgebra::blasLapack::BlasQueue<dftefe::utils::MemorySpace::HOST> *blasQueuePtr = &blasQueue;

  std::shared_ptr<dftefe::linearAlgebra::LinAlgOpContext<dftefe::utils::MemorySpace::HOST>> linAlgOpContext = 
    std::make_shared<dftefe::linearAlgebra::LinAlgOpContext<dftefe::utils::MemorySpace::HOST>>(blasQueuePtr);

  // Set up Triangulation
  const unsigned int dim = 3;
  std::shared_ptr<dftefe::basis::TriangulationBase> triangulationBase =
    std::make_shared<dftefe::basis::TriangulationDealiiParallel<dim>>(comm);
  std::vector<unsigned int>         subdivisions = {5, 5, 5};
  std::vector<bool>                 isPeriodicFlags(dim, false);
  std::vector<dftefe::utils::Point> domainVectors(dim,
                                                  dftefe::utils::Point(dim, 0.0));


  double xmax = 5.0;
  double ymax = 5.0;
  double zmax = 5.0;

  domainVectors[0][0] = xmax;
  domainVectors[1][1] = ymax;
  domainVectors[2][2] = zmax;

  // initialize the triangulation
  triangulationBase->initializeTriangulationConstruction();
  triangulationBase->createUniformParallelepiped(subdivisions,
                                                 domainVectors,
                                                 isPeriodicFlags);
  triangulationBase->finalizeTriangulationConstruction();
 
  triangulationBase->executeCoarseningAndRefinement();
  triangulationBase->finalizeTriangulationConstruction();

  // Enrichment data file consisting of g(r,\theta,\phi) = f(r)*Y_lm(\theta, \phi)
  std::string sourceDir = "/home/avirup/dft-efe/test/basis/src/";
  std::string atomDataFile = "AtomData.in";
  std::string inputFileName = sourceDir + atomDataFile;
  std::fstream fstream;

  fstream.open(inputFileName, std::fstream::in);
  
  // read the input file and create atomsymbol vector and atom coordinates vector.
  std::vector<dftefe::utils::Point> atomCoordinatesVec;
  std::vector<double> coordinates;
  coordinates.resize(dim,0.);
  std::vector<std::string> atomSymbol;
  std::string symbol;
  atomSymbol.resize(0);
  std::string line;
  while (std::getline(fstream, line)){
      std::stringstream ss(line);
      ss >> symbol; 
      for(unsigned int i=0 ; i<dim ; i++){
          ss >> coordinates[i]; 
      }
      atomCoordinatesVec.push_back(coordinates);
      atomSymbol.push_back(symbol);
  }
  dftefe::utils::mpi::MPIBarrier(comm);
      
  std::map<std::string, std::string> atomSymbolToFilename;
  for (auto i:atomSymbol )
  {
      atomSymbolToFilename[i] = sourceDir + i + ".xml";
  }

  std::vector<std::string> fieldNames{ "density", "vhartree", "vnuclear", "vtotal", "orbital" };
  std::vector<std::string> metadataNames{ "symbol", "Z", "charge", "NR", "r" };
  std::shared_ptr<dftefe::atoms::AtomSphericalDataContainer>  atomSphericalDataContainer = 
      std::make_shared<dftefe::atoms::AtomSphericalDataContainer>(atomSymbolToFilename,
                                                      fieldNames,
                                                      metadataNames);

  std::string fieldName = "vnuclear";
  double atomPartitionTolerance = 1e-6;


  // initialize the basis Manager

  unsigned int feOrder = 2;

  const dftefe::quadrature::QuadratureRuleAttributes l2ProjQuadAttr(dftefe::quadrature::QuadratureFamily::GLL,true,4);

  std::shared_ptr<dftefe::basis::FEBasisManager> basisManager =   std::make_shared<dftefe::basis::EFEBasisManagerDealii<double,dftefe::utils::MemorySpace::HOST,dim>>(
      triangulationBase ,
      atomSphericalDataContainer ,
      feOrder ,
      atomPartitionTolerance,
      atomSymbol,
      atomCoordinatesVec,
      fieldName,
      comm,
      l2ProjQuadAttr,
      linAlgOpContext);
  std::map<dftefe::global_size_type, dftefe::utils::Point> dofCoords;
  basisManager->getBasisCenters(dofCoords);

  //std::cout << (basisManager->getLocallyOwnedRanges()[0]).first << "," << (basisManager->getLocallyOwnedRanges()[0]).second << ";" << (basisManager->getLocallyOwnedRanges()[1]).first << "," << (basisManager->getLocallyOwnedRanges()[1]).second;
  
  // Set the constraints

  std::string constraintName = "HomogenousWithHanging";
  std::vector<std::shared_ptr<dftefe::basis::FEConstraintsBase<double, dftefe::utils::MemorySpace::HOST>>>
    constraintsVec;
  constraintsVec.resize(1);
  for ( unsigned int i=0 ;i < constraintsVec.size() ; i++ )
   constraintsVec[i] = std::make_shared<dftefe::basis::EFEConstraintsDealii<double, double, dftefe::utils::MemorySpace::HOST, dim>>();
   
  constraintsVec[0]->clear();
  constraintsVec[0]->makeHangingNodeConstraint(basisManager);
  constraintsVec[0]->setHomogeneousDirichletBC();
  constraintsVec[0]->close();

  std::map<std::string,
           std::shared_ptr<const dftefe::basis::Constraints<double, dftefe::utils::MemorySpace::HOST>>> constraintsMap;

  constraintsMap[constraintName] = constraintsVec[0];

  // Set up the quadrature rule
  unsigned int num1DGaussSize =4;

  dftefe::quadrature::QuadratureRuleAttributes quadAttr(dftefe::quadrature::QuadratureFamily::GAUSS,true,num1DGaussSize);

  dftefe::basis::BasisStorageAttributesBoolMap basisAttrMap;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreValues] = true;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreGradient] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreHessian] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreOverlap] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreGradNiGradNj] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreJxW] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreQuadRealPoints] = false;

  // Set up the FE Basis Data Storage
  std::shared_ptr<dftefe::basis::BasisDataStorage<double, dftefe::utils::MemorySpace::HOST>> feBasisData =
    std::make_shared<dftefe::basis::EFEBasisDataStorageDealii<double, dftefe::utils::MemorySpace::HOST,dim>>
    (basisManager, quadAttr, basisAttrMap);

  // // evaluate basis data
  feBasisData->evaluateBasisData(quadAttr, basisAttrMap);

  // Set up BasisHandler
  std::shared_ptr<dftefe::basis::BasisHandler<double, dftefe::utils::MemorySpace::HOST>> basisHandler =
    std::make_shared<dftefe::basis::EFEBasisHandlerDealii<double, double, dftefe::utils::MemorySpace::HOST,dim>>
    (basisManager, constraintsMap, comm);

  std::vector<std::pair<dftefe::global_size_type, dftefe::global_size_type>> vec = basisHandler->getLocallyOwnedRanges(constraintName);
  for (auto i:vec )
  {
    std::cout<< i.first << "," << i.second << "\n" ;
  }

  // // Set up basis Operations
  dftefe::basis::FEBasisOperations<double, double, dftefe::utils::MemorySpace::HOST,dim> feBasisOp(feBasisData,50);

  dftefe::utils::mpi::MPIFinalize();

}
