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

// Take less number of total dofs in this test case for simplicity.
// Also the enrichment spans throgh the entire triangulaion again for simplicity. 

  std::cout<<" Entering test update ghost and accumulate add of multivector\n";

  //initialize MPI

  int mpiInitFlag = 0;
  dftefe::utils::mpi::MPIInitialized(&mpiInitFlag);
  if(!mpiInitFlag)
  {
    dftefe::utils::mpi::MPIInit(NULL, NULL);
  }

  dftefe::utils::mpi::MPIComm comm = dftefe::utils::mpi::MPICommWorld;

  // Get the rank of the process
  int rank;
  dftefe::utils::mpi::MPICommRank(comm, &rank);

  // Get nProcs
  int numProcs;
  dftefe::utils::mpi::MPICommSize(comm, &numProcs);

  int blasQueue = 0;
  dftefe::linearAlgebra::blasLapack::BlasQueue<dftefe::utils::MemorySpace::HOST> *blasQueuePtr = &blasQueue;

  std::shared_ptr<dftefe::linearAlgebra::LinAlgOpContext<dftefe::utils::MemorySpace::HOST>> linAlgOpContext = 
    std::make_shared<dftefe::linearAlgebra::LinAlgOpContext<dftefe::utils::MemorySpace::HOST>>(blasQueuePtr);

  // Set up Triangulation
  const unsigned int dim = 3;
  std::shared_ptr<dftefe::basis::TriangulationBase> triangulationBase =
    std::make_shared<dftefe::basis::TriangulationDealiiParallel<dim>>(comm);
  std::vector<unsigned int>         subdivisions = {3, 3, 3};
  std::vector<bool>                 isPeriodicFlags(dim, false);
  std::vector<dftefe::utils::Point> domainVectors(dim,
                                                  dftefe::utils::Point(dim, 0.0));

  double xmax = 5.0;
  double ymax = 5.0;
  double zmax = 5.0;
  double rc = 0.5;
  unsigned int numComponents = 2;

  domainVectors[0][0] = xmax;
  domainVectors[1][1] = ymax;
  domainVectors[2][2] = zmax;
  
  // initialize the triangulation
  triangulationBase->initializeTriangulationConstruction();
  triangulationBase->createUniformParallelepiped(subdivisions,
                                                 domainVectors,
                                                 isPeriodicFlags);
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

  std::string fieldName = "orbital";
  double atomPartitionTolerance = 1e-6;

  // initialize the basis Manager

  unsigned int feOrder = 1;

  std::shared_ptr<dftefe::basis::EnrichmentClassicalInterfaceSpherical
                          <double, dftefe::utils::MemorySpace::HOST, dim>>
                          enrichClassIntfce = std::make_shared<dftefe::basis::EnrichmentClassicalInterfaceSpherical
                          <double, dftefe::utils::MemorySpace::HOST, dim>>
                          (triangulationBase,
                          atomSphericalDataContainer,
                          atomPartitionTolerance,
                          atomSymbol,
                          atomCoordinatesVec,
                          fieldName,
                          comm);

  // initialize the basis
  std::shared_ptr<dftefe::basis::FEBasisDofHandler<double, dftefe::utils::MemorySpace::HOST,dim>> basisDofHandler =  
    std::make_shared<dftefe::basis::EFEBasisDofHandlerDealii<double, double,dftefe::utils::MemorySpace::HOST,dim>>(
      enrichClassIntfce,
      feOrder,
      comm);
  std::map<dftefe::global_size_type, dftefe::utils::Point> dofCoords;
  basisDofHandler->getBasisCenters(dofCoords);

  std::cout << "Locally owned cells : " << basisDofHandler->nLocallyOwnedCells() << "\n";
  std::cout << "Total Number of dofs : " << basisDofHandler->nGlobalNodes() << "\n";

  // // Set up BasisManager
  std::shared_ptr<const dftefe::basis::BasisManager<double, dftefe::utils::MemorySpace::HOST>> basisManager =
    std::make_shared<dftefe::basis::FEBasisManager<double, double, dftefe::utils::MemorySpace::HOST,dim>>
    (basisDofHandler);

  // set up MPIPatternP2P for the constraints
  auto mpiPatternP2PHanging = basisManager->getMPIPatternP2P();

  // set up different multivectors - vh with inhomogeneous BC, vh

  std::shared_ptr<dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>
   vec = std::make_shared<
    dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>(
      mpiPatternP2PHanging, linAlgOpContext, numComponents, double());

  std::shared_ptr<const dftefe::basis::EFEBasisDofHandlerDealii<double, double,dftefe::utils::MemorySpace::HOST,dim>> efeBM =
    std::dynamic_pointer_cast<const dftefe::basis::EFEBasisDofHandlerDealii<double, double,dftefe::utils::MemorySpace::HOST,dim>>(basisDofHandler);

  std::cout << "Enrichment ghost global ids in rank " << rank << " are ";
  for (auto i:efeBM->getGhostEnrichmentGlobalIds() )
  {
    std::cout << i << ", ";
  }
  std::cout << "\n";
  dftefe::utils::mpi::MPIBarrier(comm);
  std::cout << "Enrichment local ids in rank " << rank << " are ";
  for (auto i:efeBM->getGhostEnrichmentGlobalIds() )
  {
    std::cout << basisManager->globalToLocalIndex(i) << ", ";
  }
  std::cout << "\n";
  dftefe::utils::mpi::MPIBarrier(comm);

  vec->setValue(0.0);

  int count = 0;
  for (unsigned int i = 0 ; i < vec->locallyOwnedSize() ; i++)
  {
    std::pair<bool, dftefe::size_type> a = 
      basisManager->inLocallyOwnedRanges(basisManager->localToGlobalIndex(i));
    if(a.first && a.second == basisDofHandler->
      getBasisAttributeToRangeIdMap()[dftefe::basis::BasisIdAttribute::ENRICHED]){
    count = count + 1;
    for ( unsigned int j = 0 ; j < numComponents ; j++ )
    {
      {
        *(vec->data()+i*numComponents+j) = 1000 + j + count;
      }
    }}
  }
  vec->updateGhostValues();
  vec->accumulateAddLocallyOwned();


    for(unsigned int iProc = 0 ; iProc < numProcs; iProc++)
    {
      if(iProc == rank)
      {
        for (unsigned int i = 0 ; i < vec->localSize() ; i++)
          {
            std::cout <<"rank: " << rank << " id: " << i  << " ";
            for(unsigned int j = 0 ; j<numComponents ; j++)
            {
              std::cout << *(vec->data()+i * numComponents + j) << " ";
            }
            std::cout << "\n";
          }
      }
      std::cout << std::flush ;
      dftefe::utils::mpi::MPIBarrier(comm);
    }

  int mpiFinalFlag = 0;
  dftefe::utils::mpi::MPIFinalized(&mpiFinalFlag);
  if(!mpiFinalFlag)
  {
    dftefe::utils::mpi::MPIFinalize();
  }
}
