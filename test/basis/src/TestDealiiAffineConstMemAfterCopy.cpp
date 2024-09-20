#include <basis/TriangulationBase.h>
#include <basis/TriangulationDealiiParallel.h>
#include <basis/CellMappingBase.h>
#include <basis/LinearCellMappingDealii.h>
#include <basis/EFEBasisDofHandlerDealii.h>
#include <basis/BasisDofHandler.h>
#include <basis/ConstraintsLocal.h>
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
#include <math.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h> 
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/dofs/dof_tools.h>
#include <iostream>
#include <basis/GenerateMesh.h>
#include <basis/CFEOverlapInverseOpContextGLL.h>

using namespace dftefe;
int main(int argc, char** argv)
{
  std::cout<<" Entering test Affine Const Mem \n";

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
    
  const unsigned int dim = 3;
  // Set up Triangulation
    std::shared_ptr<basis::TriangulationBase> triangulationBase =
        std::make_shared<basis::TriangulationDealiiParallel<dim>>(comm);
  std::vector<bool>                 isPeriodicFlags(dim, false);
  std::vector<utils::Point> domainVectors(dim, utils::Point(dim, 0.0));

  domainVectors[0][0] = 40;
  domainVectors[1][1] = 40;
  domainVectors[2][2] = 40;

  char* dftefe_path = getenv("DFTEFE_PATH");
  std::string sourceDir;
  // if executes if a non null value is returned
  // otherwise else executes
  if (dftefe_path != NULL) 
  {
    sourceDir = (std::string)dftefe_path + "/test/basis/src/";
  }
  else
  {
    utils::throwException(false,
                          "dftefe_path does not exist!");
  }

  std::string atomDataFile = argv[1];
  std::string inputFileName = sourceDir + atomDataFile;

  std::cout << "Reading input file: "<<inputFileName<<std::endl;

  std::fstream fstream;
  fstream.open(inputFileName, std::fstream::in);
  
  // read the input file and create atomsymbol vector and atom coordinates vector.
  std::vector<utils::Point> atomCoordinatesVec(0,utils::Point(dim, 0.0));
    std::vector<double> coordinates;
  coordinates.resize(dim,0.);
  std::vector<std::string> atomSymbolVec(0);
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
  }
  utils::mpi::MPIBarrier(comm);
  fstream.close();

  // Generate mesh
   std::shared_ptr<basis::CellMappingBase> cellMapping = std::make_shared<basis::LinearCellMappingDealii<dim>>();

  basis::GenerateMesh adaptiveMesh(atomCoordinatesVec, 
                            domainVectors,
                            0.1,
                            0.04,
                            2,
                            0.27,
                            isPeriodicFlags,
                            *cellMapping,
                            comm);

  adaptiveMesh.createMesh(*triangulationBase); 

  // initialize the basis 

  unsigned int feDegree = 2;

  std::shared_ptr<const dftefe::basis::CFEBasisDofHandlerDealii<double, dftefe::utils::MemorySpace::HOST,dim>> basisDofHandler =  
   std::make_shared<dftefe::basis::CFEBasisDofHandlerDealii<double, dftefe::utils::MemorySpace::HOST,dim>>(triangulationBase, feDegree, comm);

  std::map<dftefe::global_size_type, dftefe::utils::Point> dofCoords;
  basisDofHandler->getBasisCenters(dofCoords);

  // std::shared_ptr<const utils::ScalarSpatialFunctionReal>
  //       zeroFunction = std::make_shared
  //         <utils::ScalarZeroFunctionReal>();
            
  // // Set up BasisManager
  //   std::shared_ptr<const basis::FEBasisManager
  //     <double, double, Host,dim>>
  //   basisManager = std::make_shared
  //     <basis::FEBasisManager<double, double, Host,dim>>
  //       (basisDofHandler, zeroFunction);

  std::shared_ptr<const basis::ConstraintsLocal<double, dftefe::utils::MemorySpace::HOST>> 
    constraintsLocalIntrinsic = basisDofHandler->getIntrinsicConstraints();

  const basis::CFEConstraintsLocalDealii<double, dftefe::utils::MemorySpace::HOST, dim>
    &constraintsLocal =
    dynamic_cast<const basis::CFEConstraintsLocalDealii<double, dftefe::utils::MemorySpace::HOST, dim>&>(
      *constraintsLocalIntrinsic);

  int         myProcRank = 0;
  int         err        = utils::mpi::MPICommRank(comm, &myProcRank);

  int         nProcs = 0;
  utils::mpi::MPICommSize(comm, &nProcs);
  if(myProcRank == 0)
  {
    std::cout << "memory_consumption of dealii affine constraints with original constraint matrix"<<"\n"<<std::flush;
  }
  for(int i = 0; i < nProcs ; i++)
  {
    if(myProcRank == i)
    {
      std::cout << myProcRank << "," << constraintsLocal.memoryConsumption()/1e6<<"\n";
    }
    std::cout << std::flush;
    dftefe::utils::mpi::MPIBarrier(comm);
  }

for (int i = 0 ; i < 2 ; i ++)
{

  dealii::IndexSet locally_relevant_dofs;
  locally_relevant_dofs.clear();
  dealii::DoFTools::extract_locally_relevant_dofs(*(basisDofHandler->getDoFHandler()),
                                                  locally_relevant_dofs);
                                                  
  std::shared_ptr<basis::ConstraintsLocal<double, dftefe::utils::MemorySpace::HOST>>
    constraintsLocal1 = std::make_shared<
      basis::CFEConstraintsLocalDealii<double, dftefe::utils::MemorySpace::HOST, dim>>(locally_relevant_dofs);

  std::shared_ptr<const basis::ConstraintsLocal<double, dftefe::utils::MemorySpace::HOST>> 
    constraintsLocalIntrinsic1 = basisDofHandler->getIntrinsicConstraints();

  constraintsLocal1->copyFrom(*constraintsLocalIntrinsic1);

  constraintsLocal1->close();

  const basis::CFEConstraintsLocalDealii<double, dftefe::utils::MemorySpace::HOST, dim>
    &constraintsLocal2 =
    dynamic_cast<const basis::CFEConstraintsLocalDealii<double, dftefe::utils::MemorySpace::HOST, dim>&>(
      *constraintsLocal1);

  if(myProcRank == 0)
  {
    std::cout << "memory_consumption of new dealii affine constraints after copying from original constraint matrix"<<"\n"<<std::flush;
  }
  for(int i = 0; i < nProcs ; i++)
  {
    if(myProcRank == i)
    {
      std::cout << myProcRank << "," << constraintsLocal2.memoryConsumption()/1e6<<"\n";
    }
    std::cout << std::flush;
    dftefe::utils::mpi::MPIBarrier(comm);
  }
}

  //gracefully end MPI

  int mpiFinalFlag = 0;
  dftefe::utils::mpi::MPIFinalized(&mpiFinalFlag);
  if(!mpiFinalFlag)
  {
    dftefe::utils::mpi::MPIFinalize();
  }
}
