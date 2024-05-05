#include <basis/TriangulationBase.h>
#include <basis/TriangulationDealiiParallel.h>
#include <basis/CellMappingBase.h>
#include <basis/LinearCellMappingDealii.h>
#include <basis/CFEBasisDofHandlerDealii.h>
#include <basis/CFEConstraintsLocalDealii.h>
#include <basis/CFEBasisDataStorageDealii.h>
#include <basis/FEBasisOperations.h>
#include <basis/FEBasisManager.h>
#include <quadrature/QuadratureAttributes.h>
#include <quadrature/QuadratureRuleGauss.h>
#include <quadrature/QuadratureRuleContainer.h>
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

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_out.h>
#include <deal.II/numerics/data_out.h>

#include <iostream>

int main()
{

  std::cout<<" Entering test laplacian \n";

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
  int lapackQueue = 0;
  std::shared_ptr<dftefe::linearAlgebra::blasLapack::BlasQueue
    <dftefe::utils::MemorySpace::HOST>> blasQueuePtr = std::make_shared
      <dftefe::linearAlgebra::blasLapack::BlasQueue
        <dftefe::utils::MemorySpace::HOST>>(blasQueue);
  std::shared_ptr<dftefe::linearAlgebra::blasLapack::LapackQueue
    <dftefe::utils::MemorySpace::HOST>> lapackQueuePtr = std::make_shared
      <dftefe::linearAlgebra::blasLapack::LapackQueue
        <dftefe::utils::MemorySpace::HOST>>(lapackQueue);
  std::shared_ptr<dftefe::linearAlgebra::LinAlgOpContext
    <dftefe::utils::MemorySpace::HOST>> linAlgOpContext = 
    std::make_shared<dftefe::linearAlgebra::LinAlgOpContext
    <dftefe::utils::MemorySpace::HOST>>(blasQueuePtr, lapackQueuePtr);

  // Set up Triangulation
  const unsigned int dim = 3;
    std::shared_ptr<dftefe::basis::TriangulationBase> triangulationBase =
        std::make_shared<dftefe::basis::TriangulationDealiiParallel<dim>>(comm);
  std::vector<unsigned int>         subdivisions = {2, 2, 2};
  std::vector<bool>                 isPeriodicFlags(dim, false);
  std::vector<dftefe::utils::Point> domainVectors(dim,
                                                  dftefe::utils::Point(dim, 0.0));

  double xmax = 20.0;
  double ymax = 20.0;
  double zmax = 20.0;
  unsigned int numComponents = 1;
  unsigned int feDegree = 2;
  unsigned int num1DGaussSize = 3;

  domainVectors[0][0] = xmax;
  domainVectors[1][1] = ymax;
  domainVectors[2][2] = zmax;

  // initialize the triangulation
  triangulationBase->initializeTriangulationConstruction();
  triangulationBase->createUniformParallelepiped(subdivisions,
                                                 domainVectors,
                                                 isPeriodicFlags);
  triangulationBase->finalizeTriangulationConstruction();

  auto triaCellIter = triangulationBase->beginLocal();
  
  for( ; triaCellIter != triangulationBase->endLocal(); triaCellIter++)
  {
    dftefe::utils::Point centerPoint(dim, 0.0);
    (*triaCellIter)->center(centerPoint);
    double dist = (centerPoint[0])* (centerPoint[0]);
    dist += (centerPoint[1])* (centerPoint[1]);
    dist += (centerPoint[2])* (centerPoint[2]);
    dist = std::sqrt(dist);
    if (dist < 10)
    {
     (*triaCellIter)->setRefineFlag();
    }
  }
 
  triangulationBase->executeCoarseningAndRefinement();
  triangulationBase->finalizeTriangulationConstruction();

  std::shared_ptr<const dftefe::basis::TriangulationDealiiParallel<dim>> triDealiiPara =
  std::dynamic_pointer_cast<const dftefe::basis::TriangulationDealiiParallel<dim>>(triangulationBase);

  // std::ofstream out("grid.vtk");
  // dealii::GridOut       grid_out;
  // grid_out.write_vtk(triDealiiPara->returnDealiiTria(), out);


  // initialize the basis Manager

  std::shared_ptr<const dftefe::basis::FEBasisDofHandler<double, dftefe::utils::MemorySpace::HOST,dim>> basisDofHandler =  
   std::make_shared<dftefe::basis::CFEBasisDofHandlerDealii<double, dftefe::utils::MemorySpace::HOST,dim>>(triangulationBase, feDegree, comm);

  std::map<dftefe::global_size_type, dftefe::utils::Point> dofCoords;
  basisDofHandler->getBasisCenters(dofCoords);

  std::cout << "Locally owned cells : " <<basisDofHandler->nLocallyOwnedCells() << "\n";
  std::cout << "Total Number of dofs : " << basisDofHandler->nGlobalNodes() << "\n";

  // Set up the quadrature rule

  dftefe::quadrature::QuadratureRuleAttributes quadAttr(dftefe::quadrature::QuadratureFamily::GAUSS,true,num1DGaussSize);

  dftefe::basis::BasisStorageAttributesBoolMap basisAttrMap;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreValues] = true;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreGradient] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreHessian] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreOverlap] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreGradNiGradNj] = true;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreJxW] = true;

  // Set up the FE Basis Data Storage
  std::shared_ptr<dftefe::basis::FEBasisDataStorage<double, dftefe::utils::MemorySpace::HOST>> feBasisData =
    std::make_shared<dftefe::basis::CFEBasisDataStorageDealii<double, double, dftefe::utils::MemorySpace::HOST,dim>>
    (basisDofHandler, quadAttr, basisAttrMap);

  // evaluate basis data
  feBasisData->evaluateBasisData(quadAttr, basisAttrMap);

  std::shared_ptr<const dftefe::utils::ScalarSpatialFunctionReal>
    zeroFunction = std::make_shared<
      dftefe::utils::ScalarZeroFunctionReal>();

  // // Set up BasisManager
  std::shared_ptr<const dftefe::basis::FEBasisManager<double, double, dftefe::utils::MemorySpace::HOST,dim>> basisManagerHom =
    std::make_shared<dftefe::basis::FEBasisManager<double, double, dftefe::utils::MemorySpace::HOST,dim>>
    (basisDofHandler, zeroFunction);

  std::shared_ptr<const dftefe::basis::FEBasisManager<double,double, dftefe::utils::MemorySpace::HOST,dim>> basisManager =
    std::make_shared<dftefe::basis::FEBasisManager<double, double, dftefe::utils::MemorySpace::HOST,dim>>
    (basisDofHandler);

  // Set up basis Operations
  dftefe::basis::FEBasisOperations<double, double, dftefe::utils::MemorySpace::HOST,dim> feBasisOp(feBasisData,50);

  // set up MPIPatternP2P for the constraints
  auto mpiPatternP2PHanging = basisManager->getMPIPatternP2P();
  auto mpiPatternP2PHomwHan = basisManagerHom->getMPIPatternP2P();

  std::shared_ptr<dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>
   x = std::make_shared<
    dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>(
      mpiPatternP2PHanging, linAlgOpContext, numComponents, double());

  std::shared_ptr<dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>
   b = std::make_shared<
    dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>(
      mpiPatternP2PHomwHan, linAlgOpContext, numComponents, double());

  dftefe::utils::Point nodeLoc(dim,0.0);
  std::cout << std::boolalpha;

  // for (unsigned int iProc = 0; iProc < numProcs; iProc++)
  //   {
  //     if (iProc == rank)
  //       {
  //         for (unsigned int i = 0 ; i < b->localSize() ; i++)
  //         {
  //           basisManagerHom->getBasisCenters(i,nodeLoc);
  //           std::cout<<"id= "<<basisManagerHom->localToGlobalIndex(i)<<" x= "<<nodeLoc[0]<<" y= "<<nodeLoc[1]<<" z= "<<nodeLoc[2] << " Val= " << *(b->data()+i) << " is_C= " << (bool)basisManagerHom->getConstraints().isConstrained(basisManagerHom->localToGlobalIndex(i))<< " rank= " << rank << " is_ghost = " << (bool)basisManager->isGhostEntry(basisManagerHom->localToGlobalIndex(i)).first << std::endl;
  //         }
  //       }
  //     std::cout << std::flush;
  //     dftefe::utils::mpi::MPIBarrier(comm);
  //   }

  // x->setValue(1.0);

  auto itField  = x->begin();
  const unsigned int dofs_per_cell =
    basisDofHandler->nCellDofs(0);
  const unsigned int faces_per_cell =
    dealii::GeometryInfo<dim>::faces_per_cell;
  const unsigned int dofs_per_face =
    std::pow((basisDofHandler->getFEOrder(0)+1),2);
  std::vector<dftefe::global_size_type> cellGlobalDofIndices(dofs_per_cell);
  std::vector<dftefe::global_size_type> iFaceGlobalDofIndices(dofs_per_face);
  std::vector<bool> dofs_touched(basisDofHandler->nGlobalNodes(), false);
  auto              icell = basisDofHandler->beginLocallyOwnedCells();
  dftefe::utils::Point basisCenter(dim, 0);
  for (; icell != basisDofHandler->endLocallyOwnedCells(); ++icell)
    {
      (*icell)->cellNodeIdtoGlobalNodeId(cellGlobalDofIndices);
      for (unsigned int iFace = 0; iFace < faces_per_cell; ++iFace)
        {
          (*icell)->getFaceDoFGlobalIndices(iFace, iFaceGlobalDofIndices);
          const dftefe::size_type boundaryId = (*icell)->getFaceBoundaryId(iFace);
          if (boundaryId == 0)
            {
              for (unsigned int iFaceDof = 0; iFaceDof < dofs_per_face;
                    ++iFaceDof)
                {
                  const dftefe::global_size_type globalId =
                    iFaceGlobalDofIndices[iFaceDof];
                  if (dofs_touched[globalId])
                    continue;
                  dofs_touched[globalId] = true;
                      dftefe::size_type localId = basisManager->globalToLocalIndex(globalId) ;
                      basisManager->getBasisCenters(localId,nodeLoc);
                      *(itField + (localId)*(numComponents))  = 1;         
                }     // Face dof loop
            }
        } // Face loop
    }     // cell locally owned


  std::shared_ptr<dftefe::linearAlgebra::OperatorContext<double, double, dftefe::utils::MemorySpace::HOST>>
  AxContext = 
  std::make_shared<dftefe::electrostatics::LaplaceOperatorContextFE<double, double, dftefe::utils::MemorySpace::HOST, dim>>(
      *basisManager,
      *basisManagerHom,
      *feBasisData,
      50); 

      std::cout << std::setprecision(10);

    for(int i = 0 ; i < numComponents ; i++)
      std::cout << "b-norm before apply: " << b->l2Norms()[i] << " x-norm before apply: " <<
      x->l2Norms()[i] << "\n";

  AxContext->apply(*x, *b);

  // basisManagerHom->getConstraints().distributeParentToChild(*b, numComponents);
  // b->updateGhostValues();

    for(int i = 0 ; i < numComponents ; i++)
      std::cout << "b-norm after apply: " << b->l2Norms()[i] << " x-norm after apply: " <<
      x->l2Norms()[i] << "\n";

dftefe::utils::mpi::MPIBarrier(comm);

    for (unsigned int iProc = 0; iProc < numProcs; iProc++)
      {
      if (iProc == rank)
        {
          for (unsigned int i = 0 ; i < b->localSize() ; i++)
          {
            basisManagerHom->getBasisCenters(i,nodeLoc);
            std::cout<<"id= "<<basisManagerHom->localToGlobalIndex(i)<<" x= "<<nodeLoc[0]<<" y= "<<nodeLoc[1]<<" z= "<<nodeLoc[2] << " Val= " << *(b->data()+i) << " is_C= " << (bool)basisManagerHom->getConstraints().isConstrained(basisManagerHom->localToGlobalIndex(i))<< " rank= " << rank << " is_ghost = " << (bool)basisManagerHom->isGhostEntry(basisManagerHom->localToGlobalIndex(i)).first << std::endl;
          }
        }
        std::cout << std::flush;
        dftefe::utils::mpi::MPIBarrier(comm);
      }

  //gracefully end MPI

  int mpiFinalFlag = 0;
  dftefe::utils::mpi::MPIFinalized(&mpiFinalFlag);
  if(!mpiFinalFlag)
  {
    dftefe::utils::mpi::MPIFinalize();
  }
}
