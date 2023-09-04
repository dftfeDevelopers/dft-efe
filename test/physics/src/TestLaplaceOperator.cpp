#include <basis/TriangulationBase.h>
#include <basis/TriangulationDealiiParallel.h>
#include <basis/CellMappingBase.h>
#include <basis/LinearCellMappingDealii.h>
#include <basis/FEBasisManagerDealii.h>
#include <basis/FEConstraintsDealii.h>
#include <basis/FEBasisDataStorageDealii.h>
#include <basis/FEBasisOperations.h>
#include <basis/FEConstraintsDealii.h>
#include <basis/FEBasisHandlerDealii.h>
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
  dftefe::linearAlgebra::blasLapack::BlasQueue<dftefe::utils::MemorySpace::HOST> *blasQueuePtr = &blasQueue;

  std::shared_ptr<dftefe::linearAlgebra::LinAlgOpContext<dftefe::utils::MemorySpace::HOST>> linAlgOpContext =
    std::make_shared<dftefe::linearAlgebra::LinAlgOpContext<dftefe::utils::MemorySpace::HOST>>(blasQueuePtr);

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

  std::ofstream out("grid.vtk");
  dealii::GridOut       grid_out;
  grid_out.write_vtk(triDealiiPara->returnDealiiTria(), out);


  // initialize the basis Manager

  std::shared_ptr<dftefe::basis::FEBasisManager> basisManager =   std::make_shared<dftefe::basis::FEBasisManagerDealii<dim>>(triangulationBase, feDegree);
  std::map<dftefe::global_size_type, dftefe::utils::Point> dofCoords;
  basisManager->getBasisCenters(dofCoords);

  std::cout << "Locally owned cells : " <<basisManager->nLocallyOwnedCells() << "\n";
  std::cout << "Total Number of dofs : " << basisManager->nGlobalNodes() << "\n";

  // Set the constraints

  std::string constraintHanging = "HangingNodeConstraint"; //give BC to rho
  std::string constraintHomwHan = "HomogeneousWithHanging"; // use this to solve the laplace equation

  std::vector<std::shared_ptr<dftefe::basis::FEConstraintsBase<double, dftefe::utils::MemorySpace::HOST>>>
    constraintsVec;
  constraintsVec.resize(2);
  for ( unsigned int i=0 ;i < constraintsVec.size() ; i++ )
   constraintsVec[i] = std::make_shared<dftefe::basis::FEConstraintsDealii<double, dftefe::utils::MemorySpace::HOST, dim>>();

  constraintsVec[0]->clear();
  constraintsVec[0]->makeHangingNodeConstraint(basisManager);
  constraintsVec[0]->close();

  constraintsVec[1]->clear();
  constraintsVec[1]->makeHangingNodeConstraint(basisManager);
  constraintsVec[1]->setHomogeneousDirichletBC();
  constraintsVec[1]->close();

  std::map<std::string,
           std::shared_ptr<const dftefe::basis::Constraints<double, dftefe::utils::MemorySpace::HOST>>> constraintsMap;

  constraintsMap[constraintHanging] = constraintsVec[0];
  constraintsMap[constraintHomwHan] = constraintsVec[1];

  // Set up the quadrature rule

  dftefe::quadrature::QuadratureRuleAttributes quadAttr(dftefe::quadrature::QuadratureFamily::GAUSS,true,num1DGaussSize);

  dftefe::basis::BasisStorageAttributesBoolMap basisAttrMap;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreValues] = true;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreGradient] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreHessian] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreOverlap] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreGradNiGradNj] = true;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreJxW] = true;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreQuadRealPoints] = true;

  // Set up the FE Basis Data Storage
  std::shared_ptr<dftefe::basis::FEBasisDataStorage<double, dftefe::utils::MemorySpace::HOST>> feBasisData =
    std::make_shared<dftefe::basis::FEBasisDataStorageDealii<double, dftefe::utils::MemorySpace::HOST,dim>>
    (basisManager, quadAttr, basisAttrMap);

  // evaluate basis data
  feBasisData->evaluateBasisData(quadAttr, basisAttrMap);

  // Set up BasisHandler
  std::shared_ptr<dftefe::basis::FEBasisHandler<double, dftefe::utils::MemorySpace::HOST,dim>> basisHandler =
    std::make_shared<dftefe::basis::FEBasisHandlerDealii<double, dftefe::utils::MemorySpace::HOST,dim>>
    (basisManager, constraintsMap, comm);

  // Set up basis Operations
  dftefe::basis::FEBasisOperations<double, double, dftefe::utils::MemorySpace::HOST,dim> feBasisOp(feBasisData,50);

  // set up MPIPatternP2P for the constraints
  auto mpiPatternP2PHanging = basisHandler->getMPIPatternP2P(constraintHanging);
  auto mpiPatternP2PHomwHan = basisHandler->getMPIPatternP2P(constraintHomwHan);

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
  //           basisHandler->getBasisCenters(i,constraintHomwHan,nodeLoc);
  //           std::cout<<"id= "<<basisHandler->localToGlobalIndex(i,constraintHomwHan)<<" x= "<<nodeLoc[0]<<" y= "<<nodeLoc[1]<<" z= "<<nodeLoc[2] << " Val= " << *(b->data()+i) << " is_C= " << (bool)basisHandler->getConstraints(constraintHomwHan).isConstrained(basisHandler->localToGlobalIndex(i,constraintHomwHan))<< " rank= " << rank << " is_ghost = " << (bool)basisHandler->isGhostEntry(basisHandler->localToGlobalIndex(i,constraintHomwHan),constraintHomwHan).first << std::endl;
  //         }
  //       }
  //     std::cout << std::flush;
  //     dftefe::utils::mpi::MPIBarrier(comm);
  //   }

  // x->setValue(1.0);

  auto itField  = x->begin();
  const unsigned int dofs_per_cell =
    basisManager->nCellDofs(0);
  const unsigned int faces_per_cell =
    dealii::GeometryInfo<dim>::faces_per_cell;
  const unsigned int dofs_per_face =
    std::pow((basisManager->getFEOrder(0)+1),2);
  std::vector<dftefe::global_size_type> cellGlobalDofIndices(dofs_per_cell);
  std::vector<dftefe::global_size_type> iFaceGlobalDofIndices(dofs_per_face);
  std::vector<bool> dofs_touched(basisManager->nGlobalNodes(), false);
  auto              icell = basisManager->beginLocallyOwnedCells();
  dftefe::utils::Point basisCenter(dim, 0);
  for (; icell != basisManager->endLocallyOwnedCells(); ++icell)
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
                      dftefe::size_type localId = basisHandler->globalToLocalIndex(globalId,constraintHanging) ;
                      basisHandler->getBasisCenters(localId,constraintHanging,nodeLoc);
                      *(itField + (localId)*(numComponents))  = 1;         
                }     // Face dof loop
            }
        } // Face loop
    }     // cell locally owned


  std::shared_ptr<dftefe::linearAlgebra::OperatorContext<double, double, dftefe::utils::MemorySpace::HOST>>
  AxContext = 
  std::make_shared<dftefe::physics::LaplaceOperatorContextFE<double, double, dftefe::utils::MemorySpace::HOST, dim>>(
      *basisHandler,
      *feBasisData,
      constraintHanging,
      constraintHomwHan,
      quadAttr,
      50); 

      std::cout << std::setprecision(10);

    for(int i = 0 ; i < numComponents ; i++)
      std::cout << "b-norm before apply: " << b->l2Norms()[i] << " x-norm before apply: " <<
      x->l2Norms()[i] << "\n";

  AxContext->apply(*x, *b);

  // basisHandler->getConstraints(constraintHomwHan).distributeParentToChild(*b, numComponents);
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
            basisHandler->getBasisCenters(i,constraintHomwHan,nodeLoc);
            std::cout<<"id= "<<basisHandler->localToGlobalIndex(i,constraintHomwHan)<<" x= "<<nodeLoc[0]<<" y= "<<nodeLoc[1]<<" z= "<<nodeLoc[2] << " Val= " << *(b->data()+i) << " is_C= " << (bool)basisHandler->getConstraints(constraintHomwHan).isConstrained(basisHandler->localToGlobalIndex(i,constraintHomwHan))<< " rank= " << rank << " is_ghost = " << (bool)basisHandler->isGhostEntry(basisHandler->localToGlobalIndex(i,constraintHomwHan),constraintHomwHan).first << std::endl;
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
