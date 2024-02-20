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
#include <basis/CFEOverlapInverseOperatorContext.h>

int main()
{
  std::cout<<" Entering test CFE overlap matrix \n";

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

  // Set up Triangulation
  const unsigned int dim = 3;
    std::shared_ptr<dftefe::basis::TriangulationBase> triangulationBase =
        std::make_shared<dftefe::basis::TriangulationDealiiParallel<dim>>(comm);
  std::vector<unsigned int>         subdivisions = {5, 5 ,5};
  std::vector<bool>                 isPeriodicFlags(dim, false);
  std::vector<dftefe::utils::Point> domainVectors(dim,
                                                  dftefe::utils::Point(dim, 0.0));

  double xmax = 5.0;
  double ymax = 5.0;
  double zmax = 5.0;
  unsigned int numComponents = 1;

  domainVectors[0][0] = xmax;
  domainVectors[1][1] = ymax;
  domainVectors[2][2] = zmax;

  // initialize the triangulation
  triangulationBase->initializeTriangulationConstruction();
  triangulationBase->createUniformParallelepiped(subdivisions,
                                                 domainVectors,
                                                 isPeriodicFlags);
  triangulationBase->finalizeTriangulationConstruction();

  // initialize the basis 

  unsigned int feDegree = 3;

  std::shared_ptr<const dftefe::basis::FEBasisDofHandler<double, dftefe::utils::MemorySpace::HOST,dim>> basisDofHandler =  
   std::make_shared<dftefe::basis::CFEBasisDofHandlerDealii<double, dftefe::utils::MemorySpace::HOST,dim>>(triangulationBase, feDegree, comm);

  std::map<dftefe::global_size_type, dftefe::utils::Point> dofCoords;
  basisDofHandler->getBasisCenters(dofCoords);

  std::cout << "Locally owned cells : " << basisDofHandler->nLocallyOwnedCells() << "\n";
  std::cout << "Total Number of dofs : " << basisDofHandler->nGlobalNodes() << "\n";

  // Set up the quadrature rule
  unsigned int num1DGLLSize = 4;

  dftefe::quadrature::QuadratureRuleAttributes quadAttr(dftefe::quadrature::QuadratureFamily::GLL,true,num1DGLLSize);

  dftefe::basis::BasisStorageAttributesBoolMap basisAttrMap;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreValues] = true;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreGradient] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreHessian] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreOverlap] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreGradNiGradNj] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreJxW] = false;

  // Set up the FE Basis Data Storage
  std::shared_ptr<dftefe::basis::FEBasisDataStorage<double, dftefe::utils::MemorySpace::HOST>> feBasisData =
    std::make_shared<dftefe::basis::CFEBasisDataStorageDealii<double, double, dftefe::utils::MemorySpace::HOST,dim>>
    (basisDofHandler, quadAttr, basisAttrMap);

  // evaluate basis data
  feBasisData->evaluateBasisData(quadAttr, basisAttrMap);

  // Set up BasisManager
  std::shared_ptr<const dftefe::basis::FEBasisManager<double, double, dftefe::utils::MemorySpace::HOST,dim>> basisManager =
    std::make_shared<dftefe::basis::FEBasisManager<double, double, dftefe::utils::MemorySpace::HOST,dim>>
    (basisDofHandler);

  // Set up basis Operations
  dftefe::basis::FEBasisOperations<double, double, dftefe::utils::MemorySpace::HOST,dim> feBasisOp(feBasisData,50);

  // set up MPIPatternP2P for the constraints
  auto mpiPatternP2PHanging = basisManager->getMPIPatternP2P();

  std::shared_ptr<dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>
   X = std::make_shared<
    dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>(
      mpiPatternP2PHanging, linAlgOpContext, numComponents, double());

  std::shared_ptr<dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>
   Y = std::make_shared<
    dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>(
      mpiPatternP2PHanging, linAlgOpContext, numComponents, double());

  std::shared_ptr<dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>
   Z = std::make_shared<
    dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>(
      mpiPatternP2PHanging, linAlgOpContext, numComponents, double());

  //populate the value of the Potential at the nodes for the analytic expressions

  dftefe::size_type numLocallyOwnedCells  = basisDofHandler->nLocallyOwnedCells();
  auto itField  = X->begin();
  dftefe::utils::Point nodeLoc(dim,0.0);
  dftefe::size_type nodeCount = 0; 
  for (dftefe::size_type iCell = 0; iCell < numLocallyOwnedCells ; iCell++)
    {
      // get cell dof global ids
      std::vector<dftefe::global_size_type> cellGlobalNodeIds;
      basisDofHandler->getCellDofsGlobalIds(iCell, cellGlobalNodeIds);

      // loop over nodes of a cell
      for ( dftefe::size_type iNode = 0 ; iNode < cellGlobalNodeIds.size() ; iNode++)
        {
          // If node not constrained then get the local id and coordinates of the node
          dftefe::global_size_type globalId = cellGlobalNodeIds[iNode];
         if( !basisManager->getConstraints().isConstrained(globalId))
         {
            dftefe::size_type localId = basisManager->globalToLocalIndex(globalId) ;
            basisManager->getBasisCenters(localId,nodeLoc);
            *(itField + localId )  = ((double) rand() / (RAND_MAX));
         }
        }
    }

    // Create OperatorContext for Basisoverlap
    std::shared_ptr<const dftefe::basis::CFEOverlapOperatorContext<double,
                                                  double,
                                                  dftefe::utils::MemorySpace::HOST,
                                                  dim>> MContext =
    std::make_shared<dftefe::basis::CFEOverlapOperatorContext<double,
                                                        double,
                                                        dftefe::utils::MemorySpace::HOST,
                                                        dim>>(
                                                        *basisManager,
                                                        *basisManager,
                                                        *feBasisData,
                                                        50);


      MContext->apply(*X,*Y);
  //feBasisOp.interpolate( *dens, constraintHomwHan, *basisManager, quadValuesContainer);

  std::shared_ptr<dftefe::linearAlgebra::OperatorContext<double,
                                                   double,
                                                   dftefe::utils::MemorySpace::HOST>> MInvContext =
    std::make_shared<dftefe::basis::CFEOverlapInverseOperatorContext<double,
                                                   double,
                                                   dftefe::utils::MemorySpace::HOST,
                                                   dim>>
                                                   (*basisManager,
                                                    *MContext,
                                                    linAlgOpContext);

    MInvContext->apply(*Y,*Z);

    std::cout << "\n" <<X->l2Norms()[0] << "," << Y->l2Norms()[0] <<"," << Z->l2Norms()[0] << "\n";

  std::shared_ptr<dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>
   error = std::make_shared<
    dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>(
      mpiPatternP2PHanging, linAlgOpContext, numComponents, double());

  std::vector<double> ones(0);
  ones.resize(numComponents, (double)1.0);
  std::vector<double> nOnes(0);
  nOnes.resize(numComponents, (double)-1.0);

  dftefe::linearAlgebra::add(ones, *X, nOnes, *Z, *error);

    std::cout<<"No of dofs: "<< basisDofHandler->nGlobalNodes() <<", error norm: "<<error->l2Norms()[0]<<", relative error: "<<(error->l2Norms()[0]/Z->l2Norms()[0])<<"\n";


  //gracefully end MPI

  int mpiFinalFlag = 0;
  dftefe::utils::mpi::MPIFinalized(&mpiFinalFlag);
  if(!mpiFinalFlag)
  {
    dftefe::utils::mpi::MPIFinalize();
  }
}
