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

#include <iostream>

// operator - nabla^2 in weak form
// operand - V_H
// memoryspace - HOST

double rho(double x, double y, double z)
{
  // The function should have inhomogeneous dirichlet BC
    return 1;
}

double potential(double x, double y, double z)
{
  // The function should have inhomogeneous dirichlet BC
    return ((x)*(x) + (y)*(y) + (z)*(z))/6.0;
}

int main()
{

  std::cout<<" Entering test poisson problem classical \n";
  // Set up linAlgcontext

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
  unsigned int numComponents = 1;
  dftefe::size_type maxIter = 2e5;
  double absoluteTol = 1e-10;
  double relativeTol = 1e-12;
  double divergenceTol = 1e10;

  domainVectors[0][0] = xmax;
  domainVectors[1][1] = ymax;
  domainVectors[2][2] = zmax;

  // initialize the triangulation
  triangulationBase->initializeTriangulationConstruction();
  triangulationBase->createUniformParallelepiped(subdivisions,
                                                 domainVectors,
                                                 isPeriodicFlags);
  triangulationBase->finalizeTriangulationConstruction();

  // initialize the basis Manager

  unsigned int feDegree = 3;

  std::shared_ptr<dftefe::basis::FEBasisManager> basisManager =   std::make_shared<dftefe::basis::FEBasisManagerDealii<dim>>(triangulationBase, feDegree);
  std::map<dftefe::global_size_type, dftefe::utils::Point> dofCoords;
  basisManager->getBasisCenters(dofCoords);

  std::cout << "Locally owned cells : " <<basisManager->nLocallyOwnedCells() << "\n";
        
  // Set the constraints

  std::string constraintRho = "InHomogenousWithHangingRho";
  std::string constraintPotential = "InHomogenousWithHangingPotential";
  std::vector<std::shared_ptr<dftefe::basis::FEConstraintsBase<double, dftefe::utils::MemorySpace::HOST>>>
    constraintsVec;
  constraintsVec.resize(2, std::make_shared<dftefe::basis::FEConstraintsDealii<double, dftefe::utils::MemorySpace::HOST, dim>>());

  constraintsVec[0]->clear();
  constraintsVec[0]->makeHangingNodeConstraint(basisManager);
  constraintsVec[0]->close();

  constraintsVec[1]->clear();
  constraintsVec[1]->makeHangingNodeConstraint(basisManager);
      const unsigned int dofs_per_cell =
        basisManager->nCellDofs(0);
      const unsigned int faces_per_cell =
        dealii::GeometryInfo<dim>::faces_per_cell;
      const unsigned int dofs_per_face =
        std::pow((basisManager->getFEOrder(0)+1),2);
      std::vector<dftefe::global_size_type> cellGlobalDofIndices(dofs_per_cell);
      std::vector<dftefe::global_size_type> iFaceGlobalDofIndices(dofs_per_face);
      std::vector<bool> dofs_touched(basisManager->nGlobalNodes(), false);
      dofs_touched.resize(basisManager->nGlobalNodes(), false);
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
                      const dftefe::global_size_type nodeId =
                        iFaceGlobalDofIndices[iFaceDof];
                      if (dofs_touched[nodeId])
                        continue;
                      dofs_touched[nodeId] = true;
                      if (!constraintsVec[1]->isConstrained(nodeId))
                        {
                          basisCenter = dofCoords.find(nodeId)->second;
                          double constraintValue = 0;
                            // ((basisCenter[0])*(basisCenter[0]) + 
                            // (basisCenter[1])*(basisCenter[1]) + 
                            // (basisCenter[2])*(basisCenter[2]))/6.;  
                          constraintsVec[1]->setInhomogeneity(nodeId, constraintValue);
                        } // non-hanging node check
                    }     // Face dof loop
                }
            } // Face loop
        }     // cell locally owned
  constraintsVec[1]->close();
  
  std::vector<std::shared_ptr<dftefe::basis::Constraints<double, dftefe::utils::MemorySpace::HOST>>>
    constraintsBaseVec(constraintsVec.size(), nullptr);
  std::copy(constraintsVec.begin(), constraintsVec.end(), constraintsBaseVec.begin());

  std::map<std::string,
           std::shared_ptr<const dftefe::basis::Constraints<double, dftefe::utils::MemorySpace::HOST>>> constraintsMap;

  constraintsMap[constraintRho] = constraintsVec[0];
  constraintsMap[constraintPotential] = constraintsVec[1];

  // Set up the quadrature rule
  unsigned int num1DGaussSize = 6;

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

  // // evaluate basis data
  feBasisData->evaluateBasisData(quadAttr, basisAttrMap);

  // // Set up BasisHandler
  std::shared_ptr<dftefe::basis::FEBasisHandler<double, dftefe::utils::MemorySpace::HOST,dim>> basisHandler =
    std::make_shared<dftefe::basis::FEBasisHandlerDealii<double, dftefe::utils::MemorySpace::HOST,dim>>
    (basisManager, constraintsMap, comm);

  // Set up basis Operations
  dftefe::basis::FEBasisOperations<double, double, dftefe::utils::MemorySpace::HOST,dim> feBasisOp(feBasisData,50);


  // set up Vector for potential

  auto mpiPatternP2PPotential = basisHandler->getMPIPatternP2P(constraintPotential);
  auto mpiPatternP2PRho = basisHandler->getMPIPatternP2P(constraintRho);

  std::shared_ptr<dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>
   vh = std::make_shared<
    dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>(
      mpiPatternP2PPotential, linAlgOpContext, numComponents, double());

  std::shared_ptr<dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>
   dens = std::make_shared<
    dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>(
      mpiPatternP2PRho, linAlgOpContext, numComponents, double());

  //populate the value of the Potential at the nodes for the analytic expressions

  dftefe::size_type numLocallyOwnedCells  = basisManager->nLocallyOwnedCells();
  auto itField  = vh->begin();
  dftefe::utils::Point nodeLoc(dim,0.0);
  for (dftefe::size_type iCell = 0; iCell < numLocallyOwnedCells ; iCell++)
    {
      // get cell dof global ids
      std::vector<dftefe::global_size_type> cellGlobalNodeIds;
      basisManager->getCellDofsGlobalIds(iCell, cellGlobalNodeIds);

      // loop over nodes of a cell
      for ( dftefe::size_type iNode = 0 ; iNode < cellGlobalNodeIds.size() ; iNode++)
        {
          // If node not constrained then get the local id and coordinates of the node
          dftefe::global_size_type globalId = cellGlobalNodeIds[iNode];
         if( !constraintsVec[1]->isConstrained(globalId))
         {
            dftefe::size_type localId = basisHandler->globalToLocalIndex(globalId,constraintPotential) ;
            basisHandler->getBasisCenters(localId,constraintPotential,nodeLoc);
            *(itField + localId )  = potential(nodeLoc[0], nodeLoc[1], nodeLoc[2]);
         }
        }
    }

  // update the ghost values before calling apply Constraints
  // For a serial run, updating ghost values has no effect

  vh->updateGhostValues();
  constraintsVec[1]->distributeParentToChild(*vh, numComponents);

  //populate the value of the Density at the nodes for the analytic expressions

  numLocallyOwnedCells  = basisManager->nLocallyOwnedCells();
  itField  = dens->begin();
  dftefe::utils::Point nodeLoc1(dim,0.0);
  for (dftefe::size_type iCell = 0; iCell < numLocallyOwnedCells ; iCell++)
    {
      // get cell dof global ids
      std::vector<dftefe::global_size_type> cellGlobalNodeIds;
      basisManager->getCellDofsGlobalIds(iCell, cellGlobalNodeIds);

      // loop over nodes of a cell
      for ( dftefe::size_type iNode = 0 ; iNode < cellGlobalNodeIds.size() ; iNode++)
        {
          // If node not constrained then get the local id and coordinates of the node
          dftefe::global_size_type globalId = cellGlobalNodeIds[iNode];
         if( !constraintsVec[0]->isConstrained(globalId))
         {
            dftefe::size_type localId = basisHandler->globalToLocalIndex(globalId,constraintPotential) ;
            basisHandler->getBasisCenters(localId,constraintRho,nodeLoc1);
            *(itField + localId )  = rho(nodeLoc1[0], nodeLoc1[1], nodeLoc1[2]);
         }
        }
    }

  // update the ghost values before calling apply Constraints
  // For a serial run, updating ghost values has no effect

  dens->updateGhostValues();
  constraintsVec[0]->distributeParentToChild(*dens, numComponents);

  // create the quadrature Value Container

  std::shared_ptr<dftefe::quadrature::QuadratureRule> quadRule =
    std::make_shared<dftefe::quadrature::QuadratureRuleGauss>(dim, num1DGaussSize);

  dftefe::basis::LinearCellMappingDealii<dim> linearCellMappingDealii;
  dftefe::quadrature::QuadratureRuleContainer quadRuleContainer( quadAttr, quadRule, triangulationBase,
                                                                 linearCellMappingDealii);

  dftefe::quadrature::QuadratureValuesContainer<double, dftefe::utils::MemorySpace::HOST> quadValuesContainer(quadRuleContainer, numComponents);

  for(dftefe::size_type i = 0 ; i < quadValuesContainer.nCells() ; i++)
  {
    dftefe::size_type quadId = 0;
    for (auto j : quadRuleContainer.getCellRealPoints(i))
    {
      double a = rho( j[0], j[1], j[2]);
      double *b = &a;
      quadValuesContainer.setCellQuadValues<dftefe::utils::MemorySpace::HOST> (i, quadId, b);
      quadId = quadId + 1;
    }
  }

  //feBasisOp.interpolate( *dens, constraintRho, *basisHandler, quadAttr, quadValuesContainer);

  std::shared_ptr<dftefe::linearAlgebra::LinearSolverFunction<double,
                                                   double,
                                                   dftefe::utils::MemorySpace::HOST>> linearSolverFunction =
    std::make_shared<dftefe::physics::PoissonLinearSolverFunctionFE<double,
                                                   double,
                                                   dftefe::utils::MemorySpace::HOST,
                                                   dim>>
                                                   (basisHandler,
                                                    feBasisOp,
                                                    feBasisData,
                                                    quadValuesContainer,
                                                    quadAttr,
                                                    constraintRho,
                                                    constraintPotential,
                                                    dftefe::linearAlgebra::PreconditionerType::JACOBI ,
                                                    linAlgOpContext,
                                                    50);

  // dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST> y(
  //       mpiPatternP2PPotential, linAlgOpContext, numComponents, double());  

  // (linearSolverFunction->getAxContext()).apply(*vh, y); 

  // for (unsigned int i = 0 ; i < y.locallyOwnedSize() ; i++)
  // {
  //   std::cout << "rhs[" <<i<<"]:"<< *((linearSolverFunction->getRhs()).data()+i) << ", ";
  // }
  // std::cout << "\n" << "\n";

  // for (unsigned int i = 0 ; i < vh->locallyOwnedSize() ; i++)
  // {
  //   std::cout << "poential[" <<i<<"]:"<< *(vh->data()+i) << ", ";
  // }
  // std::cout << "\n" <<"\n";

  // for (unsigned int i = 0 ; i < y.locallyOwnedSize() ; i++)
  // {
  //   std::cout << "lhs[" <<i<<"]:"<< *(y.data()+i) << ", ";
  // }
  // std::cout << "\n" << "\n";

  // std::cout << "lhs:" << y.l2Norms()[0] << "\n";
  // std::cout << "rhs:" << (linearSolverFunction->getRhs()).l2Norms()[0] << "\n";
  

// dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST> z(
//       mpiPatternP2PPotential, linAlgOpContext, numComponents, double());
// dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST> z1(
//       mpiPatternP2PPotential, linAlgOpContext, numComponents, double());

//   AxContext->apply(*vh, y);
//   AxContext->apply(y, z);
//   AxContext->apply(z, z1);

//   std::cout << "y:" << y.l2Norms()[0] << "\n";
//   std::cout << "z:" << z.l2Norms()[0] << "\n";
//   std::cout << "z1:" << z1.l2Norms()[0] << "\n";

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

   dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST> solution;
   solution = linearSolverFunction->getSolution();
   constraintsVec[1]->distributeParentToChild(solution, numComponents);

   std::cout<<"solution norm: "<<solution.l2Norms()[0]<<", potential analytical norm: "<<vh->l2Norms()[0]<<"\n";
  
  //gracefully end MPI
  
  int mpiFinalFlag = 0;
  dftefe::utils::mpi::MPIFinalized(&mpiFinalFlag);
  if(!mpiFinalFlag)
  {
    dftefe::utils::mpi::MPIFinalize();
  }
}
