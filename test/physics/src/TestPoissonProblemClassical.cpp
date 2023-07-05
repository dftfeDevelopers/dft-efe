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

double rho(double x, double y, double z, double Rx, double Ry, double Rz, double rc)
{
  // The function should have homogeneous dirichlet BC
  double r = std::sqrt( (x-Rx)*(x-Rx) + (y-Ry)*(y-Ry) + (z-Rz)*(z-Rz) );
  double ret = 0;
  if( r > rc )
    return 0;
  else
    return -21*std::pow((r-rc),3)*(6*r*r + 3*r*rc + rc*rc)/(5*M_PI*std::pow(rc,8))*4*M_PI;
}

double potential(double x, double y, double z, double Rx, double Ry, double Rz, double rc)
{
  // The function should have inhomogeneous dirichlet BC
  double r = std::sqrt( (x-Rx)*(x-Rx) + (y-Ry)*(y-Ry) + (z-Rz)*(z-Rz) );
  if( r > rc )
    return 1/r;
  else
    return (9*std::pow(r,7)-30*std::pow(r,6)*rc
      +28*std::pow(r,5)*std::pow(rc,2)-14*std::pow(r,2)*std::pow(rc,5)
        +12*std::pow(rc,7))/(5*std::pow(rc,8));
}

int main()
{

  std::cout<<" Entering test poisson problem classical \n";

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
  std::vector<unsigned int>         subdivisions = {10, 10, 10};
  std::vector<bool>                 isPeriodicFlags(dim, false);
  std::vector<dftefe::utils::Point> domainVectors(dim,
                                                  dftefe::utils::Point(dim, 0.0));

  double xmax = 10.0;
  double ymax = 10.0;
  double zmax = 10.0;
  double rc = 0.5;
  double Rx = 3.;
  double Ry = 3.;
  double Rz = 3.;
  unsigned int numComponents = 1;
  double hMin = 0.2;
  dftefe::size_type maxIter = 2e7;
  double absoluteTol = 1e-10;
  double relativeTol = 1e-12;
  double divergenceTol = 1e10;
  double refineradius = 3*rc;

  domainVectors[0][0] = xmax;
  domainVectors[1][1] = ymax;
  domainVectors[2][2] = zmax;

  // initialize the triangulation
  triangulationBase->initializeTriangulationConstruction();
  triangulationBase->createUniformParallelepiped(subdivisions,
                                                 domainVectors,
                                                 isPeriodicFlags);
  triangulationBase->finalizeTriangulationConstruction();

  int flag = 1;
  int mpiReducedFlag = 1;
  while(mpiReducedFlag)
  {
    flag = 0;
    auto triaCellIter = triangulationBase->beginLocal();
    for( ; triaCellIter != triangulationBase->endLocal(); triaCellIter++)
    {
      (*triaCellIter)->clearRefineFlag();
      dftefe::utils::Point centerPoint(dim, 0.0); 
      (*triaCellIter)->center(centerPoint);
      double dist = (centerPoint[0] - Rx)* (centerPoint[0] - Rx);  
      dist += (centerPoint[1] - Ry)* (centerPoint[1] - Ry);
      dist += (centerPoint[2] - Rz)* (centerPoint[2] - Rz);
      dist = std::sqrt(dist);
      if (dist < refineradius && (*triaCellIter)->diameter() > hMin)
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

  // initialize the basis Manager

  unsigned int feDegree = 3;

  std::shared_ptr<dftefe::basis::FEBasisManager> basisManager =   std::make_shared<dftefe::basis::FEBasisManagerDealii<dim>>(triangulationBase, feDegree);
  std::map<dftefe::global_size_type, dftefe::utils::Point> dofCoords;
  basisManager->getBasisCenters(dofCoords);

  std::cout << "Locally owned cells : " <<basisManager->nLocallyOwnedCells() << "\n";

  // Set the constraints

  std::string constraintRho = "InHomogeneousWithHangingRho"; //give BC to rho
  std::string constraintPotential = "HomogenousWithHangingPotential"; // use this to solve the laplace equation
  std::string constraintPotentialHDB = "InHomogeneosWithHangingPotential"; // this is for getting analytical solution
  std::vector<std::shared_ptr<dftefe::basis::FEConstraintsBase<double, dftefe::utils::MemorySpace::HOST>>>
    constraintsVec;
  constraintsVec.resize(3);
  for ( unsigned int i=0 ;i < constraintsVec.size() ; i++ )
   constraintsVec[i] = std::make_shared<dftefe::basis::FEConstraintsDealii<double, dftefe::utils::MemorySpace::HOST, dim>>();

  constraintsVec[0]->clear();
  constraintsVec[0]->makeHangingNodeConstraint(basisManager);
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
                  const dftefe::global_size_type nodeId =
                    iFaceGlobalDofIndices[iFaceDof];
                  if (dofs_touched[nodeId])
                    continue;
                  dofs_touched[nodeId] = true;
                  if (!constraintsVec[0]->isConstrained(nodeId))
                    {
                      basisCenter = dofCoords.find(nodeId)->second;
                      double constraintValue = rho(basisCenter[0], basisCenter[1], basisCenter[2], Rx, Ry, Rz, rc);
                      constraintsVec[0]->setInhomogeneity(nodeId, constraintValue);
                    } // non-hanging node check
                }     // Face dof loop
            }
        } // Face loop
    }     // cell locally owned
  constraintsVec[0]->close();

  constraintsVec[1]->clear();
  constraintsVec[1]->makeHangingNodeConstraint(basisManager);
  constraintsVec[1]->setHomogeneousDirichletBC();
  constraintsVec[1]->close();

  constraintsVec[2]->clear();
  constraintsVec[2]->makeHangingNodeConstraint(basisManager);
  dofs_touched.clear();
  dofs_touched.resize(basisManager->nGlobalNodes(), false);
  icell = basisManager->beginLocallyOwnedCells();
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
                  if (!constraintsVec[2]->isConstrained(nodeId))
                    {
                      basisCenter = dofCoords.find(nodeId)->second;
                      double constraintValue = potential(basisCenter[0], basisCenter[1], basisCenter[2], Rx, Ry, Rz, rc);
                      constraintsVec[2]->setInhomogeneity(nodeId, constraintValue);
                    } // non-hanging node check
                }     // Face dof loop
            }
        } // Face loop
    }     // cell locally owned
  constraintsVec[2]->close();

  std::map<std::string,
           std::shared_ptr<const dftefe::basis::Constraints<double, dftefe::utils::MemorySpace::HOST>>> constraintsMap;

  constraintsMap[constraintRho] = constraintsVec[0];
  constraintsMap[constraintPotentialHDB] = constraintsVec[1];
  constraintsMap[constraintPotential] = constraintsVec[2];

  // Set up the quadrature rule
  unsigned int num1DGaussSize = 4;

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
  auto mpiPatternP2PRho = basisHandler->getMPIPatternP2P(constraintRho);
  auto mpiPatternP2PPotentialHDB = basisHandler->getMPIPatternP2P(constraintPotentialHDB);
  auto mpiPatternP2PPotential = basisHandler->getMPIPatternP2P(constraintPotential);

  // set up different multivectors - rho, vh with inhomogeneous BC, vh
  std::shared_ptr<dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>
   dens = std::make_shared<
    dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>(
      mpiPatternP2PRho, linAlgOpContext, numComponents, double());

  std::shared_ptr<dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>
   vhNHDB = std::make_shared<
    dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>(
      mpiPatternP2PPotentialHDB, linAlgOpContext, numComponents, double());

  std::shared_ptr<dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>
   rhsNHDB = std::make_shared<
    dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>(
      mpiPatternP2PPotentialHDB, linAlgOpContext, numComponents, double());

  std::shared_ptr<dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>
   vh = std::make_shared<
    dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>(
      mpiPatternP2PPotential, linAlgOpContext, numComponents, double());

  std::shared_ptr<dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>
   solution = std::make_shared<
    dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>(
      mpiPatternP2PPotential, linAlgOpContext, numComponents, double());

  //populate the value of the Density at the nodes for interpolating to quad points
  auto numLocallyOwnedCells  = basisManager->nLocallyOwnedCells();
  auto itField  = dens->begin();
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
        if( !basisHandler->getConstraints(constraintRho).isConstrained(globalId))
        {
          dftefe::size_type localId = basisHandler->globalToLocalIndex(globalId,constraintRho) ;
          basisHandler->getBasisCenters(localId,constraintRho,nodeLoc);
          *(itField + localId )  = rho(nodeLoc[0], nodeLoc[1], nodeLoc[2], Rx, Ry, Rz, rc);
        }
      }
    }

  // update the ghost values before calling apply Constraints
  // For a serial run, updating ghost values has no effect

  dens->updateGhostValues();
  basisHandler->getConstraints(constraintRho).distributeParentToChild(*dens, numComponents);

  // vector for lhs

  numLocallyOwnedCells  = basisManager->nLocallyOwnedCells();
  itField  = vhNHDB->begin();
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
        if( !basisHandler->getConstraints(constraintPotential).isConstrained(globalId))
        {
          dftefe::size_type localId = basisHandler->globalToLocalIndex(globalId,constraintPotential) ;
          basisHandler->getBasisCenters(localId,constraintPotential,nodeLoc);
          *(itField + localId )  = 0;
        }
      }
    }

  vhNHDB->updateGhostValues();
  basisHandler->getConstraints(constraintPotential).distributeParentToChild(*vhNHDB, numComponents);


  //populate the value of the Potential at the nodes for the analytic expressions
  numLocallyOwnedCells  = basisManager->nLocallyOwnedCells();
  itField  = vh->begin();
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
         if( !basisHandler->getConstraints(constraintPotential).isConstrained(globalId))
         {
            dftefe::size_type localId = basisHandler->globalToLocalIndex(globalId,constraintPotential) ;
            basisHandler->getBasisCenters(localId,constraintPotential,nodeLoc);
            *(itField + localId )  = potential(nodeLoc[0], nodeLoc[1], nodeLoc[2], Rx, Ry, Rz, rc);
         }
        }
    }

  // update the ghost values before calling apply Constraints
  // For a serial run, updating ghost values has no effect

  vh->updateGhostValues();
  basisHandler->getConstraints(constraintPotential).distributeParentToChild(*vh, numComponents);

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
      double a = rho( j[0], j[1], j[2], Rx, Ry, Rz, rc);
      double *b = &a;
      quadValuesContainer.setCellQuadValues<dftefe::utils::MemorySpace::HOST> (i, quadId, b);
      quadId = quadId + 1;
    }
  }

  //feBasisOp.interpolate( *dens, constraintRho, *basisHandler, quadAttr, quadValuesContainer);

  // make the rhsAddOn vector
  std::shared_ptr<const dftefe::linearAlgebra::OperatorContext<double, double, dftefe::utils::MemorySpace::HOST>>
  AxContextNHDB = std::make_shared<dftefe::physics::LaplaceOperatorContextFE
    <double, double, dftefe::utils::MemorySpace::HOST, dim>>( *basisHandler,
    *feBasisData,
    constraintPotential,
    quadAttr,
    50);

  AxContextNHDB->apply(*vhNHDB, *rhsNHDB);

  std::vector<double> zeros(0);
  zeros.resize(numComponents, (double)0.0);
  std::vector<double> nOnes(0);
  nOnes.resize(numComponents, (double)-1.0);

  dftefe::linearAlgebra::add(zeros, *rhsNHDB, nOnes, *rhsNHDB, *rhsNHDB);

//  for (unsigned int i = 0 ; i < vhNHDB->locallyOwnedSize() ; i++)
//   {
//     std::cout << "vhNHDB[" <<i<<"] : "<< *(vhNHDB->data()+i) << ","<< "rhsNHDB[" <<i<<"] : "<< *(rhsNHDB->data()+i) << "\n";;
//   }

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
                                                    constraintPotentialHDB,
                                                    *rhsNHDB,
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

   dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST> solutionHDB;
   solutionHDB = linearSolverFunction->getSolution();

   solution->setValue(0.0);

  std::vector<double> ones(0);
  ones.resize(numComponents, (double)1.0);

  dftefe::linearAlgebra::add(ones, *vhNHDB, ones, solutionHDB, *solution);

  solution->updateGhostValues();

   std::cout<<"solution norm: "<<solution->l2Norms()[0]<<", potential analytical norm: "<<vh->l2Norms()[0]<<"\n";

//  for (unsigned int i = 0 ; i < solution->locallyOwnedSize() ; i++)
//   {
//     std::cout << "solution[" <<i<<"] : "<< *(solution->data()+i) << ","<<"exact["<<i<<"] : "<<*(vh->data()+i)<<"\n";
//   }

  //gracefully end MPI

  int mpiFinalFlag = 0;
  dftefe::utils::mpi::MPIFinalized(&mpiFinalFlag);
  if(!mpiFinalFlag)
  {
    dftefe::utils::mpi::MPIFinalize();
  }
}
