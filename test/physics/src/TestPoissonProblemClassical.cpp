#include <basis/TriangulationBase.h>
#include <basis/TriangulationDealiiSerial.h>
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
    return -21*std::pow((r-rc),3)*(6*r*r + 3*r*rc + rc*rc)/(5*M_PI*std::pow(rc,8));
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
  // Set up linAlgcontext

  //dftefe::utils::mpi::MPIComm mpi_communicator = dftefe::utils::mpi::MPICommWorld;

  // initialize the MPI environment
  //dftefe::utils::mpi::MPIInit(NULL, NULL);

  // Required to solve : \nabla^2 V_H = g(r,r_c) Solve using CG in linearAlgebra
  // In the weak form the eqn is:
  // (N_i,N_j)*V_H = (N_i, g(r,r_c))
  // Input to CG are : linearSolverFnction. Reqd to create a derived class of the base.
  // For the nabla : LaplaceOperatorContextFE to get \nabla^2(A)*x = y

  //
  // initialize MPI
  //
  int mpiInitFlag = 0;
  dftefe::utils::mpi::MPIInitialized(&mpiInitFlag);
  if(!mpiInitFlag)
  {
    dftefe::utils::mpi::MPIInit(NULL, NULL);
  }

  int blasQueue = 0;
  dftefe::linearAlgebra::blasLapack::BlasQueue<dftefe::utils::MemorySpace::HOST> *blasQueuePtr = &blasQueue;

  std::shared_ptr<dftefe::linearAlgebra::LinAlgOpContext<dftefe::utils::MemorySpace::HOST>> linAlgOpContext =   
    std::make_shared<dftefe::linearAlgebra::LinAlgOpContext<dftefe::utils::MemorySpace::HOST>>(blasQueuePtr);

  //dftefe::linearAlgebra::LinAlgOpContext<dftefe::utils::MemorySpace::HOST>        linAlgOpContext(blasQueuePtr);

  // Set up Triangulation
  const unsigned int dim = 3;
  std::shared_ptr<dftefe::basis::TriangulationBase> triangulationBase =
    std::make_shared<dftefe::basis::TriangulationDealiiSerial<dim>>();
  std::vector<unsigned int>         subdivisions = {5, 5, 5};
  std::vector<bool>                 isPeriodicFlags(dim, false);
  std::vector<dftefe::utils::Point> domainVectors(dim,
                                                  dftefe::utils::Point(dim, 0.0));

  double xmax = 5.0;
  double ymax = 5.0;
  double zmax = 5.0;
  double rc = 0.5;
  double Rx = 2.5;
  double Ry = 2.5;
  double Rz = 2.5;
  unsigned int numComponents = 1;
  double hMin = 0.1;

  domainVectors[0][0] = xmax;
  domainVectors[1][1] = ymax;
  domainVectors[2][2] = zmax;

  // initialize the triangulation
  triangulationBase->initializeTriangulationConstruction();
  triangulationBase->createUniformParallelepiped(subdivisions,
                                                 domainVectors,
                                                 isPeriodicFlags);

  bool flag = true;
  while(flag)
  {
    flag = false;
    auto triaCellIter = triangulationBase->beginLocal();
    for( ; triaCellIter != triangulationBase->endLocal(); triaCellIter++)
    {
      dftefe::utils::Point centerPoint(dim, 0.0); 
      (*triaCellIter)->center(centerPoint);
      double dist = (centerPoint[0] - Rx)* (centerPoint[0] - Rx);  
      dist += (centerPoint[1] - Ry)* (centerPoint[1] - Ry);
      dist += (centerPoint[2] - Rz)* (centerPoint[2] - Rz);
      dist = std::sqrt(dist);
      if (dist < 2*rc && (*triaCellIter)->diameter() > hMin)
      {
        (*triaCellIter)->setRefineFlag();
        flag = true;
      }
    }
    triangulationBase->executeCoarseningAndRefinement();
    // Mpi_allreduce that all the flags are 1 (mpi_max)
  }

  triangulationBase->finalizeTriangulationConstruction();

  // initialize the basis Manager

  unsigned int feDegree = 3;

  std::shared_ptr<dftefe::basis::FEBasisManager> basisManager =   std::make_shared<dftefe::basis::FEBasisManagerDealii<dim>>(triangulationBase, feDegree);
  std::map<dftefe::global_size_type, dftefe::utils::Point> dofCoords;
  basisManager->getBasisCenters(dofCoords);
        
  // Set the constraints

  std::string constraintRho = "HomogenousWithHanging";
  std::string constraintPotential = "InHomogenousWithHanging";
  std::vector<std::shared_ptr<dftefe::basis::FEConstraintsBase<double, dftefe::utils::MemorySpace::HOST>>>
    constraintsVec;
  constraintsVec.resize(2, std::make_shared<dftefe::basis::FEConstraintsDealii<double, dftefe::utils::MemorySpace::HOST, dim>>());

  constraintsVec[0]->clear();
  constraintsVec[0]->makeHangingNodeConstraint(basisManager);
  constraintsVec[0]->setHomogeneousDirichletBC();
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
      auto              cell = basisManager->beginLocallyOwnedCells(),
      endc              = basisManager->endLocallyOwnedCells();
      dftefe::utils::Point basisCenter(0);
      for (; cell != endc; ++cell)
        {
          (*cell)->cellNodeIdtoGlobalNodeId(cellGlobalDofIndices);
          for (unsigned int iFace = 0; iFace < faces_per_cell; ++iFace)
            {
              (*cell)->getFaceDoFGlobalIndices(iFace, iFaceGlobalDofIndices);
              const dftefe::size_type boundaryId = (*cell)->getFaceBoundaryId(iFace);
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
                          double constraintValue = 1/std::sqrt(basisCenter[0]*basisCenter[0] + 
                            basisCenter[1]*basisCenter[1] + 
                            basisCenter[2]*basisCenter[2]);
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

  // // evaluate basis data
  feBasisData->evaluateBasisData(quadAttr, basisAttrMap);

  // // Set up BasisHandler
  std::shared_ptr<dftefe::basis::FEBasisHandler<double, dftefe::utils::MemorySpace::HOST,dim>> basisHandler =
    std::make_shared<dftefe::basis::FEBasisHandlerDealii<double, dftefe::utils::MemorySpace::HOST,dim>>
    (basisManager, constraintsMap);

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

  //dftefe::basis::Field<double, dftefe::utils::MemorySpace::HOST> fieldData( basisHandler, constraintName, 1, linAlgOpContext);

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
            *(itField + localId )  = potential(nodeLoc[0], nodeLoc[1], nodeLoc[2], Rx, Ry, Rz, rc);
         }
        }
    }

  // update the ghost values before calling apply Constraints
  // For a serial run, updating ghost values has no effect

  vh->updateGhostValues();
  constraintsVec[1]->distributeParentToChild(*vh, numComponents);

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
                                                    constraintPotential,
                                                    dftefe::linearAlgebra::PreconditionerType::JACOBI ,
                                                    mpiPatternP2PPotential,
                                                    linAlgOpContext,
                                                    50);

  dftefe::linearAlgebra::LinearAlgebraProfiler profiler;
  dftefe::size_type maxIter = 1000;
  double absoluteTol = 1e-2;
  double relativeTol = 1e-2;
  double divergenceTol = 1e-2;

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

   std::cout<<"solution norm"<<solution.l2Norms()[0]<<"potential analytical norm"<<vh->l2Norms()[0];

  // // Interpolate the nodal data to the quad points
  // // feBasisOp.interpolate( fieldData, quadAttr, quadValuesContainer);


  // // const std::vector<dftefe::utils::Point> & locQuadPoints = quadRuleContainer.getRealPoints();

  // // bool testPass = true;
  // // dftefe::size_type count = 0;
  // // for( auto it  = quadValuesContainer.begin() ; it != quadValuesContainer.end() ; it++ )
  // //   {
  // //     double xLoc = locQuadPoints[count][0];
  // //     double yLoc = locQuadPoints[count][1];
  // //     double zLoc = locQuadPoints[count][2];

  // //     double analyticValue = interpolatePolynomial (feDegree, xLoc, yLoc, zLoc,xmin,ymin,zmin);
  // //     count++;
  // //   }

  // // std::cout<<" test status = "<<testPass<<"\n";
  // // return testPass;

  //
  // gracefully end MPI
  //
  int mpiFinalFlag = 0;
  dftefe::utils::mpi::MPIFinalized(&mpiFinalFlag);
  if(!mpiFinalFlag)
  {
    dftefe::utils::mpi::MPIFinalize();
  }
}
