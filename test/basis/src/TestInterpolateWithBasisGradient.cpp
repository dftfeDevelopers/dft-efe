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
#include <quadrature/QuadratureValuesContainer.h>
#include <utils/Point.h>
#include <utils/TypeConfig.h>
#include <utils/MemorySpaceType.h>
#include <utils/MemoryStorage.h>
#include <vector>
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

double interpolatePolynomial (double x_coord, double y_coord, double z_coord)
{
  double result = 0;
  result = (x_coord)*(x_coord );
  result += ((y_coord)*(y_coord));
  result += ((z_coord)*(z_coord ));

  return result*(1.0);

}

std::vector<double> interpolatePolynomialGradient(double x_coord, double y_coord, double z_coord)
{
  std::vector<double> a(0);
  a.resize(3);
  a[0] = 2*x_coord;
  a[1] = 2*y_coord;
  a[2] = 2*z_coord;

  return a;
}

int main()
{

  std::cout<<" Entering test constraint matrix\n";
  // Set up linAlgcontext

  dftefe::utils::mpi::MPIComm mpi_communicator = dftefe::utils::mpi::MPICommWorld;

  //initialize the MPI environment
  dftefe::utils::mpi::MPIInit(NULL, NULL);

  int blasQueue = 0;
  dftefe::linearAlgebra::blasLapack::BlasQueue<dftefe::utils::MemorySpace::HOST> *blasQueuePtr = &blasQueue;

  std::shared_ptr<dftefe::linearAlgebra::LinAlgOpContext<dftefe::utils::MemorySpace::HOST>> linAlgOpContext =   std::make_shared<dftefe::linearAlgebra::LinAlgOpContext<dftefe::utils::MemorySpace::HOST>>(blasQueuePtr);

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

  domainVectors[0][0] = xmax;
  domainVectors[1][1] = ymax;
  domainVectors[2][2] = zmax;

  // initialize the triangulation
  triangulationBase->initializeTriangulationConstruction();
  triangulationBase->createUniformParallelepiped(subdivisions,
                                                 domainVectors,
                                                 isPeriodicFlags);
  //triangulationBase->finalizeTriangulationConstruction();

  auto triaCellIter = triangulationBase->beginLocal();
  
  for( ; triaCellIter != triangulationBase->endLocal(); triaCellIter++)
  {
    dftefe::utils::Point centerPoint(dim, 0.0); 
    (*triaCellIter)->center(centerPoint);
    double dist = (centerPoint[0] - 2.5)* (centerPoint[0] - 2.5);  
    dist += (centerPoint[1] - 2.5)* (centerPoint[1] - 2.5);
    dist += (centerPoint[2] - 2.5)* (centerPoint[2] - 2.5);
    dist = std::sqrt(dist); 
    if ( (centerPoint[0] < 1.0) || (dist < 1.0) )
    {
     (*triaCellIter)->setRefineFlag();
    }
  }
 
  triangulationBase->executeCoarseningAndRefinement();
  triangulationBase->finalizeTriangulationConstruction();

  // initialize the basis Manager

  unsigned int feDegree = 2;
  unsigned int numComponents = 1;

  std::shared_ptr<dftefe::basis::FEBasisManager> basisManager =   std::make_shared<dftefe::basis::FEBasisManagerDealii<dim>>(triangulationBase, feDegree);
  std::map<dftefe::global_size_type, dftefe::utils::Point> dofCoords;
  basisManager->getBasisCenters(dofCoords);

  // Set the constraints

  std::string constraintName = "InHomogenousWithHanging";
  std::vector<std::shared_ptr<dftefe::basis::FEConstraintsBase<double, dftefe::utils::MemorySpace::HOST>>>
    constraintsVec;
  constraintsVec.resize(1);
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
                      double constraintValue = interpolatePolynomial(basisCenter[0], basisCenter[1], basisCenter[2]);
                      constraintsVec[0]->setInhomogeneity(nodeId, constraintValue);
                    } // non-hanging node check
                }     // Face dof loop
            }
        } // Face loop
    }     // cell locally owned
  constraintsVec[0]->close();
  
  std::vector<std::shared_ptr<dftefe::basis::Constraints<double, dftefe::utils::MemorySpace::HOST>>>
    constraintsBaseVec(constraintsVec.size(), nullptr);
  std::copy(constraintsVec.begin(), constraintsVec.end(), constraintsBaseVec.begin());

  std::map<std::string,
           std::shared_ptr<const dftefe::basis::Constraints<double, dftefe::utils::MemorySpace::HOST>>> constraintsMap;

  constraintsMap[constraintName] = constraintsVec[0];

  // Set up the quadrature rule
  unsigned int num1DGaussSize =4;

  dftefe::quadrature::QuadratureRuleAttributes quadAttr(dftefe::quadrature::QuadratureFamily::GAUSS,true,num1DGaussSize);

  dftefe::basis::BasisStorageAttributesBoolMap basisAttrMap;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreValues] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreGradient] = true;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreHessian] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreOverlap] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreGradNiGradNj] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreJxW] = false;

  // Set up the FE Basis Data Storage
  std::shared_ptr<dftefe::basis::BasisDataStorage<double, dftefe::utils::MemorySpace::HOST>> feBasisData =
    std::make_shared<dftefe::basis::FEBasisDataStorageDealii<double, dftefe::utils::MemorySpace::HOST,dim>>
    (basisManager, quadAttr, basisAttrMap);

  // // evaluate basis data
  feBasisData->evaluateBasisData(quadAttr, basisAttrMap);

  // // Set up BasisHandler
  std::shared_ptr<dftefe::basis::BasisHandler<double, dftefe::utils::MemorySpace::HOST>> basisHandler =
    std::make_shared<dftefe::basis::FEBasisHandlerDealii<double, dftefe::utils::MemorySpace::HOST,dim>>
    (basisManager, constraintsMap);

  // Set up basis Operations
  dftefe::basis::FEBasisOperations<double, double, dftefe::utils::MemorySpace::HOST,dim> feBasisOp(feBasisData,50);

  // set up MPIPatternP2P for the constraints
  auto mpiPatternP2P = basisHandler->getMPIPatternP2P(constraintName);

  std::shared_ptr<dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>
   X = std::make_shared<
    dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>(
      mpiPatternP2P, linAlgOpContext, numComponents, double());


  //populate the value of the Potential at the nodes for the analytic expressions

  dftefe::size_type numLocallyOwnedCells  = basisManager->nLocallyOwnedCells();
  auto itField  = X->begin();
  dftefe::utils::Point nodeLoc(dim,0.0);
  dftefe::size_type nodeCount = 0; 
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
         if( !basisHandler->getConstraints(constraintName).isConstrained(globalId))
         {
            dftefe::size_type localId = basisHandler->globalToLocalIndex(globalId,constraintName) ;
            basisHandler->getBasisCenters(localId,constraintName,nodeLoc);
            for(int comp = 0 ; comp < numComponents ; comp++)
            {
              *(itField + localId*numComponents + comp )  = interpolatePolynomial(nodeLoc[0], nodeLoc[1], nodeLoc[2]);
            }
         }
        }
    }

  X->updateGhostValues();
  basisHandler->getConstraints(constraintName).distributeParentToChild(*X, X->getNumberComponents());

  // create the quadrature Value Container

  std::shared_ptr<const dftefe::quadrature::QuadratureRuleContainer> quadRuleContainer =  
                feBasisData->getQuadratureRuleContainer();                                                               

  dftefe::quadrature::QuadratureValuesContainer<double, dftefe::utils::MemorySpace::HOST> quadValuesContainer(quadRuleContainer, numComponents*dim);

  // Interpolate the nodal data to the quad points
  feBasisOp.interpolateWithBasisGradient( *X, constraintName, *basisHandler, quadValuesContainer);

  bool testPass = true;

  for (dftefe::size_type iCell = 0; iCell < numLocallyOwnedCells ; iCell++)
    {
      std::vector<dftefe::utils::Point> cellQuadPoints = quadRuleContainer->getCellRealPoints(iCell);
      for(dftefe::size_type iQuad = 0 ; iQuad < cellQuadPoints.size() ; iQuad++ )
      {
        double xLoc = cellQuadPoints[iQuad][0];
        double yLoc = cellQuadPoints[iQuad][1];
        double zLoc = cellQuadPoints[iQuad][2];
        std::vector<double> analyticValue = interpolatePolynomialGradient
          ( xLoc, yLoc, zLoc);
        std::vector<double> values(0);
        values.resize(dim);
        quadValuesContainer.getCellQuadValues<dftefe::utils::MemorySpace::HOST>(iCell, iQuad, values.data());
        for(unsigned int iDim = 0 ; iDim < dim ; iDim++ )
        {
          if ( std::abs(values[iDim] - analyticValue[iDim]) > 1e-12 )
          {
            std::cout <<"Dim = "<<iDim<<" x = "<<xLoc<<" y  = "<<yLoc<<" z = "<<zLoc<<" analVal = "<<analyticValue[iDim]<<" interValue = "<<values[iDim]<<"\n";
            testPass = false;
          } 
        }
      }
    }

  std::cout<<" test status = "<<testPass<<"\n";
  return testPass;

  dftefe::utils::mpi::MPIFinalize();
}
