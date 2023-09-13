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
  double result = 1;
  result = (x_coord)*(x_coord );
  result *= ((y_coord)*(y_coord));
  result *= ((z_coord)*(z_coord ));

  return result;

}

int main()
{

  std::cout<<" Entering test constraint matrix\n";
  // Set up linAlgcontext

  //dftefe::utils::mpi::MPIComm mpi_communicator = dftefe::utils::mpi::MPICommWorld;

  // initialize the MPI environment
  //dftefe::utils::mpi::MPIInit(NULL, NULL);

  int blasQueue = 0;
  dftefe::linearAlgebra::blasLapack::BlasQueue<dftefe::utils::MemorySpace::HOST> *blasQueuePtr = &blasQueue;

  std::shared_ptr<dftefe::linearAlgebra::LinAlgOpContext<dftefe::utils::MemorySpace::HOST>> linAlgOpContext =   std::make_shared<dftefe::linearAlgebra::LinAlgOpContext<dftefe::utils::MemorySpace::HOST>>(blasQueuePtr);

  //dftefe::linearAlgebra::LinAlgOpContext<dftefe::utils::MemorySpace::HOST>        linAlgOpContext(blasQueuePtr);

  // Set up Triangulation
  const unsigned int dim = 3;
  std::shared_ptr<dftefe::basis::TriangulationBase> triangulationBase =
    std::make_shared<dftefe::basis::TriangulationDealiiSerial<dim>>();
  std::vector<unsigned int>         subdivisions = {1, 1, 1};
  std::vector<bool>                 isPeriodicFlags(dim, false);
  std::vector<dftefe::utils::Point> domainVectors(dim,
                                                  dftefe::utils::Point(dim, 0.0));


  double xmin = 1.0;
  double ymin = 1.0;
  double zmin = 1.0;

  domainVectors[0][0] = xmin;
  domainVectors[1][1] = ymin;
  domainVectors[2][2] = zmin;

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
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreValues] = true;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreGradient] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreHessian] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreOverlap] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreGradNiGradNj] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreJxW] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreQuadRealPoints] = false;

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


  // set up Field

  dftefe::basis::Field<double, dftefe::utils::MemorySpace::HOST> fieldData( basisHandler, constraintName, 1, linAlgOpContext);


  //populate the value of the Field

  dftefe::size_type numLocallyOwnedCells  = basisManager->nLocallyOwnedCells();
  auto itField  = fieldData.begin();
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
         if( !constraintsVec[0]->isConstrained(globalId))
         {
            dftefe::size_type localId = basisHandler->globalToLocalIndex(globalId,constraintName) ;
            basisHandler->getBasisCenters(localId,constraintName,nodeLoc);

            *(itField + localId )  = interpolatePolynomial ( nodeLoc[0], nodeLoc[1], nodeLoc[2]);
           std::cout<<"UnConstrianed id = "<<globalId;
         }
         else
         {
           dftefe::size_type localId = basisHandler->globalToLocalIndex(globalId,constraintName) ;
            basisHandler->getBasisCenters(localId,constraintName,nodeLoc);
           std::cout<<"Constrained id = "<<globalId;
         }
          nodeCount++;
        }
    }



  // update the ghost values before calling apply Constraints
  // For a serial run, updating ghost values has no effect

  fieldData.updateGhostValues();
  // evaluate at the hanging nodes
  fieldData.applyConstraintsParentToChild();

  for (unsigned int i = 0 ; i < fieldData.getVector().locallyOwnedSize() ; i++)
  {
    std::cout << "data[" <<i<<"] : "<< *(fieldData.getVector().data()+i) << ",";
  }

/*
  for (dftefe::size_type iCell = 0; iCell < numLocallyOwnedCells ; iCell++)
    {
      std::vector<dftefe::global_size_type> cellGlobalNodeIds;
      basisManager->getCellDofsGlobalIds(iCell, cellGlobalNodeIds);

      for ( dftefe::size_type iNode = 0 ; iNode < cellGlobalNodeIds.size() ; iNode++)
        {
          dftefe::global_size_type globalId = cellGlobalNodeIds[iNode];
          if( !constraintsVec[0]->isConstrained(globalId))
          {
            dftefe::size_type localId = basisHandler->globalToLocalIndex(globalId,constraintName) ;

           std::cout<<"id = "<<nodeCount<<" local = "<<localId<<" inConsVal = "<<*(itField + localId )<<"\n";
          }
          nodeCount++;
        }
    }

*/
  // create the quadrature Value Container

//   std::shared_ptr<dftefe::quadrature::QuadratureRule> quadRule =
//     std::make_shared<dftefe::quadrature::QuadratureRuleGauss>(dim, num1DGaussSize);

//   dftefe::basis::LinearCellMappingDealii<dim> linearCellMappingDealii;
//   dftefe::quadrature::QuadratureRuleContainer quadRuleContainer( quadAttr, quadRule, triangulationBase,
//                                                                  linearCellMappingDealii);

//   dftefe::quadrature::QuadratureValuesContainer<double, dftefe::utils::MemorySpace::HOST> quadValuesContainer(quadRuleContainer, 1);


//   // Interpolate the nodal data to the quad points
//   feBasisOp.interpolate( fieldData, quadAttr, quadValuesContainer);


//   const std::vector<dftefe::utils::Point> & locQuadPoints = quadRuleContainer.getRealPoints();

//   bool testPass = true;
//   dftefe::size_type count = 0;
//   for( auto it  = quadValuesContainer.begin() ; it != quadValuesContainer.end() ; it++ )
//     {
//       double xLoc = locQuadPoints[count][0];
//       double yLoc = locQuadPoints[count][1];
//       double zLoc = locQuadPoints[count][2];

//       double analyticValue = interpolatePolynomial (xLoc, yLoc, zLoc);

//       if ( std::abs((*it) - analyticValue) > 1e-8 )
//         {
//          std::cout<<" id = "<<count <<" x = "<<xLoc<<" y  = "<<yLoc<<" z = "<<zLoc<<" analVal = "<<analyticValue<<" interValue = "<<(*it)<<"\n";
//          testPass = false;
// 	      } 
//       count++;
//     }

//   std::cout<<" test status = "<<testPass<<"\n";
//   return testPass;

  //dftefe::utils::mpi::MPIFinalize();
}
