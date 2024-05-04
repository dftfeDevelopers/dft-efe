#include <basis/TriangulationBase.h>
#include <basis/TriangulationDealiiSerial.h>
#include <basis/CellMappingBase.h>
#include <basis/LinearCellMappingDealii.h>
#include <basis/CFEBasisDofHandlerDealii.h>
#include <basis/CFEConstraintsLocalDealii.h>
#include <basis/CFEBasisDataStorageDealii.h>
#include <basis/FEBasisOperations.h>
#include <basis/FEBasisManager.h>
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

double interpolatePolynomial1 (double x_coord, double y_coord, double z_coord, double x_max, double y_max, double z_max)
{
  double result = 1;
  result = (x_coord-x_max)*(x_coord);
  result *= ((y_coord-y_max)*(y_coord));
  result *= ((z_coord-z_max)*(z_coord));

  return result;

}

std::vector<double> interpolatePolynomialGradient1 (double x_coord, double y_coord, double z_coord, double x_max, double y_max, double z_max)
{
  std::vector<double> a(0);
  a.resize(3);
  a[0] = (2*x_coord-x_max)*(y_coord-y_max)*(y_coord)*(z_coord-z_max)*(z_coord);
  a[1] = (x_coord)*(x_coord-x_max)*(2*y_coord - y_max)*(z_coord)*(z_coord - z_max );
  a[2] = (x_coord-x_max)*(x_coord)*(y_coord-y_max)*(y_coord)*(2*z_coord - z_max );

  return a;
}

double interpolatePolynomial2 (double x_coord, double y_coord, double z_coord, double x_max, double y_max, double z_max)
{
  double result = 1;
  result = (x_coord-x_max)*(x_coord)*(x_coord);
  result *= ((y_coord-y_max)*(y_coord)*(y_coord));
  result *= ((z_coord-z_max)*(z_coord)*(z_coord));

  return result;

}

std::vector<double> interpolatePolynomialGradient2 (double x_coord, double y_coord, double z_coord, double x_max, double y_max, double z_max)
{
  std::vector<double> a(0);
  a.resize(3);
  a[0] = (3*x_coord*x_coord-2*x_max*x_coord)*(y_coord-y_max)*(y_coord)*(y_coord)*(z_coord-z_max)*(z_coord)*(z_coord);
  a[1] = (x_coord)*(x_coord)*(x_coord-x_max)*(3*y_coord*y_coord-2*y_max*y_coord)*(z_coord)*(z_coord - z_max )*(z_coord);
  a[2] = (x_coord-x_max)*(x_coord)*(x_coord)*(y_coord-y_max)*(y_coord)*(y_coord)*(3*z_coord*z_coord-2*z_max*z_coord);

  return a;
}

int main()
{

  std::cout<<" Entering test Interpolate with basis gradient.\n";

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

  unsigned int feDegree = 3;
  unsigned int numComponents = 2;

  std::shared_ptr<const dftefe::basis::FEBasisDofHandler<double, dftefe::utils::MemorySpace::HOST,dim>> basisDofHandler =  
    std::make_shared<dftefe::basis::CFEBasisDofHandlerDealii<double, dftefe::utils::MemorySpace::HOST,dim>>(triangulationBase, feDegree);

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
    std::make_shared<dftefe::basis::CFEBasisDataStorageDealii<double, double, dftefe::utils::MemorySpace::HOST,dim>>
    (basisDofHandler, quadAttr, basisAttrMap);

  // evaluate basis data
  feBasisData->evaluateBasisData(quadAttr, basisAttrMap);

  std::shared_ptr<const dftefe::utils::ScalarSpatialFunctionReal>
    zeroFunction = std::make_shared<
    dftefe::utils::ScalarZeroFunctionReal>();

  // Set up BasisManager
  std::shared_ptr<const dftefe::basis::BasisManager<double, dftefe::utils::MemorySpace::HOST>> basisManager =
    std::make_shared<dftefe::basis::FEBasisManager<double, double, dftefe::utils::MemorySpace::HOST,dim>>
    (basisDofHandler, zeroFunction);

  // Set up basis Operations
  dftefe::basis::FEBasisOperations<double, double, dftefe::utils::MemorySpace::HOST,dim> feBasisOp(feBasisData,50);

  // set up MPIPatternP2P for the constraints
  auto mpiPatternP2P = basisManager->getMPIPatternP2P();

  std::shared_ptr<dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>
   X = std::make_shared<
    dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>(
      mpiPatternP2P, linAlgOpContext, numComponents, double());


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
            basisManager->getBasisCenters(localId, nodeLoc);
            *(itField + localId*numComponents + 0 )  = interpolatePolynomial1(nodeLoc[0], nodeLoc[1], nodeLoc[2], xmax, ymax, zmax);
            *(itField + localId*numComponents + 1 )  = interpolatePolynomial2(nodeLoc[0], nodeLoc[1], nodeLoc[2], xmax, ymax, zmax);
         }
        }
    }

  X->updateGhostValues();
  basisManager->getConstraints().distributeParentToChild(*X, X->getNumberComponents());

  // create the quadrature Value Container

  std::shared_ptr<const dftefe::quadrature::QuadratureRuleContainer> quadRuleContainer =  
                feBasisData->getQuadratureRuleContainer();                                                               

  dftefe::quadrature::QuadratureValuesContainer<double, dftefe::utils::MemorySpace::HOST> quadValuesContainer(quadRuleContainer, numComponents*dim);

  // Interpolate the nodal data to the quad points
  feBasisOp.interpolateWithBasisGradient( *X, *basisManager, quadValuesContainer);

  bool testPass = true;
           
  for (dftefe::size_type iCell = 0; iCell < numLocallyOwnedCells ; iCell++)
    {
      std::vector<dftefe::utils::Point> cellQuadPoints = quadRuleContainer->getCellRealPoints(iCell);
          std::vector<double> values(0);
          values.resize(cellQuadPoints.size() * numComponents * dim);
          quadValuesContainer.getCellValues<dftefe::utils::MemorySpace::HOST>(iCell, values.data());
          for(dftefe::size_type iQuad = 0 ; iQuad < cellQuadPoints.size() ; iQuad++ )
          {
            double xLoc = cellQuadPoints[iQuad][0];
            double yLoc = cellQuadPoints[iQuad][1];
            double zLoc = cellQuadPoints[iQuad][2];
            std::vector<double> analyticValue1(0), analyticValue2(0);
            analyticValue1.resize(dim);
            analyticValue1 = interpolatePolynomialGradient1( xLoc, yLoc, zLoc, xmax, ymax, zmax);
            analyticValue2.resize(dim);
            analyticValue2 = interpolatePolynomialGradient2( xLoc, yLoc, zLoc, xmax, ymax, zmax);
          for(unsigned int iDim = 0 ; iDim < dim ; iDim++ )
          {
            if ( std::abs(values[iDim * numComponents * cellQuadPoints.size() + 0 * cellQuadPoints.size()  + iQuad] - analyticValue1[iDim]) > 1e-8 )
            {
              std::cout << " Component = 0 Dim = "<<iDim<<" x = "<<xLoc<<" y  = "<<yLoc<<" z = "<<zLoc<<" analVal = "<<analyticValue1[iDim]<<" interValue = "<<values[iDim * numComponents * cellQuadPoints.size() + 0 * cellQuadPoints.size()  + iQuad]<<"\n";
              testPass = false;
            }
            if ( std::abs(values[iDim * numComponents * cellQuadPoints.size() + 1 * cellQuadPoints.size()  + iQuad] - analyticValue2[iDim]) > 1e-8 )
            {
              std::cout << " Component = 1 Dim = "<<iDim<<" x = "<<xLoc<<" y  = "<<yLoc<<" z = "<<zLoc<<" analVal = "<<analyticValue2[iDim]<<" interValue = "<<values[iDim * numComponents * cellQuadPoints.size() + 1 * cellQuadPoints.size()  + iQuad]<<"\n";
              testPass = false;
            }
          }
      }
    }
          
  std::cout<<" test status = "<<testPass<<"\n";
  return testPass;
}
