#include <basis/TriangulationBase.h>
#include <basis/TriangulationDealiiSerial.h>
#include <basis/CellMappingBase.h>
#include <basis/LinearCellMappingDealii.h>
#include <basis/FEBasisManagerDealii.h>
#include <basis/FEConstraintsDealii.h>
#include <basis/FEBasisDataStorageDealii.h>
#include <basis/FEBasisOperations.h>
#include <basis/FEConstraintsDealii.h>
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


double interpolatePolynomial (unsigned int feOrder, double x_coord, double y_coord, double z_coord, double xmin, double ymin, double zmin)
{
  // The function should go to 0 at the boundaries for it to be compatible with the
  // homogenous boundary conditions
  double result = 1;
  result = (x_coord - xmin)*(x_coord + xmin);
  result *= ((y_coord - ymin)*(y_coord + ymin));
  result *= ((z_coord - zmin)*(z_coord + zmin));

  return result;

}



int main()
{
  // Set up linAlgcontext

  int blasQueue = 0;
  blasLapack::BlasQueue<memorySpace> *blasQueuePtr = &blasQueue
  linearAlgebra::LinAlgOpContext           linAlgOpContext(blasQueuePtr);

  // Set up Triangulation
  const unsigned int dim = 3;
  std::shared_ptr<dftefe::basis::TriangulationBase> triangulationBase =
    std::make_shared<dftefe::basis::TriangulationDealiiSerial<dim>>();
  std::vector<unsigned int>         subdivisions = {10, 10, 10};
  std::vector<bool>                 isPeriodicFlags(dim, false);
  std::vector<dftefe::utils::Point> domainVectors(dim,
                                                  dftefe::utils::Point(dim, 0.0));

  double xmin = 10.0;
  double ymin = 10.0;
  double zmin = 10.0;

  domainVectors[0][0] = xmin;
  domainVectors[1][1] = ymin;
  domainVectors[2][2] = zmin;

  // initialize the triangulation
  triangulationBase->initializeTriangulationConstruction();
  triangulationBase->createUniformParallelepiped(subdivisions,
                                                 domainVectors,
                                                 isPeriodicFlags);
  triangulationBase->finalizeTriangulationConstruction();

  // initialize the basis Manager

  unsigned int feDegree = 3;

  std::shared_ptr<dftefe::basis::FEBasisManager> basisManager =
    std::make_shared<dftefe::basis::FEBasisManagerDealii<dim>>(triangulationBase, feDegree);


  // Set the constraints

  std::string constraintName = "HomogenousWithHangingPeriodic";
  std::vector<std::shared_ptr<dftefe::basis::FEConstraintsBase<double, dftefe::utils::MemorySpace::HOST>>>
    constraintsVec ;
  constraintsVec.resize(1, std::make_shared<dftefe::basis::FEConstraintsDealii<double, dftefe::utils::MemorySpace::HOST, dim>>());

  constraintsVec[0]->clear();
  constraintsVec[0]->makeHangingNodeConstraint(basisManager);
  constraintsVec[0]->setHomogeneousDirichletBC();
  constraintsVec[0]->close();

  std::map<std::string,
           std::shared_ptr<const Constraints<double, dftefe::utils::MemorySpace::HOST>>> constraintsMap;

  constraintsMap[constraintName] = constraintsVec[0];

  // Set up the quadrature rule
  unsigned int num1DGaussSize =4;

  std::vector<dftefe::quadrature::QuadratureRuleAttributes> quadAttr(1,
                                                                     dftefe::quadrature::QuadratureRuleAttributes
                                                                     (dftefe::quadrature::QuadratureFamily::GAUSS,true,num1DGaussSize));


  // Set up the FE Basis Data Storage
  std::shared_ptr<dftefe::basis::BasisDataStorgae<double, dftefe::utils::MemorySpace::HOST>> feBasisData =
    std::make_shared<dftefe::basis::FEBasisDataStorageDealii<double, dftefe::utils::MemorySpace::HOST,dim>>
    (basisManager,
     constraintsVec,
     quadAttr,
     true,
     false,
     false,
     false,
     false);

  // Set up BasisHandler
  std::shared_ptr<dftefe::basis::BasisHandler<double, dftefe::utils::MemorySpace::HOST>> basisHandler =
    std::make_shared<dftefe::basis::FEBasisHandlerDealii<double, dftefe::utils::MemorySpace::HOST,dim>>
    (basisManager, constraintsMap);

  // Set up basis Operations
  dftefe::basis::FEBasisOperations<double, dftefe::utils::MemorySpace::HOST,dim> feBasisOp(feBasisData,50);


  // set up Field

  dftefe::basis::Field<double, dftefe::utils::MemorySpace::HOST> fieldData(basisHandler,constraintName,linAlgOpContext);


  //populate the value of the Field

  size_type numLocallyOwnedCells  = basisManager->nLocallyOwnedCells();
  auto itField  = fieldData.begin();
  dftefe::Point nodeLoc;
  for (size_type iCell = 0; iCell < numLocallyOwnedCells ; iCell++)
    {
      std::vector<global_size_type> cellGlobalNodeIds;
      basisManager->getCellDofsGlobalIds(iCell, cellGlobalNodeIds);

      for ( size_type iNode = 0 ; iNode < cellGlobalNodeIds.size() ; iNode++)
        {
          global_size_type = globalId = cellGlobalNodeIds[iNode]
          if( !constraintsVec[0]->isConstrained(globalId))
          {
            size_type localId = basisHandler->globalToLocalIndex(globalId,constraintName) ;
            basisHandler->getBasisCenters(localId,constraintName,nodeLoc);

            *(itField + localId )  = interpolatePolynomial (feDegree, nodeLoc[0], nodeLoc[1], nodeLoc[2],xmin,ymin,zmin);
          }

        }
    }



  // update the ghost values before calling apply Constraints
  // For a serial run, updating ghost values has no effect

  fieldData.updateGhostValues();
  fieldData.applyConstraintsParentToChild();

  // create the quadrature Value Container

  std::shared_ptr<dftefe::quadrarture::QuadratureRule> quadRule =
    std::make_shared<dftefe::quadrarture::QuadratureRuleGauss>(dim, num1DGaussSize);

  dftefe::basis::LinearCellMappingDealii<dim> linearCellMappingDealii;
  dftefe::quadrarture::QuadratureRuleContainer quadRuleContainer( quadRule, triangulationBase,
                                                                 linearCellMappingDealii);

  dftefe::quadrarture::QuadratureValuesContainer<double, dftefe::utils::MemorySpace::HOST> quadValuesContainer(quadRuleContainer, 1);


  // Interpolate the nodal data to the quad points
  feBasisOp.interpolate( fieldData, quadAttr[0], quadValuesContainer);


  std::vector<dftefe::utils::Point> & locQuadPoints = quadRuleContainer.getRealPoints();

  bool testPass = true;
  for( auto it  = quadValuesContainer.begin() ; it != quadValuesContainer.end() ; it++ )
    {
      size_type jQuad = it->first;
      double xLoc = locQuadPoints[jQuad][0];
      double yLoc = locQuadPoints[jQuad][1];
      double zLoc = locQuadPoints[jQuad][2];

      analyticValue = interpolatePolynomial (feDegree, xLoc, yLoc, zLoc,xmin,ymin,zmin);

      if ( std::abs((*it) - analyticValue) > 1e-8 )
        testPass = false;

    }

  std::cout<<" test status = "<<testPass<<"\n";
  return testPass;
}