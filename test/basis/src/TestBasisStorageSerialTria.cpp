#include <basis/TriangulationBase.h>
#include <basis/TriangulationDealiiSerial.h>
#include <basis/CellMappingBase.h>
#include <basis/LinearCellMappingDealii.h>
#include <basis/CFEBasisDofHandlerDealii.h>
#include <basis/CFEConstraintsLocalDealii.h>
#include <basis/CFEBasisDataStorageDealii.h>
#include <quadrature/QuadratureAttributes.h>
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

int main()
{
  const unsigned int dim = 3;
  std::shared_ptr<dftefe::basis::TriangulationBase> triangulationBase =
    std::make_shared<dftefe::basis::TriangulationDealiiSerial<dim>>();
  std::vector<unsigned int>         subdivisions = {1, 1, 1};
  std::vector<bool>                 isPeriodicFlags(dim, false);
  std::vector<dftefe::utils::Point> domainVectors(dim,
                                                  dftefe::utils::Point(dim, 0.0));
  domainVectors[0][0] = 10.0;
  domainVectors[1][1] = 10.0;
  domainVectors[2][2] = 10.0;
  triangulationBase->initializeTriangulationConstruction();
  triangulationBase->createUniformParallelepiped(subdivisions,
                                                 domainVectors,
                                                 isPeriodicFlags);
  triangulationBase->finalizeTriangulationConstruction();

 unsigned int feDegree = 2;

  std::shared_ptr<const dftefe::basis::FEBasisDofHandler<double, dftefe::utils::MemorySpace::HOST,dim>> dofHandler =  
   std::make_shared<dftefe::basis::CFEBasisDofHandlerDealii<double, dftefe::utils::MemorySpace::HOST,dim>>(triangulationBase, feDegree);

  unsigned int num1DGaussSize =4;

  dftefe::quadrature::QuadratureRuleAttributes quadAttr = 
            dftefe::quadrature::QuadratureRuleAttributes(dftefe::quadrature::QuadratureFamily::GAUSS,true,num1DGaussSize);


  dftefe::basis::BasisStorageAttributesBoolMap basisAttrMap;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreValues] = true;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreGradient] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreHessian] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreOverlap] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreGradNiGradNj] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreJxW] = false;

  // Set up the FE Basis Data Storage

  dftefe::basis::CFEBasisDataStorageDealii<double, double, dftefe::utils::MemorySpace::HOST,dim> 
    feBasisData (dofHandler, quadAttr, basisAttrMap);

  // evaluate basis data
    feBasisData.evaluateBasisData(quadAttr, basisAttrMap);

  const dftefe::utils::MemoryStorage<double, dftefe::utils::MemorySpace::HOST> &shapeFuncValue =
    feBasisData.getBasisDataInAllCells();

  auto it = shapeFuncValue.begin();
 for (auto it = shapeFuncValue.begin(); it != shapeFuncValue.end(); it++)
   {
     std::cout<<" value = "<<*it<<"\n";
   }



  // Dealii traingulations

  dealii::Point<dim> vector1(domainVectors[0][0],
                             domainVectors[0][1],
                             domainVectors[0][2]);
  dealii::Point<dim> vector2(domainVectors[1][0],
                             domainVectors[1][1],
                             domainVectors[1][2]);
  dealii::Point<dim> vector3(domainVectors[2][0],
                             domainVectors[2][1],
                             domainVectors[2][2]);

  unsigned int dealiiSubDivisions[dim];
  dealiiSubDivisions[0] = 1.0;
  dealiiSubDivisions[1] = 1.0;
  dealiiSubDivisions[2] = 1.0;

  //
  // Generate coarse mesh
  //
  dealii::Point<dim> basisVectors[dim] = {vector1, vector2, vector3};
  dealii::Triangulation<dim> dealiiTria;
  dealiiTria.clear();
  dealii::GridGenerator::subdivided_parallelepiped<dim>(
    dealiiTria, dealiiSubDivisions, basisVectors);

  dealii::DoFHandler<dim> dealiiDofHandler(dealiiTria);
  const dealii::FE_Q<dim> dealiiFE(feDegree);
  dealiiDofHandler.distribute_dofs(dealiiFE);

  dealii::QGauss<dim> quadrature_formula(num1DGaussSize);
  dealii::FEValues<dim> dealiiFeVal(dealiiFE,
                        quadrature_formula,
                        dealii::update_values | dealii::update_JxW_values);


  const unsigned int dofs_per_cell = dealiiFE.n_dofs_per_cell();
  const unsigned int quadSize = quadrature_formula.size();

  unsigned int iCell = 0;

  bool testPass = true;
  auto cell = dealiiDofHandler.begin_active();
  dealiiFeVal.reinit(cell);

  for ( unsigned int iNode = 0 ; iNode < dofs_per_cell ; iNode++)
    {
      for ( unsigned int jQuad = 0; jQuad < quadSize ; jQuad ++)
        {
          if ( std::abs(dealiiFeVal.shape_value(iNode, jQuad) - *it) > 1e-6 )
            {
              testPass  = false  ;
            }
          it++;
        }
    }

  std::cout<<" test status = "<<testPass<<"\n";
  return testPass;
}