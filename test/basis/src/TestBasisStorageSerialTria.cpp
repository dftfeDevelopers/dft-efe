#include <basis/TriangulationBase.h>
#include <basis/TriangulationDealiiSerial.h>
#include <basis/CellMappingBase.h>
#include <basis/LinearCellMappingDealii.h>
#include <basis/FEBasisManagerDealii.h>
#include <basis/FEConstraintsDealii.h>
#include <basis/FEBasisDataStorageDealii.h>
#include <quadrature/QuadratureAttributes.h>
#include <utils/Point.h>
#include <utils/TypeConfig.h>
#include <utils/MemorySpaceType.h>
#include <utils/MemoryStorage.h>
#include <vector>
#include <memory>
#include <deal.II/fe/fe_q.h>



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

  std::shared_ptr<dftefe::basis::FEBasisManager> dofHandler =
    std::make_shared<dftefe::basis::FEBasisManagerDealii<dim>>(triangulationBase);

  unsigned int feDegree = 2;

  dofHandler->reinit(triangulationBase, feDegree);

  std::vector<std::shared_ptr<dftefe::basis::FEConstraintsBase<double>>>
    constraintsVec ;
  constraintsVec.resize(1, std::make_shared<dftefe::basis::FEConstraintsDealii<dim,double>>());

  constraintsVec[0]->clear();
  constraintsVec[0]->makeHangingNodeConstraint(dofHandler);
  constraintsVec[0]->setHomogeneousDirichletBC();
  constraintsVec[0]->close();

  std::vector<dftefe::quadrature::QuadratureRuleAttributes> quadAttr(1,
           dftefe::quadrature::QuadratureRuleAttributes(dftefe::quadrature::QuadratureFamily::GAUSS,true,4));


  dftefe::basis::FEBasisDataStorageDealii<double, dftefe::utils::MemorySpace::HOST,dim> feBasisData(dofHandler,
                                                                              constraintsVec,
                                                                              quadAttr,
                                                                              true,
                                                                              false,
                                                                              false,
                                                                              false,
                                                                              false);

  const dftefe::utils::MemoryStorage<double, dftefe::utils::MemorySpace::HOST> &shapeFuncValue =
    feBasisData.getBasisDataInAllCells(quadAttr[0]);

//  for (auto it = shapeFuncValue.begin(); it != shapeFuncValue.end(); it++)
//    {
//      std::cout<<" value = "<<*it<<"\n";
//    }


  






}