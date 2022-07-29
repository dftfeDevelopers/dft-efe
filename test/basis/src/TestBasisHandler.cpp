//#include <mpi.h>
#include <basis/TriangulationBase.h>
#include <basis/TriangulationDealiiSerial.h>
#include <basis/CellMappingBase.h>
#include <basis/LinearCellMappingDealii.h>
#include <utils/Point.h>
#include <utils/TypeConfig.h>
#include <vector>
#include <memory>


int
main()
{
  std::shared_ptr<dftefe::basis::TriangulationBase> triangulationBase =
    std::make_shared<dftefe::basis::TriangulationDealiiSerial<3>>();

  std::vector<unsigned int>         subdivisions = {1, 1, 1};
  std::vector<bool>                 isPeriodicFlags(3, false);
  std::vector<dftefe::utils::Point> domainVectors(3,
                                                  dftefe::utils::Point(3, 0.0));
  domainVectors[0][0] = 10.0;
  domainVectors[1][1] = 10.0;
  domainVectors[2][2] = 10.0;
  triangulationBase->initializeTriangulationConstruction();
  triangulationBase->createUniformParallelepiped(subdivisions,
                                                 domainVectors,
                                                 isPeriodicFlags);
  triangulationBase->finalizeTriangulationConstruction();
  std::cout << triangulationBase->nCells() << std::endl;
  dftefe::basis::TriangulationBase::TriangulationCellIterator it =
    triangulationBase->beginLocal();
  unsigned int iCell = 0;
}
