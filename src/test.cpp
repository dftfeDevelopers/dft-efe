#include <basis/TriangulationBase.h>
#include <basis/TriangulationDealiiSerial.h>
#include <basis/ParentToChildCellsManagerBase.h>
#include <basis/ParentToChildCellsManagerDealii.h>
#include <utils/Point.h>
#include <vector>
#include <memory>

void
printVertices(const std::vector<dftefe::utils::Point> &points)
{
  const unsigned int N = points.size();
  for (unsigned int i = 0; i < N; ++i)
    std::cout << points[i][0] << "\t" << points[i][1] << "\t" << points[i][2]
              << std::endl;
}

int
main()
{
  dftefe::basis::TriangulationBase *triangulationBase =
    new dftefe::basis::TriangulationDealiiSerial<3>();

  std::vector<unsigned int>         subdivisions = {2, 2, 2};
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
  std::cout << triangulationBase->nLocalCells() << std::endl;

  dftefe::basis::TriangulationBase::cellIterator it =
    triangulationBase->beginLocal();
  dftefe::basis::ParentToChildCellsManagerBase *parentToChildCellsManager =
    new dftefe::basis::ParentToChildCellsManagerDealii<3>();
  unsigned int iCell = 0;
  for (; it != triangulationBase->endLocal(); ++it)
    {
      std::vector<dftefe::utils::Point> points(0, dftefe::utils::Point(3, 0.0));
      (*it)->getVertices(points);
      printVertices(points);

      std::vector<std::shared_ptr<const dftefe::basis::TriangulationCellBase>>
        childCells = parentToChildCellsManager->createChildCells(*(*it));

      std::cout << "Printing for iCell: " << iCell << std::endl;
      std::cout << "==========================================================="
                << std::endl;
      std::cout << "Number child cells: " << childCells.size() << std::endl;
      for (unsigned int iChild = 0; iChild < childCells.size(); ++iChild)
        {
          std::vector<dftefe::utils::Point> points(0,
                                                   dftefe::utils::Point(3,
                                                                        0.0));
          childCells[iChild]->getVertices(points);
          std::cout << "Printing for iChild: " << iChild << std::endl;
          std::cout
            << "------------------------------------------------------------"
            << std::endl;
          printVertices(points);
        }

      parentToChildCellsManager->popLast();
      iCell++;
    }

  delete parentToChildCellsManager;
  delete triangulationBase;
}
