#include <basis/TriangulationBase.h>
#include <basis/TriangulationDealiiSerial.h>
#include <utils/Point.h>
#include <vector>

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

  std::vector<unsigned int>         subdivisions = {5, 5, 5};
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
  for (; it != triangulationBase->endLocal(); ++it)
    {
      std::vector<dftefe::utils::Point> points(0, dftefe::utils::Point(3, 0.0));
      (*it)->getVertices(points);
      printVertices(points);
    }
}
