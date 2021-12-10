#include <basis/TriangulationBase.h>
#include <basis/TriangulationDealiiSerial.h>
#include <utils/Point.h>
#include <vector>
#include <basis/LinearCellMappingDealii.h>
#include <basis/CellMappingBase.h>

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

  std::vector<unsigned int>         subdivisions = {1, 1, 1};
  std::vector<bool>                 isPeriodicFlags(3, false);
  std::vector<dftefe::utils::Point> domainVectors(3,
                                                  dftefe::utils::Point(3, 0.0));
  domainVectors[0][0] = 1.0;
  domainVectors[1][1] = 1.0;
  domainVectors[2][2] = 1.0;
  triangulationBase->initializeTriangulationConstruction();
  triangulationBase->createUniformParallelepiped(subdivisions,
                                                 domainVectors,
                                                 isPeriodicFlags);
  triangulationBase->finalizeTriangulationConstruction();
  std::cout << triangulationBase->nLocalCells() << std::endl;
  dftefe::basis::CellMappingBase *mapping =
    new dftefe::basis::LinearCellMappingDealii<3>();


  dftefe::basis::TriangulationBase::cellIterator it =
    triangulationBase->beginLocal();
  for (; it != triangulationBase->endLocal(); ++it)
    {
      std::vector<dftefe::utils::Point> points(0, dftefe::utils::Point(3, 0.0));
      (*it)->getVertices(points);
      printVertices(points);
      std::vector<dftefe::utils::Point> paramPoint(0,
                                                   dftefe::utils::Point(3,
                                                                        0.0));
      bool                              pointInside;
      for (unsigned int i = 0; i < points.size(); i++)
        {
          (*it)->getParametricPoint(points[i], *mapping, paramPoint[i]);
          //          mapping->getParametricPoint(points[i], it ,paramPoint[i],
          //          pointInside );
        }
      printVertices(paramPoint);
    }
}
