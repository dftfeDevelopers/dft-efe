#include "TriangulationBase.h"
#include "TriangulationDealiiSerial.h"
#include <utils/Point.h>
#include <vector>
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
  triangulationBase->createUniformParallelepiped(subdivisions,
                                                 domainVectors,
                                                 isPeriodicFlags);
}
