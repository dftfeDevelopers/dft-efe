#include <utils/Exceptions.h>
#include <utils/Point.h>
#include "ParentToChildCellsDealii.h"
#include <utils/DealiiConversions.h>
namespace dftefe
{
  namespace basis
  {
    template <unsigned int dim>
    ParentToChildCellsDealii::ParentToChildCellsDealii()
    {}

    template <unsigned int dim>
    ParentToChildCellsDealii::~ParentToChilCellsDealii()
    {
      delete &d_triangulationDealiiSerial;
    }

    template <unsigned int dim>
    std::vector<std::shared_ptr<const TriangulationCellBase>>
    ParentToChildCellsDealii::getChildCells(
      const TriangulationCellBase &parentCell)
    {
      std::vector<utils::Point> vertices(0, Point(dim, 0.0));
      parentCell.getVertices(vertices);
      d_triangulationDealiiSerial.initializeTriangulationConstruction();
      d_triangulationDealiiSerial.createSingleCellTriangulation(vertices);
      d_triangulationDealiiSerial.refineGlobal(1);
      d_triangulationDealiiSerial.finalizeTriangulationConstruction();
      const size_type numberCells = d_triangulationDealiiSerial.nLocalCells();
      std::vector<std::shared_ptr<const TriangulationCellBase>> returnValue(
        numberCells);

      TriangulationCellBase::const_cellIterator cellIter =
        d_triangulationDealiiSerial.beginLocal();
      unsigned int iCell = 0;
      for (; cellIter != d_triangulationDealiiSerial.endLocal(); ++cellIter)
        {
          returnValue[iCell] = *cellIter;
          iCell++;
        }

      return returnValue;
    }
    template <unsigned int dim>
    void
    ParentToChildCellsDealii::cleanup()
    {}

  } // end of namespace basis

} // end of namespace dftefe
