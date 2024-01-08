#include <utils/Exceptions.h>
#include <utils/Point.h>
namespace dftefe
{
  namespace basis
  {
    template <unsigned int dim>
    ParentToChildCellsManagerDealii<dim>::ParentToChildCellsManagerDealii()
      : d_triangulationDealiiSerialVector(0)
    {}

    template <unsigned int dim>
    ParentToChildCellsManagerDealii<dim>::~ParentToChildCellsManagerDealii()
    {}

    template <unsigned int dim>
    std::vector<std::shared_ptr<const TriangulationCellBase>>
    ParentToChildCellsManagerDealii<dim>::createChildCells(
      const TriangulationCellBase &parentCell)
    {
      std::vector<utils::Point> vertices(0, utils::Point(dim, 0.0));
      parentCell.getVertices(vertices);
      auto triangulationDealiiSerial =
        std::make_shared<TriangulationDealiiSerial<dim>>();
      d_triangulationDealiiSerialVector.push_back(triangulationDealiiSerial);
      triangulationDealiiSerial->initializeTriangulationConstruction();
      triangulationDealiiSerial->createSingleCellTriangulation(vertices);
      triangulationDealiiSerial->refineGlobal(1);
      triangulationDealiiSerial->finalizeTriangulationConstruction();
      const size_type numberCells =
        triangulationDealiiSerial->nLocallyOwnedCells();
      std::vector<std::shared_ptr<const TriangulationCellBase>> returnValue(
        numberCells);

      TriangulationBase::const_TriangulationCellIterator cellIter =
        triangulationDealiiSerial->beginLocal();
      unsigned int iCell = 0;
      for (; cellIter != triangulationDealiiSerial->endLocal(); ++cellIter)
        {
          returnValue[iCell] = *cellIter;
          iCell++;
        }

      return returnValue;
    }

    template <unsigned int dim>
    void
    ParentToChildCellsManagerDealii<dim>::popLast()
    {
      d_triangulationDealiiSerialVector.pop_back();
    }

  } // end of namespace basis

} // end of namespace dftefe
