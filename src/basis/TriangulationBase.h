#ifndef dftefeTriangulationBase_h
#define dftefeTriangulationBase_h

#include <utils/TypeConfig.h>
#include <utils/Point.h>
#include "TriangulationCellBase.h"
#include <vector>
#include <memory>
namespace dftefe
{
  namespace basis
  {
    /**
     * @brief An abstract class for the triangulation class. The derived class specialises this class to dealii and otehr specialisations if required.
     **/
    class TriangulationBase
    {
    public:
      typedef std::vector<std::shared_ptr<TriangulationCellBase>>::iterator
        cellIterator;
      typedef std::vector<std::shared_ptr<TriangulationCellBase>>::
        const_iterator const_cellIterator;
      virtual void
      initializeTriangulationConstruction() = 0;
      virtual void
      finalizeTriangulationConstruction() = 0;
      virtual void
      createUniformParallelepiped(
        const std::vector<unsigned int> &subdivisions,
        const std::vector<utils::Point> &domainVectors,
        const std::vector<bool> &        isPeriodicFlags) = 0;
      virtual void
      shiftTriangulation(const utils::Point &origin) = 0;
      virtual void
      refineGlobal(const unsigned int times = 1) = 0;
      virtual void
      coarsenGlobal(const unsigned int times = 1) = 0;
      virtual void
      clearUserFlags() = 0;
      virtual void
      executeCoarseningAndRefinement() = 0;
      virtual unsigned int
      nLocalCells() const = 0;
      virtual size_type
      nGlobalCells() const = 0;
      virtual std::vector<size_type>
      getBoundaryIds() const = 0;
      virtual cellIterator
      beginLocal() = 0;
      virtual cellIterator
      endLocal() = 0;
      virtual const_cellIterator
      beginLocal() const = 0;
      virtual const_cellIterator
      endLocal() const = 0;
      virtual unsigned int
      getDim() const = 0;

    }; // end of class TriangulationBase
  }    // end of namespace basis
} // end of namespace dftefe
#endif
