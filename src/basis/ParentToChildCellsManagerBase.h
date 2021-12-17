#ifndef dftefeParentToChildCellsManagerBase_h
#define dftefeParentToChildCellsManagerBase_h

#include <utils/TypeConfig.h>
#include "TriangulationCellBase.h"
#include <vector>
#include <memory>
namespace dftefe
{
  namespace basis
  {
    class ParentToChildCellsManagerBase
    {
    public:
      virtual ~ParentToChildCellsManagerBase() = default;
      virtual std::vector<std::shared_ptr<const TriangulationCellBase>>
      createChildCells(const TriangulationCellBase &parentCell) = 0;

      virtual void
      popLast() = 0;
    };

  } // end of namespace basis

} // end of namespace dftefe
#endif // dftefeParentToChildCellsManagerBase_h
