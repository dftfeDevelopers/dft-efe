#ifndef dftefeParentToChildCellsBase_h
#define dftefeParentToChildCellsBase_h

#include <utils/TypeConfig.h>
#include "TriangulationCellBase.h"
#include <vector>
#include <memory>
namespace dftefe
{
  namespace basis
  {
    class ParentToChildCellsBase
    {
    public:
      virtual std::vector<std::shared_ptr<const TriangulationCellBase>>
      getChildCells(const TriangulationCellBase &parentCell) = 0;

      virtual void
      cleanup() = 0;
    };

  } // end of namespace basis

} // end of namespace dftefe
#endif // dftefeParentToChildCellsBase_h
