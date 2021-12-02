#ifndef dftefeParentToChildCellsDealii_h
#define dftefeParentToChildCellsDealii_h

#include <utils/TypeConfig.h>
#include "ParentToChildCellsBase.h"
#include "TriangulationCellBase.h"
#include "TriangulationDealiiSerial.h"
#include <vector>
#include <memory>
namespace dftefe
{
  namespace basis
  {
    template <unsigned int dim>
    class ParentToChildCellsDealii : public ParentToChildCellsBase
    {
    public:
      ParentToChildCellsDealii();
      ~ParentToChildCellsDealii();

      std::vector<std::shared_ptr<const TriangulationCellBase>>
      getChildCells(const TriangulationCellBase &parentCell) override;

      void
      cleanup() override;

    private:
      TriangulationDealiiSerial<dim> d_triangulationDealiiSerial;
    };

  } // end of namespace basis

} // end of namespace dftefe
#endif // dftefeParentToChildCellsDealii_h
