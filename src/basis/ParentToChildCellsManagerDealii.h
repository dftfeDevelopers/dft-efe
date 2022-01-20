#ifndef dftefeParentToChildCellsManagerDealii_h
#define dftefeParentToChildCellsManagerDealii_h

#include <utils/TypeConfig.h>
#include "ParentToChildCellsManagerBase.h"
#include "TriangulationCellBase.h"
#include "TriangulationDealiiSerial.h"
#include <vector>
#include <memory>
namespace dftefe
{
  namespace basis
  {
    template <unsigned int dim>
    class ParentToChildCellsManagerDealii : public ParentToChildCellsManagerBase
    {
    public:
      ParentToChildCellsManagerDealii();
      ~ParentToChildCellsManagerDealii();

      std::vector<std::shared_ptr<const TriangulationCellBase>>
      createChildCells(const TriangulationCellBase &parentCell) override;

      void
      popLast() override;

    private:
      std::vector<std::shared_ptr<TriangulationDealiiSerial<dim>>>
        d_triangulationDealiiSerialVector;
    };

  } // end of namespace basis

} // end of namespace dftefe
#include "ParentToChildCellsManagerDealii.t.cpp"
#endif // dftefeParentToChildCellsManagerDealii_h
