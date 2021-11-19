#ifndef dftefeCellMappingBase_h
#define dftefeCellMappingBase_h

#include <vector>
#include <utils/Point.h>
namespace dftefe
{
  namespace basis
  {
    //
    // forward declarations
    //
    class TriaCellBase;

    enum class CellMappingType
    {
      LINEAR
      //
      // can add other mapping types
      // like curvilinear maps
    };

    /**
     * @brief An abstract class to map a real point to parametric point and vice-versa
     */
    class CellMappingBase
    {
      CellMappingBase();
      virtual ~CellMappingBase();

      virtual void
      getParametricPoint(const dftefe::utils::Point &realPoint,
                         const TriaCellBase &        triaCellBase,
                         dftefe::utils::Point &      parametricPoint) const = 0;

      virtual void
      getParametricPoints(
        const std::vector<dftefe::utils::Point> &realPoints,
        const TriaCellBase &                     triaCellBase,
        std::vector<dftefe::utils::Point> &      parametricPoints) const = 0;

      virtual void
      getRealPoint(const dftefe::utils::Point &parametricPoint,
                   const TriaCellBase &        triaCellBase,
                   dftefe::utils::Point &      realPoint) const = 0;


      virtual void
      getRealPoints(const std::vector<dftefe::utils::Point> &parametricPoints,
                    const TriaCellBase &                     triaCellBase,
                    std::vector<dftefe::utils::Point> &realPoints) const = 0;


    }; // end of class CellMappingBase


  } // namespace basis

} // namespace dftefe

#endif
