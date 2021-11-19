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
    class TriangulationCellBase;

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
    public:
      virtual void
      getJxW(const TriangulationCellBase &            triaCellBase,
             const std::vector<dftefe::utils::Point> &paramPoints,
             const std::vector<double> &              weights,
             std::vector<double> &                    valuesJxW) const = 0;

      virtual void
      getParametricPoint(const dftefe::utils::Point & realPoint,
                         const TriangulationCellBase &triaCellBase,
                         dftefe::utils::Point &       parametricPoint,
                         bool &                       isPointInside) const = 0;

      virtual void
      getParametricPoints(const std::vector<dftefe::utils::Point> &realPoints,
                          const TriangulationCellBase &            triaCellBase,
                          std::vector<utils::Point> &parametricPoints,
                          std::vector<bool> &        arePointsInside) const = 0;

      virtual void
      getRealPoint(const dftefe::utils::Point & parametricPoint,
                   const TriangulationCellBase &triaCellBase,
                   dftefe::utils::Point &       realPoint) const = 0;


      virtual void
      getRealPoints(const std::vector<dftefe::utils::Point> &parametricPoints,
                    const TriangulationCellBase &            triaCellBase,
                    std::vector<dftefe::utils::Point> &realPoints) const = 0;

    }; // end of class CellMappingBase


  } // namespace basis

} // namespace dftefe

#endif
