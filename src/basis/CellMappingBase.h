#ifndef dftefeCellMappingBase_h
#define dftefeCellMappingBase_h

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
      getParametricPoint(const utils::Point &realPoint,
                         const TrialCellBase &        triaCellBase,
                         utils::Point &parametricPoint) const = 0;

      virtual void
      getRealPoint(const utils::Point &parametricPoint,
                   const TriaCellBase &         triaCellBase,
                   utils::Point &realPoint) const = 0;

    }; // end of class CellMappingBase


  } // namespace basis

} // namespace dftefe

#endif
