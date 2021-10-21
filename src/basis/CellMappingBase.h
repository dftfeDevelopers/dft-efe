#ifndef dftefeCellMappingBase_h
#define dftefeCellMappingBase_h

namespace dftefe {

  namespace basis {

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
    class CellMappingBase {

      CellMappingBase();
      virtual ~CellMappingBase();

      virtual std::shared_ptr<Point>
	getParametricPoint(std::shared_ptr<const Point> realPoint,
	    const TrialCellBase & triaCellBase) const = 0;

      virtual std::shared_ptr<Point>
	getRealPoint(std::shared_ptr<const Point> parametricPoint,
	    const TriaCellBase & triaCellBase ) const = 0;

    }; // end of class CellMappingBase 


  } // end of basis namespace 

} // end of dftefe namespace 

#endif
