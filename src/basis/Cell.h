#ifndef dftefeCell_h
#define dftefeCell_h

#include <utils/Point.h>
#include <utils/TypeConfig.h>
#include "CellMapping.h"

#include <memory>
namespace dftefe {

  /**
   * @brief An abstract class for a finite element cell (can be of any dimension)
   * This is created primarily to be a wrapper around deal.ii cells, so as to avoid 
   * the cascading of template parameters. 
  **/
  class Cell {

    public:
      Cell();
      ~Cell();

      virtual 
	std::vector<std::shared_ptr<Point> > 
	getVertices() const = 0;

      virtual
	std::vector<std::shared_ptr<Point> > 
	getNodalPoints() const = 0;

      virtual
	std::shared_ptr<Point> 
	getVertex(size_type i) const = 0;

      virtual
	size_type
	getId() const = 0;

      virtual
	bool
	isPointInside(std::shared_ptr<const Point> point) const = 0;

      virtual 
	bool 
	isAtBoundary(const unsigned int i) const = 0;

      virtual
	bool 
	isAtBoundary() const = 0;
	
      virtual
	void
	setRefineFlag() = 0;

      virtual
	void
	clearRefineFlag() = 0;

      virtual
	void
	setCoarsenFlag() = 0;

      virtual
	void
	clearCoarsenFlag() = 0;

      virtual 
	bool
	isActive() const  = 0;

      virtual 
	bool
	isLocallyOwned() const  = 0;

      virtual 
	bool
	isGhost() const  = 0;

      virtual 
	bool
	isArtificial() const  = 0;

      virtual
	int
	getDim() const  = 0;

      virtual 
	std::shared_ptr<Point>
	getParametricPoint(std::shared_ptr<const Point> realPoint,
	    		  const CellMapping & cellMapping) const = 0;

      virtual 
	std::shared_ptr<Point>
	getRealPoint(std::shared_ptr<const Point> parametricPoint, 
	    	     const CellMapping & cellMapping) const = 0;

  }; // end of class Cell

} // end of namespace dftefe
#endif // dftefeCell_h
