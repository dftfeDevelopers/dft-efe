#ifndef dftefeTria_h
#define dftefeTria_h

#include <utils/Point.h>
#include <utils/TypeConfig.h>
#include "FECellLinearMapping.h"
#include "TriaBase.h"

#include <memory>
namespace dftefe
{
namespace basis
{

/**
 * @brief An abstract class to interface the Triangulation cell class of dealii
 *  This is pure virtual class. the derived class of this provides the function implementation.
 *  This is done to prevent the template to propagate all across teh code
 **/
  template< unsigned int dim>
class Tria : public TriaBase
{
public:
    Tria();
  ~Tria();

  std::vector<std::shared_ptr<Point>>
  getVertices() const ;

  std::shared_ptr<Point>
  getVertex(size_type i) const ;

  size_type
  getId() const ;

  bool
  isPointInside(std::shared_ptr<const Point> point) const ;

  bool
  isAtBoundary(const unsigned int i) const ;

  bool
  isAtBoundary() const ;

  void
  setRefineFlag() ;

  void
  clearRefineFlag() ;

  void
  setCoarsenFlag() ;

  void
  clearCoarsenFlag() ;

  bool
  isActive() const ;

  bool
  isLocallyOwned() const ;

  bool
  isGhost() const ;

  bool
  isArtificial() const ;

  int
  getDim() const ;

  std::shared_ptr<Point>
  getParametricPoint(std::shared_ptr<const Point> realPoint,
                     const FECellMapping &          feCellMapping) const ;

  std::shared_ptr<Point>
  getRealPoint(std::shared_ptr<const Point> parametricPoint,
               const FECellMapping &          feCellMapping) const ;

}; // end of class Tria
}// end of namespace basis

} // end of namespace dftefe
#endif // dftefeTria_h

