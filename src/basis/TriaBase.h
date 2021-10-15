#ifndef dftefeTriaBase_h
#define dftefeTriaBase_h

#include <utils/Point.h>
#include <utils/TypeConfig.h>
#include "FECellLinearMapping.h"

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
class TriaBase
{
public:
    TriaBase();
  ~TriaBase();

  virtual std::vector<std::shared_ptr<Point>>
  getVertices() const = 0;

  virtual std::shared_ptr<Point>
  getVertex(size_type i) const = 0;

  virtual size_type
  getId() const = 0;

  virtual bool
  isPointInside(std::shared_ptr<const Point> point) const = 0;

  virtual bool
  isAtBoundary(const unsigned int i) const = 0;

  virtual bool
  isAtBoundary() const = 0;

  virtual void
  setRefineFlag() = 0;

  virtual void
  clearRefineFlag() = 0;

  virtual void
  setCoarsenFlag() = 0;

  virtual void
  clearCoarsenFlag() = 0;

  virtual bool
  isActive() const = 0;

  virtual bool
  isLocallyOwned() const = 0;

  virtual bool
  isGhost() const = 0;

  virtual bool
  isArtificial() const = 0;

  virtual int
  getDim() const = 0;

  virtual std::shared_ptr<Point>
  getParametricPoint(std::shared_ptr<const Point> realPoint,
                     const FECellMapping &          feCellMapping) const = 0;

  virtual std::shared_ptr<Point>
  getRealPoint(std::shared_ptr<const Point> parametricPoint,
               const FECellMapping &          feCellMapping) const = 0;

}; // end of class TriaBase
}// end of namespace basis

} // end of namespace dftefe
#endif // dftefeTriaBase_h
