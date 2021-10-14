#ifndef dftefeDealiiFE_h
#define dftefeDealiiFE_h

#include <utils/Point.h>
#include <utils/TypeConfig.h>
#include "FECellLinearMapping.h"
#include "FEBase.h"

#include <memory>
namespace dftefe
{
 namespace basis
{
 /**
  * @brief An derived class that implements the interface for the dealii's cell class
  *  This class is derived from FECell. This is done so that the template is not propagated all across the code.
  **/
 tempate<dim>
 class DealiiFE : public FEBase
 {
 public:
     DealiiFE();
   ~DealiiFE();

   std::vector<std::shared_ptr<utils::Point>>
   getVertices() const;

   std::vector<std::shared_ptr<utils::Point>>
   getNodalPoints() const ;

   std::shared_ptr<Point>
   getVertex(size_type i) const ;

   size_type
   getId() const ;

   bool
   isPointInside(std::shared_ptr<const utils::Point> point) const;

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
   getParametricPoint(std::shared_ptr<const utils::Point> realPoint,
                      const FECellMapping &          feCellMapping) const ;

   std::shared_ptr<Point>
   getRealPoint(std::shared_ptr<const utils::Point> parametricPoint,
                const FECellMapping &          feCellMapping) const ;
     
 private:
     dealii::

 }; // end of class DealiiFE

 }//end of namespace basis
} // end of namespace dftefe
#endif // dftefeDealiiFE_h
