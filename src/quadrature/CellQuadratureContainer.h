#ifndef dftefeCellQuadratureContainer_h
#define dftefeCellQuadratureContainer_h

#include "QuadratureRule.h"
#include <utils/Point.h>
#include <basis/CellMappingBase.h>
#include "Memory.h"

namespace dftefe {

  namespace quadrature {

    class CellQuadratureContainer {

     CellQuadratureContainer(std::shared_ptr<const QuadratureRule> quadratureRule,
	 std::shared_ptr<const TriangulationBase> triangulation,
	  const basis::CellMappingBase & cellMapping);
     
     CellQuadratureContainer(std::vector<std::shared_ptr<const QuadratureRule> > quadratureRule,
	 std::shared_ptr<const TriangulationBase> triangulation,
	  const basis::CellMappingBase & cellMapping);

      const std::vector<Point> &
	getRealPoints() const;

      const std::vector<Point> &
	getRealPoints(const unsigned int cellId) const;
      
      const std::vector<Point> &
	getParametricPoints(const unsigned int cellId) const;

      const std::vector<double> &
	getQuadratureWeights(const unsigned int cellId) const;
      
      const std::vector<double> &
	getJxW() const;
      
      const std::vector<double> &
	getJxW(const unsigned int cellId) const;

      const QuadratureRule &
	getQuadratureRule() const ;


      private:
      std::vector<std::shared_ptr<QuadratureRule> > d_quadratureRuleVec;
      std::vector<Points> d_realPoints;
      std::vector<double> d_JxW;
    };
  } // end of namespace quadrature

} // end of namespace dftefe
