#ifndef dftefeCellQuadratureContainer_h
#define dftefeCellQuadratureContainer_h

#include "QuadratureRule.h"
#include <utils/TypeConfig.h>
#include <utils/Point.h>
#include <basis/TriangulationBase.h>
#include <basis/CellMappingBase.h>
#include <memory>

namespace dftefe {

  namespace quadrature {

    class CellQuadratureContainer {

     CellQuadratureContainer(std::shared_ptr<const QuadratureRule> quadratureRule,
	 std::shared_ptr<const basis::TriangulationBase> triangulation,
	  const basis::CellMappingBase & cellMapping);
     
     CellQuadratureContainer(std::vector<std::shared_ptr<const QuadratureRule> > quadratureRule,
	 std::shared_ptr<const basis::TriangulationBase> triangulation,
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
      std::vector<size_type> nCellQuadPoints;
      std::vector<Points> d_realPoints;
      std::vector<double> d_JxW;
    };
  } // end of namespace quadrature

} // end of namespace dftefe
