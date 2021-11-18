#include "CellQuadratureContainer.h"
namespace dftefe {

  namespace quadrature {


    CellQuadratureContainer::CellQuadratureContainer(
	std::shared_ptr<const QuadratureRule> quadratureRule,
	 std::shared_ptr<const basis::TriangulationBase> triangulation,
	  const basis::CellMappingBase & cellMapping)
    {
	size_type numCells = triangulation.nLocalCells();
	d_quadratureRuleVec.resize(numCells, quadratureRule);
	unsigned int numQuadPoints = 0;
	for(unsigned int iCell = 0; iCell < numCells; ++iCell)
	{
	  numQuadPoints += d_quadratureRuleVec[iCell].nPoints();
	}

	d_realPoints.resize(numQuadPoints);
	for(unsigned int iCell = 0; iCell < numCells; ++iCell)
	{
	  const std::vector<Point> parametricPoints = 
	    d_quadratureRuleVec[iCell].getPoints();
	  std::vector<Point> realPoints(0);
	  cellMapp

	}
    }

  } // end of namespace quadrature
} // end of namespace dftefe
