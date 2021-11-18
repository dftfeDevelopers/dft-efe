#include <utils/Exceptions.h>
#include "QuadratureRule.h"

namespace dftefe {

  namespace quadrature {


	QuadratureRule:: QuadratureRule()
	{

	}

	QuadratureRule:: QuadratureRule(const unsigned int dim,
	  const std::vector<Point> & points,
	  const std::vector<double> & weights):
	  d_dim(dim),
	  d_isTensorStructured(false),
	  d_points(points),
	  d_weights(weights),
	  d_numPoints(points.size())
	{

	  utils::throwException(d_numPoints > 0,
		  "Empty points passed to create a quadratureRule");
	  utils::throwException(d_points.size() == d_weights.size(),
		  "The number of points and the associated weights are of different sizes");
	  utils::throwException(d_dim == d_points[0].getDim(),
		  "The dimension of the quadrature rule and the quadrature points do not match");

	}

    const std::vector<Point> &
	QuadratureRule::getPoints() const
	{
	  return d_points;
	}

      const std::vector<Point> &
	QuadratureRule::get1DPoints() const
	{
	  utils::throwException(d_isTensorStructured,
		  "No notion of 1D quad points for non-tensor structure quadrature rule");
	  utils::throwException(d_num1DPoints > 0,
		  "Empty 1D points in the tensor structured quadrature rule. Perhaps it"
		  "has not been constructed properly via a derived class of QuadratureRule");
	  return d_1DPoints;
	}

    const std::vector<double> &
	QuadratureRule::getWeights() const
	{

	  return d_weights;
	}

      const std::vector<double> &
	QuadratureRule::get1DWeights() const
	{
	  utils::throwException(d_isTensorStructured,
		  "No notion of 1D quad points for non-tensor structure quadrature rule");
	  utils::throwException(d_num1DPoints > 0,
		  "Empty 1D points in the tensor structured quadrature rule. Perhaps it"
		  "has not been constructed properly via a derived class of QuadratureRule");

	  return d_1DWeights;

	}

      bool 
	QuadratureRule::isTensorStructured() const
	{

	  return d_isTensorStructured;
	}

	size_type
	QuadratureRule::nPoints() const
	{

	  return d_numPoints;

	}

	size_type
	QuadratureRule::n1DPoints() const
	{

	  return d_num1DPoints;

	}

  } // end of namespace quadrature


} // end of namespace dftefe
