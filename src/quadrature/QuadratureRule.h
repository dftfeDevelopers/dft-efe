#ifndef dftefeQuadratureRule_h
#define dftefeQuadratureRule_h

#include <utils/Point.h>
#include <utils/TypeConfig.h>
#include <vector>

namespace dftefe {

  namespace quadrature {

    class QuadratureRule 
    {

      public:
	QuadratureRule(const unsigned int dim,
	    const std::vector<Point> & points,
	    const std::vector<double> & weights);

	const std::vector<Point> &
	  virtual getPoints() const;

	const std::vector<Point> &
	  virtual get1DPoints() const;

	const std::vector<double> &
	  virtual getWeights() const ;

	const std::vector<double> &
	  virtual get1DWeights() const;

	bool 
	  virtual isTensorStructured() const;

	size_type
	  nPoints() const;

	size_type
	  n1DPoints() const;

      protected:
	QuadratureRule();

	unsigned int dim;
	unsigned int d_numPoints;
	unsigned int d_num1DPoints;
	std::vector<Point> d_points;
	std::vector<Point> d_1DPoints;
	std::vector<double> d_weights;
	std::vector<double> d_1DWeights;
	bool d_isTensorStructured;

    };

  } // end of namespace quadrature

} // end of namespace dftefe

#endif // dftefeQuadratureRule_h
