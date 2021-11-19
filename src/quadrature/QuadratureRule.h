#ifndef dftefeQuadratureRule_h
#define dftefeQuadratureRule_h

#include <utils/Point.h>
#include <utils/TypeConfig.h>
#include <vector>

namespace dftefe
{
  namespace quadrature
  {
    class QuadratureRule
    {
    public:
      QuadratureRule(const unsigned int               dim,
                     const std::vector<utils::Point> &points,
                     const std::vector<double> &      weights);

      virtual const std::vector<utils::Point> &
      getPoints() const;

      virtual const std::vector<utils::Point> &
      get1DPoints() const;

      virtual const std::vector<double> &
      getWeights() const;

      virtual const std::vector<double> &
      get1DWeights() const;

      virtual bool
      isTensorStructured() const;

      size_type
      nPoints() const;

      size_type
      n1DPoints() const;

      unsigned int
      getDim() const;

    protected:
      QuadratureRule();

      unsigned int              d_dim;
      size_type                 d_numPoints;
      size_type                 d_num1DPoints;
      std::vector<utils::Point> d_points;
      std::vector<utils::Point> d_1DPoints;
      std::vector<double>       d_weights;
      std::vector<double>       d_1DWeights;
      bool                      d_isTensorStructured;
    };

  } // end of namespace quadrature

} // end of namespace dftefe

#endif // dftefeQuadratureRule_h
