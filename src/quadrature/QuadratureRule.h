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

      const std::vector<utils::Point> &virtual getPoints() const;

      const std::vector<utils::Point> &virtual get1DPoints() const;

      const std::vector<double> &virtual getWeights() const;

      const std::vector<double> &virtual get1DWeights() const;

      bool virtual isTensorStructured() const;

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
