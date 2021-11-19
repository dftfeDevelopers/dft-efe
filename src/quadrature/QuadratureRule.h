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
      QuadratureRule(const unsigned int                       dim,
                     const std::vector<dftefe::utils::Point> &points,
                     const std::vector<double> &              weights);

      virtual const std::vector<dftefe::utils::Point> &
      getPoints() const;

      virtual const std::vector<dftefe::utils::Point> &
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

    protected:
      QuadratureRule();

      unsigned int                      d_dim;
      unsigned int                      d_numPoints;
      unsigned int                      d_num1DPoints;
      std::vector<dftefe::utils::Point> d_points;
      std::vector<dftefe::utils::Point> d_1DPoints;
      std::vector<double>               d_weights;
      std::vector<double>               d_1DWeights;
      bool                              d_isTensorStructured;
    };

  } // end of namespace quadrature

} // end of namespace dftefe

#endif // dftefeQuadratureRule_h
