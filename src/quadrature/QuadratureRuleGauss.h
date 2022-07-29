#ifndef dftefeQuadratureRuleGauss_h
#define dftefeQuadratureRuleGauss_h

#include <quadrature/QuadratureRule.h>

namespace dftefe
{
  namespace quadrature
  {
    class QuadratureRuleGauss : public QuadratureRule
    {
    public:
      QuadratureRuleGauss(const size_type dim, const size_type order1D);
    };
  } // end of namespace quadrature

} // end of namespace dftefe


#endif // dftefeQuadratureRuleGauss_h
