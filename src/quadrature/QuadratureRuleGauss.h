#ifndef dftefeQuadratureRuleGauss_h
#define dftefeQuadratureRuleGauss_h

#include "QuadratureRule.h"

namespace dftefe
{
  namespace quadrature
  {
    class QuadratureRuleGauss : public QuadratureRule
    {
    public:
      QuadratureRuleGauss(const unsigned int dim, const unsigned int order1D);
    };
  } // end of namespace quadrature

} // end of namespace dftefe


#endif // dftefeQuadratureRuleGauss_h
