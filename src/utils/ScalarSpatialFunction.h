#ifndef dftefeScalarSpatialFunction_h
#define dftefeScalarSpatialFunction_h
#include "Point.h"
#include "Function.h"
#include <complex>
namespace dftefe
{
  namespace utils
  {
    template <typename T>
    using ScalarSpatialFunction = Function<utils::Point, T>;

    using ScalarSpatialFunctionReal = Function<utils::Point, double>;

    using ScalarSpatialFunctionComplex =
      Function<utils::Point, std::complex<double>>;

  } // end of namespace utils

} // end of namespace dftefe
#endif // dftefeScalarSpatialFunction_h
