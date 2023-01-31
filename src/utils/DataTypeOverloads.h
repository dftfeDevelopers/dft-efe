#ifndef dftefeDataTypeOverloads_h
#define dftefeDataTypeOverloads_h

#include <complex>
#include <algorithm>
namespace dftefe
{
  namespace utils
  {
    inline unsigned int
    abs_(unsigned int a)
    {
      return a;
    }

    inline int
    abs_(int a)
    {
      return std::abs(a);
    }

    inline double
    abs_(double a)
    {
      return std::abs(a);
    }

    inline float
    abs_(float a)
    {
      return std::abs(a);
    }

    inline double
    abs_(std::complex<double> a)
    {
      return std::abs(a);
    }

    inline float
    abs_(std::complex<float> a)
    {
      return std::abs(a);
    }

    inline unsigned int
    absSq(unsigned int a)
    {
      return a * a;
    }

    inline int
    absSq(int a)
    {
      return a * a;
    }

    inline double
    absSq(double a)
    {
      return a * a;
    }

    inline float
    absSq(float a)
    {
      return a * a;
    }

    inline double
    absSq(std::complex<double> a)
    {
      return a.real() * a.real() + a.imag() * a.imag();
    }

    inline float
    absSq(std::complex<float> a)
    {
      return a.real() * a.real() + a.imag() * a.imag();
    }

    template <typename ValueType>
    inline bool
    absCompare(ValueType a, ValueType b)
    {
      return (abs_(a) < abs_(b));
    }

  } // namespace utils

} // namespace dftefe

#endif
