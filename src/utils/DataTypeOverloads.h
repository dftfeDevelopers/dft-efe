#ifndef dftefeDataTypeOverloads_h
#define dftefeDataTypeOverloads_h

#include <complex>
#include <algorithm>
#include <utils/TypeConfig.h>
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

    // Get real part

    template <typename T>
    inline typename RealType<T>::Type
    realPart(const T &x)
    {
      throwException(false, "realPart() not implemented for datatype.");
      return 0;
    }

    template <>
    inline RealType<float>::Type
    realPart(const float &x)
    {
      return x;
    }

    template <>
    inline RealType<double>::Type
    realPart(const double &x)
    {
      return x;
    }

    template <>
    inline RealType<std::complex<float>>::Type
    realPart(const std::complex<float> &x)
    {
      return x.real();
    }

    template <>
    inline RealType<std::complex<double>>::Type
    realPart(const std::complex<double> &x)
    {
      return x.real();
    }

    // Get imaginary part

    template <typename T>
    inline typename RealType<T>::Type
    imagPart(const T &x)
    {
      throwException(false, "imagPart() not implemented for datatype.");
      return 0;
    }

    template <>
    inline RealType<float>::Type
    imagPart(const float &x)
    {
      return x;
    }

    template <>
    inline RealType<double>::Type
    imagPart(const double &x)
    {
      return x;
    }

    template <>
    inline RealType<std::complex<float>>::Type
    imagPart(const std::complex<float> &x)
    {
      return x.imag();
    }

    template <>
    inline RealType<std::complex<double>>::Type
    imagPart(const std::complex<double> &x)
    {
      return x.imag();
    }

    // Get the complex conjugate

    template <typename T>
    inline T
    conjugate(const T &x)
    {
      throwException(false, "conjugate() not implemented for datatype.");
      return 0;
    }

    template <>
    inline float
    conjugate(const float &x)
    {
      return x;
    }

    template <>
    inline double
    conjugate(const double &x)
    {
      return x;
    }

    template <>
    inline std::complex<float>
    conjugate(const std::complex<float> &x)
    {
      return std::conj(x);
    }

    template <>
    inline std::complex<double>
    conjugate(const std::complex<double> &x)
    {
      return std::conj(x);
    }

  } // namespace utils

} // namespace dftefe

#endif
