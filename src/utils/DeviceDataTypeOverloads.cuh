#ifndef dftefeDeviceDataTypeOverloads_cuh
#define dftefeDeviceDataTypeOverloads_cuh

#include <complex>
#include <cuComplex.h>
namespace dftefe
{
  namespace utils
  {
    template <typename NumType>
    __device__ NumType
    mult(NumType x, NumType y);

    inline double *
    makeDataTypeGPUCompatible(double *x)
    {
      return reinterpret_cast<double *>(x);
    }

    inline const double *
    makeDataTypeGPUCompatible(const double *x)
    {
      return reinterpret_cast<const double *>(x);
    }

    inline float *
    makeDataTypeGPUCompatible(float *x)
    {
      return reinterpret_cast<float *>(x);
    }

    inline const float *
    makeDataTypeGPUCompatible(const float *x)
    {
      return reinterpret_cast<const float *>(x);
    }

    inline cuDoubleComplex *
    makeDataTypeGPUCompatible(std::complex<double> *x)
    {
      return reinterpret_cast<cuDoubleComplex *>(x);
    }

    inline const cuDoubleComplex *
    makeDataTypeGPUCompatible(const std::complex<double> *x)
    {
      return reinterpret_cast<const cuDoubleComplex *>(x);
    }

    inline cuFloatComplex *
    makeDataTypeGPUCompatible(std::complex<float> *x)
    {
      return reinterpret_cast<cuFloatComplex *>(x);
    }

    inline const cuFloatComplex *
    makeDataTypeGPUCompatible(const std::complex<float> *x)
    {
      return reinterpret_cast<const cuFloatComplex *>(x);
    }
  } // namespace utils

} // namespace dftefe

#endif
