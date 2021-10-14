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
    mult(NumType a, NumType b);

    __device__ cuDoubleComplex
    mult(double a, cuDoubleComplex b);

    __device__ cuDoubleComplex
    mult(cuDoubleComplex a, double b);

    __device__ cuFloatComplex
    mult(double a, cuFloatComplex b);

    __device__ cuFloatComplex
    mult(cuFloatComplex a, double b);

    template <typename NumType>
    __device__ NumType
    add(NumType a, NumType b);

    inline double *
    makeDataTypeGPUCompatible(double *a)
    {
      return reinterpret_cast<double *>(a);
    }

    inline const double *
    makeDataTypeGPUCompatible(const double *a)
    {
      return reinterpret_cast<const double *>(a);
    }

    inline float *
    makeDataTypeGPUCompatible(float *a)
    {
      return reinterpret_cast<float *>(a);
    }

    inline const float *
    makeDataTypeGPUCompatible(const float *a)
    {
      return reinterpret_cast<const float *>(a);
    }

    inline cuDoubleComplex *
    makeDataTypeGPUCompatible(std::complex<double> *a)
    {
      return reinterpret_cast<cuDoubleComplex *>(a);
    }

    inline const cuDoubleComplex *
    makeDataTypeGPUCompatible(const std::complex<double> *a)
    {
      return reinterpret_cast<const cuDoubleComplex *>(a);
    }

    inline cuFloatComplex *
    makeDataTypeGPUCompatible(std::complex<float> *a)
    {
      return reinterpret_cast<cuFloatComplex *>(a);
    }

    inline const cuFloatComplex *
    makeDataTypeGPUCompatible(const std::complex<float> *a)
    {
      return reinterpret_cast<const cuFloatComplex *>(a);
    }
  } // namespace utils

} // namespace dftefe

#endif
