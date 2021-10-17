#ifndef dftefeDeviceDataTypeOverloads_cuh
#define dftefeDeviceDataTypeOverloads_cuh

#include <complex>
#include <cuComplex.h>
namespace dftefe
{
  namespace utils
  {
    __inline__ __device__ double
    mult(double a, double b)
    {
      return a * b;
    }

    __inline__ __device__ float
    mult(float a, float b)
    {
      return a * b;
    }

    __inline__ __device__ cuDoubleComplex
    mult(cuDoubleComplex a, cuDoubleComplex b)
    {
      return cuCmul(a, b);
    }

    __inline__ __device__ cuFloatComplex
    mult(cuFloatComplex a, cuFloatComplex b)
    {
      return cuCmulf(a, b);
    }

    __inline__ __device__ double
    add(double a, double b)
    {
      return a + b;
    }

    __inline__ __device__ float
    add(float a, float b)
    {
      return a + b;
    }

    __inline__ __device__ cuDoubleComplex
    add(cuDoubleComplex a, cuDoubleComplex b)
    {
      return cuCadd(a, b);
    }


    __inline__ __device__ cuFloatComplex
    add(cuFloatComplex a, cuFloatComplex b)
    {
      return cuCaddf(a, b);
    }

    __inline__ __device__ cuDoubleComplex
    mult(double a, cuDoubleComplex b)
    {
      return make_cuDoubleComplex(a * b.x, a * b.y);
    }

    __inline__ __device__ cuDoubleComplex
    mult(cuDoubleComplex a, double b)
    {
      return make_cuDoubleComplex(b * a.x, b * a.y);
    }

    __inline__ __device__ cuFloatComplex
    mult(double a, cuFloatComplex b)
    {
      return make_cuFloatComplex(a * b.x, a * b.y);
    }

    __inline__ __device__ cuFloatComplex
    mult(cuFloatComplex a, double b)
    {
      return make_cuFloatComplex(b * a.x, b * a.y);
    }

    inline double *
    makeDataTypeDeviceCompatible(double *a)
    {
      return reinterpret_cast<double *>(a);
    }

    inline const double *
    makeDataTypeDeviceCompatible(const double *a)
    {
      return reinterpret_cast<const double *>(a);
    }

    inline float *
    makeDataTypeDeviceCompatible(float *a)
    {
      return reinterpret_cast<float *>(a);
    }

    inline const float *
    makeDataTypeDeviceCompatible(const float *a)
    {
      return reinterpret_cast<const float *>(a);
    }

    inline cuDoubleComplex *
    makeDataTypeDeviceCompatible(std::complex<double> *a)
    {
      return reinterpret_cast<cuDoubleComplex *>(a);
    }

    inline const cuDoubleComplex *
    makeDataTypeDeviceCompatible(const std::complex<double> *a)
    {
      return reinterpret_cast<const cuDoubleComplex *>(a);
    }

    inline cuFloatComplex *
    makeDataTypeDeviceCompatible(std::complex<float> *a)
    {
      return reinterpret_cast<cuFloatComplex *>(a);
    }

    inline const cuFloatComplex *
    makeDataTypeDeviceCompatible(const std::complex<float> *a)
    {
      return reinterpret_cast<const cuFloatComplex *>(a);
    }
  } // namespace utils

} // namespace dftefe

#endif
