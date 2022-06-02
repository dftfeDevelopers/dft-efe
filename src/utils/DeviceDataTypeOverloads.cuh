#ifndef dftefeDeviceDataTypeOverloads_cuh
#define dftefeDeviceDataTypeOverloads_cuh

#include <complex>
#include <cuComplex.h>
#include "TypeConfig.h"
namespace dftefe
{
  namespace utils
  {
    // abs obverloads

    __inline__ __device__ double
    abs(double a)
    {
      return fabs(a);
    }

    __inline__ __device__ double
    abs(cuDoubleComplex a)
    {
      return cuCabs(a);
    }

    __inline__ __device__ double
    abs(cuFloatComplex a)
    {
      return cuCabsf(a);
    }

    // mult for homogeneous types e.g. (double, double)
    __inline__ __device__ size_type
    mult(size_type a, size_type b)
    {
      return a * b;
    }

    __inline__ __device__ int
    mult(int a, int b)
    {
      return a * b;
    }

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

    __inline__ __device__ size_type
    add(size_type a, size_type b)
    {
      return a + b;
    }

    __inline__ __device__ int
    add(int a, int b)
    {
      return a + b;
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

    __inline__ __device__ size_type
    sub(size_type a, size_type b)
    {
      return a - b;
    }

    __inline__ __device__ int
    sub(int a, int b)
    {
      return a - b;
    }

    __inline__ __device__ double
    sub(double a, double b)
    {
      return a - b;
    }

    __inline__ __device__ float
    sub(float a, float b)
    {
      return a - b;
    }

    __inline__ __device__ cuDoubleComplex
    sub(cuDoubleComplex a, cuDoubleComplex b)
    {
      return cuCsub(a, b);
    }

    __inline__ __device__ cuFloatComplex
    sub(cuFloatComplex a, cuFloatComplex b)
    {
      return cuCsubf(a, b);
    }

    __inline__ __device__ size_type
    div(size_type a, size_type b)
    {
      return a / b;
    }

    __inline__ __device__ int
    div(int a, int b)
    {
      return a / b;
    }

    __inline__ __device__ double
    div(double a, double b)
    {
      return a / b;
    }

    __inline__ __device__ float
    div(float a, float b)
    {
      return a / b;
    }

    __inline__ __device__ cuDoubleComplex
    div(cuDoubleComplex a, cuDoubleComplex b)
    {
      return cuCdiv(a, b);
    }

    __inline__ __device__ cuFloatComplex
    div(cuFloatComplex a, cuFloatComplex b)
    {
      return cuCdivf(a, b);
    }

    // mult for heterogeneous types e.g. (double, complex)
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

    inline int *
    makeDataTypeDeviceCompatible(int *a)
    {
      return a;
    }

    inline const int *
    makeDataTypeDeviceCompatible(const int *a)
    {
      return a;
    }

    inline size_type *
    makeDataTypeDeviceCompatible(size_type *a)
    {
      return a;
    }

    inline const size_type *
    makeDataTypeDeviceCompatible(const size_type *a)
    {
      return a;
    }

    inline double *
    makeDataTypeDeviceCompatible(double *a)
    {
      return a;
    }

    inline const double *
    makeDataTypeDeviceCompatible(const double *a)
    {
      return a;
    }

    inline float *
    makeDataTypeDeviceCompatible(float *a)
    {
      return a;
    }

    inline const float *
    makeDataTypeDeviceCompatible(const float *a)
    {
      return a;
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

    inline int
    makeDataTypeDeviceCompatible(int a)
    {
      return a;
    }

    inline size_type
    makeDataTypeDeviceCompatible(size_type a)
    {
      return a;
    }

    inline double
    makeDataTypeDeviceCompatible(double a)
    {
      return a;
    }

    inline float
    makeDataTypeDeviceCompatible(float a)
    {
      return a;
    }

    inline cuDoubleComplex
    makeDataTypeDeviceCompatible(std::complex<double> a)
    {
      return make_cuDoubleComplex(a.real(), a.imag());
    }

    inline cuFloatComplex
    makeDataTypeDeviceCompatible(std::complex<float> a)
    {
      return make_cuFloatComplex(a.real(), a.imag());
    }

    __inline__ __device__ void
    setRealValue(cuFloatComplex *a, double value )
    {
      *a =  make_cuFloatComplex(value, 0.0);
    }

    __inline__ __device__ void
    setRealValue(cuDoubleComplex *a, double value )
    {
      *a =  make_cuDoubleComplex(value, 0.0);
    }

    __inline__ __device__ void
    setRealValue(float *a, double value )
    {
      *a =  value;
    }

    __inline__ __device__ void
    setRealValue(double *a, double value )
    {
      *a =  value;
    }




  } // namespace utils

} // namespace dftefe

#endif
