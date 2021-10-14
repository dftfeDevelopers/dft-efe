#ifdef DFTEFE_WITH_DEVICE_CUDA
#  include "DeviceDataTypeOverloads.cuh"

namespace dftefe
{
  namespace utils
  {
    __device__ double
    mult(double a, double b)
    {
      return a * b;
    }

    __device__ cuDoubleComplex
    mult(cuDoubleComplex a, cuDoubleComplex b)
    {
      return cuCmul(a, b);
    }

    __device__ double
    add(double a, double b)
    {
      return a + b;
    }

    __device__ cuDoubleComplex
    add(cuDoubleComplex a, cuDoubleComplex b)
    {
      return cuCadd(a, b);
    }
    __device__ cuDoubleComplex
    mult(double a, cuDoubleComplex b)
    {
      return make_cuDoubleComplex(a * b.x, a * b.y);
    }
    __device__ cuDoubleComplex
    mult(cuDoubleComplex a, double b)
    {
      return make_cuDoubleComplex(b * a.x, b * a.y);
    }
    __device__ cuFloatComplex
    mult(double a, cuFloatComplex b)
    {
      return make_cuFloatComplex(a * b.x, a * b.y);
    }
    __device__ cuFloatComplex
    mult(cuFloatComplex a, double b)
    {
      return make_cuFloatComplex(b * a.x, b * a.y);
    }
  } // namespace utils
} // namespace dftefe
#endif
