#ifdef DFTEFE_WITH_DEVICE_CUDA
#  include "DeviceDataTypeOverloads.cuh"

namespace dftefe
{
  namespace utils
  {
    __device__ double
    mult(double x, double y)
    {
      return x * y;
    }

    __device__ cuDoubleComplex
    mult(cuDoubleComplex x, cuDoubleComplex y)
    {
      return cuCmul(x, y);
    }
  } // namespace utils
} // namespace dftefe
#endif
