#ifdef DFTEFE_WITH_DEVICE_CUDA
#  include "DeviceDataTypeOverloads.cuh"
#  include "VectorKernels.h"
namespace dftefe
{
  namespace utils
  {
    template <typename NumberType>
    __global__ void
    addKernel(const NumberType  a,
              size_t            size,
              const NumberType *V1,
              NumberType       *V2)
    {
      const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      for (unsigned int i = globalThreadId; i < size;
           i += blockDim.x * gridDim.x)
        {
          V2[i] = add(mult(a, V1[i]), V2[i]);
        }
    }

    template <typename NumberType>
    void
    addKernel(const NumberType                               a,
              const Vector<NumberType, MemorySpace::DEVICE> &V1,
              Vector<NumberType, MemorySpace::DEVICE>       &V2)
    {
      // todo: add assert condition: V1.size() == V2.size();
    }
  } // namespace utils
} // namespace dftefe
#endif