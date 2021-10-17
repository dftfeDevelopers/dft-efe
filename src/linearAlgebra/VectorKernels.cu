#ifdef DFTEFE_WITH_DEVICE_CUDA
#  include "DeviceDataTypeOverloads.cuh"
#  include "VectorKernels.h"
namespace dftefe
{
  namespace linearAlgebra
  {
    namespace
    {
      template <typename NumberType>
      __global__ void
      addCUDAKernel(const NumberType  a,
                    size_type         size,
                    const NumberType *u,
                    NumberType *      v)
      {
        const unsigned int globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        for (unsigned int i = globalThreadId; i < size;
             i += blockDim.x * gridDim.x)
          {
            v[i] = dftefe::utils::add(dftefe::utils::mult(a, u[i]), v[i]);
          }
      }

      template <typename NumberType>
      __global__ void
      addCUDAKernel(size_type size, const NumberType *u, NumberType *v)
      {
        const unsigned int globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        for (unsigned int i = globalThreadId; i < size;
             i += blockDim.x * gridDim.x)
          {
            v[i] = dftefe::utils::add(u[i], v[i]);
          }
      }

    } // namespace

    template <typename NumberType>
    void
    VectorKernels<NumberType, dftefe::utils::MemorySpace::DEVICE>::add(
      const size_type   size,
      const NumberType *u,
      NumberType *      v)
    {
      addCUDAKernel<<<size / 256 + 1, 256>>>(
        size,
        dftefe::utils::makeDataTypeDeviceCompatible(u),
        dftefe::utils::makeDataTypeDeviceCompatible(v));
    }

    template class VectorKernels<double, dftefe::utils::MemorySpace::DEVICE>;
    template class VectorKernels<float, dftefe::utils::MemorySpace::DEVICE>;
    template class VectorKernels<std::complex<double>,
                                 dftefe::utils::MemorySpace::DEVICE>;
    template class VectorKernels<std::complex<float>,
                                 dftefe::utils::MemorySpace::DEVICE>;
  } // namespace linearAlgebra
} // namespace dftefe
#endif
