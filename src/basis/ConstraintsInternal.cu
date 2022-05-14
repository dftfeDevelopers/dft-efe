#ifdef DFTEFE_WITH_DEVICE_CUDA

#  include "ConstraintsInternal.h"
#  include <utils/DeviceKernelLauncher.h>
#  include <utils/DeviceDataTypeOverloads.cuh>
#  include <utils/DataTypeOverloads.h>
namespace dftefe
{
  namespace basis
  {
    template <typename ValueType>
    __global__ void
    setzeroKernel(const size_type  contiguousBlockSize,
                  ValueType *      xVec,
                  const size_type *constraintLocalRowIdsUnflattened,
                  const size_type  numConstraints,
                  const size_type  blockSize)
    {
      const global_size_type globalThreadId =
        blockIdx.x * blockDim.x + threadIdx.x;
      const global_size_type numberEntries =
        numConstraints * contiguousBlockSize;

      for (global_size_type index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          const unsigned int blockIndex      = index / contiguousBlockSize;
          const unsigned int intraBlockIndex = index % contiguousBlockSize;
          xVec[constraintLocalRowIdsUnflattened[blockIndex] * blockSize +
               intraBlockIndex]              = 0;
        }
    }


    template <typename ValueType>
    void
    ConstraintsInternal<ValueType, dftefe::utils::MemorySpace::DEVICE>::
      constraintsSetConstrainedNodesToZero(
        linearAlgebra::Vector<ValueType, dftefe::utils::MemorySpace::DEVICE>
          &       vectorData,
        size_type blockSize,
        utils::MemoryStorage<global_size_type,
                             dftefe::utils::MemorySpace::DEVICE>
          &rowConstraintsIdsLocal)
    {
      const size_type numConstrainedDofs = rowConstraintsIdsLocal.size();

      if (numConstrainedDofs == 0)
        return;

      setzeroKernel<<<min((blockSize + 255) / 256 * numConstrainedDofs, 30000),
                      256>>>(blockSize,
                             dftefe::utils::makeDataTypeDeviceCompatible(
                               vectorData.begin()),
                             dftefe::utils::makeDataTypeDeviceCompatible(
                               rowConstraintsIdsLocal.begin()),
                             numConstrainedDofs,
                             blockSize);
    }


    template class ConstraintsInternal<double,
                                       dftefe::utils::MemorySpace::DEVICE>;
    template class ConstraintsInternal<float,
                                       dftefe::utils::MemorySpace::DEVICE>;
    template class ConstraintsInternal<std::complex<double>,
                                       dftefe::utils::MemorySpace::DEVICE>;
    template class ConstraintsInternal<std::complex<float>,
                                       dftefe::utils::MemorySpace::DEVICE>;


  } // namespace basis
} // namespace dftefe

#endif
