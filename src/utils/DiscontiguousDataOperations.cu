/******************************************************************************
 * Copyright (c) 2021.                                                        *
 * The Regents of the University of Michigan and DFT-EFE developers.          *
 *                                                                            *
 * This file is part of the DFT-EFE code.                                     *
 *                                                                            *
 * DFT-EFE is free software: you can redistribute it and/or modify            *
 *   it under the terms of the Lesser GNU General Public License as           *
 *   published by the Free Software Foundation, either version 3 of           *
 *   the License, or (at your option) any later version.                      *
 *                                                                            *
 * DFT-EFE is distributed in the hope that it will be useful, but             *
 *   WITHOUT ANY WARRANTY; without even the implied warranty                  *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                     *
 *   See the Lesser GNU General Public License for more details.              *
 *                                                                            *
 * You should have received a copy of the GNU Lesser General Public           *
 *   License at the top level of DFT-EFE distribution.  If not, see           *
 *   <https://www.gnu.org/licenses/>.                                         *
 ******************************************************************************/

/*
 * @author Bikash Kanungo
 */

#ifdef DFTEFE_WITH_DEVICE_CUDA
#  include <utils/DeviceKernelLauncher.h>
#  include <utils/DeviceDataTypeOverloads.cuh>
#  include <utils/DiscontiguousDataOperations.h>
#  include <utils/Exceptions.h>
#  include <complex>
#  include <algorithm>

namespace dftefe
{
  namespace utils
  {
    namespace
    {
      template <typename ValueType>
      __global__ void
      copyFromDiscontiguousMemoryDeviceKernel(const size_type  N,
                                              const size_type  blockSize,
                                              const ValueType *src,
                                              ValueType *      dst,
                                              const size_type *discontIds)
      {
        const size_type globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
        for (size_type i = globalThreadId; i < N * blockSize;
             i += blockDim.x * gridDim.x)
          {
            const size_type blockId      = i / blockSize;
            const size_type intraBlockId = i - blockId * blockSize;
            dst[i] = src[discontIds[blockId] * blockSize + intraBlockId];
          }
      }

      template <typename ValueType>
      __global__ void
      copyToDiscontiguousMemoryDeviceKernel(const size_type  N,
                                            const size_type  blockSize,
                                            const ValueType *src,
                                            ValueType *      dst,
                                            const size_type *discontIds)
      {
        const size_type globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
        for (size_type i = globalThreadId; i < N * blockSize;
             i += blockDim.x * gridDim.x)
          {
            const size_type blockId      = i / blockSize;
            const size_type intraBlockId = i - blockId * blockSize;
            dst[discontIds[blockId] * blockSize + intraBlockId] = src[i];
          }
      }

      template <typename ValueType>
      __global__ void
      addToDiscontiguousMemoryDeviceKernel(const size_type  N,
                                           const size_type  blockSize,
                                           const ValueType *src,
                                           ValueType *      dst,
                                           const size_type *discontIds)
      {
        const size_type globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
        for (size_type i = globalThreadId; i < N * blockSize;
             i += blockDim.x * gridDim.x)
          {
            const size_type blockId      = i / blockSize;
            const size_type intraBlockId = i - blockId * blockSize;

            // FIXME: atomicAdd will not work for complex ValueType,
            // Implement workaround with temporary real and imaginary double
            // arrays
            atomicAdd(&dst[discontIds[blockId] * blockSize + intraBlockId],
                      src[i]);
          }
      }
    } // namespace

    template <typename ValueType>
    void
    DiscontiguousDataOperations<ValueType, utils::MemorySpace::DEVICE>::
      copyFromDiscontiguousMemory(const ValueType *src,
                                  ValueType *      dst,
                                  const size_type *discontIds,
                                  const size_type  N,
                                  const size_type  blockSize)
    {
      copyFromDiscontiguousMemoryDeviceKernel<<<N / dftefe::utils::BLOCK_SIZE +
                                                  1,
                                                dftefe::utils::BLOCK_SIZE>>>(
        N,
        blockSize,
        dftefe::utils::makeDataTypeDeviceCompatible(src),
        dftefe::utils::makeDataTypeDeviceCompatible(dst),
        dftefe::utils::makeDataTypeDeviceCompatible(discontIds));
    }

    template <typename ValueType>
    void
    DiscontiguousDataOperations<ValueType, utils::MemorySpace::DEVICE>::
      copyToDiscontiguousMemory(const ValueType *src,
                                ValueType *      dst,
                                const size_type *discontIds,
                                const size_type  N,
                                const size_type  blockSize)
    {
      copyToDiscontiguousMemoryDeviceKernel<<<N / dftefe::utils::BLOCK_SIZE + 1,
                                              dftefe::utils::BLOCK_SIZE>>>(
        N,
        blockSize,
        dftefe::utils::makeDataTypeDeviceCompatible(src),
        dftefe::utils::makeDataTypeDeviceCompatible(dst),
        dftefe::utils::makeDataTypeDeviceCompatible(discontIds));
    }

    template <typename ValueType>
    void
    DiscontiguousDataOperations<ValueType, utils::MemorySpace::DEVICE>::
      addToDiscontiguousMemory(const ValueType *src,
                               ValueType *      dst,
                               const size_type *discontIds,
                               const size_type  N,
                               const size_type  blockSize)
    {
      addToDiscontiguousMemoryDeviceKernel<<<N / dftefe::utils::BLOCK_SIZE + 1,
                                             dftefe::utils::BLOCK_SIZE>>>(
        N,
        blockSize,
        dftefe::utils::makeDataTypeDeviceCompatible(src),
        dftefe::utils::makeDataTypeDeviceCompatible(dst),
        dftefe::utils::makeDataTypeDeviceCompatible(discontIds));
    }

    template class DiscontiguousDataOperations<
      double,
      dftefe::utils::MemorySpace::DEVICE>;
    template class DiscontiguousDataOperations<
      float,
      dftefe::utils::MemorySpace::DEVICE>;
    template class DiscontiguousDataOperations<
      std::complex<double>,
      dftefe::utils::MemorySpace::DEVICE>;
    template class DiscontiguousDataOperations<
      std::complex<float>,
      dftefe::utils::MemorySpace::DEVICE>;
  } // end of namespace utils
} // end of namespace dftefe
#endif // DFTEFE_WITH_DEVICE_CUDA
