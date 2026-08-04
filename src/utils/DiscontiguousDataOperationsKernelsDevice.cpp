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

#ifdef DFTEFE_WITH_DEVICE
#  include <utils/DeviceKernelLauncherHelpers.h>
#  include <utils/DeviceDataTypeOverloads.h>
#  include <utils/DeviceTypeConfigHalfPrec.h>
#  include <utils/MPICommunicatorP2PKernels.h>
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
      DFTEFE_CREATE_KERNEL(
        void,
        copyFromDiscontiguousMemoryDeviceKernel,
        {
          for (size_type i = globalThreadId; i < N * blockSize;
               i += nThreadsPerBlock * nThreadBlock)
            {
              const size_type blockId      = i / blockSize;
              const size_type intraBlockId = i - blockId * blockSize;
              dftefe::utils::copyValue(
                dst + i, src[discontIds[blockId] * blockSize + intraBlockId]);
            }
        },
        const size_type  N,
        const size_type  blockSize,
        const ValueType *src,
        ValueType *      dst,
        const size_type *discontIds);

      template <typename ValueType>
      DFTEFE_CREATE_KERNEL(
        void,
        copyToDiscontiguousMemoryDeviceKernel,
        {
          for (size_type i = globalThreadId; i < N * blockSize;
               i += nThreadsPerBlock * nThreadBlock)
            {
              const size_type blockId      = i / blockSize;
              const size_type intraBlockId = i - blockId * blockSize;
              dftefe::utils::copyValue(dst + discontIds[blockId] * blockSize +
                                         intraBlockId,
                                       src[i]);
            }
        },
        const size_type  N,
        const size_type  blockSize,
        const ValueType *src,
        ValueType *      dst,
        const size_type *discontIds);

      template <typename ValueType>
      DFTEFE_CREATE_KERNEL(
        void,
        addToDiscontiguousMemoryDeviceKernel,
        {
          for (size_type i = globalThreadId; i < N * blockSize;
               i += nThreadsPerBlock * nThreadBlock)
            {
              const size_type blockId      = i / blockSize;
              const size_type intraBlockId = i - blockId * blockSize;
              dftefe::utils::atomicAddWrapper(
                dst + discontIds[blockId] * blockSize + intraBlockId, src[i]);
            }
        },
        const size_type  N,
        const size_type  blockSize,
        const ValueType *src,
        ValueType *      dst,
        const size_type *discontIds);

      template <>
      DFTEFE_CREATE_KERNEL(
        void,
        addToDiscontiguousMemoryDeviceKernel,
        {
          for (size_type i = globalThreadId; i < N * blockSize;
               i += nThreadsPerBlock * nThreadBlock)
            {
              const size_type blockId      = i / blockSize;
              const size_type intraBlockId = i - blockId * blockSize;

              auto *add_real = reinterpret_cast<float *>(
                dst + discontIds[blockId] * blockSize + intraBlockId);
              auto *add_imag = add_real + 1;

              dftefe::utils::atomicAddWrapper(
                add_real, dftefe::utils::realPartDevice(src[i]));
              dftefe::utils::atomicAddWrapper(
                add_imag, dftefe::utils::imagPartDevice(src[i]));
            }
        },
        const size_type                          N,
        const size_type                          blockSize,
        const dftefe::utils::deviceFloatComplex *src,
        dftefe::utils::deviceFloatComplex *      dst,
        const size_type *                        discontIds);

      template <>
      DFTEFE_CREATE_KERNEL(
        void,
        addToDiscontiguousMemoryDeviceKernel,
        {
          for (size_type i = globalThreadId; i < N * blockSize;
               i += nThreadsPerBlock * nThreadBlock)
            {
              const size_type blockId      = i / blockSize;
              const size_type intraBlockId = i - blockId * blockSize;

              auto *add_real = reinterpret_cast<double *>(
                dst + discontIds[blockId] * blockSize + intraBlockId);
              auto *add_imag = add_real + 1;

              dftefe::utils::atomicAddWrapper(
                add_real, dftefe::utils::realPartDevice(src[i]));
              dftefe::utils::atomicAddWrapper(
                add_imag, dftefe::utils::imagPartDevice(src[i]));
            }
        },
        const size_type                           N,
        const size_type                           blockSize,
        const dftefe::utils::deviceDoubleComplex *src,
        dftefe::utils::deviceDoubleComplex *      dst,
        const size_type *                         discontIds);

    } // namespace

    template <typename ValueType>
    void
    DiscontiguousDataOperations<ValueType, utils::MemorySpace::DEVICE>::
      copyFromDiscontiguousMemory(const ValueType *     src,
                                  ValueType *           dst,
                                  const size_type *     discontIds,
                                  const size_type       N,
                                  const size_type       blockSize,
                                  utils::deviceStream_t streamId)
    {
      DFTEFE_LAUNCH_KERNEL(
        copyFromDiscontiguousMemoryDeviceKernel,
        (N * blockSize) / dftefe::utils::DEVICE_BLOCK_SIZE + 1,
        dftefe::utils::DEVICE_BLOCK_SIZE,
        streamId,
        N,
        blockSize,
        dftefe::utils::makeDataTypeDeviceCompatible(src),
        dftefe::utils::makeDataTypeDeviceCompatible(dst),
        dftefe::utils::makeDataTypeDeviceCompatible(discontIds));
    }

    template <typename ValueType>
    void
    DiscontiguousDataOperations<ValueType, utils::MemorySpace::DEVICE>::
      copyToDiscontiguousMemory(const ValueType *     src,
                                ValueType *           dst,
                                const size_type *     discontIds,
                                const size_type       N,
                                const size_type       blockSize,
                                utils::deviceStream_t streamId)
    {
      DFTEFE_LAUNCH_KERNEL(
        copyToDiscontiguousMemoryDeviceKernel,
        (N * blockSize) / dftefe::utils::DEVICE_BLOCK_SIZE + 1,
        dftefe::utils::DEVICE_BLOCK_SIZE,
        streamId,
        N,
        blockSize,
        dftefe::utils::makeDataTypeDeviceCompatible(src),
        dftefe::utils::makeDataTypeDeviceCompatible(dst),
        dftefe::utils::makeDataTypeDeviceCompatible(discontIds));
    }

    template <typename ValueType>
    void
    DiscontiguousDataOperations<ValueType, utils::MemorySpace::DEVICE>::
      addToDiscontiguousMemory(const ValueType *     src,
                               ValueType *           dst,
                               const size_type *     discontIds,
                               const size_type       N,
                               const size_type       blockSize,
                               utils::deviceStream_t streamId)
    {
      DFTEFE_LAUNCH_KERNEL(
        addToDiscontiguousMemoryDeviceKernel,
        (N * blockSize) / dftefe::utils::DEVICE_BLOCK_SIZE + 1,
        dftefe::utils::DEVICE_BLOCK_SIZE,
        streamId,
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
#endif // DFTEFE_WITH_DEVICE
