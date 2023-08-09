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
 * @author Sambit Das.
 */

#ifdef DFTEFE_WITH_DEVICE_CUDA
#  include <utils/DeviceKernelLauncher.h>
#  include <utils/DeviceDataTypeOverloads.cuh>
#  include <utils/MPICommunicatorP2PKernels.h>
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
      gatherSendBufferDeviceKernel(
        const size_type  totalFlattenedSize,
        const size_type  blockSize,
        const ValueType *dataArray,
        const size_type *ownedLocalIndicesForTargetProcs,
        ValueType *      sendBuffer)
      {
        const size_type globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
        for (size_type i = globalThreadId; i < totalFlattenedSize;
             i += blockDim.x * gridDim.x)
          {
            const size_type blockId      = i / blockSize;
            const size_type intraBlockId = i - blockId * blockSize;

            sendBuffer[i] =
              dataArray[ownedLocalIndicesForTargetProcs[blockId] * blockSize +
                        intraBlockId];
          }
      }

      template <typename ValueType>
      __global__ void
      accumAddFromRecvBufferDeviceKernel(
        const size_type  totalFlattenedSize,
        const size_type  blockSize,
        const ValueType *recvBuffer,
        const size_type *ownedLocalIndicesForTargetProcs,
        ValueType *      dataArray)
      {
        const size_type globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
        for (size_type i = globalThreadId; i < totalFlattenedSize;
             i += blockDim.x * gridDim.x)
          {
            const size_type blockId      = i / blockSize;
            const size_type intraBlockId = i - blockId * blockSize;

            // FIXME: would not complex ValueType, workaround with temporary
            // real and imaginary double arrays needs to be implemented
            atomicAdd(
              &dataArray[ownedLocalIndicesForTargetProcs[blockId] * blockSize +
                         intraBlockId],
              recvBuffer[i]);
          }
      }

      template <typename ValueType>
      __global__ void
      insertFromRecvBufferDeviceKernel(const size_type  totalFlattenedSize,
                                       const size_type  blockSize,
                                       const ValueType *recvBuffer,
                                       const size_type *ghostLocalIndices,
                                       ValueType *      dataArray)
      {
        const size_type globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
        for (size_type i = globalThreadId; i < totalFlattenedSize;
             i += blockDim.x * gridDim.x)
          {
            const size_type blockId      = i / blockSize;
            const size_type intraBlockId = i - blockId * blockSize;

            dataArray[ghostLocalIndices[blockId] * blockSize + intraBlockId] =
              recvBuffer[i];
          }
      }

    } // namespace

    template <typename ValueType>
    void
    MPICommunicatorP2PKernels<ValueType, utils::MemorySpace::DEVICE>::
      gatherLocallyOwnedEntriesSendBufferToTargetProcs(
        const MemoryStorage<ValueType, utils::MemorySpace::DEVICE> &dataArray,
        const MemoryStorage<size_type, utils::MemorySpace::DEVICE>
          &             ownedLocalIndicesForTargetProcs,
        const size_type blockSize,
        MemoryStorage<ValueType, utils::MemorySpace::DEVICE> &sendBuffer)
    {
      gatherSendBufferDeviceKernel<<<
        ownedLocalIndicesForTargetProcs.size() / dftefe::utils::BLOCK_SIZE + 1,
        dftefe::utils::BLOCK_SIZE>>>(
        ownedLocalIndicesForTargetProcs.size() * blockSize,
        blockSize,
        dftefe::utils::makeDataTypeDeviceCompatible(dataArray.data()),
        dftefe::utils::makeDataTypeDeviceCompatible(
          ownedLocalIndicesForTargetProcs.data()),
        dftefe::utils::makeDataTypeDeviceCompatible(sendBuffer.data()));
    }


    template <typename ValueType>
    void
    MPICommunicatorP2PKernels<ValueType, utils::MemorySpace::DEVICE>::
      accumAddLocallyOwnedContrRecvBufferFromTargetProcs(
        const MemoryStorage<ValueType, utils::MemorySpace::DEVICE> &recvBuffer,
        const utils::MemoryStorage<size_type, utils::MemorySpace::DEVICE>
          &             ownedLocalIndicesForTargetProcs,
        const size_type blockSize,
        MemoryStorage<ValueType, dftefe::utils::MemorySpace::DEVICE> &dataArray)
    {
      accumAddFromRecvBufferDeviceKernel<<<
        ownedLocalIndicesForTargetProcs.size() / dftefe::utils::BLOCK_SIZE + 1,
        dftefe::utils::BLOCK_SIZE>>>(
        ownedLocalIndicesForTargetProcs.size() * blockSize,
        blockSize,
        dftefe::utils::makeDataTypeDeviceCompatible(recvBuffer.data()),
        dftefe::utils::makeDataTypeDeviceCompatible(
          ownedLocalIndicesForTargetProcs.data()),
        dftefe::utils::makeDataTypeDeviceCompatible(dataArray.data()));
    }


    template <typename ValueType>
    void
    MPICommunicatorP2PKernels<ValueType, utils::MemorySpace::DEVICE>::
      insertLocalGhostsRecvBufferFromGhostProcs(
        const MemoryStorage<ValueType, utils::MemorySpace::DEVICE> &recvBuffer,
        const utils::MemoryStorage<size_type, utils::MemorySpace::DEVICE>
          &             ghostLocalIndices,
        const size_type blockSize,
        MemoryStorage<ValueType, dftefe::utils::MemorySpace::DEVICE> &dataArray)
    {
      insertFromRecvBufferDeviceKernel<<<
        ghostLocalIndices.size() / dftefe::utils::BLOCK_SIZE + 1,
        dftefe::utils::BLOCK_SIZE>>>(
        ghostLocalIndices.size() * blockSize,
        blockSize,
        dftefe::utils::makeDataTypeDeviceCompatible(recvBuffer.data()),
        dftefe::utils::makeDataTypeDeviceCompatible(ghostLocalIndices.data()),
        dftefe::utils::makeDataTypeDeviceCompatible(dataArray.data()));
    }

    template class MPICommunicatorP2PKernels<
      double,
      dftefe::utils::MemorySpace::DEVICE>;
    template class MPICommunicatorP2PKernels<
      float,
      dftefe::utils::MemorySpace::DEVICE>;
    template class MPICommunicatorP2PKernels<
      std::complex<double>,
      dftefe::utils::MemorySpace::DEVICE>;
    template class MPICommunicatorP2PKernels<
      std::complex<float>,
      dftefe::utils::MemorySpace::DEVICE>;

  } // namespace utils
} // namespace dftefe
#endif
