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

#include <utils/MPICommunicatorP2PKernels.h>
#include <utils/Exceptions.h>
#include <complex>
#include <algorithm>


namespace dftefe
{
  namespace utils
  {
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    MPICommunicatorP2PKernels<ValueType, memorySpace>::
      gatherLocallyOwnedEntriesSendBufferToTargetProcs(
        const MemoryStorage<ValueType, memorySpace> &dataArray,
        const SizeTypeVector &                 ownedLocalIndicesForTargetProcs,
        const size_type                        blockSize,
        MemoryStorage<ValueType, memorySpace> &sendBuffer)
    {
      for (size_type i = 0; i < ownedLocalIndicesForTargetProcs.size(); ++i)
        for (size_type j = 0; j < blockSize; ++j)
          sendBuffer.data()[i * blockSize + j] =
            dataArray
              .data()[ownedLocalIndicesForTargetProcs.data()[i] * blockSize +
                      j];
    }


    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    MPICommunicatorP2PKernels<ValueType, memorySpace>::
      accumAddLocallyOwnedContrRecvBufferFromTargetProcs(
        const MemoryStorage<ValueType, memorySpace> &recvBuffer,
        const SizeTypeVector &                 ownedLocalIndicesForTargetProcs,
        const size_type                        blockSize,
        MemoryStorage<ValueType, memorySpace> &dataArray)
    {
      for (size_type i = 0; i < ownedLocalIndicesForTargetProcs.size(); ++i)
        for (size_type j = 0; j < blockSize; ++j)
          dataArray
            .data()[ownedLocalIndicesForTargetProcs.data()[i] * blockSize +
                    j] += recvBuffer.data()[i * blockSize + j];
    }


    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    MPICommunicatorP2PKernels<ValueType, memorySpace>::
      insertLocalGhostsRecvBufferFromGhostProcs(
        const MemoryStorage<ValueType, memorySpace> &recvBuffer,
        const SizeTypeVector &                       ghostLocalIndices,
        const size_type                              blockSize,
        MemoryStorage<ValueType, memorySpace> &      dataArray)
    {
      for (size_type i = 0; i < ownedLocalIndicesForTargetProcs.size(); ++i)
        for (size_type j = 0; j < blockSize; ++j)
          dataArray.data()[ghostLocalIndices.data()[i] * blockSize + j] =
            recvBuffer.data()[i * blockSize + j];
    }

    template class MPICommunicatorP2PKernels<double,
                                             dftefe::utils::MemorySpace::HOST>;
    template class MPICommunicatorP2PKernels<float,
                                             dftefe::utils::MemorySpace::HOST>;
    template class MPICommunicatorP2PKernels<std::complex<double>,
                                             dftefe::utils::MemorySpace::HOST>;
    template class MPICommunicatorP2PKernels<std::complex<float>,
                                             dftefe::utils::MemorySpace::HOST>;

#ifdef DFTEFE_WITH_DEVICE
    template class MPICommunicatorP2PKernels<
      double,
      dftefe::utils::MemorySpace::HOST_PINNED>;
    template class MPICommunicatorP2PKernels<
      float,
      dftefe::utils::MemorySpace::HOST_PINNED>;
    template class MPICommunicatorP2PKernels<
      std::complex<double>,
      dftefe::utils::MemorySpace::HOST_PINNED>;
    template class MPICommunicatorP2PKernels<
      std::complex<float>,
      dftefe::utils::MemorySpace::HOST_PINNED>;
#endif


  } // namespace utils
} // namespace dftefe
