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
 * @author Sambit Das
 */

#ifndef dftefeMPICommunicatorP2P_h
#define dftefeMPICommunicatorP2P_h

#include <utils/MemorySpaceType.h>
#include <utils/MPITypes.h>
#include <utils/MPIPatternP2P.h>
#include <utils/TypeConfig.h>
#include <utils/MemoryStorage.h>
#include <utils/DeviceTypeConfig.h>
#include <functional>


namespace dftefe
{
  namespace utils
  {
    namespace mpi
    {
      enum class communicationProtocol
      {
        mpiHost,
        mpiDevice
      };

      enum class communicationPrecision
      {
        standard // same as valueType
      };

      template <typename ValueType, MemorySpace memorySpace>
      class MPICommunicatorP2P
      {
      public:
        // NOTE: getStream is a type-erased callback (rather than, say, a
        // LinAlgOpContext&) so that this utils-layer class never has to
        // depend on linearAlgebra::LinAlgOpContext -- dft-efe-linalg links
        // dft-efe-utils, not the reverse. Callers (Vector/MultiVector) pass
        // a lambda that calls linAlgOpContext.getBlasStream(); it's invoked
        // fresh on every communication call, so LinAlgOpContext's default-
        // /non-default-stream exclusivity gating still triggers correctly,
        // instead of being resolved once and going stale.
        MPICommunicatorP2P(
          std::shared_ptr<const MPIPatternP2P<memorySpace>> mpiPatternP2P,
          const size_type                                   blockSize,
          std::function<utils::deviceStream_t()>            getStream);

        void
        updateGhostValues(MemoryStorage<ValueType, memorySpace> &dataArray,
                          const size_type communicationChannel = 0);

        void
        accumulateAddLocallyOwned(
          MemoryStorage<ValueType, memorySpace> &dataArray,
          const size_type                        communicationChannel = 0);


        void
        updateGhostValuesBegin(MemoryStorage<ValueType, memorySpace> &dataArray,
                               const size_type communicationChannel = 0);

        void
        updateGhostValuesEnd(MemoryStorage<ValueType, memorySpace> &dataArray);

        void
        accumulateAddLocallyOwnedBegin(
          MemoryStorage<ValueType, memorySpace> &dataArray,
          const size_type                        communicationChannel = 0);

        void
        accumulateAddLocallyOwnedEnd(
          MemoryStorage<ValueType, memorySpace> &dataArray);

        std::shared_ptr<const MPIPatternP2P<memorySpace>>
        getMPIPatternP2P() const;

        int
        getBlockSize() const;

        void
        setCommunicationPrecision(communicationPrecision precision);

      private:
        std::shared_ptr<const MPIPatternP2P<memorySpace>> d_mpiPatternP2P;

        size_type d_blockSize;

        std::function<utils::deviceStream_t()> d_getStream;

        MemoryStorage<ValueType, memorySpace> d_targetDataBuffer;

        MemoryStorage<ValueType, memorySpace> d_ghostDataBuffer;


#ifdef DFTEFE_WITH_DEVICE
        MemoryStorage<ValueType, MemorySpace::HOST_PINNED>
          d_ghostDataCopyHostPinned;

        MemoryStorage<ValueType, MemorySpace::HOST_PINNED>
          d_sendRecvBufferHostPinned;
#endif // DFTEFE_WITH_DEVICE

        std::vector<MPIRequest> d_requestsUpdateGhostValues;
        std::vector<MPIRequest> d_requestsAccumulateAddLocallyOwned;
        MPIComm                 d_mpiCommunicator;

        communicationProtocol  d_commProtocol;
        communicationPrecision d_commPrecision;
      };

    } // namespace mpi
  }   // namespace utils
} // namespace dftefe
#include "MPICommunicatorP2P.t.cpp"
#endif // dftefeMPICommunicatorP2P_h
