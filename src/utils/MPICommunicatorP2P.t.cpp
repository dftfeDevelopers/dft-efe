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

#include <utils/MPICommunicatorP2P.h>
#include <utils/DiscontiguousDataOperations.h>
#include <utils/Exceptions.h>
#include <utils/MPIErrorCodeHandler.h>


namespace dftefe
{
  namespace utils
  {
    namespace mpi
    {
      template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
      MPICommunicatorP2P<ValueType, memorySpace>::MPICommunicatorP2P(
        std::shared_ptr<const MPIPatternP2P<memorySpace>> mpiPatternP2P,
        const size_type                                   blockSize)
        : d_mpiPatternP2P(mpiPatternP2P)
        , d_blockSize(blockSize)
      {
        d_mpiCommunicator = d_mpiPatternP2P->mpiCommunicator();
        d_targetDataBuffer.resize(
          d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs().size() *
            blockSize,
          0.0);
        d_ghostDataBuffer.resize((d_mpiPatternP2P->localGhostSize()) *
                                   blockSize,
                                 0.0);
        d_requestsUpdateGhostValues.resize(
          d_mpiPatternP2P->getGhostProcIds().size() +
          d_mpiPatternP2P->getTargetProcIds().size());
        d_requestsAccumulateAddLocallyOwned.resize(
          d_mpiPatternP2P->getGhostProcIds().size() +
          d_mpiPatternP2P->getTargetProcIds().size());

#if defined(DFTEFE_WITH_DEVICE) && !defined(DFTEFE_WITH_DEVICE_AWARE_MPI)
        if (memorySpace == MemorySpace::DEVICE)
          {
            d_ghostDataCopyHostPinned.resize(d_mpiPatternP2P->localGhostSize() *
                                               blockSize,
                                             0.0);
            d_sendRecvBufferHostPinned.resize(
              d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs().size() *
                blockSize,
              0.0);
          }
#endif // defined(DFTEFE_WITH_DEVICE) && !defined(DFTEFE_WITH_DEVICE_AWARE_MPI)
      }


      template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
      void
      MPICommunicatorP2P<ValueType, memorySpace>::updateGhostValues(
        MemoryStorage<ValueType, memorySpace> &dataArray,
        const size_type                        communicationChannel)
      {
#ifdef DFTEFE_WITH_MPI
        updateGhostValuesBegin(dataArray, communicationChannel);
        updateGhostValuesEnd(dataArray);
#endif // DFTEFE_WITH_MPI
      }

      template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
      void
      MPICommunicatorP2P<ValueType, memorySpace>::updateGhostValuesBegin(
        MemoryStorage<ValueType, memorySpace> &dataArray,
        const size_type                        communicationChannel)
      {
#ifdef DFTEFE_WITH_MPI
        // initiate non-blocking receives from ghost processors
        ValueType *recvArrayStartPtr = d_ghostDataBuffer.begin();

#  if defined(DFTEFE_WITH_DEVICE) && !defined(DFTEFE_WITH_DEVICE_AWARE_MPI)
        if (memorySpace == MemorySpace::DEVICE)
          recvArrayStartPtr = d_ghostDataCopyHostPinned.begin();
#  endif // defined(DFTEFE_WITH_DEVICE) && //
         // !defined(DFTEFE_WITH_DEVICE_AWARE_MPI)

        for (size_type i = 0; i < (d_mpiPatternP2P->getGhostProcIds()).size();
             ++i)
          {
#  if defined(DFTEFE_WITH_DEVICE) && !defined(DFTEFE_WITH_DEVICE_AWARE_MPI)
            const int err = MPIIrecv<MemorySpace::HOST_PINNED>(
              recvArrayStartPtr,
              (d_mpiPatternP2P->getGhostLocalIndicesRanges().data()[2 * i + 1] -
               d_mpiPatternP2P->getGhostLocalIndicesRanges().data()[2 * i]) *
                d_blockSize * sizeof(ValueType),
              MPIByte,
              d_mpiPatternP2P->getGhostProcIds().data()[i],
              static_cast<size_type>(
                MPITags::MPI_P2P_COMMUNICATOR_SCATTER_TAG) +
                communicationChannel,
              d_mpiCommunicator,
              &d_requestsUpdateGhostValues[i]);
#  else
            const int err = MPIIrecv<memorySpace>(
              recvArrayStartPtr,
              (d_mpiPatternP2P->getGhostLocalIndicesRanges().data()[2 * i + 1] -
               d_mpiPatternP2P->getGhostLocalIndicesRanges().data()[2 * i]) *
                d_blockSize * sizeof(ValueType),
              MPIByte,
              d_mpiPatternP2P->getGhostProcIds().data()[i],
              static_cast<size_type>(
                MPITags::MPI_P2P_COMMUNICATOR_SCATTER_TAG) +
                communicationChannel,
              d_mpiCommunicator,
              &d_requestsUpdateGhostValues[i]);
#  endif


            const std::pair<bool, std::string> isSuccessAndMessage =
              MPIErrorCodeHandler::getIsSuccessAndMessage(err);
            throwException(isSuccessAndMessage.first,
                           isSuccessAndMessage.second);

            recvArrayStartPtr +=
              (d_mpiPatternP2P->getGhostLocalIndicesRanges().data()[2 * i + 1] -
               d_mpiPatternP2P->getGhostLocalIndicesRanges().data()[2 * i]) *
              d_blockSize;
          }

        // gather locally owned entries into a contiguous send buffer
        const size_type *ownedLocalIndicesForTargetProcsPtr =
          (d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs()).data();

        const size_type numTotalOwnedIndicesForTargetProcs =
          d_mpiPatternP2P->getTotalOwnedIndicesForTargetProcs();

        DiscontiguousDataOperations<ValueType, memorySpace>::
          copyFromDiscontiguousMemory(dataArray.data(),
                                      d_targetDataBuffer.data(),
                                      ownedLocalIndicesForTargetProcsPtr,
                                      numTotalOwnedIndicesForTargetProcs,
                                      d_blockSize);

        // initiate non-blocking sends to target processors
        ValueType *sendArrayStartPtr = d_targetDataBuffer.begin();

#  if defined(DFTEFE_WITH_DEVICE) && !defined(DFTEFE_WITH_DEVICE_AWARE_MPI)
        if (memorySpace == MemorySpace::DEVICE)
          {
            MemoryTransfer<MemorySpace::HOST_PINNED, memorySpace>
              memoryTransfer;
            memoryTransfer.copy(d_sendRecvBufferHostPinned.size(),
                                d_sendRecvBufferHostPinned.begin(),
                                d_targetDataBuffer.begin());

            sendArrayStartPtr = d_sendRecvBufferHostPinned.begin();
          }
#  endif // defined(DFTEFE_WITH_DEVICE) &&
         // !defined(DFTEFE_WITH_DEVICE_AWARE_MPI)

        for (size_type i = 0; i < (d_mpiPatternP2P->getTargetProcIds()).size();
             ++i)
          {
#  if defined(DFTEFE_WITH_DEVICE) && !defined(DFTEFE_WITH_DEVICE_AWARE_MPI)
            const int err = MPIIsend<MemorySpace::HOST_PINNED>(
              sendArrayStartPtr,
              d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs().data()[i] *
                d_blockSize * sizeof(ValueType),
              MPIByte,
              d_mpiPatternP2P->getTargetProcIds().data()[i],
              static_cast<size_type>(
                MPITags::MPI_P2P_COMMUNICATOR_SCATTER_TAG) +
                communicationChannel,

              d_mpiCommunicator,
              &d_requestsUpdateGhostValues
                [d_mpiPatternP2P->getGhostProcIds().size() + i]);
#  else
            const int err = MPIIsend<memorySpace>(
              sendArrayStartPtr,
              d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs().data()[i] *
                d_blockSize * sizeof(ValueType),
              MPIByte,
              d_mpiPatternP2P->getTargetProcIds().data()[i],
              static_cast<size_type>(
                MPITags::MPI_P2P_COMMUNICATOR_SCATTER_TAG) +
                communicationChannel,

              d_mpiCommunicator,
              &d_requestsUpdateGhostValues
                [d_mpiPatternP2P->getGhostProcIds().size() + i]);
#  endif


            const std::pair<bool, std::string> isSuccessAndMessage =
              MPIErrorCodeHandler::getIsSuccessAndMessage(err);
            throwException(isSuccessAndMessage.first,
                           isSuccessAndMessage.second);

            sendArrayStartPtr +=
              d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs().data()[i] *
              d_blockSize;
          }
#endif // DFTEFE_WITH_MPI
      }


      template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
      void
      MPICommunicatorP2P<ValueType, memorySpace>::updateGhostValuesEnd(
        MemoryStorage<ValueType, memorySpace> &dataArray)
      {
#ifdef DFTEFE_WITH_MPI
        // wait for all send and recv requests to be completed
        if (d_requestsUpdateGhostValues.size() > 0)
          {
            const int err = MPIWaitall(d_requestsUpdateGhostValues.size(),
                                       d_requestsUpdateGhostValues.data(),
                                       MPIStatusesIgnore);
            const std::pair<bool, std::string> isSuccessAndMessage =
              MPIErrorCodeHandler::getIsSuccessAndMessage(err);
            throwException(isSuccessAndMessage.first,
                           isSuccessAndMessage.second);

#  if defined(DFTEFE_WITH_DEVICE) && !defined(DFTEFE_WITH_DEVICE_AWARE_MPI)
            if (memorySpace == MemorySpace::DEVICE)
              {
                MemoryTransfer<memorySpace, MemorySpace::HOST_PINNED>
                  memoryTransfer;
                memoryTransfer.copy(d_ghostDataCopyHostPinned.size(),
                                    d_ghostDataBuffer.data(),
                                    d_ghostDataCopyHostPinned.data());
              }
#  endif // defined(DFTEFE_WITH_DEVICE) && //
         // !defined(DFTEFE_WITH_DEVICE_AWARE_MPI)

            // Copy ghost buffer receieved to the ghost part of the data.
            // Set the starting reference of the destination to the ghost part
            // of the data. NOTE: It assumes that the ghost part of data follows
            // the owned part
            ValueType *dataGhostPtr =
              dataArray.data() +
              d_mpiPatternP2P->localOwnedSize() * d_blockSize;
            const size_type *ghostLocalIndicesForGhostProcsPtr =
              (d_mpiPatternP2P->getGhostLocalIndicesForGhostProcs()).data();

            const size_type numGhostIndices = d_mpiPatternP2P->localGhostSize();

            DiscontiguousDataOperations<ValueType, memorySpace>::
              copyToDiscontiguousMemory(d_ghostDataBuffer.data(),
                                        dataGhostPtr,
                                        ghostLocalIndicesForGhostProcsPtr,
                                        numGhostIndices,
                                        d_blockSize);
          }
#endif // DFTEFE_WITH_MPI
      }


      template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
      void
      MPICommunicatorP2P<ValueType, memorySpace>::accumulateAddLocallyOwned(
        MemoryStorage<ValueType, memorySpace> &dataArray,
        const size_type                        communicationChannel)
      {
        accumulateAddLocallyOwnedBegin(dataArray, communicationChannel);
        accumulateAddLocallyOwnedEnd(dataArray);
      }

      template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
      void
      MPICommunicatorP2P<ValueType, memorySpace>::
        accumulateAddLocallyOwnedBegin(
          MemoryStorage<ValueType, memorySpace> &dataArray,
          const size_type                        communicationChannel)
      {
#ifdef DFTEFE_WITH_MPI
        // initiate non-blocking receives from target processors
        ValueType *recvArrayStartPtr = d_targetDataBuffer.begin();
#  if defined(DFTEFE_WITH_DEVICE) && !defined(DFTEFE_WITH_DEVICE_AWARE_MPI)
        if (memorySpace == MemorySpace::DEVICE)
          recvArrayStartPtr = d_sendRecvBufferHostPinned.begin();
#  endif // defined(DFTEFE_WITH_DEVICE) && //
         // !defined(DFTEFE_WITH_DEVICE_AWARE_MPI)

        for (size_type i = 0; i < (d_mpiPatternP2P->getTargetProcIds()).size();
             ++i)
          {
#  if defined(DFTEFE_WITH_DEVICE) && !defined(DFTEFE_WITH_DEVICE_AWARE_MPI)
            const int err = MPIIrecv<MemorySpace::HOST_PINNED>(
              recvArrayStartPtr,
              d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs().data()[i] *
                d_blockSize * sizeof(ValueType),
              MPIByte,
              d_mpiPatternP2P->getTargetProcIds().data()[i],
              static_cast<size_type>(MPITags::MPI_P2P_COMMUNICATOR_GATHER_TAG) +
                communicationChannel,
              d_mpiCommunicator,
              &d_requestsAccumulateAddLocallyOwned[i]);
#  else
            const int err = MPIIrecv<memorySpace>(
              recvArrayStartPtr,
              d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs().data()[i] *
                d_blockSize * sizeof(ValueType),
              MPIByte,
              d_mpiPatternP2P->getTargetProcIds().data()[i],
              static_cast<size_type>(MPITags::MPI_P2P_COMMUNICATOR_GATHER_TAG) +
                communicationChannel,
              d_mpiCommunicator,
              &d_requestsAccumulateAddLocallyOwned[i]);
#  endif

            const std::pair<bool, std::string> isSuccessAndMessage =
              MPIErrorCodeHandler::getIsSuccessAndMessage(err);
            throwException(isSuccessAndMessage.first,
                           isSuccessAndMessage.second);


            recvArrayStartPtr +=
              d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs().data()[i] *
              d_blockSize;
          }


        // Gather ghost data into send buffer,
        // Set the starting reference of the source data to the ghost part of
        // the data. NOTE: It assumes that the ghost part of data follows the
        // owned part.
        const ValueType *dataGhostPtr =
          dataArray.data() + d_mpiPatternP2P->localOwnedSize() * d_blockSize;

        const size_type *ghostLocalIndicesForGhostProcsPtr =
          (d_mpiPatternP2P->getGhostLocalIndicesForGhostProcs()).data();

        const size_type numGhostIndices = d_mpiPatternP2P->localGhostSize();

        DiscontiguousDataOperations<ValueType, memorySpace>::
          copyFromDiscontiguousMemory(dataGhostPtr,
                                      d_ghostDataBuffer.data(),
                                      ghostLocalIndicesForGhostProcsPtr,
                                      numGhostIndices,
                                      d_blockSize);

        // initiate non-blocking sends to ghost processors
        ValueType *sendArrayStartPtr = d_ghostDataBuffer.data();

#  if defined(DFTEFE_WITH_DEVICE) && !defined(DFTEFE_WITH_DEVICE_AWARE_MPI)
        if (memorySpace == MemorySpace::DEVICE)
          {
            MemoryTransfer<MemorySpace::HOST_PINNED, memorySpace>
              memoryTransfer;
            memoryTransfer.copy(d_ghostDataCopyHostPinned.size(),
                                d_ghostDataCopyHostPinned.begin(),
                                d_ghostDataBuffer.begin());

            sendArrayStartPtr = d_ghostDataCopyHostPinned.begin();
          }
#  endif // defined(DFTEFE_WITH_DEVICE) &&
         // !defined(DFTEFE_WITH_DEVICE_AWARE_MPI)

        for (size_type i = 0; i < (d_mpiPatternP2P->getGhostProcIds()).size();
             ++i)
          {
#  if defined(DFTEFE_WITH_DEVICE) && !defined(DFTEFE_WITH_DEVICE_AWARE_MPI)
            const int err = MPIIsend<MemorySpace::HOST_PINNED>(
              sendArrayStartPtr,
              (d_mpiPatternP2P->getGhostLocalIndicesRanges().data()[2 * i + 1] -
               d_mpiPatternP2P->getGhostLocalIndicesRanges().data()[2 * i]) *
                d_blockSize * sizeof(ValueType),
              MPIByte,
              d_mpiPatternP2P->getGhostProcIds().data()[i],
              static_cast<size_type>(MPITags::MPI_P2P_COMMUNICATOR_GATHER_TAG) +
                communicationChannel,
              d_mpiCommunicator,
              &d_requestsAccumulateAddLocallyOwned
                [(d_mpiPatternP2P->getTargetProcIds()).size() + i]);
#  else
            const int err = MPIIsend<memorySpace>(
              sendArrayStartPtr,
              (d_mpiPatternP2P->getGhostLocalIndicesRanges().data()[2 * i + 1] -
               d_mpiPatternP2P->getGhostLocalIndicesRanges().data()[2 * i]) *
                d_blockSize * sizeof(ValueType),
              MPIByte,
              d_mpiPatternP2P->getGhostProcIds().data()[i],
              static_cast<size_type>(MPITags::MPI_P2P_COMMUNICATOR_GATHER_TAG) +
                communicationChannel,
              d_mpiCommunicator,
              &d_requestsAccumulateAddLocallyOwned
                [(d_mpiPatternP2P->getTargetProcIds()).size() + i]);
#  endif


            const std::pair<bool, std::string> isSuccessAndMessage =
              MPIErrorCodeHandler::getIsSuccessAndMessage(err);
            throwException(isSuccessAndMessage.first,
                           isSuccessAndMessage.second);

            sendArrayStartPtr +=
              (d_mpiPatternP2P->getGhostLocalIndicesRanges().data()[2 * i + 1] -
               d_mpiPatternP2P->getGhostLocalIndicesRanges().data()[2 * i]) *
              d_blockSize;
          }
#endif // DFTEFE_WITH_MPI
      }


      template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
      void
      MPICommunicatorP2P<ValueType, memorySpace>::accumulateAddLocallyOwnedEnd(
        MemoryStorage<ValueType, memorySpace> &dataArray)
      {
#ifdef DFTEFE_WITH_MPI
        // wait for all send and recv requests to be completed
        if (d_requestsAccumulateAddLocallyOwned.size() > 0)
          {
            const int err =
              MPIWaitall(d_requestsAccumulateAddLocallyOwned.size(),
                         d_requestsAccumulateAddLocallyOwned.data(),
                         MPIStatusesIgnore);

            const std::pair<bool, std::string> isSuccessAndMessage =
              MPIErrorCodeHandler::getIsSuccessAndMessage(err);
            throwException(isSuccessAndMessage.first,
                           isSuccessAndMessage.second);

#  if defined(DFTEFE_WITH_DEVICE) && !defined(DFTEFE_WITH_DEVICE_AWARE_MPI)
            if (memorySpace == MemorySpace::DEVICE)
              {
                MemoryTransfer<MemorySpace::HOST_PINNED, memorySpace>
                  memoryTransfer;
                memoryTransfer.copy(d_sendRecvBufferHostPinned.size(),
                                    d_sendRecvBufferHostPinned.data(),
                                    d_targetDataBuffer.data());
              }
#  endif // defined(DFTEFE_WITH_DEVICE) &&
         // !defined(DFTEFE_WITH_DEVICE_AWARE_MPI)
          }

        // accumulate add into locally owned entries from recv buffer
        const size_type *ownedLocalIndicesForTargetProcsPtr =
          (d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs()).data();

        const size_type numTotalOwnedIndicesForTargetProcs =
          d_mpiPatternP2P->getTotalOwnedIndicesForTargetProcs();

        DiscontiguousDataOperations<ValueType, memorySpace>::
          addToDiscontiguousMemory(d_targetDataBuffer.data(),
                                   dataArray.data(),
                                   ownedLocalIndicesForTargetProcsPtr,
                                   numTotalOwnedIndicesForTargetProcs,
                                   d_blockSize);
#endif // DFTEFE_WITH_MPI
      }

      template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
      std::shared_ptr<const MPIPatternP2P<memorySpace>>
      MPICommunicatorP2P<ValueType, memorySpace>::getMPIPatternP2P() const
      {
        return d_mpiPatternP2P;
      }

      template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
      int
      MPICommunicatorP2P<ValueType, memorySpace>::getBlockSize() const
      {
        return d_blockSize;
      }

    } // namespace mpi
  }   // namespace utils
} // namespace dftefe
