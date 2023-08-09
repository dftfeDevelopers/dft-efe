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
#include <utils/MPICommunicatorP2PKernels.h>
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
        d_sendRecvBuffer.resize(
          d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs().size() *
            blockSize,
          0.0);
       d_ghostDataRecvBuffer.resize(
          d_mpiPatternP2P->getGhostLocalIndicesForGhostProcs().size() *
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
        ValueType *recvArrayStartPtr =d_ghostDataRecvBuffer.begin();

#  if defined(DFTEFE_WITH_DEVICE) && !defined(DFTEFE_WITH_DEVICE_AWARE_MPI)
        if (memorySpace == MemorySpace::DEVICE)
          recvArrayStartPtr = d_ghostDataCopyHostPinned.begin();
#  endif // defined(DFTEFE_WITH_DEVICE) &&
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
        MPICommunicatorP2PKernels<ValueType, memorySpace>::
          gatherLocallyOwnedEntriesSendBufferToTargetProcs(
            dataArray,
            d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs(),
            d_blockSize,
            d_sendRecvBuffer);

        // initiate non-blocking sends to target processors
        ValueType *sendArrayStartPtr = d_sendRecvBuffer.begin();

#  if defined(DFTEFE_WITH_DEVICE) && !defined(DFTEFE_WITH_DEVICE_AWARE_MPI)
        if (memorySpace == MemorySpace::DEVICE)
          {
            MemoryTransfer<MemorySpace::HOST_PINNED, memorySpace>
              memoryTransfer;
            memoryTransfer.copy(d_sendRecvBufferHostPinned.size(),
                                d_sendRecvBufferHostPinned.begin(),
                                d_sendRecvBuffer.begin());

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
                                    dataArray.begin() +
                                      d_mpiPatternP2P->localOwnedSize() *
                                        d_blockSize,
                                    d_ghostDataCopyHostPinned.data());
              }
#  endif // defined(DFTEFE_WITH_DEVICE) &&
         // !defined(DFTEFE_WITH_DEVICE_AWARE_MPI)
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
        ValueType *recvArrayStartPtr = d_sendRecvBuffer.begin();
#  if defined(DFTEFE_WITH_DEVICE) && !defined(DFTEFE_WITH_DEVICE_AWARE_MPI)
        if (memorySpace == MemorySpace::DEVICE)
          recvArrayStartPtr = d_sendRecvBufferHostPinned.begin();
#  endif // defined(DFTEFE_WITH_DEVICE) &&
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



        // initiate non-blocking sends to ghost processors
        ValueType *sendArrayStartPtr =
          dataArray.begin() + d_mpiPatternP2P->localOwnedSize() * d_blockSize;

#  if defined(DFTEFE_WITH_DEVICE) && !defined(DFTEFE_WITH_DEVICE_AWARE_MPI)
        if (memorySpace == MemorySpace::DEVICE)
          {
            MemoryTransfer<MemorySpace::HOST_PINNED, memorySpace>
              memoryTransfer;
            memoryTransfer.copy(d_ghostDataCopyHostPinned.size(),
                                d_ghostDataCopyHostPinned.begin(),
                                dataArray.begin() +
                                  d_mpiPatternP2P->localOwnedSize() *
                                    d_blockSize);

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
                                    d_sendRecvBuffer.data());
              }
#  endif // defined(DFTEFE_WITH_DEVICE) &&
         // !defined(DFTEFE_WITH_DEVICE_AWARE_MPI)
          }

        // accumulate add into locally owned entries from recv buffer
        MPICommunicatorP2PKernels<ValueType, memorySpace>::
          accumAddLocallyOwnedContrRecvBufferFromTargetProcs(
            d_sendRecvBuffer,
            d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs(),
            d_blockSize,
            dataArray);
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
