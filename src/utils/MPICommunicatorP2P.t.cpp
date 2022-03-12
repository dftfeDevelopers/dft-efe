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


namespace dftefe
{
  namespace utils
  {
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    MPICommunicatorP2P<ValueType, memorySpace>::MPICommunicatorP2P(
      std::shared_ptr<const MPIPatternP2P<memorySpace>> mpiPatternP2P,
      const size_type                                   blockSize)
      : d_mpiPatternP2P(mpiPatternP2P)
      , d_blockSize(blockSize)
    {
#ifdef DFTEFE_WITH_MPI
      d_mpiCommunicator = d_mpiPatternP2P->mpiCommunicator();
      d_sendRecvBuffer.resize(
        d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs().size() * blockSize);
      d_requestsUpdateGhostValues.resize(
        d_mpiPatternP2P->getGhostProcIds().size() +
        d_mpiPatternP2P->getTargetProcIds().size());
      d_requestsAccumulateAddLocallyOwned.resize(
        d_mpiPatternP2P->getGhostProcIds().size() +
        d_mpiPatternP2P->getTargetProcIds().size());
#endif
    }


    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    MPICommunicatorP2P<ValueType, memorySpace>::updateGhostValues(
      MemoryStorage<ValueType, memorySpace> &dataArray,
      const size_type                        communicationChannel)
    {
#ifdef DFTEFE_WITH_MPI
      updateGhostValuesBegin(dataArray, communicationChannel);
      updateGhostValuesEnd();
#endif
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    MPICommunicatorP2P<ValueType, memorySpace>::updateGhostValuesBegin(
      MemoryStorage<ValueType, memorySpace> &dataArray,
      const size_type                        communicationChannel)
    {
#ifdef DFTEFE_WITH_MPI
      // initiate non-blocking receives from ghost processors
      ValueType *recvArrayStartPtr =
        dataArray.begin() + d_mpiPatternP2P->localOwnedSize() * d_blockSize;
      for (size_type i = 0; i < (d_mpiPatternP2P->getGhostProcIds()).size();
           ++i)
        {
          const int err = MPI_Irecv(
            recvArrayStartPtr,
            (d_mpiPatternP2P->getGhostLocalIndicesRanges().data()[2 * i + 1] -
             d_mpiPatternP2P->getGhostLocalIndicesRanges().data()[2 * i]) *
              d_blockSize * sizeof(ValueType),
            MPI_BYTE,
            d_mpiPatternP2P->getGhostProcIds().data()[i],
            static_cast<size_type>(MPITags::MPI_P2P_COMMUNICATOR_SCATTER_TAG) +
              communicationChannel,
            d_mpiCommunicator,
            &d_requestsUpdateGhostValues[i]);

          std::string errMsg = "Error occured while using MPI_Irecv. "
                               "Error code: " +
                               std::to_string(err);
          throwException(err == MPI_SUCCESS, errMsg);

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
      for (size_type i = 0; i < (d_mpiPatternP2P->getTargetProcIds()).size();
           ++i)
        {
          const int err = MPI_Isend(
            sendArrayStartPtr,
            d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs().data()[i] *
              d_blockSize * sizeof(ValueType),
            MPI_BYTE,
            d_mpiPatternP2P->getTargetProcIds().data()[i],
            static_cast<size_type>(MPITags::MPI_P2P_COMMUNICATOR_SCATTER_TAG) +
              communicationChannel,

            d_mpiCommunicator,
            &d_requestsUpdateGhostValues
              [d_mpiPatternP2P->getGhostProcIds().size() + i]);

          std::string errMsg = "Error occured while using MPI_Isend. "
                               "Error code: " +
                               std::to_string(err);
          throwException(err == MPI_SUCCESS, errMsg);

          sendArrayStartPtr +=
            d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs().data()[i] *
            d_blockSize;
        }

#endif
    }


    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    MPICommunicatorP2P<ValueType, memorySpace>::updateGhostValuesEnd()
    {
#ifdef DFTEFE_WITH_MPI
      // wait for all send and recv requests to be completed
      if (d_requestsUpdateGhostValues.size() > 0)
        {
          const int err = MPI_Waitall(d_requestsUpdateGhostValues.size(),
                                      d_requestsUpdateGhostValues.data(),
                                      MPI_STATUSES_IGNORE);

          std::string errMsg = "Error occured while using MPI_Waitall. "
                               "Error code: " +
                               std::to_string(err);
          throwException(err == MPI_SUCCESS, errMsg);
        }
#endif
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
    MPICommunicatorP2P<ValueType, memorySpace>::accumulateAddLocallyOwnedBegin(
      MemoryStorage<ValueType, memorySpace> &dataArray,
      const size_type                        communicationChannel)
    {
#ifdef DFTEFE_WITH_MPI
      // initiate non-blocking receives from target processors
      ValueType *recvArrayStartPtr = d_sendRecvBuffer.begin();
      for (size_type i = 0; i < (d_mpiPatternP2P->getTargetProcIds()).size();
           ++i)
        {
          const int err = MPI_Irecv(
            recvArrayStartPtr,
            d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs().data()[i] *
              d_blockSize * sizeof(ValueType),
            MPI_BYTE,
            d_mpiPatternP2P->getTargetProcIds().data()[i],
            static_cast<size_type>(MPITags::MPI_P2P_COMMUNICATOR_GATHER_TAG) +
              communicationChannel,
            d_mpiCommunicator,
            &d_requestsAccumulateAddLocallyOwned[i]);

          std::string errMsg = "Error occured while using MPI_Irecv. "
                               "Error code: " +
                               std::to_string(err);
          throwException(err == MPI_SUCCESS, errMsg);

          recvArrayStartPtr +=
            d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs().data()[i] *
            d_blockSize;
        }



      // initiate non-blocking sends to ghost processors
      ValueType *sendArrayStartPtr =
        dataArray.begin() + d_mpiPatternP2P->localOwnedSize() * d_blockSize;
      for (size_type i = 0; i < (d_mpiPatternP2P->getTargetProcIds()).size();
           ++i)
        {
          const int err = MPI_Isend(
            sendArrayStartPtr,
            (d_mpiPatternP2P->getGhostLocalIndicesRanges().data()[2 * i + 1] -
             d_mpiPatternP2P->getGhostLocalIndicesRanges().data()[2 * i]) *
              d_blockSize * sizeof(ValueType),
            MPI_BYTE,
            (d_mpiPatternP2P->getGhostProcIds())[i],
            static_cast<size_type>(MPITags::MPI_P2P_COMMUNICATOR_GATHER_TAG) +
              communicationChannel,
            d_mpiCommunicator,
            &d_requestsAccumulateAddLocallyOwned[i]);

          std::string errMsg = "Error occured while using MPI_Isend. "
                               "Error code: " +
                               std::to_string(err);
          throwException(err == MPI_SUCCESS, errMsg);

          sendArrayStartPtr +=
            (d_mpiPatternP2P->getGhostLocalIndicesRanges().data()[2 * i + 1] -
             d_mpiPatternP2P->getGhostLocalIndicesRanges().data()[2 * i]) *
            d_blockSize;
        }


#endif
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
            MPI_Waitall(d_requestsAccumulateAddLocallyOwned.size(),
                        d_requestsAccumulateAddLocallyOwned.data(),
                        MPI_STATUSES_IGNORE);

          std::string errMsg = "Error occured while using MPI_Waitall. "
                               "Error code: " +
                               std::to_string(err);
          throwException(err == MPI_SUCCESS, errMsg);
        }

      // accumulate add into locally owned entries from recv buffer
      MPICommunicatorP2PKernels<ValueType, memorySpace>::
        accumAddLocallyOwnedContrRecvBufferFromTargetProcs(
          d_sendRecvBuffer,
          d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs(),
          d_blockSize,
          dataArray);
#endif
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    std::shared_ptr<const MPIPatternP2P<memorySpace>>
    MPICommunicatorP2P<ValueType, memorySpace>::getMPIPatternP2P() const
    {
      return d_mpiPatternP2P;
    }
  } // namespace utils
} // namespace dftefe
