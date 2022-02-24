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
      const size_type                                          blockSize)
      : d_mpiPatternP2P(mpiPatternP2P)
      , d_blockSize(blockSize)
    {
#ifdef DFTEFE_WITH_MPI        
      d_mpiCommunicator=d_mpiPatternP2P->mpiCommunicator();
      d_sendRecvBuffer.resize(d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs()*blockSize);
      d_requestsScatterToGhost.resize(d_mpiPatternP2P->getGhostProcIds().size()+d_mpiPatternP2P->getTargetProcIds().size());
      d_requestsGatherFromGhost.resize(d_mpiPatternP2P->getGhostProcIds().size()+d_mpiPatternP2P->getTargetProcIds().size());
#endif      
    }


    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    MPICommunicatorP2P<ValueType, memorySpace>::scatterToGhost(
      MemoryStorage<ValueType, memorySpace> &dataArray,
      const size_type                        communicationChannel)
    {
#ifdef DFTEFE_WITH_MPI
      scatterToGhostBegin(dataArray,
                          communicationChannel);
      scatterToGhostEnd();
#endif
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    MPICommunicatorP2P<ValueType, memorySpace>::scatterToGhostBegin(
      MemoryStorage<ValueType, memorySpace> &dataArray,
      const size_type                        communicationChannel)
    {
#ifdef DFTEFE_WITH_MPI
      // initiate non-blocking receives from ghost processors 
      ValueType * recvArrayStartPtr=dataArray.begin()+d_mpiPatternP2P->localOwnedSize()*d_blockSize;
      for (size_type i = 0; i < (d_mpiPatternP2P->getGhostProcIds()).size(); ++i)
        {
          const int err =
            MPI_Irecv(recvArrayStartPtr,
                      (d_mpiPatternP2P->getGhostLocalIndicesRanges()[2*i+1]-d_mpiPatternP2P->getGhostLocalIndicesRanges()[2*i])*d_blockSize* sizeof(ValueType),
                      MPI_BYTE,
                      d_mpiPatternP2P->getGhostProcIds()[i],
                      static_cast<size_type>(MPITags::MPI_P2P_COMMUNICATOR_SCATTER_TAG)+communicationChannel,
                      d_mpiCommunicator,
                      &d_requestsScatterToGhost[i]);

          std::string errMsg= "Error occured while using MPI_Irecv. "
                               "Error code: " +
                               std::to_string(err);
          throwException(err == MPI_SUCCESS, errMsg);

          recvArrayStartPtr +=
            (d_mpiPatternP2P->getGhostLocalIndicesRanges()[2 * i + 1] -
             d_mpiPatternP2P->getGhostLocalIndicesRanges()[2 * i]) *
            d_blockSize;
        }

      // gather locally owned entries into a contiguous send buffer
      MPICommunicatorP2PKernels<ValueType,memorySpace>::gatherLocallyOwnedEntriesToSendBuffer(dataArray,
                                                                       d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs(),
                                                                       d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs(),
                                                                       d_blockSize,
                                                                       d_sendRecvBuffer);

      // initiate non-blocking sends to target processors
      ValueType * sendArrayStartPtr=d_sendRecvBuffer.begin();
      for (unsigned int i = 0; i < (d_mpiPatternP2P->getTargetProcIds()).size(); ++i)
        {
          const int err =
            MPI_Isend(sendArrayStartPtr,
                      d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs()[i]*d_blockSize* sizeof(ValueType),
                      MPI_BYTE,
                      d_mpiPatternP2P->getTargetProcIds()[i],
                      static_cast<size_type>(MPITags::MPI_P2P_COMMUNICATOR_SCATTER_TAG)+communicationChannel,
                      d_mpiCommunicator,
                      &d_requestsScatterToGhost[d_mpiPatternP2P->getGhostProcIds().size()+i]);

          std::string errMsg= "Error occured while using MPI_Isend. "
                               "Error code: " +
                               std::to_string(err);
          throwException(err == MPI_SUCCESS, errMsg);

          sendArrayStartPtr += d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs()[i]*d_blockSize;
        }

#endif
    }


    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    MPICommunicatorP2P<ValueType, memorySpace>::scatterToGhostEnd()
    {
#ifdef DFTEFE_WITH_MPI
      // wait for all send and recv requests to be completed      
      if (d_requestsScatterToGhost.size() > 0)
        {
          const int err =
            MPI_Waitall(d_requestsScatterToGhost.size(), 
                        d_requestsScatterToGhost.data(),
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
    MPICommunicatorP2P<ValueType, memorySpace>::gatherFromGhost(
      MemoryStorage<ValueType, memorySpace> &dataArray,
      const size_type                        communicationChannel)
    {
      gatherFromGhostBegin(dataArray,
                           communicationChannel);
      gatherFromGhostEnd(dataArray);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    MPICommunicatorP2P<ValueType, memorySpace>::gatherFromGhostBegin(
      MemoryStorage<ValueType, memorySpace> &dataArray,
      const size_type                        communicationChannel)
    {
#ifdef DFTEFE_WITH_MPI
      // initiate non-blocking receives from target processors 
      ValueType * recvArrayStartPtr=d_sendRecvBuffer.begin();
      for (size_type i = 0; i < (d_mpiPatternP2P->getTargetProcIds()).size(); ++i)
        {
          const int err =
            MPI_Irecv(recvArrayStartPtr,
                      d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs()[i]*d_blockSize* sizeof(ValueType),
                      MPI_BYTE,
                      d_mpiPatternP2P->getTargetProcIds()[i],
                      static_cast<size_type>(MPITags::MPI_P2P_COMMUNICATOR_GATHER_TAG)+communicationChannel,
                      d_mpiCommunicator,
                      &d_requestsGatherFromGhost[i]);

          std::string errMsg= "Error occured while using MPI_Irecv. "
                               "Error code: " +
                               std::to_string(err);
          throwException(err == MPI_SUCCESS, errMsg);

          recvArrayStartPtr +=d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs()[i]*d_blockSize;
        }     



      // initiate non-blocking sends to ghost processors
      ValueType * sendArrayStartPtr=dataArray.begin()+d_mpiPatternP2P->localOwnedSize()*d_blockSize;
      for (unsigned int i = 0; i < (d_mpiPatternP2P->getTargetProcIds()).size(); ++i)
        {
          const int err =
            MPI_Isend(sendArrayStartPtr,
                      (d_mpiPatternP2P->getGhostLocalIndicesRanges()[2*i+1]-d_mpiPatternP2P->getGhostLocalIndicesRanges()[2*i])*d_blockSize* sizeof(ValueType),
                      MPI_BYTE,
                      d_mpiPatternP2P->getGhostProcIds()[i],
                      static_cast<size_type>(MPITags::MPI_P2P_COMMUNICATOR_GATHER_TAG)+communicationChannel,
                      d_mpiCommunicator,
                      &d_requestsGatherFromGhost[i]);

          std::string errMsg= "Error occured while using MPI_Isend. "
                               "Error code: " +
                               std::to_string(err);
          throwException(err == MPI_SUCCESS, errMsg);

          sendArrayStartPtr += (d_mpiPatternP2P->getGhostLocalIndicesRanges()[2*i+1]-d_mpiPatternP2P->getGhostLocalIndicesRanges()[2*i])*d_blockSize;

        }

      
#endif
    }


    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    MPICommunicatorP2P<ValueType, memorySpace>::gatherFromGhostEnd(
      MemoryStorage<ValueType, memorySpace> &dataArray)
    {
#ifdef DFTEFE_WITH_MPI

      // wait for all send and recv requests to be completed
      if (d_requestsGatherFromGhost.size()>0)
        {
          const int err =
            MPI_Waitall(d_requestsGatherFromGhost.size(),
                        d_requestsGatherFromGhost.data(),
                         MPI_STATUSES_IGNORE);
                  
          std::string errMsg= "Error occured while using MPI_Waitall. "
                               "Error code: " +
                               std::to_string(err);
          throwException(err == MPI_SUCCESS, errMsg);
        }

      // accumulate add into locally owned entries from recv buffer
      MPICommunicatorP2PKernels<ValueType,memorySpace>::accumulateAddRecvBufferToLocallyOwnedEntries(d_sendRecvBuffer,
                                                                       d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs(),
                                                                       d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs(),
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
