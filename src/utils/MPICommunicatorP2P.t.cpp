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

#include <utils/Exceptions.h>


namespace dftefe
{
  namespace utils
  {
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    MPICommunicatorP2P<ValueType, memorySpace>::MPICommunicatorP2P(
      const std::shared_ptr<const MPIPatternP2P<memorySpace>> &mpiPatternP2P,
      const size_type                                          blockSize)
      : d_mpiPatternP2P(mpiPatternP2P)
      , d_blockSize(blockSize)
    {
#ifdef DFTEFE_WITH_MPI
      d_mpiCommunicator = d_mpiPatternP2P->mpiCommunicator();
      d_sendRecvBuffer.resize(
        d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs() * blockSize);
      d_recvRequestsScatterToGhost.resize(
        (d_mpiPatternP2P->getGhostProcIds()).size());
      d_sendRequestsScatterToGhost.resize(
        (d_mpiPatternP2P->getTargetProcIds()).size());
      d_recvRequestsGatherFromGhost.resize(
        (d_mpiPatternP2P->getTargetProcIds()).size());
      d_sendRequestsGatherFromGhost.resize(
        (d_mpiPatternP2P->getGhostProcIds()).size());
#endif
    }


    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    MPICommunicatorP2P<ValueType, memorySpace>::scatterToGhost(
      MemoryStorage<ValueType, memorySpace> &dataArray,
      const size_type                        communicationChannel)
    {
#ifdef DFTEFE_WITH_MPI
      scatterToGhostBegin(dataArray);
      scatterToGhostEnd(dataArray);
#endif
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    MPICommunicatorP2P<ValueType, memorySpace>::scatterToGhostBegin(
      MemoryStorage<ValueType, memorySpace> &dataArray,
      const size_type                        communicationChannel)
    {
#ifdef DFTEFE_WITH_MPI
      ValueType *recvArrayStartPtr =
        dataArray.begin() + d_mpiPatternP2P->localOwnedSize() * d_blockSize;
      for (size_type i = 0; i < (d_mpiPatternP2P->getGhostProcIds()).size();
           ++i)
        {
          const int err = MPI_Irecv(
            recvArrayStartPtr,
            (d_mpiPatternP2P->getGhostLocalIndicesRanges()[2 * i + 1] -
             d_mpiPatternP2P->getGhostLocalIndicesRanges()[2 * i]) *
              d_blockSize * sizeof(ValueType),
            MPI_BYTE,
            d_mpiPatternP2P->getGhostProcIds().[i],
            MPI_P2P_COMMUNICATOR_SCATTER_TAG + communicationChannel,
            d_mpiCommunicator,
            &d_recvRequestsScatterToGhost[i]);

          std::string errMsg = "Error occured while using MPI_Irecv. "
                               "Error code: " +
                               std::to_string(err);
          throwException(err == MPI_SUCCESS, errMsg);

          recvArrayStartPtr +=
            (d_mpiPatternP2P->getGhostLocalIndicesRanges()[2 * i + 1] -
             d_mpiPatternP2P->getGhostLocalIndicesRanges()[2 * i]) *
            d_blockSize;
        }


      ValueType *sendArrayStartPtr = d_sendRecvBuffer.begin();
      for (unsigned int i = 0; i < (d_mpiPatternP2P->getTargetProcIds()).size();
           ++i)
        {
          const int err =
            MPI_Isend(sendArrayStart,
                      (d_mpiPatternP2P->getNumOwnedIndicesForTargetProc(i)*d_blockSize* sizeof(ValueType),
                      MPI_BYTE,
                      d_mpiPatternP2P->getTargetProcIds().[i],
                      MPI_P2P_COMMUNICATOR_SCATTER_TAG+communicationChannel,
                      d_mpiCommunicator,
                      &d_sendRequestsScatterToGhost[i]);

          std::string errMsg= "Error occured while using MPI_Isend. "
                               "Error code: " +
                               std::to_string(err);
          throwException(err == MPI_SUCCESS, errMsg);

          sendArrayStartPtr += d_mpiPatternP2P->getNumOwnedIndicesForTargetProc(i)*d_blockSize;
        }

#endif
    }


    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    MPICommunicatorP2P<ValueType, memorySpace>::scatterToGhostEnd(
      MemoryStorage<ValueType, memorySpace> &dataArray,
      const size_type                        communicationChannel)
    {
#ifdef DFTEFE_WITH_MPI
      if (d_sendRequestsScatterToGhost.size() > 0)
        {
          const int ierr = MPI_Waitall(d_sendRequestsScatterToGhost.size(),
                                       d_sendRequestsScatterToGhost.data(),
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
      gatherFromGhostBegin(dataArray);
      gatherFromGhostEnd(dataArray);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    MPICommunicatorP2P<ValueType, memorySpace>::gatherFromGhostBegin(
      MemoryStorage<ValueType, memorySpace> &dataArray,
      const size_type                        communicationChannel)
    {
#ifdef DFTEFE_WITH_MPI
#endif
    }


    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    MPICommunicatorP2P<ValueType, memorySpace>::gatherFromGhostEnd(
      MemoryStorage<ValueType, memorySpace> &dataArray,
      const size_type                        communicationChannel)
    {
#ifdef DFTEFE_WITH_MPI
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
