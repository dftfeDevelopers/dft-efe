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

#include <utils/MPIRequestersNBX.h>
#include <utils/MPIWrapper.h>
#include <utils/MPITags.h>
#include <utils/Exceptions.h>
#include <string>
namespace dftefe
{
  namespace utils
  {
    namespace mpi
    {
      MPIRequestersNBX::MPIRequestersNBX(
        const std::vector<size_type> &targetIDs,
        const MPIComm &               comm)
        : d_targetIDs(targetIDs)
        , d_comm(comm)
        , d_recvBuffers(0)
        , d_recvRequests(0)
      {
        d_myRank           = 0;
        d_numProcessors    = 1;
        int         err    = MPICommSize(d_comm, &d_numProcessors);
        std::string errMsg = "Error occured while using MPI_Comm_size. "
                             "Error code: " +
                             std::to_string(err);
        throwException(err == MPISuccess, errMsg);

        err    = MPICommRank(d_comm, &d_myRank);
        errMsg = "Error occured while using MPI_Comm_rank. "
                 "Error code: " +
                 std::to_string(err);
        throwException(err == MPISuccess, errMsg);
      }

      std::vector<size_type>
      MPIRequestersNBX::getRequestingRankIds()
      {
        startLocalSend();

        while (haveAllLocalSendReceived() == false)
          probeAndReceiveIncomingMsg();

        signalLocalSendCompletion();

        while (haveAllIncomingMsgsReceived() == false)
          probeAndReceiveIncomingMsg();

        finish();

        return std::vector<size_type>(d_requestingProcesses.begin(),
                                      d_requestingProcesses.end());
      }

      void
      MPIRequestersNBX::startLocalSend()
      {
#ifdef DFTEFE_WITH_MPI
        const size_type numTargets = d_targetIDs.size();
        const int       tag = static_cast<int>(MPITags::MPI_REQUESTERS_NBX_TAG);

        d_sendRequests.resize(numTargets);
        d_sendBuffers.resize(numTargets);
        for (unsigned int i = 0; i < numTargets; ++i)
          {
            const unsigned int rank = d_targetIDs[i];
            throwException<DomainError>(
              rank < d_numProcessors,
              "Target rank " + std::to_string(rank) +
                " is outside the range of number of processors(i.e., " +
                std::to_string(d_numProcessors) + ")");

            int &sendBuffer = d_sendBuffers[i];
            auto err        = MPIIssend(
              &sendBuffer, 1, MPIInt, rank, tag, d_comm, &d_sendRequests[i]);

            std::string errMsg = "Error occured while using MPI_ISsend. "
                                 "Error code: " +
                                 std::to_string(err);
            throwException(err == MPISuccess, errMsg);
          }
#endif
      }

      void
      MPIRequestersNBX::probeAndReceiveIncomingMsg()
      {
#ifdef DFTEFE_WITH_MPI

        const int tag = static_cast<int>(MPITags::MPI_REQUESTERS_NBX_TAG);

        // Check if there is an incoming message to be received.
        // If yes, extract the source rank and then receive the
        // message i
        MPIStatus status;
        int       foundIncomingMsg;
        int       err =
          MPIIprobe(MPIAnySource, tag, d_comm, &foundIncomingMsg, &status);
        std::string errMsg = "Error occured while using MPI_Iprobe. "
                             "Error code: " +
                             std::to_string(err);
        throwException(err == MPISuccess, errMsg);

        if (foundIncomingMsg != 0)
          {
            // Get the rank of the source process
            // and add it to the set of requesting processes
            const auto sourceRank = status.MPI_SOURCE;

            //
            // Check if the source process has already sent message.
            // It is supposed to send message only once
            //
            bool hasRankAlreadySent = (d_requestingProcesses.find(sourceRank) !=
                                       d_requestingProcesses.end());
            errMsg =
              "Process " + std::to_string(sourceRank) +
              " is sending message to " + std::to_string(d_myRank) +
              " second time.\n"
              "The NBX algorithm is designed to receive at most one incoming"
              " message from any source process.";
            throwException(hasRankAlreadySent == false, errMsg);
            d_requestingProcesses.insert(sourceRank);

            //
            // get the current size of receive buffers
            //
            size_type N = d_recvBuffers.size();

            //
            // increase the size of receive buffers and
            // receive requests by 1 to allocate memory
            // for this found incoming message
            //
            int dummyVal = 0;
            d_recvBuffers.push_back(dummyVal);
            MPIRequest request;
            d_recvRequests.push_back(request);

            err    = MPIIrecv(&d_recvBuffers[N],
                           1,
                           MPIInt,
                           sourceRank,
                           tag,
                           d_comm,
                           &d_recvRequests[N]);
            errMsg = "Error occured while using MPI_Irecv. "
                     "Error code: " +
                     std::to_string(err);
            throwException(err == MPISuccess, errMsg);
          }
#endif
      }

      bool
      MPIRequestersNBX::haveAllLocalSendReceived()
      {
        if (d_sendRequests.size() > 0)
          {
            int         allLocalSendCompletedFlag;
            const auto  err    = MPITestall(d_sendRequests.size(),
                                        d_sendRequests.data(),
                                        &allLocalSendCompletedFlag,
                                        MPIStatusesIgnore);
            std::string errMsg = "Error occured while using MPI_TestAll. "
                                 " Error code: " +
                                 std::to_string(err);
            throwException(err == MPISuccess, errMsg);

            return allLocalSendCompletedFlag != 0;
          }
        else
          return true;
      }

      void
      MPIRequestersNBX::signalLocalSendCompletion()
      {
        const auto  err    = MPIIbarrier(d_comm, &d_barrierRequest);
        std::string errMsg = "Error occured while using MPI_Ibarrier. "
                             " Error code: " +
                             std::to_string(err);
        throwException(err == MPISuccess, errMsg);
      }

      bool
      MPIRequestersNBX::haveAllIncomingMsgsReceived()
      {
        int         allProcessorsInvokedIBarrier;
        const auto  err    = MPITest(&d_barrierRequest,
                                 &allProcessorsInvokedIBarrier,
                                 MPIStatusesIgnore);
        std::string errMsg = "Error occured while using MPI_Test. "
                             " Error code: " +
                             std::to_string(err);
        throwException(err == MPISuccess, errMsg);
        return allProcessorsInvokedIBarrier != 0;
      }

      void
      MPIRequestersNBX::finish()
      {
#ifdef DFTEFE_WITH_MPI
        if (d_sendRequests.size() > 0)
          {
            const int   err    = MPIWaitall(d_sendRequests.size(),
                                       d_sendRequests.data(),
                                       MPIStatusesIgnore);
            std::string errMsg = "Error occured while using MPI_Waitall. "
                                 " Error code: " +
                                 std::to_string(err);
            throwException(err == MPISuccess, errMsg);
          }

        if (d_recvRequests.size() > 0)
          {
            const int   err    = MPIWaitall(d_recvRequests.size(),
                                       d_recvRequests.data(),
                                       MPIStatusesIgnore);
            std::string errMsg = "Error occured while using MPI_Waitall. "
                                 " Error code: " +
                                 std::to_string(err);
            throwException(err == MPISuccess, errMsg);
          }

        int         err    = MPIWait(&d_barrierRequest, MPIStatusIgnore);
        std::string errMsg = "Error occured while using MPI_Wait. "
                             " Error code: " +
                             std::to_string(err);
        throwException(err == MPISuccess, errMsg);

#  ifndef NDEBUG
        // note: MPI_Ibarrier seems to make problem during testing, this
        // additional Barrier seems to help
        err    = MPIBarrier(d_comm);
        errMsg = "Error occured while using MPI_Barrier. "
                 " Error code: " +
                 std::to_string(err);
        throwException(err == MPISuccess, errMsg);
#  endif
#endif
      }
    } // namespace mpi
  }   // namespace utils
} // namespace dftefe
