
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

#include <utils/MPIPatternP2P.h>
#include <utils/MPITags.h>
#include <utils/Exceptions.h>
#include <string>

namespace dftefe
{
  namespace utils
  {
    getGhostProcIdToLocalGhostIndicesMap(
      const std::vector<global_size_type> &ghostIndices,
      const global_size_type               ownedRangeStart,
      const global_size_type               ownedRangeEnd,
      std::map<size_type, std::vector<size_type>>
        &             ghostProcIdToLocalGhostIndices;
      const MPI_Comm &mpiComm)
    {
      ghostProcIds.resize(0);
      int         nprocs = 1;
      int         err    = MPI_Comm_size(mpiComm, &nprocs);
      std::string errMsg = "Error occured while using MPI_Comm_size. "
                           "Error code: " +
                           std::to_string(err);
      throwException(err == MPI_SUCCESS, errMsg);
      std::vector<global_size_type> locallyOwnedRanges(2 * nProcs);
      std::vector<int>              recvCounts(nProcs, 2);
      std::vector<int>              displs(nProcs, 0);
      for (unsigned int i = 0; i < nprocs; ++i)
        displs[i] = 2 * i;

      std::vector<global_size_type> ownedRanges = {ownedRangeStart,
                                                   ownedRangeEnd};
      MPI_Allgatherv(&ownedRanges[0],
                     2,
                     MPI_UNSIGNED_LONG,
                     &locallyOwnedRanges[0],
                     &recvCounts[0],
                     &displs[0],
                     MPI_UNSIGNED_LONG,
                     mpiComm);

      //
      // NOTE: The locally owned ranges need not be ordered as per the
      // processor ranks. That is ranges for processor 0, 1, ...., P-1 given by
      // [N_0,N_1), [N_1, N_2), [N_2, N_3), ..., [N_{P-1},N_P) need not honor
      // the fact that N_0, N_1, ..., N_P are increasing. However, it is more
      // efficient to perform search operations in a sorted vector. Thus, we
      // perform a sort on the end of each locally owned range and also keep
      // track of the indices during the sort
      //

      // vector to store the end of each locally owned ranges
      std::vector<global_size_type> locallyOwnedRangesEnd(nprocs);

      //
      // Vector to keep track of the  indices of locallyOwnedRangesEnd
      // during sort
      std::vector<size_type> locallyOwnedRangesEndProcIds(nprocs);
      for (unsigned int i = 0; i < nprocs; ++i)
        {
          locallyOwnedRangesEnd[i]        = locallyOwnedRanges[2 * i + 1];
          locallyOwnedRangesEndProcIds[i] = i;
        }

      std::sort(locallyOwnedRangesEndProcIds.begin(),
                locallyOwnedRangesEndProcIds.end(),
                [&locallyOwnedRangesEnd](size_type x, size_type y) {
                  return locallyOwnedRanges[x] < locallyOwnedRanges[y];
                });

      std::sort(locallyOwnedRangesEnd.begin(), locallyOwnedRangesEnd.end());

      const size_type numGhosts = ghostIndices.size();
      for (unsigned int iGhost = 0; iGhost < numGhosts; ++iGhost)
        {
          global_size_type ghostIndex = ghostIndices[iGhost];
          auto             up  = std::upper_bound(locallyOwnedRangesEnd.begin(),
                                     locallyOwnedRangesEnd.end(),
                                     ghostIndex);
          std::string      msg = "Ghost index " + std::to_string(ghostIndex) +
                            " not found in any of the processors";
          DFTEFE_AssertWithMsg(up != locallyOwnedRangesEnd.end(), msg);
          size_type upPos  = std::distance(locallyOwnedRangesEnd.begin(), up);
          size_type procId = locallyOwnedRangesEndProcIds[upPos];
          ghostProcIdToLocalGhostIndices[procId].push_back(iGhost);
        }
    }


#ifdef DFTEFE_WITH_MPI

    template <dftefe::utils::MemorySpace memorySpace>
    MPIPatternP2P::MPIPatternP2P(
      const std::pair<global_size_type, global_size_type> locallyOwnedRange,
      const std::vector<dftefe::global_size_type> &       ghostIndices,
      const MPI_Comm &                                    mpiComm)
      : d_locallyOwnedRange(locallyOwnedRange)
      , d_mpiComm(mpiComm)
      , d_ghostIndices(0)
      , d_numGhostProcs(0)
      , d_ghostProcIds(0)
      , d_numGhostIndicesInGhostProcs(0)
      , d_flattenedLocalGhostIndices(0)
      , d_toSendProcIds(0)
      , d_numOwnedIndicesToSendToProcs(0)
      , d_flattenedLocalToSendIndices(0)
    {
      d_myRank           = 0;
      d_nprocs           = 1;
      int         err    = MPI_Comm_size(d_mpiComm, &d_nprocs);
      std::string errMsg = "Error occured while using MPI_Comm_size. "
                           "Error code: " +
                           std::to_string(err);
      throwException(err == MPI_SUCCESS, errMsg);

      err    = MPI_Comm_rank(d_mpiComm, &d_myRank);
      errMsg = "Error occured while using MPI_Comm_rank. "
               "Error code: " +
               std::to_string(err);
      throwException(err == MPI_SUCCESS, errMsg);

      std::set<global_size_type> ghostIndicesSet(ghostIndices.begin(),
                                                 ghostIndices.end());
      MemoryTransfer<memorySpace, MemorySpace::HOST> memoryTransfer;

      d_ghostIndices.resize(ghostIndices.size());
      memoryTransfer.copy(d_ghostIndices.size(),
                          d_ghostIndices.begin(),
                          ghostIndices.begin());

      std::map<size_type, std::vector<size_type>>
        ghostProcIdToLocalGhostIndices;
      getGhostProcIdToLocalGhostIndicesMap(ghostIndices,
                                           d_locallyOwnedRange.first,
                                           d_locallyOwnedRange.second,
                                           ghostProcIdToLocalGhostIndices;
                                           d_mpiComm);

      d_numGhostProcs = ghostProcIdToLocalGhostIndices.size();
      d_ghostProcIds.resize(numGhostProcs);
      d_numGhostIndicesInGhostProcs(numGhostProcs);

      std::vector<size_type> ghostProcIdsTmp(numGhostProcs);
      std::vector<size_type> numGhostIndicesTmp(numGhostProcs);
      std::vector<size_type> flattenedLocalGhostIndicesTmp(0);
      auto                   it = ghostProcIdToLocalGhostIndices.begin();
      for (unsigned int iGhostProc = 0; iGhostProc < d_numGhostProcs;
           ++iGhostProc)
        {
          ghostProcIdsTmp[iGhostProc] = it->first;
          const std::vector<size_type> localGhostIndicesInGhostProc =
            it->second;
          numGhostIndicesInGhostProcsTmp[iGhostProc] =
            localGhostIndicesInGhostProc.size();

          //
          // Append localGhostIndicesInGhostProc to
          // flattenedLocalGhostIndicesTmp
          //
          std::copy(localGhostIndicesInGhostProc.begin(),
                    localGhostIndicesInGhostProc.end(),
                    back_inserter(flattenedLocalGhostIndicesTmp));
        }

      std::string msg = "In rank " + std::to_string(d_myRank) +
                        " mismatch of"
                        " the sizes of ghost indices";
      DFTEFE_AssertWithMsg(flattenedLocalGhostIndicesTmp.size() ==
                             ghostIndices.size(),
                           msg);

      memoryTransfer.copy(d_numGhostProcs,
                          d_ghostProcIds.begin(),
                          ghostProcIdsTmp.begin());

      memoryTransfer.copy(d_numGhostProcs,
                          d_numGhostIndices.begin(),
                          numGhostIndicesTmp.begin());

      memoryTransfer.copy(ghostIndices.size(),
                          d_flattenedLocalGhostIndices.begin(),
                          flattenedLocalGhostIndicesTmp.begin());
    }
#endif

  } // end of namespace utils

} // end of namespace dftefe
