
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

#include <utils/Exceptions.h>
#include <utils/MPITags.h>
#include <utils/MPIRequestersBase.h>
#include <utils/MPIRequestersNBX.h>
#include <string>
#include <map>
#include <set>
#include <iostream>
#include <memory>
#include <numeric>

namespace dftefe
{
  namespace utils
  {
    namespace
    {
#ifdef DFTEFE_WITH_MPI
      void
      getAllOwnedRanges(const global_size_type         ownedRangeStart,
                        const global_size_type         ownedRangeEnd,
                        std::vector<global_size_type> &allOwnedRanges,
                        const MPI_Comm &               mpiComm)
      {
        int         nprocs = 1;
        int         err    = MPI_Comm_size(mpiComm, &nprocs);
        std::string errMsg = "Error occured while using MPI_Comm_size. "
                             "Error code: " +
                             std::to_string(err);
        throwException(err == MPI_SUCCESS, errMsg);
        std::vector<int> recvCounts(nprocs, 2);
        std::vector<int> displs(nprocs, 0);
        allOwnedRanges.resize(2 * nprocs);
        for (unsigned int i = 0; i < nprocs; ++i)
          displs[i] = 2 * i;

        std::vector<global_size_type> ownedRanges = {ownedRangeStart,
                                                     ownedRangeEnd};
        MPI_Allgatherv(&ownedRanges[0],
                       2,
                       MPI_UNSIGNED_LONG,
                       &allOwnedRanges[0],
                       &recvCounts[0],
                       &displs[0],
                       MPI_UNSIGNED_LONG,
                       mpiComm);
      }

      void
      getGhostProcIdToLocalGhostIndicesMap(
        const std::vector<global_size_type> &ghostIndices,
        const std::vector<global_size_type> &allOwnedRanges,
        std::map<size_type, std::vector<size_type>>
          &             ghostProcIdToLocalGhostIndices,
        const MPI_Comm &mpiComm)
      {
        int         nprocs = 1;
        int         err    = MPI_Comm_size(mpiComm, &nprocs);
        std::string errMsg = "Error occured while using MPI_Comm_size. "
                             "Error code: " +
                             std::to_string(err);
        throwException(err == MPI_SUCCESS, errMsg);

        //
        // NOTE: The locally owned ranges need not be ordered as per the
        // processor ranks. That is ranges for processor 0, 1, ...., P-1 given
        // by [N_0,N_1), [N_1, N_2), [N_2, N_3), ..., [N_{P-1},N_P) need not
        // honor the fact that N_0, N_1, ..., N_P are increasing. However, it is
        // more efficient to perform search operations in a sorted vector. Thus,
        // we perform a sort on the end of each locally owned range and also
        // keep track of the indices during the sort
        //

        // vector to store the end of each locally owned ranges
        std::vector<global_size_type> locallyOwnedRangesEnd(nprocs);

        //
        // Vector to keep track of the  indices of locallyOwnedRangesEnd
        // during sort
        std::vector<size_type> locallyOwnedRangesEndProcIds(nprocs);
        for (unsigned int i = 0; i < nprocs; ++i)
          {
            locallyOwnedRangesEnd[i]        = allOwnedRanges[2 * i + 1];
            locallyOwnedRangesEndProcIds[i] = i;
          }

        std::sort(locallyOwnedRangesEndProcIds.begin(),
                  locallyOwnedRangesEndProcIds.end(),
                  [&locallyOwnedRangesEnd](size_type x, size_type y) {
                    return locallyOwnedRangesEnd[x] < locallyOwnedRangesEnd[y];
                  });

        std::sort(locallyOwnedRangesEnd.begin(), locallyOwnedRangesEnd.end());

        const size_type numGhosts = ghostIndices.size();
        for (unsigned int iGhost = 0; iGhost < numGhosts; ++iGhost)
          {
            global_size_type ghostIndex = ghostIndices[iGhost];
            auto        up  = std::upper_bound(locallyOwnedRangesEnd.begin(),
                                       locallyOwnedRangesEnd.end(),
                                       ghostIndex);
            std::string msg = "Ghost index " + std::to_string(ghostIndex) +
                              " not found in any of the processors";
            throwException(up != locallyOwnedRangesEnd.end(), msg);
            size_type upPos  = std::distance(locallyOwnedRangesEnd.begin(), up);
            size_type procId = locallyOwnedRangesEndProcIds[upPos];
            ghostProcIdToLocalGhostIndices[procId].push_back(iGhost);
          }
      }
#endif
      struct RangeMetaData {
	global_size_type Id;
	size_type rangeId;
	bool isRangeStart;
      };

      bool compareRangeMetaData(const RangeMetaData & x, const RangeMetaData & y)
      {
	if(x.Id == y.Id) return (!x.isRangeStart);
	else return x.Id < y.Id;
      }
      std::vector<size_type>
	getOverlappingRangeIds(const std::vector<global_size_type> & ranges)
	{
	  size_type numRanges = ranges.size()/2;
	  std::vector<RangeMetaData> rangeMetaDataVec(0);
	  for(unsigned int i = 0; i < numRanges; ++i)
	  {

	    RangeMetaData left;
	    left.Id = ranges[2*i];
	    left.rangeId = i;
	    left.isRangeStart = true;

	    RangeMetaData right;
	    right.Id = ranges[2*i+1];
	    right.rangeId = i;
	    right.isRangeStart = false;

	    rangeMetaDataVec.push_back(left);
	    rangeMetaDataVec.push_back(right);

	  }
	  std::sort(rangeMetaDataVec.begin(), rangeMetaDataVec.end(), 
	      compareRangeMetaData);
	  int currentOpen = -1;
	  bool added = false;
	  std::vector<size_type> returnValue(0);
	  for(unsigned int i = 0; i  < rangeMetaDataVec.size(); ++i)
	  {
	    size_type rangeId = rangeMetaDataVec[i].rangeId;
	    if(rangeMetaDataVec[i].isRangeStart)
	    {
	      if(currentOpen == -1)
	      {
		currentOpen = rangeId;
		added = false;
	      }
	      else
	      {
		if(!added)
		{
		  returnValue.push_back(currentOpen);
		  added = true;
		}
		returnValue.push_back(rangeId);
		if(ranges[2*rangeId+1] > ranges[2*currentOpen+1])
		{
		  currentOpen = rangeId;
		  added = true;
		}
	      }
	    }
	    else
	    {
	      if(rangeId == currentOpen)
	      {
		currentOpen = -1;
		added = false;
	      }
	    }
	  }
	  return returnValue;
	}

    } // namespace

#ifdef DFTEFE_WITH_MPI

    template <dftefe::utils::MemorySpace memorySpace>
      MPIPatternP2P<memorySpace>::MPIPatternP2P(
	  const std::pair<global_size_type, global_size_type> locallyOwnedRange,
	  const std::vector<dftefe::global_size_type> &       ghostIndices,
	  const MPI_Comm &                                    mpiComm)
      : d_locallyOwnedRange(locallyOwnedRange)
	, d_mpiComm(mpiComm)
	       , d_allOwnedRanges(0)
	       , d_ghostIndices(0)
	       , d_numGhostIndices(0)
	       , d_numGhostProcs(0)
	       , d_ghostProcIds(0)
	       , d_numGhostIndicesInGhostProcs(0)
	       , d_numTargetProcs(0)
	       , d_flattenedLocalGhostIndices(0)
	       , d_targetProcIds(0)
	       , d_numOwnedIndicesForTargetProcs(0)
	       , d_flattenedLocalTargetIndices(0)
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

      ///////////////////////////////////////////////////
      //////////// Ghost Data Evaluation Begin //////////
      ///////////////////////////////////////////////////
      std::set<global_size_type> ghostIndicesSet(ghostIndices.begin(),
	  ghostIndices.end());

      d_numGhostIndices = ghostIndices.size();
      MemoryTransfer<memorySpace, MemorySpace::HOST> memoryTransfer;
      d_ghostIndices.resize(d_numGhostIndices);
      memoryTransfer.copy(d_numGhostIndices,
	  d_ghostIndices.begin(),
	  &ghostIndices[0]);

      std::vector<global_size_type> allOwnedRanges(0);
      getAllOwnedRanges(d_locallyOwnedRange.first,
	  d_locallyOwnedRange.second,
	  allOwnedRanges,
	  d_mpiComm);

      std::vector<size_type> overlappingRangeIds = 
	getOverlappingRangeIds(allOwnedRanges);
      throwException<LogicError>(overlappingRangIds.size()==0,
	  "Detected overlapping ranges among the locallyOwnedRanges passed "
	  "to MPIPatternP2P");

      d_allOwnedRanges.resize(2 * d_nprocs);
      memoryTransfer.copy(2 * d_nprocs,
	  d_allOwnedRanges.begin(),
	  &allOwnedRanges[0]);

      std::map<size_type, std::vector<size_type>>
	ghostProcIdToLocalGhostIndices;
      getGhostProcIdToLocalGhostIndicesMap(ghostIndices,
	  allOwnedRanges,
	  ghostProcIdToLocalGhostIndices,
	  d_mpiComm);

      d_numGhostProcs = ghostProcIdToLocalGhostIndices.size();
      d_ghostProcIds.resize(d_numGhostProcs);
      d_numGhostIndicesInGhostProcs.resize(d_numGhostProcs);

      std::vector<size_type> ghostProcIdsTmp(d_numGhostProcs);
      std::vector<size_type> numGhostIndicesInGhostProcsTmp(d_numGhostProcs);
      std::vector<size_type> flattenedLocalGhostIndicesTmp(0);
      auto                   it = ghostProcIdToLocalGhostIndices.begin();
      unsigned int           iGhostProc = 0;
      for (; it != ghostProcIdToLocalGhostIndices.end(); ++it)
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
	++iGhostProc;
      }

      std::string msg = "In rank " + std::to_string(d_myRank) +
	" mismatch of"
	" the sizes of ghost indices. Expected size: " +
	std::to_string(d_numGhostIndices) + " Received size: " +
	std::to_string(flattenedLocalGhostIndicesTmp.size());
      throwException<DomainError>(flattenedLocalGhostIndicesTmp.size() ==
	  d_numGhostIndices,
	  msg);
      memoryTransfer.copy(d_numGhostProcs,
	  d_ghostProcIds.begin(),
	  &ghostProcIdsTmp[0]);

      memoryTransfer.copy(d_numGhostProcs,
	  d_numGhostIndicesInGhostProcs.begin(),
	  &numGhostIndicesInGhostProcsTmp[0]);

      d_flattenedLocalGhostIndices.resize(d_numGhostIndices);
      memoryTransfer.copy(d_numGhostIndices,
	  d_flattenedLocalGhostIndices.begin(),
	  &flattenedLocalGhostIndicesTmp[0]);
      ///////////////////////////////////////////////////
      //////////// Ghost Data Evaluation End / //////////
      ///////////////////////////////////////////////////


      ///////////////////////////////////////////////////
      //////////// Target Data Evaluation Begin ////////
      ///////////////////////////////////////////////////
      std::shared_ptr<utils::MPIRequestersBase> mpirequesters =
	std::make_shared<utils::MPIRequestersNBX>(ghostProcIdsTmp, d_mpiComm);
      const std::vector<size_type> targetProcIdsTmp =
	mpirequesters->getRequestingRankIds();
      d_numTargetProcs = targetProcIdsTmp.size();
      d_targetProcIds.resize(d_numTargetProcs);
      memoryTransfer.copy(d_numTargetProcs,
	  d_targetProcIds.begin(),
	  &targetProcIdsTmp[0]);
      std::vector<size_type> numOwnedIndicesForTargetProcsTmp(d_numTargetProcs);

      std::vector<MPI_Request> sendRequests(d_numGhostProcs);
      std::vector<MPI_Status>  sendStatuses(d_numGhostProcs);
      std::vector<MPI_Request> recvRequests(d_numTargetProcs);
      std::vector<MPI_Status>  recvStatuses(d_numTargetProcs);
      const int tag = static_cast<int>(MPITags::MPI_P2P_PATTERN_TAG);
      for (unsigned int iGhost = 0; iGhost < d_numGhostProcs; ++iGhost)
      {
	const size_type numGhostIndicesInProc =
	  numGhostIndicesInGhostProcsTmp[iGhost];
	const int   ghostProcId = ghostProcIdsTmp[iGhost];
	int         err         = MPI_Isend(&numGhostIndicesInProc,
	    1,
	    MPI_UNSIGNED,
	    ghostProcId,
	    tag,
	    d_mpiComm,
	    &sendRequests[iGhost]);
	std::string errMsg      = "Error occured while using MPI_Isend. "
	  "Error code: " +
	  std::to_string(err);
	throwException(err == MPI_SUCCESS, errMsg);
      }

      for (unsigned int iTarget = 0; iTarget < d_numTargetProcs; ++iTarget)
      {
	const int targetProcId = targetProcIdsTmp[iTarget];
	int       err = MPI_Irecv(&numOwnedIndicesForTargetProcsTmp[iTarget],
	    1,
	    MPI_UNSIGNED,
	    targetProcId,
	    tag,
	    d_mpiComm,
	    &recvRequests[iTarget]);
	std::string errMsg = "Error occured while using MPI_Irecv. "
	  "Error code: " +
	  std::to_string(err);
	throwException(err == MPI_SUCCESS, errMsg);
      }

      err =
	MPI_Waitall(d_numGhostProcs, sendRequests.data(), sendStatuses.data());
      errMsg = "Error occured while using MPI_Waitall. "
	"Error code: " +
	std::to_string(err);
      throwException(err == MPI_SUCCESS, errMsg);

      err =
	MPI_Waitall(d_numTargetProcs, recvRequests.data(), recvStatuses.data());
      errMsg = "Error occured while using MPI_Waitall. "
	"Error code: " +
	std::to_string(err);
      throwException(err == MPI_SUCCESS, errMsg);

      size_type totalOwnedIndicesForTargetProcs =
	std::accumulate(numOwnedIndicesForTargetProcsTmp.begin(),
	    numOwnedIndicesForTargetProcsTmp.end(),
	    0);
      std::vector<size_type> flattenedLocalTargetIndicesTmp(
	  totalOwnedIndicesForTargetProcs);

      size_type startIndex = 0;
      for (unsigned int iGhost = 0; iGhost < d_numGhostProcs; ++iGhost)
      {
	const int numGhostIndicesInProc =
	  numGhostIndicesInGhostProcsTmp[iGhost];
	const int ghostProcId = ghostProcIdsTmp[iGhost];

	// We need to send what is the local index in the ghost processor
	// (i.e., the processor that owns the current processor's ghost
	// index)
	std::vector<size_type> localIndicesForGhostProc(
	    numGhostIndicesInProc);
	for (unsigned iIndex = 0; iIndex < numGhostIndicesInProc; ++iIndex)
	{
	  const size_type ghostLocalIndex =
	    flattenedLocalGhostIndicesTmp[startIndex + iIndex];
	  const global_size_type ghostGlobalIndex =
	    ghostIndices[ghostLocalIndex];
	  const global_size_type ghostProcOwnedIndicesStart =
	    allOwnedRanges[2 * ghostProcId];
	  localIndicesForGhostProc[iIndex] =
	    (size_type)(ghostGlobalIndex - ghostProcOwnedIndicesStart);
	}

	int         err    = MPI_Isend(&localIndicesForGhostProc[0],
	    numGhostIndicesInProc,
	    MPI_UNSIGNED,
	    ghostProcId,
	    tag,
	    d_mpiComm,
	    &sendRequests[iGhost]);
	std::string errMsg = "Error occured while using MPI_Isend. "
	  "Error code: " +
	  std::to_string(err);
	throwException(err == MPI_SUCCESS, errMsg);
	startIndex += numGhostIndicesInProc;
      }

      startIndex = 0;
      for (unsigned int iTarget = 0; iTarget < d_numTargetProcs; ++iTarget)
      {
	const int targetProcId = targetProcIdsTmp[iTarget];
	const int numOwnedIndicesForTarget =
	  numOwnedIndicesForTargetProcsTmp[iTarget];
	int err = MPI_Irecv(&flattenedLocalTargetIndicesTmp[startIndex],
	    numOwnedIndicesForTarget,
	    MPI_UNSIGNED,
	    targetProcId,
	    tag,
	    d_mpiComm,
	    &recvRequests[iTarget]);
	std::string errMsg = "Error occured while using MPI_Irecv. "
	  "Error code: " +
	  std::to_string(err);
	throwException(err == MPI_SUCCESS, errMsg);
	startIndex += numOwnedIndicesForTarget;
      }

      err =
	MPI_Waitall(d_numGhostProcs, sendRequests.data(), sendStatuses.data());
      errMsg = "Error occured while using MPI_Waitall. "
	"Error code: " +
	std::to_string(err);
      throwException(err == MPI_SUCCESS, errMsg);

      err =
	MPI_Waitall(d_numTargetProcs, recvRequests.data(), recvStatuses.data());
      errMsg = "Error occured while using MPI_Waitall. "
	"Error code: " +
	std::to_string(err);
      throwException(err == MPI_SUCCESS, errMsg);

      d_numOwnedIndicesForTargetProcs.resize(d_numTargetProcs);
      memoryTransfer.copy(d_numTargetProcs,
	  d_numOwnedIndicesForTargetProcs.begin(),
	  &numOwnedIndicesForTargetProcsTmp[0]);

      d_flattenedLocalTargetIndices.resize(totalOwnedIndicesForTargetProcs);
      memoryTransfer.copy(totalOwnedIndicesForTargetProcs,
	  d_flattenedLocalTargetIndices.begin(),
	  &flattenedLocalTargetIndicesTmp[0]);

      ///////////////////////////////////////////////////
      //////////// Target Data Evaluation End ////////
      ///////////////////////////////////////////////////
    }
#endif

    template <dftefe::utils::MemorySpace memorySpace>
      const typename utils::MPIPatternP2P<memorySpace>::GlobalSizeTypeVector &
      MPIPatternP2P<memorySpace>::getGhostIndices() const
      {
	return d_ghostIndices;
      }

    template <dftefe::utils::MemorySpace memorySpace>
      const typename utils::MPIPatternP2P<memorySpace>::SizeTypeVector &
      MPIPatternP2P<memorySpace>::getGhostProcIds() const
      {
	return d_ghostProcIds;
      }

    template <dftefe::utils::MemorySpace memorySpace>
      typename utils::MPIPatternP2P<memorySpace>::SizeTypeVector
      MPIPatternP2P<memorySpace>::getGhostLocalIndices(
	  const size_type procId) const
      {
	size_type cumulativeIndices     = 0;
	size_type numGhostIndicesInProc = 0;
	auto      itProcIds             = d_ghostProcIds.begin();
	auto      itNumGhostIndices     = d_numGhostIndicesInGhostProcs.begin();
	for (; itProcIds != d_ghostProcIds.end(); ++itProcIds)
	{
	  numGhostIndicesInProc = *itNumGhostIndices;
	  if (procId == *itProcIds)
	    break;

	  cumulativeIndices += numGhostIndicesInProc;
	  ++itNumGhostIndices;
	}

	std::string msg =
	  "The processor Id " + std::to_string(procId) +
	  " does not contain any ghost indices for the current processor"
	  " (i.e., processor Id " +
	  std::to_string(d_myRank) + ")";
	throwException<InvalidArgument>(itProcIds != d_ghostProcIds.end(), msg);

	SizeTypeVector returnValue(numGhostIndicesInProc);
	MemoryTransfer<memorySpace, memorySpace>::copy(
	    numGhostIndicesInProc,
	    returnValue.begin(),
	    d_flattenedLocalGhostIndices.begin() + cumulativeIndices);

	return returnValue;
      }

    template <dftefe::utils::MemorySpace memorySpace>
      const typename utils::MPIPatternP2P<memorySpace>::SizeTypeVector &
      MPIPatternP2P<memorySpace>::getTargetProcIds() const
      {
	return d_targetProcIds;
      }

    template <dftefe::utils::MemorySpace memorySpace>
      typename utils::MPIPatternP2P<memorySpace>::SizeTypeVector
      MPIPatternP2P<memorySpace>::getOwnedLocalIndices(
	  const size_type procId) const
      {
	size_type cumulativeIndices      = 0;
	size_type numOwnedIndicesForProc = 0;
	auto      itProcIds              = d_targetProcIds.begin();
	auto      itNumOwnedIndices = d_numOwnedIndicesForTargetProcs.begin();
	for (; itProcIds != d_targetProcIds.end(); ++itProcIds)
	{
	  numOwnedIndicesForProc = *itNumOwnedIndices;
	  if (procId == *itProcIds)
	    break;

	  cumulativeIndices += numOwnedIndicesForProc;
	  ++itNumOwnedIndices;
	}

	std::string msg = "There are no owned indices for "
	  " target processor Id " +
	  std::to_string(procId) +
	  " in the current processor"
	  " (i.e., processor Id " +
	  std::to_string(d_myRank) + ")";
	throwException<InvalidArgument>(itProcIds != d_targetProcIds.end(), msg);

	SizeTypeVector returnValue(numOwnedIndicesForProc);
	MemoryTransfer<memorySpace, memorySpace>::copy(
	    numOwnedIndicesForProc,
	    returnValue.begin(),
	    d_flattenedLocalTargetIndices.begin() + cumulativeIndices);

	return returnValue;
      }


    template <dftefe::utils::MemorySpace memorySpace>
      size_type
      MPIPatternP2P<memorySpace>::getNumOwnedIndicesForTargetProcs() const
      {
	return d_numOwnedIndicesForTargetProcs;
      }


#ifdef DFTEFE_WITH_MPI
    template <dftefe::utils::MemorySpace memorySpace>
      const MPI_Comm &
      MPIPatternP2P<memorySpace>::mpiCommunicator() const
      {
	return d_mpiComm;
      }
#endif

    template <dftefe::utils::MemorySpace memorySpace>
      size_type
      MPIPatternP2P<memorySpace>::nmpiProcesses() const
      {
	return d_nprocs;
      }

    template <dftefe::utils::MemorySpace memorySpace>
      size_type
      MPIPatternP2P<memorySpace>::thisProcessId() const
      {
	return d_myRank;
      }

  } // end of namespace utils
} // end of namespace dftefe
