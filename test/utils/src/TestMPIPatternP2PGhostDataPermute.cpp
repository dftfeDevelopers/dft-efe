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

#include <utils/TypeConfig.h>
#include <utils/Exceptions.h>
#include <utils/MPITypes.h>
#include <utils/MPIWrapper.h>
#include <utils/MPIPatternP2P.h>

#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <ctime>
#include <utility>

using size_type = dftefe::size_type;
using global_size_type = dftefe::global_size_type;

// Takes a vector v, swaps the last element with 
// a randomly selected entry, pops the last element
// (reduces the size of vector by 1), and returns
// the randomly selected entry
size_type swapAndPopLast(std::vector<size_type> & v)
{
    int n = v.size();
    srand(time(NULL));
    int index = rand() % n;
    int num = v[index];
    std::swap(v[index], v[n - 1]);
    v.pop_back();
    return num;
}

// Function to generate n non-repeating random numbers 
// from 0 to n-1.
// It follows the Fisher–Yates shuffle Algorithm which has O(n) time 
// complexity. The assumption here is, we are given a function rand() that 
// generates a random number in O(1) time. 
// The idea is to start from the last element, swap it with a randomly 
// selected element from the whole array (including last). 
// Now consider the array from 0 to n-2 (size reduced by 1), 
// and repeat the process till we hit the first element.
//
std::vector<size_type>
permute(size_type n)
{
    std::vector<size_type> v(n);
    for (size_type i = 0; i < n; i++)
        v[i] = i;
    
    std::vector<size_type> returnValue(n);
    int count = 0;
    while (v.size()) {
        returnValue[count] = swapAndPopLast(v);
        count++;
    }
    return returnValue;
}


global_size_type getAGhostIndex(const global_size_type numGlobalIndices,
    const global_size_type ownedIndexStart,
    const global_size_type ownedIndexEnd)
{
  global_size_type ghostIndex = std::rand() % numGlobalIndices;
  if(ghostIndex >= ownedIndexStart && ghostIndex < ownedIndexEnd)
    return getAGhostIndex(numGlobalIndices,ownedIndexStart,ownedIndexEnd);
  else 
   return ghostIndex;
}

void
getOwnedRanges(std::vector<global_size_type> & startIds,
               std::vector<global_size_type> & endIds,
               size_type maxOwnedIds,
               size_type nprocs)
{
    startIds.resize(nprocs,0);
    endIds.resize(nprocs,0);
    std::vector<size_type> startIdsTmp(nprocs,0);
    std::vector<size_type> endIdsTmp(nprocs,0);
    global_size_type offset = 0;
    for(size_type i = 0; i < nprocs; ++i)
    {
        size_type nLocallyOwned = std::rand() % maxOwnedIds;
        startIdsTmp[i] = offset;
        endIdsTmp[i] = offset + nLocallyOwned;
        offset += nLocallyOwned;
    }

    // permute the ranges
    std::vector<size_type> permutation = permute(nprocs);
    for(size_type i = 0; i < nprocs; ++i)
    {
        size_type id = permutation[i];
        startIds[i] = startIdsTmp[id];
        endIds[i] = endIdsTmp[id];
    }
}

void
getProcIdToLocalGhostIndices(
        const std::vector<global_size_type> &ghostIndices,
        const std::vector<global_size_type> & ownedStartIdInProcs,
        const std::vector<global_size_type> & ownedEndIdInProcs,
        std::map<size_type, std::vector<size_type>>
        &            procIdToLocalGhostIndices,
        const size_type nprocs)
{
    //
    // NOTE: The locally owned ranges need not be ordered as per the
    // processor ranks. That is ranges for processor 0, 1, ...., P-1 given
    // by [N_0,N_1), [N_1, N_2), [N_2, N_3), ..., [N_{P-1},N_P) need not
    // honor the fact that N_0, N_1, ..., N_P are increasing. However, it
    // is more efficient to perform search operations in a sorted vector.
    // Thus, we perform a sort on the end of each locally owned range and
    // also keep track of the indices during the sort
    //
    std::vector<std::pair<size_type, global_size_type>> 
        procIdAndOwnedEndIdPairs(0);
    for (unsigned int i = 0; i < nprocs; ++i)
    {
        //only add procs whose locallyOwnedRange is greater than zero
        if(ownedEndIdInProcs[i] - ownedStartIdInProcs[i] > 0)
            procIdAndOwnedEndIdPairs.push_back(std::make_pair(i,ownedEndIdInProcs[i]));
    }

    // sort based on end id of the locally owned range
    std::sort(procIdAndOwnedEndIdPairs.begin(),
            procIdAndOwnedEndIdPairs.end(),
            [](auto &left, auto &right) {
            return left.second < right.second;});

    const size_type nProcsWithNonZeroRange = procIdAndOwnedEndIdPairs.size();
    std::vector<global_size_type> locallyOwnedRangesEnd(nProcsWithNonZeroRange, 0);
    std::vector<size_type> locallyOwnedRangesEndProcIds(nProcsWithNonZeroRange, 0);
    for (unsigned int i = 0; i < nProcsWithNonZeroRange; ++i)
    {
        locallyOwnedRangesEndProcIds[i] = procIdAndOwnedEndIdPairs[i].first;
        locallyOwnedRangesEnd[i] = procIdAndOwnedEndIdPairs[i].second;
    }

    const size_type numGhosts = ghostIndices.size();
    for (unsigned int iGhost = 0; iGhost < numGhosts; ++iGhost)
    {
        global_size_type ghostIndex = ghostIndices[iGhost];
        auto        up  = std::upper_bound(locallyOwnedRangesEnd.begin(),
                locallyOwnedRangesEnd.end(),
                ghostIndex);
        std::string msg = "Ghost index " + std::to_string(ghostIndex) +
            " not found in any of the processors";
        dftefe::utils::throwException(up != locallyOwnedRangesEnd.end(), msg);
        size_type upPos =
            std::distance(locallyOwnedRangesEnd.begin(), up);
        size_type procId = locallyOwnedRangesEndProcIds[upPos];
        procIdToLocalGhostIndices[procId].push_back(iGhost);
    }
}


int main()
{

#ifdef DFTEFE_WITH_MPI

    // initialize the MPI environment
    dftefe::utils::mpi::MPIInit(NULL, NULL);

    // Get the number of processes
    int numProcs;
    dftefe::utils::mpi::MPICommSize(dftefe::utils::mpi::MPICommWorld, &numProcs);

    // Get the rank of the process
    int rank;
    dftefe::utils::mpi::MPICommRank(dftefe::utils::mpi::MPICommWorld, &rank);

    size_type maxNumOwnedIndices = 1000;
    size_type maxNumGhostIndices = 50;
    
    //
    // Get a [start,end) ranges for each processor
    // such that the [start,end) across all processors
    // form a disjoint and contiguous set of non-negative integers.
    // The [start,end) need not be in increasing order of processor
    // id (i.e., if i > j , [start, end) for i-th processor can be
    // a lower interval than that of the j-th proccesor.
    // The following function, explicitly jumbles things so that
    // the [start,end) are not in order of proccesor id. This is
    // done to test the MPIPatternP2P more rigorously. We first
    // initialize [start, end) for all processors in root proccesor
    // and then do an MPI_Allreduce to broadcast it to other processors.
    //
    std::vector<global_size_type> ownedStartIdInProcsTmp(numProcs,0);
    std::vector<global_size_type> ownedEndIdInProcsTmp(numProcs,0);
    if(rank == 0)
    {
        getOwnedRanges(ownedStartIdInProcsTmp, 
                ownedEndIdInProcsTmp, 
                maxNumOwnedIndices, 
                numProcs);
    }

    std::vector<global_size_type> ownedStartIdInProcs(numProcs,0);
    std::vector<global_size_type> ownedEndIdInProcs(numProcs,0);
   int err =  dftefe::utils::mpi::MPIAllreduce<dftefe::utils::MemorySpace::HOST>(
           &ownedStartIdInProcsTmp[0], 
            &ownedStartIdInProcs[0], 
            numProcs, 
            dftefe::utils::mpi::MPIUnsignedLong,
            dftefe::utils::mpi::MPISum,
            dftefe::utils::mpi::MPICommWorld);

    err = dftefe::utils::mpi::MPIAllreduce<dftefe::utils::MemorySpace::HOST>( 
            &ownedEndIdInProcsTmp[0], 
            &ownedEndIdInProcs[0], 
            numProcs, 
            dftefe::utils::mpi::MPIUnsignedLong,
            dftefe::utils::mpi::MPISum,
            dftefe::utils::mpi::MPICommWorld);

    const global_size_type ownedIndexStart = ownedStartIdInProcs[rank];
    const global_size_type ownedIndexEnd = ownedEndIdInProcs[rank];
    const global_size_type numGlobalIndices = 
        *std::max_element(ownedEndIdInProcs.begin(), ownedEndIdInProcs.end());
    const size_type numOwnedIndices = ownedIndexEnd - ownedIndexStart;
    
    const size_type numGhostIndices = (numProcs==1)? 0:std::rand()%maxNumGhostIndices;
    std::set<global_size_type> ghostIndicesSet;
    std::map<size_type, std::vector<size_type>> procIdToLocalGhostIndices; 
    for(unsigned int iProc = 0; iProc < numProcs; ++iProc)
        procIdToLocalGhostIndices[iProc] = std::vector<size_type>(0);

    for(unsigned int i = 0; i < numGhostIndices; ++i)
    {
        global_size_type ghostIndex = getAGhostIndex(numGlobalIndices,
                ownedIndexStart, ownedIndexEnd);
        if(ghostIndicesSet.find(ghostIndex) != ghostIndicesSet.end())
        { 
            --i;
            continue;
        }
        else
            ghostIndicesSet.insert(ghostIndex);
    }

    std::pair<global_size_type, global_size_type> 
        locallyOwnedRange(std::make_pair(ownedIndexStart,ownedIndexEnd));
    std::vector<global_size_type> ghostIndices(numGhostIndices);
    std::copy(ghostIndicesSet.begin(), ghostIndicesSet.end(), 
            ghostIndices.begin());

    getProcIdToLocalGhostIndices(ghostIndices,
            ownedStartIdInProcs,
            ownedEndIdInProcs,
            procIdToLocalGhostIndices,
            numProcs);

    std::string ghostIndicesStr = "";
    for(unsigned int iGhost = 0; iGhost < numGhostIndices; ++iGhost)
        ghostIndicesStr += std::to_string(ghostIndices[iGhost]) + " ";

    //std::cout << "Ghost indices for Proc[" << rank << "]: " << ghostIndicesStr << std::endl;
    //if(rank == 0)
    //{
    //    std::cout << "Number global indices: " << numGlobalIndices << std::endl;
    //    std::cout << "Locally Owned Range for procs" << std::endl;
    //    for(unsigned int i = 0; i < numProcs; ++i)
    //    {
    //        std::cout << "[" << i << "]: " << ownedStartIdInProcs[i] << " " 
    //            << ownedEndIdInProcs[i] << std::endl;
    //    }
    //}

    
    dftefe::utils::mpi::MPIPatternP2P<dftefe::utils::MemorySpace::HOST> 
        mpiPatternP2P(locallyOwnedRange,
                ghostIndices,
                dftefe::utils::mpi::MPICommWorld);


    for(unsigned int iProc = 0; iProc < numProcs; ++iProc)
    {
        int numGhostInProc = procIdToLocalGhostIndices[iProc].size();
        if(numGhostInProc > 0)
        {
            auto ghostLocalIndicesFromMPIPatternP2P = 
                mpiPatternP2P.getGhostLocalIndices(iProc);
            size_type numGhostIndicesFromMPIPatternP2P = 
                ghostLocalIndicesFromMPIPatternP2P.size();
            std::vector<size_type> expectedGhostLocalIndices = 
                procIdToLocalGhostIndices[iProc];
            std::string msg = "In rank " + std::to_string(rank) + 
                " mismatch in size of ghostLocalIndices corresponding to rank " + 
                std::to_string(iProc) + " The expected size is " + 
                std::to_string(numGhostInProc) + " and the size returned is " +
                std::to_string(numGhostIndicesFromMPIPatternP2P);
            dftefe::utils::throwException(
                    numGhostInProc==numGhostIndicesFromMPIPatternP2P, 
                    msg);

            std::vector<size_type> ghostLocalIndicesFromMPIPatternP2PSTL(
                    numGhostInProc);
            auto itGhostLocalIndicesFromMPIPatternP2P = 
                ghostLocalIndicesFromMPIPatternP2P.begin();
            size_type count = 0;
            for(; itGhostLocalIndicesFromMPIPatternP2P != 
                    ghostLocalIndicesFromMPIPatternP2P.end(); 
                    ++itGhostLocalIndicesFromMPIPatternP2P)
            {
                ghostLocalIndicesFromMPIPatternP2PSTL[count] = 
                    *itGhostLocalIndicesFromMPIPatternP2P;
                ++count;
            }

            std::sort(ghostLocalIndicesFromMPIPatternP2PSTL.begin(), 
                    ghostLocalIndicesFromMPIPatternP2PSTL.end());
            std::sort(expectedGhostLocalIndices.begin(),
                    expectedGhostLocalIndices.end());
            std::string msg1 = "";
            std::string msg2 = "";
            for(unsigned int iGhost = 0; iGhost < numGhostInProc;
                    ++iGhost)
            {
                msg1 += std::to_string(ghostLocalIndicesFromMPIPatternP2PSTL[iGhost]) +
                    " ";
                msg2 += std::to_string(expectedGhostLocalIndices[iGhost]) + " ";
            }

            msg = "In rank " + std::to_string(rank) + " mismatch of local ghost indices"
                " corresponding to rank " + std::to_string(iProc) + ".\n" 
                " Expected local ghost indices: " + msg2 + "\n"
                " Received local ghost indices: " + msg1;

            dftefe::utils::throwException(ghostLocalIndicesFromMPIPatternP2PSTL ==
                    expectedGhostLocalIndices, msg);
        }
    }

    dftefe::utils::mpi::MPIFinalize();
#endif 
}
