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


global_size_type getAGhostIndex(const std::pair<global_size_type, global_size_type> globalRange,
    const std::pair<global_size_type, global_size_type> locallyOwnedRange)
{
  global_size_type rangeSize = globalRange.second - globalRange.first;
  global_size_type ghostIndex = globalRange.first + (std::rand() % rangeSize);
  if(ghostIndex >= locallyOwnedRange.first && ghostIndex < locallyOwnedRange.second)
    return getAGhostIndex(globalRange, locallyOwnedRange);
  else 
   return ghostIndex;
}

void
getOwnedMultipleRanges(std::vector<std::vector<std::pair<global_size_type, global_size_type>>> & allOwnedRanges,
               const std::vector<size_type> & maxOwnedIds,
               const std::vector<size_type> & gapsBetweenRanges,
               const size_type nprocs,
               const double zeroLocallyOwnedProbability)
{
    const size_type nRanges = maxOwnedIds.size();
    allOwnedRanges.resize(nRanges, 
            std::vector<std::pair<global_size_type,global_size_type>>(nprocs));
    
    global_size_type cumulativeOverRanges = 0;
    for(size_type iRange = 0; iRange < nRanges; ++iRange)
    {
        cumulativeOverRanges += gapsBetweenRanges[iRange];
        std::vector<size_type> startIdsInRange(nprocs,0);
        std::vector<size_type> endIdsInRange(nprocs,0);
        global_size_type offset = 0;
        for(size_type i = 0; i < nprocs; ++i)
        {
            const double prob = (std::rand() + 0.0)/RAND_MAX;
            size_type nLocallyOwned = 0;
            if(prob < zeroLocallyOwnedProbability && maxOwnedIds[iRange] > 0) 
                nLocallyOwned = std::rand() % maxOwnedIds[iRange];
            startIdsInRange[i] = offset + cumulativeOverRanges;
            endIdsInRange[i] = startIdsInRange[i] + nLocallyOwned;
            offset += nLocallyOwned;
        }

        // permute the ranges
        std::vector<size_type> permutation = permute(nprocs);
        for(size_type i = 0; i < nprocs; ++i)
        {
            size_type id = permutation[i];
            allOwnedRanges[iRange][i].first = startIdsInRange[id];
            allOwnedRanges[iRange][i].second = endIdsInRange[id];
        }

        cumulativeOverRanges += offset; 
    }
}

void
getGlobalRanges(const std::vector<std::vector<std::pair<global_size_type, global_size_type>>> & allOwnedRanges,
        std::vector<std::pair<global_size_type, global_size_type>> & globalRanges)
{
    const size_type nRanges = allOwnedRanges.size();
    const size_type nProcs = allOwnedRanges[0].size();
    globalRanges.resize(nRanges);
    for(size_type iRange = 0; iRange < nRanges; ++iRange)
    {
      std::vector<global_size_type> ranges(2*nProcs,0);
      for(size_type iProc = 0; iProc < nProcs; ++iProc)
      {
          ranges[2*iProc] =  allOwnedRanges[iRange][iProc].first;
          ranges[2*iProc+1] =  allOwnedRanges[iRange][iProc].second;
      }

      globalRanges[iRange].first = *std::min_element(ranges.begin(), ranges.end());
      globalRanges[iRange].second = *std::max_element(ranges.begin(), ranges.end());
    }
}

void
getProcIdToLocalGhostIndices(
        const std::vector<global_size_type> &ghostIndices,
        const std::vector<std::vector<std::pair<global_size_type, global_size_type>>> & allOwnedRanges,
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
    const size_type nRanges = allOwnedRanges.size();
    const size_type numGhosts = ghostIndices.size();
    for (size_type iGhost = 0; iGhost < numGhosts; ++iGhost)
    {
        for(size_type iRange = 0; iRange < nRanges; ++iRange)
        {
            for (size_type iProc = 0; iProc < nprocs; ++iProc)
            {
                if(ghostIndices[iGhost] >=  allOwnedRanges[iRange][iProc].first &&
                        ghostIndices[iGhost] <  allOwnedRanges[iRange][iProc].second)
                {
                    procIdToLocalGhostIndices[iProc].push_back(iGhost);
                    break;
                }
            }
        }
    }
}


int main()
{

#ifdef DFTEFE_WITH_MPI

    // initialize the MPI environment
    dftefe::utils::mpi::MPIInit(NULL, NULL);

    // Get the number of processes
    int numProcs;
    int err = dftefe::utils::mpi::MPICommSize(dftefe::utils::mpi::MPICommWorld, &numProcs);

    // Get the rank of the process
    int rank;
    err  = dftefe::utils::mpi::MPICommRank(dftefe::utils::mpi::MPICommWorld, &rank);

    size_type nRanges = 2;
    std::vector<size_type> maxNumOwnedIndices = {1000, 50};
    std::vector<size_type> maxNumGhostIndices = {50, 5};
    std::vector<size_type> gapsBetweenRanges = {0, 500};

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
    std::vector<std::vector<std::pair<global_size_type, global_size_type>>> allOwnedRanges(nRanges, 
            std::vector<std::pair<global_size_type,global_size_type>>(numProcs));
    std::vector<global_size_type> ownedStartIdsInProcsTmp(nRanges*numProcs,0);
    std::vector<global_size_type> ownedEndIdsInProcsTmp(nRanges*numProcs,0);
    if(rank == 0)
    {
        getOwnedMultipleRanges(allOwnedRanges, 
                maxNumOwnedIndices, 
                gapsBetweenRanges,
                numProcs,
                0.5);
        for(size_type iRange = 0; iRange < nRanges; ++iRange)
        {
            for(size_type iProc = 0; iProc < numProcs; ++iProc)
            {
                ownedStartIdsInProcsTmp[iRange*numProcs + iProc] = 
                    allOwnedRanges[iRange][iProc].first;
                ownedEndIdsInProcsTmp[iRange*numProcs + iProc] = 
                    allOwnedRanges[iRange][iProc].second;
            }
        }
    }

    std::vector<global_size_type> ownedStartIdsInProcs(nRanges*numProcs,0);
    std::vector<global_size_type> ownedEndIdsInProcs(nRanges*numProcs,0);

    err =  dftefe::utils::mpi::MPIAllreduce<dftefe::utils::MemorySpace::HOST>(
            &ownedStartIdsInProcsTmp[0], 
            &ownedStartIdsInProcs[0], 
            nRanges*numProcs, 
            dftefe::utils::mpi::MPIUnsignedLong,
            dftefe::utils::mpi::MPISum,
            dftefe::utils::mpi::MPICommWorld);

    err = dftefe::utils::mpi::MPIAllreduce<dftefe::utils::MemorySpace::HOST>( 
            &ownedEndIdsInProcsTmp[0], 
            &ownedEndIdsInProcs[0], 
            nRanges*numProcs, 
            dftefe::utils::mpi::MPIUnsignedLong,
            dftefe::utils::mpi::MPISum,
            dftefe::utils::mpi::MPICommWorld);

    for(size_type iRange = 0; iRange < nRanges; ++iRange)
    {
        for(size_type iProc = 0; iProc < numProcs; ++iProc)
        {
            allOwnedRanges[iRange][iProc].first = ownedStartIdsInProcs[iRange*numProcs + iProc]; 
            allOwnedRanges[iRange][iProc].second = ownedEndIdsInProcs[iRange*numProcs + iProc]; 
        }
    }


    std::vector<std::pair<global_size_type, global_size_type>> globalRanges(nRanges);
    getGlobalRanges(allOwnedRanges, globalRanges);

    //if(rank == 0)
    //{
    //    for(size_type iRange = 0; iRange < nRanges; ++iRange)
    //        std::cout << "Global Range " << iRange << " : " << 
    //            globalRanges[iRange].first << " " << 
    //            globalRanges[iRange].second << std::endl;
    //}

    std::vector<std::pair<global_size_type, global_size_type>> locallyOwnedRanges(nRanges);
    size_type nOwnedIndices = 0;
    std::string locallyOwnedRangesStr = "";
    for(size_type iRange = 0; iRange < nRanges; ++iRange)
    {
        locallyOwnedRanges[iRange].first = allOwnedRanges[iRange][rank].first;
        locallyOwnedRanges[iRange].second = allOwnedRanges[iRange][rank].second;
        nOwnedIndices += locallyOwnedRanges[iRange].second - locallyOwnedRanges[iRange].first;
        locallyOwnedRangesStr += "(" + std::to_string(locallyOwnedRanges[iRange].first) + " " +  
            std::to_string(locallyOwnedRanges[iRange].second) + ")\t";
    }

    //std::cout << "Proc " << rank << " locally owned ranges: " << locallyOwnedRangesStr << std::endl;

    // 
    // Ceil the maxNumGhostIndices[iRange] with the size of globalRanges[iRange]
    // minus the size of the locallyOwnedRanges[iRange]
    // Otherwise finding a ghost index might enter into an infinite loop. Why so?
    // We can only pick a ghost index from the set of indices (say S) belonging to the globalRanges[iRange]
    // but not to locallyOwnedRanges[iRange]. In our approach, we keep on randomly picking from the set S,
    // until the required number of ghost indices are picke. The required number of ghost indices is set to 
    // a number <= maxNumGhostIndices[iRange]. Thus, if  maxNumGhostIndices[iRange] is greater than the 
    // size of the set S, we will never have enough available indices to pick and thereby be stuck in 
    // an infinite loop.

    // 
    //
    for(size_type iRange = 0; iRange < nRanges; ++iRange)
    {
        size_type N = (globalRanges[iRange].second-globalRanges[iRange].first)
            - (locallyOwnedRanges[iRange].second-locallyOwnedRanges[iRange].first);
        maxNumGhostIndices[iRange] = std::min(N,maxNumGhostIndices[iRange]);
    }

    std::set<global_size_type> ghostIndicesSet;
    for(size_type iRange = 0; iRange < nRanges; ++iRange)
    {
        //
        // proceed only if the maxNumGhostIndices is positive
        //
        if(maxNumGhostIndices[iRange] > 0)
        {
            const size_type numGhostInRange = (numProcs==1) ? 
                0 : std::rand()%maxNumGhostIndices[iRange];
            for(unsigned int i = 0; i < numGhostInRange; ++i)
            {
                global_size_type ghostIndex = getAGhostIndex(globalRanges[iRange],
                        locallyOwnedRanges[iRange]);
                if(ghostIndicesSet.find(ghostIndex) != ghostIndicesSet.end())
                { 
                    --i;
                    continue;
                }
                else
                    ghostIndicesSet.insert(ghostIndex);
            }
        }
    }

    const size_type nGhostIndices = ghostIndicesSet.size();
    std::vector<global_size_type> ghostIndices(nGhostIndices);
    std::copy(ghostIndicesSet.begin(), ghostIndicesSet.end(), 
            ghostIndices.begin());

    std::map<size_type, std::vector<size_type>> procIdToLocalGhostIndices; 
    for(unsigned int iProc = 0; iProc < numProcs; ++iProc)
        procIdToLocalGhostIndices[iProc] = std::vector<size_type>(0);

    getProcIdToLocalGhostIndices(ghostIndices,
            allOwnedRanges,
            procIdToLocalGhostIndices,
            numProcs);

    std::string ghostIndicesStr = "";
    for(unsigned int iGhost = 0; iGhost < nGhostIndices; ++iGhost)
        ghostIndicesStr += std::to_string(ghostIndices[iGhost]) + " ";

    //std::cout << "Proc " << rank << " ghostIndices: " << ghostIndicesStr << std::endl;

    dftefe::utils::mpi::MPIPatternP2P<dftefe::utils::MemorySpace::HOST> 
        mpiPatternP2P(locallyOwnedRanges,
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
