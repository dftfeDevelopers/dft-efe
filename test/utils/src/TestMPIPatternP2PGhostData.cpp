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

using size_type = dftefe::size_type;
using global_size_type = dftefe::global_size_type;

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

  size_type numOwnedIndices = 1000;
  size_type maxNumGhostIndices = 50;
 
  std::srand(std::time(nullptr)*rank);
  const global_size_type numGlobalIndices = numProcs*numOwnedIndices; 
  const global_size_type ownedIndexStart = rank*numOwnedIndices;
  const global_size_type ownedIndexEnd = ownedIndexStart + numOwnedIndices;
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

  for(unsigned int i = 0; i < numGhostIndices; ++i)
  {
    const size_type ghostIndex = ghostIndices[i];
    const size_type owningRank = (ghostIndex/numOwnedIndices);
    procIdToLocalGhostIndices[owningRank].push_back(i);
  }

  
  std::string ghostIndicesStr = "";
  for(unsigned int iGhost = 0; iGhost < numGhostIndices; ++iGhost)
    ghostIndicesStr += std::to_string(ghostIndices[iGhost]) + " ";

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
        mpiPatternP2P.getGhostLocalIndicesForGhostProc(iProc);
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
      
      //std::string msg0 = "Rank: " + std::to_string(rank) + " ghost rank: " 
      //  + std::to_string(iProc) + "\nExpected local ghost indices: " + msg2 +
      //  "\nReceived local ghost indices: " + msg1;
      //std::cout << msg0 << std::endl;
    }
  }

  dftefe::utils::mpi::MPIFinalize();
#endif 
}
