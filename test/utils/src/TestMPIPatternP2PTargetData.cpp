
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

#ifdef DFTEFE_WITH_MPI
#include <mpi.h>
#endif

#include <utils/TypeConfig.h>
#include <utils/Exceptions.h>
#include <utils/MPIPatternP2P.h>

#include <iostream>
#include <fstream>
#include <sstream>
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
  MPI_Init(NULL, NULL);

  // Get the number of processes
  int numProcs;
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

  // Get the rank of the process
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  size_type numOwnedIndices = 1000;
  size_type maxNumGhostIndices = 50;
 
  std::srand(std::time(nullptr)*rank);
  const global_size_type numGlobalIndices = numProcs*numOwnedIndices; 
  const global_size_type ownedIndexStart = rank*numOwnedIndices;
  const global_size_type ownedIndexEnd = ownedIndexStart + numOwnedIndices;
  const size_type numGhostIndices = std::rand()%maxNumGhostIndices;
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

  std::string filename = "GlobalIndices" + std::to_string(rank);
  std::ofstream outfile(filename.c_str());
  for(unsigned int i = 0; i < numOwnedIndices; ++i)
    outfile << ownedIndexStart + i << " ";
  for(unsigned int i = 0; i < numGhostIndices; ++i)
    outfile << ghostIndices[i] << " ";
  
  outfile.close();
  MPI_Barrier(MPI_COMM_WORLD);

  std::map<size_type, std::vector<size_type>> procIdToOwnedLocalIndices;
  for(unsigned int iProc = 0; iProc < numProcs; ++iProc)
  {
    if(iProc != rank)
    {
      std::string readfilename = "GlobalIndices" + std::to_string(iProc);
      std::ifstream readfile;
      std::string line;
      readfile.open(readfilename.c_str());
      dftefe::utils::throwException(readfile.is_open(), "The file " + readfilename + 
	  " does not exist");
      std::vector<global_size_type> globalIndices(0);
      while(std::getline(readfile,line))
      {
	dftefe::utils::throwException(!line.empty(),
	    "Empty or invalid line in file " + readfilename + " Line: " + line);
	std::istringstream lineString(line);
	std::string word;
	while(lineString >> word)
	  globalIndices.push_back(std::stoi(word));
      }

      for(unsigned int i = 0; i < globalIndices.size(); ++i)
      {
	global_size_type globalIndex = globalIndices[i];
	if(globalIndex >= ownedIndexStart && globalIndex < ownedIndexEnd)
	  procIdToOwnedLocalIndices[iProc].push_back(globalIndex-ownedIndexStart);
      }

      readfile.close();
    }
  }

  dftefe::utils::MPIPatternP2P<dftefe::utils::MemorySpace::HOST> 
    mpiPatternP2P(locallyOwnedRange,
	ghostIndices,
	MPI_COMM_WORLD);

  auto targetProcIds = mpiPatternP2P.getTargetProcIds();
  for(unsigned int iProc = 0; iProc < numProcs; ++iProc)
  {
    const std::vector<size_type> procOwnedLocalIndices = 
      procIdToOwnedLocalIndices[iProc];
    const size_type numOwnedIndices = procOwnedLocalIndices.size();
    if(numOwnedIndices > 0)
    {
      auto ownedLocalIndicesFromMPIPatternP2P = 
	mpiPatternP2P.getOwnedLocalIndices(iProc);
      size_type numOwnedIndicesFromMPIPatternP2P = 
	ownedLocalIndicesFromMPIPatternP2P.size();
      std::string msg = "In rank " + std::to_string(rank) + 
	" mismatch in size of ownedLocalIndices corresponding to rank " + 
	std::to_string(iProc) + " The expected size is " + 
	std::to_string(numOwnedIndices) + " and the size returned is " +
	std::to_string(numOwnedIndicesFromMPIPatternP2P);
      dftefe::utils::throwException(
	  numOwnedIndices==numOwnedIndicesFromMPIPatternP2P, 
	  msg);

      std::vector<size_type> ownedLocalIndicesFromMPIPatternP2PSTL(
	  numOwnedIndices);
      auto it = ownedLocalIndicesFromMPIPatternP2P.begin();
      size_type count = 0;
      for(; it != ownedLocalIndicesFromMPIPatternP2P.end(); ++it)
      {
	ownedLocalIndicesFromMPIPatternP2PSTL[count] = *it;
	++count;
      }

      std::sort(ownedLocalIndicesFromMPIPatternP2PSTL.begin(), 
	  ownedLocalIndicesFromMPIPatternP2PSTL.end());
      std::vector<size_type> expectedOwnedLocalIndices = 
	procOwnedLocalIndices;
      std::sort(expectedOwnedLocalIndices.begin(),
	  expectedOwnedLocalIndices.end());
      std::string msg1 = "";
      std::string msg2 = "";
      for(unsigned int iOwned = 0; iOwned < numOwnedIndices;
	  ++iOwned)
      {
	msg1 += std::to_string(ownedLocalIndicesFromMPIPatternP2PSTL[iOwned]) +
	  " ";
	msg2 += std::to_string(expectedOwnedLocalIndices[iOwned]) + " ";
      }

      msg = "In rank " + std::to_string(rank) + " mismatch of local owned indices"
	" corresponding to rank " + std::to_string(iProc) + ".\n" 
	" Expected local owned indices: " + msg2 + "\n"
	" Received local owned indices: " + msg1;

      dftefe::utils::throwException(ownedLocalIndicesFromMPIPatternP2PSTL ==
	  expectedOwnedLocalIndices, msg);

    }

  }
  MPI_Finalize();
#endif  
}
