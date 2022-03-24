
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
 * @author Sambit Das
 */

#ifdef DFTEFE_WITH_MPI
#include <mpi.h>
#endif

#include <utils/TypeConfig.h>
#include <utils/Exceptions.h>
#include <utils/MPIPatternP2P.h>
#include <utils/MPICommunicatorP2P.h>

#include <complex>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <ctime>

using size_type = dftefe::size_type;
using global_size_type = dftefe::global_size_type;

using MemoryStorageDoubleHost    = dftefe::utils::MemoryStorage<double, dftefe::utils::MemorySpace::HOST>;
using MemoryStorageComplexDoubleHost    = dftefe::utils::MemoryStorage<std::complex<double>, dftefe::utils::MemorySpace::HOST>;

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
  for(size_type iProc = 0; iProc < numProcs; ++iProc)
    procIdToLocalGhostIndices[iProc] = std::vector<size_type>(0);

  for(size_type i = 0; i < numGhostIndices; ++i)
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
  for(size_type i = 0; i < numOwnedIndices; ++i)
    outfile << ownedIndexStart + i << " ";
  for(size_type i = 0; i < numGhostIndices; ++i)
    outfile << ghostIndices[i] << " ";
  
  outfile.close();
  MPI_Barrier(MPI_COMM_WORLD);

  std::map<size_type, std::vector<size_type>> procIdToOwnedLocalIndices;
  for(size_type iProc = 0; iProc < numProcs; ++iProc)
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

      for(size_type i = 0; i < globalIndices.size(); ++i)
      {
	global_size_type globalIndex = globalIndices[i];
	if(globalIndex >= ownedIndexStart && globalIndex < ownedIndexEnd)
	  procIdToOwnedLocalIndices[iProc].push_back(globalIndex-ownedIndexStart);
      }

      readfile.close();
    }
  }

  std::shared_ptr<const dftefe::utils::MPIPatternP2P<dftefe::utils::MemorySpace::HOST>>
    mpiPatternP2PPtr= std::make_shared<dftefe::utils::MPIPatternP2P<dftefe::utils::MemorySpace::HOST>>(locallyOwnedRange,
	ghostIndices,
	MPI_COMM_WORLD);

  // test double and block size=1
  const size_type ownedSize=mpiPatternP2PPtr->localOwnedSize(); 
  const size_type ownedPlusGhostSize=mpiPatternP2PPtr->localOwnedSize()+mpiPatternP2PPtr->localGhostSize();
  std::vector<double> dVecStd1(ownedPlusGhostSize,0.0);
  for(size_type i = 0; i < ownedSize; ++i)
    dVecStd1[i] = mpiPatternP2PPtr->localToGlobal(i);
  
  MemoryStorageDoubleHost memStorage1(ownedPlusGhostSize);
  memStorage1.copyFrom<dftefe::utils::MemorySpace::HOST>(dVecStd1.data());

  dftefe::utils::MPICommunicatorP2P<double,dftefe::utils::MemorySpace::HOST> mpiCommunicatorP2P1(mpiPatternP2PPtr,1);

  mpiCommunicatorP2P1.updateGhostValues(memStorage1); 

  memStorage1.copyTo<dftefe::utils::MemorySpace::HOST>(dVecStd1.data()); 

  for(size_type i = ownedSize; i < ownedPlusGhostSize; ++i)
  {
      const double expectedVal=mpiPatternP2PPtr->localToGlobal(i);
      std::string msg = "In rank " + std::to_string(rank) + " mismatch of ghost value for double and block size=1 case"
    " Expected ghost value: " + std::to_string(expectedVal) + "\n"
    " Received ghost value: " + std::to_string(dVecStd1[i]);
      dftefe::utils::throwException(std::abs(dVecStd1[i]-expectedVal) <=1e-10, msg);
  }


  // test std::complex<double> and block size=1
  std::vector<std::complex<double>> dVecStd2(ownedPlusGhostSize,0.0);
  for(size_type i = 0; i < ownedSize; ++i)
    dVecStd2[i] = std::complex<double>(mpiPatternP2PPtr->localToGlobal(i),-mpiPatternP2PPtr->localToGlobal(i));
  
  MemoryStorageComplexDoubleHost memStorage2(ownedPlusGhostSize);
  memStorage2.copyFrom<dftefe::utils::MemorySpace::HOST>(dVecStd2.data());

  dftefe::utils::MPICommunicatorP2P<std::complex<double>,dftefe::utils::MemorySpace::HOST> mpiCommunicatorP2P2(mpiPatternP2PPtr,1);

  mpiCommunicatorP2P2.updateGhostValues(memStorage2); 

  memStorage2.copyTo<dftefe::utils::MemorySpace::HOST>(dVecStd2.data()); 

  for(size_type i = ownedSize; i < ownedPlusGhostSize; ++i)
  {
      const std::complex<double> expectedVal= std::complex<double>(mpiPatternP2PPtr->localToGlobal(i),-mpiPatternP2PPtr->localToGlobal(i));
      std::string msg = "In rank " + std::to_string(rank) + " mismatch of ghost value for std::complex<double> and block size=1 case"
    " Expected ghost real value: " + std::to_string(expectedVal.real()) + "\n"
    " Received ghost real value: " + std::to_string(dVecStd2[i].real()) + "\n"
    " Expected ghost imag value: " + std::to_string(expectedVal.imag()) + "\n"
    " Received ghost imag value: " + std::to_string(dVecStd2[i].imag());    
      dftefe::utils::throwException(std::abs(dVecStd2[i].real()-expectedVal.real()) <=1e-10 
                                         && std::abs(dVecStd2[i].imag()-expectedVal.imag()) <=1e-10, msg);
  }


  // test double and block size=3
  const size_type blockSize=3;
  const size_type ownedSizeMultivector=(mpiPatternP2PPtr->localOwnedSize())*blockSize; 
  const size_type ownedPlusGhostSizeMultivector=(mpiPatternP2PPtr->localOwnedSize()+mpiPatternP2PPtr->localGhostSize())*blockSize;
  std::vector<double> dVecStd3(ownedPlusGhostSizeMultivector,0.0);
  for(size_type i = 0; i < ownedSizeMultivector; ++i)
    dVecStd3[i] = mpiPatternP2PPtr->localToGlobal(i/blockSize)*blockSize+i%blockSize;
  
  MemoryStorageDoubleHost memStorage3(ownedPlusGhostSizeMultivector);
  memStorage3.copyFrom<dftefe::utils::MemorySpace::HOST>(dVecStd3.data());

  dftefe::utils::MPICommunicatorP2P<double,dftefe::utils::MemorySpace::HOST> mpiCommunicatorP2P3(mpiPatternP2PPtr,blockSize);

  mpiCommunicatorP2P3.updateGhostValues(memStorage3); 

  memStorage3.copyTo<dftefe::utils::MemorySpace::HOST>(dVecStd3.data()); 

  for(size_type i = ownedSizeMultivector; i < ownedPlusGhostSizeMultivector; ++i)
  {
      const double expectedVal=mpiPatternP2PPtr->localToGlobal(i/blockSize)*blockSize+i%blockSize;
      std::string msg = "In rank " + std::to_string(rank) + " mismatch of ghost value for double and block size=3 case"
    " Expected ghost value: " + std::to_string(expectedVal) + "\n"
    " Received ghost value: " + std::to_string(dVecStd3[i]);
      dftefe::utils::throwException(std::abs(dVecStd3[i]-expectedVal) <=1e-10, msg);
  }

  MPI_Finalize();
#endif  
}
