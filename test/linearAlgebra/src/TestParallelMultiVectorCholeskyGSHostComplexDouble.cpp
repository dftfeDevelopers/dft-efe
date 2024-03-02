/******************************************************************************
* Copyright (c) 2022.                                                        *
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
* @author Avirup Sircar
*/

/*
* Take a random distributed multiVector X. 
* Get the orthogonalized X0.
* Do (I-X0*X0^T)X < tolerance
*/

#include <linearAlgebra/BlasLapack.h>
#include <vector>
#include <iostream>
#include <linearAlgebra/OrthonormalizationFunctions.h>

#include <utils/TypeConfig.h>
#include <utils/Exceptions.h>
#include <utils/MPITypes.h>
#include <utils/MPIWrapper.h>
#include <utils/MPIPatternP2P.h>

#include <string>
#include <cstdlib>
#include <ctime>

using namespace dftefe;
using size_type = size_type;
using global_size_type = global_size_type;
using ValueType = std::complex<double>;

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

  const utils::MemorySpace Host = utils::MemorySpace::HOST;

  linearAlgebra::blasLapack::BlasQueue<Host> queue;
  std::shared_ptr<linearAlgebra::LinAlgOpContext<Host>> linAlgOpContext = 
    std::make_shared<linearAlgebra::LinAlgOpContext<Host>>(&queue);
  
  // initialize the MPI environment
  utils::mpi::MPIInit(NULL, NULL);

  // Get the number of processes
  int numProcs;
  utils::mpi::MPICommSize(utils::mpi::MPICommWorld, &numProcs);

  // Get the rank of the process
  int rank;
  utils::mpi::MPICommRank(utils::mpi::MPICommWorld, &rank);

  size_type numOwnedIndices = 10;
  size_type maxNumGhostIndices = 7;
 
  std::srand(std::time(nullptr)*rank);
  const global_size_type numGlobalIndices = numProcs*numOwnedIndices;
  const global_size_type ownedIndexStart = rank*numOwnedIndices;
  const global_size_type ownedIndexEnd = ownedIndexStart + numOwnedIndices;
  const size_type numGhostIndices = (numProcs==1)? 0:std::rand()%maxNumGhostIndices;
  std::set<global_size_type> ghostIndicesSet;

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

  // create mpipatternP2P
  std::shared_ptr<utils::mpi::MPIPatternP2P<Host>> mpiPatternP2P
    = std::make_shared<utils::mpi::MPIPatternP2P<Host>> 
      (locallyOwnedRange, ghostIndices, 
        utils::mpi::MPICommWorld);

  size_type numComponents = 5;
  linearAlgebra::MultiVector<ValueType, Host> X(mpiPatternP2P,linAlgOpContext,numComponents);

  size_type vecSize = X.locallyOwnedSize();
  size_type numVec = X.getNumberComponents();                                            

  for(size_type i = 0 ; i < vecSize ; i++)
  {
    for(size_type j = 0 ; j < numVec ; j++)
    {
      (X.data()+i*numVec+j)->real(static_cast<double>(std::rand())/RAND_MAX);
      (X.data()+i*numVec+j)->imag(static_cast<double>(std::rand())/RAND_MAX);
    }
  }

  X.updateGhostValues();

  for(size_type procId = 0 ; procId < numProcs ; procId++)
  {
  if(procId == rank)
  {
  std::cout << "X: \n" ;
  for(size_type j = 0 ; j < numVec ; j++)
  {
    std::cout << "[";
    for(size_type i = 0 ; i < vecSize ; i++)
    { 
      std::cout << *(X.data()+i*numVec+j) << ",";
    }
    std::cout << "]\n";
  }
  }
  std::cout << std::flush;
  dftefe::utils::mpi::MPIBarrier(utils::mpi::MPICommWorld);
  }
  
  linearAlgebra::MultiVector<ValueType, Host> orthogonalizedX(X, 0.0);
  linearAlgebra::MultiVector<ValueType, Host> X0X0HX(X, 0.0);

  linearAlgebra::OrthonormalizationFunctions<ValueType, Host>::CholeskyGramSchmidt(X, orthogonalizedX);
  utils::MemoryStorage<ValueType, Host> X0HX(numVec*numVec);

  orthogonalizedX.updateGhostValues();

  for(size_type procId = 0 ; procId < numProcs ; procId++)
  {
  if(procId == rank)
  {
  std::cout << "orthogonalizedX: \n" ;
  for(size_type j = 0 ; j < numVec ; j++)
  {
    std::cout << "[";
    for(size_type i = 0 ; i < vecSize ; i++)
    { 
      std::cout << *(orthogonalizedX.data()+i*numVec+j) << ",";
    }
    std::cout << "]\n";
  }
  }
    std::cout << std::flush;
  dftefe::utils::mpi::MPIBarrier(utils::mpi::MPICommWorld);
  }

  linearAlgebra::blasLapack::gemm<ValueType,
          ValueType,
          Host>
      ( linearAlgebra::blasLapack::Layout::ColMajor,
        linearAlgebra::blasLapack::Op::NoTrans,
        linearAlgebra::blasLapack::Op::ConjTrans,
        numVec,
        numVec,
        vecSize,
        1,
        X.data(),
        numVec,
        orthogonalizedX.data(),
        numVec,
        0,
        X0HX.data(),
        numVec,
        *linAlgOpContext);

  // MPI_AllReduce to get the S from all procs
  
  int err = utils::mpi::MPIAllreduce<Host>(
    utils::mpi::MPIInPlace,
    X0HX.data(),
    X0HX.size(),
    utils::mpi::Types<ValueType>::getMPIDatatype(),
    utils::mpi::MPISum,
    utils::mpi::MPICommWorld);

  std::pair<bool, std::string> mpiIsSuccessAndMsg = utils::mpi::MPIErrIsSuccessAndMsg(err);
  utils::throwException(mpiIsSuccessAndMsg.first,
                        "MPI Error:" + mpiIsSuccessAndMsg.second);        


  linearAlgebra::blasLapack::gemm<ValueType,
          ValueType,
          Host>
      (linearAlgebra::blasLapack::Layout::ColMajor,
       linearAlgebra::blasLapack::Op::NoTrans,
       linearAlgebra::blasLapack::Op::NoTrans,
        numVec,
        vecSize,
        numVec,
        1,
        X0HX.data(),
        numVec,
        orthogonalizedX.data(),
        numVec,
        0,
        X0X0HX.data(),
        numVec,
        *linAlgOpContext);

    X0X0HX.updateGhostValues();

  for(size_type procId = 0 ; procId < numProcs ; procId++)
  {
  if(procId == rank)
  {
  std::cout << "X0X0HX: \n" ;
  for(size_type j = 0 ; j < numVec ; j++)
  {
    std::cout << "[";
    for(size_type i = 0 ; i < vecSize ; i++)
    { 
      std::cout << *(X0X0HX.data()+i*numVec+j) << ",";
    }
    std::cout << "]\n";
  }
    }
    std::cout << std::flush;
  dftefe::utils::mpi::MPIBarrier(utils::mpi::MPICommWorld);
  }

  for(size_type i = 0; i < X0X0HX.localSize(); ++i)
   {
    for(size_type j = 0 ; j < X0X0HX.numVectors(); j++)
    {
      if((std::fabs((X0X0HX.data() + i*numComponents + j)->real()- 
        (X.data() + i*numComponents + j)->real()) > 1e-12) && 
        (std::fabs((X0X0HX.data() + i*numComponents + j)->imag()- 
        (X.data() + i*numComponents + j)->imag()) > 1e-12))
      {
          std::string msg = "At index " + std::to_string(i) +
                            " mismatch of entries";
          std::cout << msg << "\n";
          throw std::runtime_error(msg);
      }
    }
   }

  utils::mpi::MPIFinalize();
#endif 
}
