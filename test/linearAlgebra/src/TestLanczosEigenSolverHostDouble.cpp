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
* Take a hermitian matrix with same matrix stored in eah processors.
* Eg: diagonal matrix with values as -3, -2 , [-1,1], 5, 7
* Take a distributed multivector as initial guess.   
* Print out the extreme eigenvalues
*/

#include <linearAlgebra/BlasLapack.h>
#include <vector>
#include <iostream>
#include <linearAlgebra/HermitianIterativeEigenSolver.h>
#include <linearAlgebra/LanczosExtremeEigenSolver.h>
#include <linearAlgebra/IdentityOperatorContext.h>

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
using ValueType = double;
const utils::MemorySpace Host = utils::MemorySpace::HOST;

void
generateDiagonalColMajorMatrix(std::vector<ValueType> &colMajorA, global_size_type globalSize, int rank)
{

  dftefe::utils::mpi::MPIBarrier(utils::mpi::MPICommWorld);
  if(rank == 0)
  {
    ValueType min = -1.0, max = 1.0;
    for( global_size_type i = 0 ; i < globalSize ; i++ )
    {
      colMajorA[i*globalSize + i] = (max - min) * ( (ValueType)std::rand() / (ValueType)RAND_MAX ) + min;
    }

    colMajorA[0*globalSize + 0] = -4;
    colMajorA[1*globalSize + 1] = -8;
    colMajorA[(globalSize-2)*globalSize + globalSize-2] = 7;
    colMajorA[(globalSize-1)*globalSize + globalSize-1] = 13.44;

  }

  int err = utils::mpi::MPIAllreduce<Host>(
    utils::mpi::MPIInPlace,
    colMajorA.data(),
    colMajorA.size(),
    utils::mpi::Types<ValueType>::getMPIDatatype(),
    utils::mpi::MPISum,
    utils::mpi::MPICommWorld);

}

void generateHermitianPosDefColMajorMatrix(std::vector<ValueType> &colMajorA, global_size_type globalSize, int rank)
{

  std::vector<ValueType> colMajorACopy(0);
  std::vector<ValueType> a(globalSize*globalSize, 0);
  dftefe::utils::mpi::MPIBarrier(utils::mpi::MPICommWorld);
  if(rank == 0)
  {
    for( global_size_type i = 0 ; i < globalSize ; i++ )
    {
      for( global_size_type j = 0 ; j < globalSize ; j++ )
      {
        a[j*globalSize + i] = static_cast<ValueType>(std::rand()) / RAND_MAX;
      }
    }

    for( global_size_type i = 0 ; i < globalSize ; i++ )
    {
      for( global_size_type j = 0 ; j < globalSize ; j++ )
      {
        colMajorA[j*globalSize + i] = (a[j*globalSize + i] + a[i*globalSize + j])/2.0 ;
      }
    }

    colMajorACopy = colMajorA;
    std::vector<ValueType> eigenValues(globalSize);

    lapack::heevd(lapack::Job::NoVec,
                lapack::Uplo::Lower,
                globalSize,
                colMajorACopy.data(),
                globalSize,
                eigenValues.data());

    for( global_size_type i = 0 ; i < globalSize ; i++ )
    {
      colMajorA[i*globalSize + i] = colMajorA[i*globalSize + i] + std::max(0.0,-eigenValues[0]) + 
              (1 - 0.1) * ((ValueType)std::rand() / (ValueType)RAND_MAX ) + 0.1;
    } 

  }

  int err = utils::mpi::MPIAllreduce<Host>(
    utils::mpi::MPIInPlace,
    colMajorA.data(),
    colMajorA.size(),
    utils::mpi::Types<ValueType>::getMPIDatatype(),
    utils::mpi::MPISum,
    utils::mpi::MPICommWorld);

}

// Create an operatorContext class

  //
  // A test OperatorContext for Ax
  //
  template <typename ValueTypeOperator,
	   typename ValueTypeOperand>
	     class OperatorContextA: public 
        linearAlgebra::OperatorContext<ValueTypeOperator, ValueTypeOperand ,Host>
	   {
	     public:
	       OperatorContextA(const std::vector<ValueTypeOperator> & globalA, const unsigned int globalSize):
		      d_globalSize(globalSize), d_globalA(globalA)
	     {}

	     void
		   apply(linearAlgebra::MultiVector<ValueTypeOperand, Host> &x,
		     linearAlgebra::MultiVector<linearAlgebra::blasLapack::scalar_type<ValueTypeOperator, ValueTypeOperand>,
		     Host> &y) const override 
		  {
        using ValueType = linearAlgebra::blasLapack::scalar_type<ValueTypeOperator, ValueTypeOperand>;

        std::vector<std::pair<global_size_type, global_size_type>> locallyOwnedRanges = 
          x.getMPIPatternP2P()->getLocallyOwnedRanges();
        std::pair<global_size_type, global_size_type> locallyOwnedRange = locallyOwnedRanges[0];
        size_type locallyOwnedSize = x.locallyOwnedSize();

        utils::MemoryStorage<ValueType, Host>
          Xglobal(d_globalSize*x.getNumberComponents(),(ValueType)0.0);

          std::copy (x.data(), 
                      x.data()+x.getNumberComponents()*x.locallyOwnedSize(), 
                      Xglobal.data()+locallyOwnedRange.first*x.getNumberComponents());

        int err = utils::mpi::MPIAllreduce<Host>(
          utils::mpi::MPIInPlace,
          Xglobal.data(),
          Xglobal.size(),
          utils::mpi::Types<ValueType>::getMPIDatatype(),
          utils::mpi::MPISum,
          utils::mpi::MPICommWorld);
        
        linearAlgebra::blasLapack::gemm<ValueType, ValueType, Host>(
            linearAlgebra::blasLapack::Layout::ColMajor,
            linearAlgebra::blasLapack::Op::NoTrans,
            linearAlgebra::blasLapack::Op::Trans,
            x.getNumberComponents(),
            locallyOwnedSize,
            d_globalSize,
            (ValueType)1.0,
            Xglobal.data(),
            x.getNumberComponents(),
            d_globalA.data()+locallyOwnedRange.first,
            d_globalSize,
            (ValueType)0.0,
            y.data(),
            x.getNumberComponents(),
            *x.getLinAlgOpContext());

        y.updateGhostValues();

		  }
	     private:
	       unsigned int d_globalSize;
	       std::vector<ValueTypeOperator> d_globalA;
	   };// end of clas OperatorContextA

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

  std::shared_ptr<linearAlgebra::blasLapack::BlasQueue<Host>> blasqueue;
  std::shared_ptr<linearAlgebra::blasLapack::LapackQueue<Host>> lapackqueue;
  std::shared_ptr<linearAlgebra::LinAlgOpContext<Host>> linAlgOpContext = 
    std::make_shared<linearAlgebra::LinAlgOpContext<Host>>(blasqueue, lapackqueue);
  
  // initialize the MPI environment
  utils::mpi::MPIInit(NULL, NULL);

  // Get the number of processes
  int numProcs;
  utils::mpi::MPICommSize(utils::mpi::MPICommWorld, &numProcs);

  // Get the rank of the process
  int rank;
  utils::mpi::MPICommRank(utils::mpi::MPICommWorld, &rank);

  size_type numOwnedIndices = 5;
  size_type maxNumGhostIndices = 3;
  size_type numEigWantedLow = 1;
  size_type numEigWantedUp = 1;
 
  std::srand(std::time(nullptr)*(rank+1));
  const global_size_type numGlobalIndices = numProcs*numOwnedIndices;
  const global_size_type ownedIndexStart = rank*numOwnedIndices;
  const global_size_type ownedIndexEnd = ownedIndexStart + numOwnedIndices;
  const size_type numGhostIndices = (numProcs==1 || maxNumGhostIndices == 0)? 0:std::rand()%maxNumGhostIndices;
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

  global_size_type globalSize = numGlobalIndices;

  std::vector<ValueType> colMajorA(globalSize*globalSize, 0.0);
  std::vector<ValueType> colMajorB(globalSize*globalSize, 0.0);
  std::vector<ValueType> colMajorBCopy(0);
  std::vector<ValueType> colMajorBInv(0);
  std::vector<ValueType> eigenVectorsBenchMark(globalSize*globalSize, 0.0);
  std::vector<ValueType> eigenValuesBenchMark(globalSize);

  generateHermitianPosDefColMajorMatrix(colMajorA, globalSize, rank);

  generateHermitianPosDefColMajorMatrix(colMajorB, globalSize, rank);
  colMajorBInv = colMajorB;

  linearAlgebra::blasLapack::inverse<ValueType, Host>((size_type)globalSize, colMajorBInv.data(), *linAlgOpContext);

  std::shared_ptr<linearAlgebra::OperatorContext<ValueType, ValueType, Host>> opContextA
    = std::make_shared<OperatorContextA<ValueType, ValueType>> 
      (colMajorA, globalSize);

  std::shared_ptr<linearAlgebra::OperatorContext<ValueType, ValueType, Host>> opContextB
    = std::make_shared<OperatorContextA<ValueType, ValueType>> 
      (colMajorB, globalSize);

  std::shared_ptr<linearAlgebra::OperatorContext<ValueType, ValueType, Host>> opContextBInv
    = std::make_shared<OperatorContextA<ValueType, ValueType>> 
      (colMajorBInv, globalSize);
  
  std::vector<ValueType> tolerance(numEigWantedLow + numEigWantedUp);
  for(auto &i : tolerance)
    i = 1e-4;

  std::shared_ptr<linearAlgebra::HermitianIterativeEigenSolver<ValueType, ValueType, Host>> lanczos
    = std::make_shared<linearAlgebra::LanczosExtremeEigenSolver<ValueType, ValueType, Host>> 
        (100,
         numEigWantedLow,
         numEigWantedUp,
         tolerance,
         1e-8,
         mpiPatternP2P,
         linAlgOpContext);

  linearAlgebra::MultiVector<ValueType, Host> eigenVectors(mpiPatternP2P,linAlgOpContext,numEigWantedLow+numEigWantedUp);
  linearAlgebra::MultiVector<ValueType, Host> BxEigenVectorsProjectionToBenchmark(eigenVectors, (ValueType)0);
  std::vector<ValueType> eigenValues(numEigWantedLow+numEigWantedUp);
  lanczos->solve(*opContextA,
                  eigenValues,
                  eigenVectors,
                  true,
                  *opContextB,
                  *opContextBInv);

  std::cout << "EigenValues: ";
  for(auto i : eigenValues)
  {
    std::cout << i << ",";
  }
  std::cout << "\n";

    for(size_type procId = 0 ; procId < numProcs ; procId++)
    {
    if(procId == rank)
    {
    std::cout << "EigenVectors: \n" ;
    for(size_type j = 0 ; j < eigenVectors.getNumberComponents() ; j++)
    {
      std::cout << "[";
      for(size_type i = 0 ; i < eigenVectors.locallyOwnedSize() ; i++)
      { 
        std::cout << *(eigenVectors.data()+i*eigenVectors.getNumberComponents()+j) << ",";
      }
      std::cout << "]\n";
    }
    }
    std::cout << std::flush;
    dftefe::utils::mpi::MPIBarrier(utils::mpi::MPICommWorld);
    }

  std::vector<ValueType> eigenVectorsBenchMarkProjection(globalSize*globalSize, (ValueType)0);

  eigenVectorsBenchMark = colMajorA;
  colMajorBCopy = colMajorB;
  lapack::hegv(1, 
              lapack::Job::Vec, 
              lapack::Uplo::Lower, 
              globalSize,
              eigenVectorsBenchMark.data(), 
              globalSize,
              colMajorBCopy.data(), 
              globalSize,
              eigenValuesBenchMark.data() );

  // eigenValuesBenchMark.resize(globalSize);
  // eigenVectorsBenchMark = colMajorA;
  // lapack::heevd(lapack::Job::Vec,
  //             lapack::Uplo::Lower,
  //             globalSize,
  //             eigenVectorsBenchMark.data(),
  //             globalSize,
  //             eigenValuesBenchMark.data()); 	

  for(size_type procId = 0 ; procId < numProcs ; procId++)
  {
  if(procId == 0)
  {
  std::cout << "eigenValuesBenchMark: \n" ;
  for(size_type j = 0 ; j < globalSize ; j++)
  {
    std::cout << *(eigenValuesBenchMark.data()+j) << ",";
  }
  std::cout << "\n";
  // std::cout << "eigenVectorsBenchMark: \n" ;
  // for(size_type j = 0 ; j < globalSize ; j++)
  // {
  //   std::cout << "[";
  //   for(size_type i = 0 ; i < globalSize ; i++)
  //   { 
  //     std::cout << *(eigenVectorsBenchMark.data()+i*globalSize+j) << ",";
  //   }
  //   std::cout << "]\n";
  // }
  }
  std::cout << std::flush;
  dftefe::utils::mpi::MPIBarrier(utils::mpi::MPICommWorld);
  }

// check the projection error in the eigenvectors (I-XX^HB)Y < tolerance

  linearAlgebra::blasLapack::gemm<ValueType, ValueType, Host>(
      linearAlgebra::blasLapack::Layout::ColMajor,
      linearAlgebra::blasLapack::Op::NoTrans,
      linearAlgebra::blasLapack::Op::ConjTrans,
      globalSize,
      globalSize,
      globalSize,
      (ValueType)1.0,
      eigenVectorsBenchMark.data(),
      globalSize,
      eigenVectorsBenchMark.data(),
      globalSize,
      (ValueType)0.0,
      eigenVectorsBenchMarkProjection.data(),
      globalSize,
      *linAlgOpContext);

  linearAlgebra::MultiVector<ValueType, Host> BxEigenVectors(mpiPatternP2P,linAlgOpContext,numEigWantedLow+numEigWantedUp);
  opContextB->apply(eigenVectors, BxEigenVectors);

      utils::MemoryStorage<ValueType, Host>
        BxEigenVectorsglobal(globalSize*BxEigenVectors.getNumberComponents(),(ValueType)0.0);

        std::copy (BxEigenVectors.data(), 
                    BxEigenVectors.data()+eigenVectors.getNumberComponents()*eigenVectors.locallyOwnedSize(), 
                    BxEigenVectorsglobal.data()+locallyOwnedRange.first*eigenVectors.getNumberComponents());

        int err = utils::mpi::MPIAllreduce<Host>(
          utils::mpi::MPIInPlace,
          BxEigenVectorsglobal.data(),
          BxEigenVectorsglobal.size(),
          utils::mpi::Types<ValueType>::getMPIDatatype(),
          utils::mpi::MPISum,
          utils::mpi::MPICommWorld);

  linearAlgebra::blasLapack::gemm<ValueType, ValueType, Host>(
      linearAlgebra::blasLapack::Layout::ColMajor,
      linearAlgebra::blasLapack::Op::NoTrans,
      linearAlgebra::blasLapack::Op::Trans,
      eigenVectors.getNumberComponents(),
      eigenVectors.locallyOwnedSize(),
      globalSize,
      (ValueType)1.0,
      BxEigenVectorsglobal.data(),
      eigenVectors.getNumberComponents(),
      eigenVectorsBenchMarkProjection.data()+locallyOwnedRange.first,
      globalSize,
      (ValueType)0.0,
      BxEigenVectorsProjectionToBenchmark.data(),
      eigenVectors.getNumberComponents(),
      *eigenVectors.getLinAlgOpContext());

      // for(size_type procId = 0 ; procId < numProcs ; procId++)
      // {
      // if(procId == rank)
      // {
      // std::cout << "BxEigenVectorsProjectionToBenchmark: \n" ;
      // for(size_type j = 0 ; j < eigenVectors.getNumberComponents() ; j++)
      // {
      //   std::cout << "[";
      //   for(size_type i = 0 ; i < eigenVectors.locallyOwnedSize() ; i++)
      //   { 
      //     std::cout << *(BxEigenVectorsProjectionToBenchmark.data()+i*eigenVectors.getNumberComponents()+j) << ",";
      //   }
      //   std::cout << "]\n";
      // }
      // }
      // std::cout << std::flush;
      // dftefe::utils::mpi::MPIBarrier(utils::mpi::MPICommWorld);
      // }

  bool isEigenVectorsSame = false;
  for(size_type i = 0; i < BxEigenVectorsProjectionToBenchmark.locallyOwnedSize(); ++i)
   {
    for(size_type j = 0 ; j < BxEigenVectorsProjectionToBenchmark.getNumberComponents(); j++)
    {
      if(std::fabs(*(BxEigenVectorsProjectionToBenchmark.data() + i*BxEigenVectorsProjectionToBenchmark.getNumberComponents() + j)- 
        *(eigenVectors.data() + i*eigenVectors.getNumberComponents() + j)) > 1e-12)
      {
          std::string msg = "At index " + std::to_string(i) +
                            " mismatch of entries";
          std::cout << msg << "\n";
          isEigenVectorsSame = false;
          throw std::runtime_error(msg);
      }
      else
        isEigenVectorsSame = true;
    }
   }

   if(isEigenVectorsSame)
   {
    std::cout << "Hurray!! EigenVectors are same from lapack and lanczos.\n";
   }

/*

  // Test to see that OperatorContextA is working fine for parallel case

  linearAlgebra::MultiVector<ValueType, Host> X(mpiPatternP2P,linAlgOpContext,numComponents);

  size_type vecSize = X.locallyOwnedSize();
  size_type numVec = X.getNumberComponents();

  for(size_type i = 0 ; i < vecSize ; i++)
  {
    for(size_type j = 0 ; j < numVec ; j++)
    {
      *(X.data()+i*numVec+j) = static_cast<ValueType>(std::rand()) / RAND_MAX;
    }
  }

  X.updateGhostValues();

  linearAlgebra::MultiVector<ValueType, Host> Y(X, (ValueType)0.0);

  opContextA->apply(X,Y);

  for(size_type procId = 0 ; procId < numProcs ; procId++)
  {
  if(procId == rank)
  {
  std::cout << "Y: \n" ;
  for(size_type j = 0 ; j < numVec ; j++)
  {
    std::cout << "[";
    for(size_type i = 0 ; i < vecSize ; i++)
    { 
      std::cout << *(Y.data()+i*numVec+j) << ",";
    }
    std::cout << "]\n";
  }
  }
  std::cout << std::flush;
  dftefe::utils::mpi::MPIBarrier(utils::mpi::MPICommWorld);
  }

  std::vector<ValueType> Xmpireduced(globalSize*numVec,(ValueType)0.0);

  std::vector<ValueType> Ybenchmark(globalSize*numVec,(ValueType)0.0);

  std::copy (X.data(), 
              X.data()+numVec*vecSize, 
              Xmpireduced.data()+locallyOwnedRange.first*numVec);

  err = utils::mpi::MPIAllreduce<Host>(
    utils::mpi::MPIInPlace,
    Xmpireduced.data(),
    Xmpireduced.size(),
    utils::mpi::Types<ValueType>::getMPIDatatype(),
    utils::mpi::MPISum,
    utils::mpi::MPICommWorld);

    linearAlgebra::blasLapack::gemm<ValueType, ValueType, Host>(
        linearAlgebra::blasLapack::Layout::ColMajor,
        linearAlgebra::blasLapack::Op::NoTrans,
        linearAlgebra::blasLapack::Op::Trans,
        numVec,
        globalSize,
        globalSize,
        (ValueType)1.0,
        Xmpireduced.data(),
        numVec,
        colMajorA.data(),
        globalSize,
        (ValueType)0.0,
        Ybenchmark.data(),
        numVec,
        *X.getLinAlgOpContext());

      for(size_type procId = 0 ; procId < numProcs ; procId++)
      {
      if(procId == 0)
      {
      std::cout << "Ybenchmark: \n" ;
      for(size_type j = 0 ; j < numVec ; j++)
      {
        std::cout << "[";
        for(size_type i = 0 ; i < globalSize ; i++)
        { 
          std::cout << *(Ybenchmark.data()+i*numVec+j) << ",";
        }
        std::cout << "]\n";
      }
      }
      std::cout << std::flush;
      dftefe::utils::mpi::MPIBarrier(utils::mpi::MPICommWorld);
      }
      */

  utils::mpi::MPIFinalize();
#endif 
}
