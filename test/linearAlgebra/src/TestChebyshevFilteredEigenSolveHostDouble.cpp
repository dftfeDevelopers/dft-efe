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
* Take few eigenvalues and eigenvectors and form a hermitian matrix
* Choose eigenvalues such that some of them lie in (a,b) and some in (b,c)
* In such cases, test the ChebyshevFilteredEigenSolver class
*/

#include <linearAlgebra/BlasLapack.h>
#include <vector>
#include <iostream>
#include <linearAlgebra/HermitianIterativeEigenSolver.h>
#include <linearAlgebra/ChebyshevFilteredEigenSolver.h>
#include <linearAlgebra/LanczosExtremeEigenSolver.h>
#include <linearAlgebra/IdentityOperatorContext.h>
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
using ValueType = double;
using RealType = double;
const utils::MemorySpace Host = utils::MemorySpace::HOST;

void generateHermitianPosDefColMajorMatrix(global_size_type globalSize, 
                                          int rank,
                                          std::vector<ValueType> &colMajorA)
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

void generateHermitianColMajorMatrix(const std::vector<ValueType> &colMajorB, 
                                    std::vector<RealType> &eigenValues, 
                                    int rank, 
                                    std::shared_ptr<linearAlgebra::LinAlgOpContext<Host>> linAlgOpContext,
                                    std::vector<ValueType> &colMajorA)
{
  // A = LQTQ^HL^H where Q consists of orthogonal vectors q_i's Q^HQ = I
  // T consists of diagonal eigenvalues \lambda_i's
  // B = LQQ^HL^H
  // Solution of A * x_i = \lambda_i B x_i
  // So, A = (LX) T (LX)^H 
  // L is the cholesky of B

  size_type globalSize = eigenValues.size();

  linearAlgebra::MultiVector<ValueType, Host> Q(globalSize,
                                                globalSize,
                                                linAlgOpContext);

  linearAlgebra::MultiVector<ValueType, Host> Qortho(globalSize,
                                                    globalSize,
                                                    linAlgOpContext);

  std::vector<ValueType> temp1(globalSize*globalSize, 0), temp2(globalSize*globalSize, 0);
  std::vector<RealType> T(globalSize*globalSize, 0);
  dftefe::utils::mpi::MPIBarrier(utils::mpi::MPICommWorld);
  if(rank == 0)
  {
    for( global_size_type i = 0 ; i < globalSize ; i++ )
    {
      for( global_size_type j = 0 ; j < globalSize ; j++ )
      {
        *(Q.data() + j*globalSize + i) = static_cast<RealType>(std::rand()) / RAND_MAX;
      }
    }
  }

  int err = utils::mpi::MPIAllreduce<Host>(
    utils::mpi::MPIInPlace,
    Q.data(),
    Q.globalSize()*Q.getNumberComponents(),
    utils::mpi::Types<ValueType>::getMPIDatatype(),
    utils::mpi::MPISum,
    utils::mpi::MPICommWorld);

    // orthogonalize the eigenvectors

    linearAlgebra::OrthonormalizationFunctions<ValueType,
              ValueType,
              Host>::CholeskyGramSchmidt(Q, Qortho);

    std::vector<ValueType> L(0);
    L = colMajorB;

    linearAlgebra::blasLapack::potrf<ValueType, Host>(linearAlgebra::blasLapack::Uplo::Lower, 
                                                      globalSize, 
                                                      L.data(), 
                                                      globalSize, 
                                                      *linAlgOpContext);

    for (size_type i = 0; i < globalSize; i++) // column
      {
        for (size_type j = 0; j < globalSize; j++) // row
          {
            if (i < j) // if colid < rowid i.e. upper tri
              {
                *(L.data() + j * globalSize + i) = (ValueType)0.0;
              }
          }
      }     

    // get LQortho = temp1
    linearAlgebra::blasLapack::gemm<RealType, ValueType, Host>(
      linearAlgebra::blasLapack::Layout::ColMajor,
      linearAlgebra::blasLapack::Op::NoTrans,
      linearAlgebra::blasLapack::Op::ConjTrans,
      globalSize,
      globalSize,
      globalSize,
      (ValueType)1.0,
      L.data(),
      globalSize,
      Qortho.data(),
      globalSize,
      (ValueType)0.0,
      temp1.data(),
      globalSize,
      *linAlgOpContext);


    for( global_size_type i = 0 ; i < globalSize ; i++ )
    {
      T[i*globalSize + i] = eigenValues[i];
    }

    // get T (temp1)^H = temp2
    linearAlgebra::blasLapack::gemm<RealType, ValueType, Host>(
        linearAlgebra::blasLapack::Layout::ColMajor,
        linearAlgebra::blasLapack::Op::NoTrans,
        linearAlgebra::blasLapack::Op::ConjTrans,
        globalSize,
        globalSize,
        globalSize,
        (ValueType)1.0,
        T.data(),
        globalSize,
        temp1.data(),
        globalSize,
        (ValueType)0.0,
        temp2.data(),
        globalSize,
        *linAlgOpContext);

    // get temp1 * temp2 = A
    linearAlgebra::blasLapack::gemm<ValueType, ValueType, Host>(
        linearAlgebra::blasLapack::Layout::ColMajor,
        linearAlgebra::blasLapack::Op::NoTrans,
        linearAlgebra::blasLapack::Op::NoTrans,
        globalSize,
        globalSize,
        globalSize,
        (ValueType)1.0,
        temp1.data(),
        globalSize,
        temp2.data(),
        globalSize,
        (ValueType)0.0,
        colMajorA.data(),
        globalSize,
        *linAlgOpContext);
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

//Restriction : a<b<c
void  
getEigenValuesInRange(RealType a, RealType b, RealType c, global_size_type globalSize, bool isLowestEigenValueDegenerate, std::vector<RealType> &eigenValues)
{
  eigenValues.resize(0);
  global_size_type midIndex = (global_size_type)(globalSize*0.5);
  global_size_type count = 0;
  while(count < midIndex)
  {
    RealType val = (b - a) * ((RealType)std::rand() / RAND_MAX ) + a;
    if(std::find(std::begin(eigenValues),std::end(eigenValues),val) == std::end(eigenValues)) {
        eigenValues.push_back(val);
        count++;
    }
  }
  count = 0;
  while(count < globalSize-midIndex)
  {
    RealType val = (c - b) * ((RealType)std::rand() / RAND_MAX ) + b;
    if(std::find(std::begin(eigenValues),std::end(eigenValues),val) == std::end(eigenValues)) {
        eigenValues.push_back(val);
        count++;
    }
  }
  std::sort(eigenValues.begin(), eigenValues.end());
  if(isLowestEigenValueDegenerate)
  {
    eigenValues[1] = eigenValues[0];
  }
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

  size_type numOwnedIndices = 100;
  size_type maxNumGhostIndices = 50;
  RealType a = -10;
  RealType b = 10;
  RealType c = 100;
  size_type numWantedEigenValues = 40;
 
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

  std::vector<ValueType> colMajorA(globalSize*globalSize, 0.0), colMajorB(globalSize*globalSize, 0.0);
  std::vector<RealType> eigenValues(0);

  getEigenValuesInRange(a, b, c, globalSize, false, eigenValues);

  generateHermitianPosDefColMajorMatrix(globalSize, rank, colMajorB);
  generateHermitianColMajorMatrix(colMajorB, eigenValues, rank, linAlgOpContext, colMajorA);

  std::vector<ValueType> colMajorBInv(0);
  colMajorBInv = colMajorB;
  linearAlgebra::blasLapack::inverse<ValueType, Host>((size_type)globalSize, colMajorBInv.data(), *linAlgOpContext);

  std::cout << "eigenValues: ";
  for(auto i : eigenValues)
  {
    std::cout << i << ",";
  }
  std::cout << "\n";
  
  std::shared_ptr<linearAlgebra::OperatorContext<ValueType, ValueType, Host>> opContextA
    = std::make_shared<OperatorContextA<ValueType, ValueType>> 
      (colMajorA, globalSize);

  std::shared_ptr<linearAlgebra::OperatorContext<ValueType, ValueType, Host>> opContextB
    = std::make_shared<OperatorContextA<ValueType, ValueType>> 
      (colMajorB, globalSize);

  std::shared_ptr<linearAlgebra::OperatorContext<ValueType, ValueType, Host>> opContextBInv
    = std::make_shared<OperatorContextA<ValueType, ValueType>> 
      (colMajorBInv, globalSize);

  std::vector<ValueType> tol = {1e-6,1e-6};
  std::shared_ptr<linearAlgebra::HermitianIterativeEigenSolver<ValueType, ValueType, Host>> lanczos
    = std::make_shared<linearAlgebra::LanczosExtremeEigenSolver<ValueType, ValueType, Host>> 
        (100,
         1,
         1,
         tol,
         1e-8,
         mpiPatternP2P,
         linAlgOpContext);

  linearAlgebra::MultiVector<ValueType, Host> eigenVectorsLanczos;
  std::vector<ValueType> eigenValuesLanczos(2);
  lanczos->solve(*opContextA,
                  eigenValuesLanczos,
                  eigenVectorsLanczos,
                  false,
                  *opContextB,
                  *opContextBInv);

  for(size_type procId = 0 ; procId < numProcs ; procId++)
  {
  if(procId == 0)
  {
  std::cout << "eigenValuesLanczos: \n" ;
  for(size_type j = 0 ; j < 2 ; j++)
  {
    std::cout << *(eigenValuesLanczos.data()+j) << ",";
  }
  std::cout<<"\n";
  }
  std::cout << std::flush;
  dftefe::utils::mpi::MPIBarrier(utils::mpi::MPICommWorld);
  }              

  ValueType wantedSpectrumUpperBound = b; //(eigenValuesLanczos[1] - eigenValuesLanczos[0])*((ValueType)(numWantedEigenValues+(int)(0.05*numWantedEigenValues))/globalSize) + eigenValuesLanczos[0];

  std::cout << "wantedSpectrumUpperBound: "<<wantedSpectrumUpperBound<<std::endl;
  linearAlgebra::MultiVector<ValueType, Host> eigenSubspaceGuess(mpiPatternP2P,linAlgOpContext,numWantedEigenValues);

  dftefe::utils::mpi::MPIBarrier(utils::mpi::MPICommWorld);

    for( global_size_type i = 0 ; i < eigenSubspaceGuess.locallyOwnedSize() ; i++ )
    {
      for( global_size_type j = 0 ; j < numWantedEigenValues ; j++ )
      {
        *(eigenSubspaceGuess.data() + i*numWantedEigenValues + j) = static_cast<RealType>(std::rand()) / RAND_MAX;
      }
    }

  eigenSubspaceGuess.updateGhostValues();

  std::shared_ptr<linearAlgebra::HermitianIterativeEigenSolver<ValueType, ValueType, Host>> chfsi
    = std::make_shared<linearAlgebra::ChebyshevFilteredEigenSolver<ValueType, ValueType, Host>>
                          (eigenValuesLanczos[0],
                           wantedSpectrumUpperBound,
                           eigenValuesLanczos[1],
                           100,
                           1e-14,
                           eigenSubspaceGuess,
                           50);

  std::vector<RealType>             eigenValuesCHFSI(numWantedEigenValues);
  linearAlgebra::MultiVector<ValueType, Host> eigenVectorsCHFSI(mpiPatternP2P,linAlgOpContext,numWantedEigenValues);

  linearAlgebra::EigenSolverError err = chfsi->solve(*opContextA,
              eigenValuesCHFSI,
              eigenVectorsCHFSI,
              true,
              *opContextB,
              *opContextBInv);

  std::cout << err.msg << "\n";

  std::vector<ValueType> eigenVectorsBenchMark(globalSize*globalSize, (ValueType)0);
  std::vector<ValueType> eigenValuesBenchMark(globalSize, (ValueType)0);

  eigenVectorsBenchMark = colMajorA;
  std::vector<ValueType> colMajorBCopy(0);
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
  std::cout << "eigenValuesCHFSI: \n" ;
  for(size_type j = 0 ; j < numWantedEigenValues ; j++)
  {
    std::cout << *(eigenValuesCHFSI.data()+j) << ",";
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

  std::vector<ValueType> eigenVectorsBenchMarkProjection(globalSize*globalSize, (ValueType)0);

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

  linearAlgebra::MultiVector<ValueType, Host> BxEigenVectors(mpiPatternP2P,linAlgOpContext,numWantedEigenValues);
  opContextB->apply(eigenVectorsCHFSI, BxEigenVectors);

      utils::MemoryStorage<ValueType, Host>
        BxEigenVectorsglobal(globalSize*BxEigenVectors.getNumberComponents(),(ValueType)0.0);

        std::copy (BxEigenVectors.data(), 
                    BxEigenVectors.data()+eigenVectorsCHFSI.getNumberComponents()*eigenVectorsCHFSI.locallyOwnedSize(), 
                    BxEigenVectorsglobal.data()+locallyOwnedRange.first*eigenVectorsCHFSI.getNumberComponents());

        int mpierr = utils::mpi::MPIAllreduce<Host>(
          utils::mpi::MPIInPlace,
          BxEigenVectorsglobal.data(),
          BxEigenVectorsglobal.size(),
          utils::mpi::Types<ValueType>::getMPIDatatype(),
          utils::mpi::MPISum,
          utils::mpi::MPICommWorld);

  linearAlgebra::MultiVector<ValueType, Host> BxEigenVectorsProjectionToBenchmark(eigenVectorsCHFSI, (ValueType)0);

  linearAlgebra::blasLapack::gemm<ValueType, ValueType, Host>(
      linearAlgebra::blasLapack::Layout::ColMajor,
      linearAlgebra::blasLapack::Op::NoTrans,
      linearAlgebra::blasLapack::Op::Trans,
      eigenVectorsCHFSI.getNumberComponents(),
      eigenVectorsCHFSI.locallyOwnedSize(),
      globalSize,
      (ValueType)1.0,
      BxEigenVectorsglobal.data(),
      eigenVectorsCHFSI.getNumberComponents(),
      eigenVectorsBenchMarkProjection.data()+locallyOwnedRange.first,
      globalSize,
      (ValueType)0.0,
      BxEigenVectorsProjectionToBenchmark.data(),
      eigenVectorsCHFSI.getNumberComponents(),
      *eigenVectorsCHFSI.getLinAlgOpContext());

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
        *(eigenVectorsCHFSI.data() + i*eigenVectorsCHFSI.getNumberComponents() + j)) > 1e-12)
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

  utils::mpi::MPIFinalize();
#endif 
}
