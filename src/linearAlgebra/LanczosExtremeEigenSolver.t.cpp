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
 * @author Avirup Sircar
 */

#include <linearAlgebra/BlasLapackTypedef.h>
#include <linearAlgebra/BlasLapack.h>
#include <utils/DataTypeOverloads.h>
#include <cstdlib>
#include <ctime>

namespace dftefe
{
  namespace linearAlgebra
  {
    namespace LanczosExtremeEigenSolverInternal
    {
      template <typename ValueTypeOperator, typename ValueTypeOperand, utils::MemorySpace memorySpace>
      void LanczosEigenSolverImpl(Vector<typename LanczosExtremeEigenSolver<ValueTypeOperator, 
                                  ValueTypeOperand,  memorySpace>::ValueType, memorySpace> initialGuess,
                                size_type krylovSubspaceSize,
                                const LanczosExtremeEigenSolver<ValueTypeOperator, 
                                    ValueTypeOperand,  memorySpace>::OpContext &A,
                                const LanczosExtremeEigenSolver<ValueTypeOperator, 
                                    ValueTypeOperand,  memorySpace>::OpContext &B,
                                const LanczosExtremeEigenSolver<ValueTypeOperator, 
                                    ValueTypeOperand,  memorySpace>::OpContext &BInv,
                                std::vector<typename LanczosExtremeEigenSolver<ValueTypeOperator, 
                                    ValueTypeOperand,  memorySpace>::RealType> &eigenValues,
                                std::vector<typename LanczosExtremeEigenSolver<ValueTypeOperator, 
                                    ValueTypeOperand,  memorySpace>::ValueType> &eigenVectorsKrylovSubspaceSTL,
                                bool             computeEigenVectors)
      {
        using ValueType = LanczosExtremeEigenSolver<ValueTypeOperator, ValueTypeOperand,  memorySpace>::ValueType;
        using RealType = LanczosExtremeEigenSolver<ValueTypeOperator, ValueTypeOperand,  memorySpace>::RealType;
        ValueType alpha, beta = (ValueType)0.0, norm;

        std::vector<ValueType> T(krylovSubspaceSize *
                                  krylovSubspaceSize,
                                (ValueType)0.0);

        Vector<ValueType, memorySpace> temp(initialGuess, 0.0);
        Vector<ValueType, memorySpace> v(initialGuess, 0.0);
        Vector<ValueType, memorySpace> q(initialGuess, 0.0);
        Vector<ValueType, memorySpace> qPrev(initialGuess, 0.0);

        // normalize the initialGuess with B norm set q = b/norm
        // compute B-norm = (initGuess)^TB(initGuess)

        B.apply(initialGuess, temp);

        blasLapack::dotMultiVector<ValueTypeOperand, ValueType, memorySpace>(
          initialGuess.locallyOwnedSize(),
          1,
          initialGuess.data(),
          temp.data(),
          Op::ConjTrans,
          Op::NoTrans,
          &norm,
          *initialGuess.getLinAlgOpContext());

          // MemTransfer ??

        int err = utils::mpi::MPIAllreduce<memorySpace>(
          utils::mpi::MPIInPlace,
          &norm,
          1,
          utils::Types<ValueType>::getMPIDatatype(),
          utils::mpi::MPISum,
          initialGuess.getMPIPatternP2P()->mpiCommunicator());

        mpiIsSuccessAndMsg = utils::mpi::MPIErrIsSuccessAndMsg(err);
        utils::throwException(mpiIsSuccessAndMsg.first,
                              "MPI Error:" + mpiIsSuccessAndMsg.second);

        for (size_type h = 0; h < initialGuess.locallyOwnedSize(); h++)
        {
          *(q.data() + h) = *(initialGuess.data() + h) / norm;
        }

        for (size_type iter = 0; iter < krylovSubspaceSize - 1; iter++)
          {
            // v = BInv A q_i

            temp.setValue((ValueType)0.0);
            A.apply(q, temp);
            BInv.apply(temp, v);

            // get \alpha = q_i^TAq_i

            blasLapack::dotMultiVector<ValueType, ValueType, memorySpace>(
              q.locallyOwnedSize(),
              1,
              q.data(),
              temp.data(),
              Op::ConjTrans,
              Op::NoTrans,
              &alpha,
              *q.getLinAlgOpContext());

            T[iter * krylovSubspaceSize + iter] = alpha;

            ValueType nAlpha = (ValueType)(-1.0) * alpha,
                      nBeta  = (ValueType)(-1.0) * beta;

            // get v = v - \alpha_iq_i - \beta_i-1q_i-1

            add(ones, v, nAlpha, q, v);
            add(ones, v, nBeta, qPrev, v);

            // compute \beta_i = bnorm v

            temp.setValue((ValueType)0.0);
            B.apply(v, temp);

            blasLapack::dotMultiVector<ValueTypeOperand, ValueType, memorySpace>(
              v.locallyOwnedSize(),
              1,
              v.data(),
              temp.data(),
              Op::ConjTrans,
              Op::NoTrans,
              &beta,
              *v.getLinAlgOpContext());

            int err = utils::mpi::MPIAllreduce<memorySpace>(
              utils::mpi::MPIInPlace,
              &beta,
              1,
              utils::Types<ValueType>::getMPIDatatype(),
              utils::mpi::MPISum,
              initialGuess.getMPIPatternP2P()->mpiCommunicator());

            mpiIsSuccessAndMsg = utils::mpi::MPIErrIsSuccessAndMsg(err);
            utils::throwException(mpiIsSuccessAndMsg.first,
                                  "MPI Error:" + mpiIsSuccessAndMsg.second);

            T[iter * krylovSubspaceSize + (iter + 1)]   = beta;
            T[(iter + 1) * krylovSubspaceSize + (iter)] = beta;

            // get q_i+1 = v/\beta_i

            for (size_type h = 0; h < v.locallyOwnedSize(); h++)
              {
                *(q.data() + h) = *(v.data() + h) / beta;
              }
          }

        temp.setValue((ValueType)0.0);
        A.apply(q, temp);
        BInv.apply(temp, v);

        // get \alpha = q_i^TAq_i

        blasLapack::dotMultiVector<ValueType, ValueType, memorySpace>(
          q.locallyOwnedSize(),
          1,
          q.data(),
          temp.data(),
          Op::ConjTrans,
          Op::NoTrans,
          &alpha,
          *q.getLinAlgOpContext());

        T[iter * krylovSubspaceSize + iter] = alpha;

        if(computeEigenVectors)
        {
          lapack::heevd(lapack::Job::Vec,
                        lapack::Uplo::Lower,
                        krylovSubspaceSize,
                        T.data(),
                        krylovSubspaceSize,
                        eigenValues)

          eigenVectorsKrylovSubspaceSTL = T;
        }
        else
        {
          lapack::heevd(lapack::Job::NoVec,
                        lapack::Uplo::Lower,
                        krylovSubspaceSize,
                        T.data(),
                        krylovSubspaceSize,
                        eigenValues)
        }
      }
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    LanczosExtremeEigenSolver<ValueTypeOperator,
                              ValueTypeOperand,
                              memorySpace>::
      LanczosExtremeEigenSolver(
        const size_type                              maxKrylovSubspaceSize,
        const size_type                              numLowerExtermeEigenValues,
        const size_type                              numUpperExtermeEigenValues,
        std::vector<double>                          tolerance,
        const Vector<ValueTypeOperand, memorySpace>  &initialGuess)
    {
      reinit(maxKrylovSubspaceSize,
             numLowerExtermeEigenValues,
             numUpperExtermeEigenValues,
             tolerance,
             initialGuess);
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    LanczosExtremeEigenSolver<ValueTypeOperator,
                              ValueTypeOperand,
                              memorySpace>::
      LanczosExtremeEigenSolver(const size_type     maxKrylovSubspaceSize,
                                const size_type     numLowerExtermeEigenValues,
                                const size_type     numUpperExtermeEigenValues,
                                std::vector<double> tolerance,
                                std::shared_ptr<utils::mpi::MPIPatternP2P<memorySpace>> mpiPatternP2P,
                                std::shared_ptr<LinAlgOpContext<memorySpace>> linAlgOpContext)
    {
      reinit(
        maxKrylovSubspaceSize,
        numLowerExtermeEigenValues,
        numUpperExtermeEigenValues,
        tolerance,
        mpiPatternP2P,
        linAlgOpContext);
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    void
    LanczosExtremeEigenSolver<ValueTypeOperator,
                              ValueTypeOperand,
                              memorySpace>::
      reinit(const size_type     maxKrylovSubspaceSize,
             const size_type     numLowerExtermeEigenValues,
             const size_type     numUpperExtermeEigenValues,
             std::vector<double> tolerance,
             const Vector<ValueTypeOperand, memorySpace> &initialGuess)
    {
      d_maxKrylovSubspaceSize = maxKrylovSubspaceSize;
      d_initialGuess = initialGuess;
      d_numLowerExtermeEigenValues = numLowerExtermeEigenValues;
      d_numUpperExtermeEigenValues = numUpperExtermeEigenValues;
      d_tolerance = tolerance;
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    void
    LanczosExtremeEigenSolver<ValueTypeOperator,
                              ValueTypeOperand,
                              memorySpace>::
      reinit(const size_type     maxKrylovSubspaceSize,
             const size_type     numLowerExtermeEigenValues,
             const size_type     numUpperExtermeEigenValues,
             std::vector<double> tolerance,
             std::shared_ptr<utils::mpi::MPIPatternP2P<memorySpace>> mpiPatternP2P,
             std::shared_ptr<LinAlgOpContext<memorySpace>> linAlgOpContext)
    {
      std::srand(std::time(nullptr)*rank);
      d_initialGuess(mpiPatternP2P,linAlgOpContext);                        

      // todo
      for(size_type i = 0 ; i <  d_initialGuess.locallyOwnedSize() ; i++)
      {
        *(d_initialGuess.data()+i) = static_cast<ValueType>((std::rand()) / RAND_MAX);
      }

      d_maxKrylovSubspaceSize = maxKrylovSubspaceSize;
      d_numLowerExtermeEigenValues = numLowerExtermeEigenValues;
      d_numUpperExtermeEigenValues = numUpperExtermeEigenValues;
      d_tolerance = tolerance;
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    Error
    LanczosExtremeEigenSolver<ValueTypeOperator,
                              ValueTypeOperand,
                              memorySpace>::
    solve(const LanczosExtremeEigenSolver<ValueTypeOperator, ValueTypeOperand,  memorySpace>::OpContext &A,
          std::vector<LanczosExtremeEigenSolver<ValueTypeOperator, ValueTypeOperand,  memorySpace>::RealType> &                           eigenValues,
          MultiVector<LanczosExtremeEigenSolver<ValueTypeOperator, ValueTypeOperand,  memorySpace>::ValueType, memorySpace> &eigenVectors,
          bool             computeEigenVectors,
          const LanczosExtremeEigenSolver<ValueTypeOperator, ValueTypeOperand,  memorySpace>::OpContext &B,
          const LanczosExtremeEigenSolver<ValueTypeOperator, ValueTypeOperand,  memorySpace>::OpContext &BInv)
    {

      utils::throwException(!computeEigenVectors,
                            "Computing eigenvectors for Lanczos have not been implemented yet.");

      utils::throwException(!d_maxKrylovSubspaceSize >= d_numLowerExtermeEigenValues+d_numUpperExtermeEigenValues,
                            "Maximum Krylov subspace size should be more than number of required eigenPairs.");

      using ValueType =
        LanczosExtremeEigenSolver<ValueTypeOperator, ValueTypeOperand,  memorySpace>::ValueType;
      using RealType = LanczosExtremeEigenSolver<ValueTypeOperator, ValueTypeOperand,  memorySpace>::RealType;

      eigenValues.clear();
      std::vector<RealType> eigenValuesPrev(0);
      std::vector<bool> isToleranceReached;

      std::vector<ValueType>  eigenVectorsKrylovSubspaceSTL(0);

      // solve the Lanczos until the eigenValues are calculated till tolerance

      for(size_type krylovSubspaceSize = d_numLowerExtermeEigenValues+d_numUpperExtermeEigenValues ; 
          krylovSubspaceSize <= d_maxKrylovSubspaceSize ; krylovSubspaceSize++)
      {
        LanczosExtremeEigenSolverInternal::LanczosEigenSolverImpl<ValueTypeOperator,
                                                                  ValueTypeOperand,
                                                                  memorySpace>
                                                                  (d_initialGuess,
                                                                  krylovSubspaceSize,
                                                                  A,
                                                                  B,
                                                                  BInv,
                                                                  eigenValues,
                                                                  eigenVectorsKrylovSubspaceSTL,
                                                                  computeEigenVectors);

        eigenValuesPrev.clear();                                                          
        eigenValuesPrev = eigenValues;
        eigenValues.clear();
        std::sort(values.begin(), values.end());
        sie_type numValuesInsertedInLowerEnd= 0;
        size_type numValuesInsertedInUpperEnd = 0;
        for(size_type id = 0 ; id < values.size() ; id++)
        {
          if(numValuesInsertedInLowerEnd <= d_numLowerExtermeEigenValues)
          {
            eigenValues.push_back(values[id]);
            numValuesInsertedInLowerEnd += 1;
          }
          else
            break;
        }
        for(size_type id = values.size()-1 ; id >=0 ; id--)
        {
          else if(numValuesInsertedInUpperEnd <= d_numUpperExtermeEigenValues)
          {
            eigenValues.push_back(values[id]);
            numValuesInsertedInUpperEnd += 1;
          }
          else
            break;
        }
        isToleranceReached.clear();
        for(size_type eigId = 0 ; eigId <= d_numLowerExtermeEigenValues+d_numUpperExtermeEigenValues
            ; eigId++)
          {
            isToleranceReached.push_back((eigenValuesPrev[eigId]-eigenValues[eigId]<=d_tolerance[eigId]));
          }
        if(std::all_of(isToleranceReached.begin(), isToleranceReached.end(), [](bool v) { return v; });)
        {
          break;
        }
      }

      // copy the eigenVectors from eigenVectorsKrylovSubspaceSTL to eigenVectors if it is reqd
      // TODO

    }
  } // end of namespace linearAlgebra
} // end of namespace dftefe
