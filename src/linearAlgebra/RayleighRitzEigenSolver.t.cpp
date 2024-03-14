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
#include <linearAlgebra/OrthonormalizationFunctions.h>
#include <string>

namespace dftefe
{
  namespace linearAlgebra
  {
    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    RayleighRitzEigenSolver<ValueTypeOperator, ValueTypeOperand, memorySpace>::
      RayleighRitzEigenSolver(
        const MultiVector<ValueTypeOperand, memorySpace> &X,
        const double                                      illConditionTolerance)
    {
      reinit(X);
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    void
    RayleighRitzEigenSolver<ValueTypeOperator, ValueTypeOperand, memorySpace>::
      reinit(const MultiVector<ValueTypeOperand, memorySpace> &X,
             const double illConditionTolerance)
    {
      d_X                     = X;
      d_illConditionTolerance = illConditionTolerance;
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    EigenSolverError
    RayleighRitzEigenSolver<ValueTypeOperator, ValueTypeOperand, memorySpace>::
      solve(const OpContext &                    A,
            std::vector<RealType> &              eigenValues,
            MultiVector<ValueType, memorySpace> &eigenVectors,
            bool                                 computeEigenVectors,
            const OpContext &                    B,
            const OpContext &                    BInv)
    {
      EigenSolverError retunValue;

      size_type                           numVec  = d_X.getNumberComponents();
      size_type                           vecSize = d_X.locallyOwnedSize();
      MultiVector<ValueType, memorySpace> temp(d_X, (ValueType)0);

      // allocate memory for overlap matrix
      utils::MemoryStorage<ValueType, memorySpace> S(
        numVec * numVec, utils::Types<ValueType>::zero);

      std::vector<RealType> eigenValuesS(numVec);

      utils::MemoryStorage<ValueType, memorySpace> XprojectedA(
        numVec * numVec, utils::Types<ValueType>::zero);

      // Compute overlap matrix S = X^H B X

      B.apply(d_X, temp);

      linearAlgebra::blasLapack::gemm<ValueType, ValueTypeOperand, memorySpace>(
        linearAlgebra::blasLapack::Layout::ColMajor,
        linearAlgebra::blasLapack::Op::NoTrans,
        linearAlgebra::blasLapack::Op::ConjTrans,
        numVec,
        numVec,
        vecSize,
        (ValueType)1,
        temp.data(),
        numVec,
        d_X.data(),
        numVec,
        (ValueType)0,
        S.data(),
        numVec,
        *d_X.getLinAlgOpContext());

      utils::MemoryStorage<ValueType, utils::MemorySpace::HOST> Shost(S.size());
      utils::MemoryTransfer<utils::MemorySpace::HOST, memorySpace>::copy(
        S.size(), Shost.data(), S.data());

      // TODO: Copy only the real part because S is real
      // Reason: Reduced flops.

      // MPI_AllReduce to get the S from all procs

      int err = utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
        utils::mpi::MPIInPlace,
        Shost.data(),
        Shost.size(),
        utils::mpi::Types<ValueType>::getMPIDatatype(),
        utils::mpi::MPISum,
        d_X.getMPIPatternP2P()->mpiCommunicator());

      std::pair<bool, std::string> mpiIsSuccessAndMsg =
        utils::mpi::MPIErrIsSuccessAndMsg(err);
      utils::throwException(mpiIsSuccessAndMsg.first,
                            "MPI Error:" + mpiIsSuccessAndMsg.second);

      eigenValuesS.resize(numVec);
      lapack::heevd(lapack::Job::NoVec,
                    lapack::Uplo::Lower,
                    numVec,
                    Shost.data(),
                    numVec,
                    eigenValuesS.data());

      if (eigenValuesS[0] > d_illConditionTolerance)
        {
          // No orthogonalization required
          // Compute projected hamiltonian = X^H A X

          A.apply(d_X, temp);

          linearAlgebra::blasLapack::
            gemm<ValueType, ValueTypeOperand, memorySpace>(
              linearAlgebra::blasLapack::Layout::ColMajor,
              linearAlgebra::blasLapack::Op::NoTrans,
              linearAlgebra::blasLapack::Op::ConjTrans,
              numVec,
              numVec,
              vecSize,
              (ValueType)1,
              temp.data(),
              numVec,
              d_X.data(),
              numVec,
              (ValueType)0,
              XprojectedA.data(),
              numVec,
              *d_X.getLinAlgOpContext());

          // Solve generalized eigenvalue problem

          lapack::hegv(1,
                       lapack::Job::Vec,
                       lapack::Uplo::Lower,
                       numVec,
                       XprojectedA.data(),
                       numVec,
                       S.data(),
                       numVec,
                       eigenValues.data());

          // Rotation X_febasis = XQ.

          blasLapack::gemm<ValueType, ValueTypeOperand, memorySpace>(
            blasLapack::Layout::ColMajor,
            blasLapack::Op::Trans,
            blasLapack::Op::Trans,
            numVec,
            vecSize,
            numVec,
            (ValueType)1,
            XprojectedA.data(),
            numVec,
            d_X.data(),
            vecSize,
            (ValueType)0,
            eigenVectors.data(),
            numVec,
            *d_X.getLinAlgOpContext());
        }
      else
        {
          MultiVector<ValueType, memorySpace> XOrtho(d_X, (ValueType)0);

          // B orthogonalization required of X -> X_O

          OrthonormalizationFunctions<ValueTypeOperator,
                                      ValueTypeOperand,
                                      memorySpace>::CholeskyGramSchmidt(d_X,
                                                                        XOrtho,
                                                                        B);

          // Compute projected hamiltonian = X_O^H A X_O

          A.apply(XOrtho, temp);

          linearAlgebra::blasLapack::gemm<ValueType, ValueType, memorySpace>(
            linearAlgebra::blasLapack::Layout::ColMajor,
            linearAlgebra::blasLapack::Op::NoTrans,
            linearAlgebra::blasLapack::Op::ConjTrans,
            numVec,
            numVec,
            vecSize,
            (ValueType)1,
            temp.data(),
            numVec,
            XOrtho.data(),
            numVec,
            (ValueType)0,
            XprojectedA.data(),
            numVec,
            *d_X.getLinAlgOpContext());

          // Solve the standard eigenvalue problem

          lapack::heevd(lapack::Job::Vec,
                        lapack::Uplo::Lower,
                        numVec,
                        XprojectedA.data(),
                        numVec,
                        eigenValues.data());

          // Rotation X_febasis = X_O Q.

          blasLapack::gemm<ValueType, ValueType, memorySpace>(
            blasLapack::Layout::ColMajor,
            blasLapack::Op::Trans,
            blasLapack::Op::Trans,
            numVec,
            vecSize,
            numVec,
            (ValueType)1,
            XprojectedA.data(),
            numVec,
            XOrtho.data(),
            vecSize,
            (ValueType)0,
            eigenVectors.data(),
            numVec,
            *d_X.getLinAlgOpContext());
        }

      return retunValue;
    }
  } // end of namespace linearAlgebra
} // end of namespace dftefe
