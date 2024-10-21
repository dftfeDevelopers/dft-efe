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
      RayleighRitzEigenSolver()
    {}

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    EigenSolverError
    RayleighRitzEigenSolver<ValueTypeOperator, ValueTypeOperand, memorySpace>::
      solve(const OpContext &                           A,
            MultiVector<ValueTypeOperand, memorySpace> &X,
            std::vector<RealType> &                     eigenValues,
            MultiVector<ValueType, memorySpace> &       eigenVectors,
            bool                                        computeEigenVectors)
    {
      EigenSolverError     retunValue;
      EigenSolverErrorCode err;
      LapackError          lapackReturn;

      size_type                           numVec  = X.getNumberComponents();
      size_type                           vecSize = X.locallyOwnedSize();
      MultiVector<ValueType, memorySpace> temp(X, (ValueType)0);

      utils::MemoryStorage<ValueType, memorySpace> XprojectedA(
        numVec * numVec, utils::Types<ValueType>::zero);

      // Compute projected hamiltonian = X_O^H A X_O

      A.apply(X, temp);

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
        X.data(),
        numVec,
        (ValueType)0,
        XprojectedA.data(),
        numVec,
        *X.getLinAlgOpContext());

      // TODO: Copy only the real part because XprojectedA is real
      // Reason: Reduced flops.

      // MPI_AllReduce to get the XprojectedA from all procs

      int mpierr = utils::mpi::MPIAllreduce<memorySpace>(
        utils::mpi::MPIInPlace,
        XprojectedA.data(),
        XprojectedA.size(),
        utils::mpi::Types<ValueType>::getMPIDatatype(),
        utils::mpi::MPISum,
        X.getMPIPatternP2P()->mpiCommunicator());

      std::pair<bool, std::string> mpiIsSuccessAndMsg =
        utils::mpi::MPIErrIsSuccessAndMsg(mpierr);
      utils::throwException(mpiIsSuccessAndMsg.first,
                            "MPI Error:" + mpiIsSuccessAndMsg.second);

      // Solve the standard eigenvalue problem

      if (computeEigenVectors)
        {
          utils::MemoryStorage<RealType, memorySpace> eigenValuesMemSpace(
            numVec);

          lapackReturn = blasLapack::heevd<ValueType, memorySpace>(
            blasLapack::Job::Vec,
            blasLapack::Uplo::Lower,
            numVec,
            XprojectedA.data(),
            numVec,
            eigenValuesMemSpace.data(),
            *X.getLinAlgOpContext());

          eigenValuesMemSpace.template copyTo<utils::MemorySpace::HOST>(
            eigenValues.data(), numVec, 0, 0);

          // Rotation X_febasis = X_O Q.

          blasLapack::gemm<ValueType, ValueType, memorySpace>(
            blasLapack::Layout::ColMajor,
            blasLapack::Op::Trans,
            blasLapack::Op::NoTrans,
            numVec,
            vecSize,
            numVec,
            (ValueType)1,
            XprojectedA.data(),
            numVec,
            X.data(),
            numVec,
            (ValueType)0,
            eigenVectors.data(),
            numVec,
            *X.getLinAlgOpContext());
        }
      else
        {
          utils::MemoryStorage<RealType, memorySpace> eigenValuesMemSpace(
            numVec);

          lapackReturn = blasLapack::heevd<ValueType, memorySpace>(
            blasLapack::Job::NoVec,
            blasLapack::Uplo::Lower,
            numVec,
            XprojectedA.data(),
            numVec,
            eigenValuesMemSpace.data(),
            *X.getLinAlgOpContext());

          eigenValuesMemSpace.template copyTo<utils::MemorySpace::HOST>(
            eigenValues.data(), numVec, 0, 0);
        }

      if (lapackReturn.err == LapackErrorCode::FAILED_STANDARD_EIGENPROBLEM)
        {
          err        = EigenSolverErrorCode::LAPACK_ERROR;
          retunValue = EigenSolverErrorMsg::isSuccessAndMsg(err);
          retunValue.msg += lapackReturn.msg;
        }
      else
        {
          err        = EigenSolverErrorCode::SUCCESS;
          retunValue = EigenSolverErrorMsg::isSuccessAndMsg(err);
        }

      return retunValue;
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    EigenSolverError
    RayleighRitzEigenSolver<ValueTypeOperator, ValueTypeOperand, memorySpace>::
      solve(const OpContext &                           A,
            const OpContext &                           B,
            MultiVector<ValueTypeOperand, memorySpace> &X,
            std::vector<RealType> &                     eigenValues,
            MultiVector<ValueType, memorySpace>
              &  eigenVectors, /* M ortho eigenvecs(in)/ out*/
            bool computeEigenVectors)
    {
      EigenSolverError     retunValue;
      EigenSolverErrorCode err;
      LapackError          lapackReturn;

      size_type                           numVec  = X.getNumberComponents();
      size_type                           vecSize = X.locallyOwnedSize();
      MultiVector<ValueType, memorySpace> temp(X, (ValueType)0);

      // allocate memory for overlap matrix
      utils::MemoryStorage<ValueType, memorySpace> S(
        numVec * numVec, utils::Types<ValueType>::zero);

      utils::MemoryStorage<ValueType, memorySpace> XprojectedA(
        numVec * numVec, utils::Types<ValueType>::zero);

      // Compute overlap matrix S = X^H B X

      B.apply(X, temp);

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
        X.data(),
        numVec,
        (ValueType)0,
        S.data(),
        numVec,
        *X.getLinAlgOpContext());

      // TODO: Copy only the real part because S is real
      // Reason: Reduced flops.

      // MPI_AllReduce to get the S from all procs

      int mpierr = utils::mpi::MPIAllreduce<memorySpace>(
        utils::mpi::MPIInPlace,
        S.data(),
        S.size(),
        utils::mpi::Types<ValueType>::getMPIDatatype(),
        utils::mpi::MPISum,
        X.getMPIPatternP2P()->mpiCommunicator());

      std::pair<bool, std::string> mpiIsSuccessAndMsg =
        utils::mpi::MPIErrIsSuccessAndMsg(mpierr);
      utils::throwException(mpiIsSuccessAndMsg.first,
                            "MPI Error:" + mpiIsSuccessAndMsg.second);

      // No orthogonalization required
      // Compute projected hamiltonian = X^H A X

      A.apply(X, temp);

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
        X.data(),
        numVec,
        (ValueType)0,
        XprojectedA.data(),
        numVec,
        *X.getLinAlgOpContext());

      // TODO: Copy only the real part because XprojectedA is real
      // Reason: Reduced flops.

      // MPI_AllReduce to get the XprojectedA from all procs

      mpierr = utils::mpi::MPIAllreduce<memorySpace>(
        utils::mpi::MPIInPlace,
        XprojectedA.data(),
        XprojectedA.size(),
        utils::mpi::Types<ValueType>::getMPIDatatype(),
        utils::mpi::MPISum,
        X.getMPIPatternP2P()->mpiCommunicator());

      mpiIsSuccessAndMsg = utils::mpi::MPIErrIsSuccessAndMsg(mpierr);
      utils::throwException(mpiIsSuccessAndMsg.first,
                            "MPI Error:" + mpiIsSuccessAndMsg.second);

      // Solve generalized eigenvalue problem

      if (computeEigenVectors)
        {
          utils::MemoryStorage<RealType, memorySpace> eigenValuesMemSpace(
            numVec);

          lapackReturn =
            blasLapack::hegv<ValueType, memorySpace>(1,
                                                     blasLapack::Job::Vec,
                                                     blasLapack::Uplo::Lower,
                                                     numVec,
                                                     XprojectedA.data(),
                                                     numVec,
                                                     S.data(),
                                                     numVec,
                                                     eigenValuesMemSpace.data(),
                                                     *X.getLinAlgOpContext());

          eigenValuesMemSpace.template copyTo<utils::MemorySpace::HOST>(
            eigenValues.data(), numVec, 0, 0);


          // Rotation X_febasis = XQ. /* X_i, Y_i scratch of block size ;  */

          blasLapack::gemm<ValueType, ValueType, memorySpace>(
            blasLapack::Layout::ColMajor,
            blasLapack::Op::Trans,
            blasLapack::Op::NoTrans,
            numVec,
            vecSize,
            numVec,
            (ValueType)1,
            XprojectedA.data(),
            numVec,
            X.data(),
            numVec,
            (ValueType)0,
            eigenVectors.data(),
            numVec,
            *X.getLinAlgOpContext());
        }
      else
        {
          utils::MemoryStorage<RealType, memorySpace> eigenValuesMemSpace(
            numVec);

          lapackReturn =
            blasLapack::hegv<ValueType, memorySpace>(1,
                                                     blasLapack::Job::NoVec,
                                                     blasLapack::Uplo::Lower,
                                                     numVec,
                                                     XprojectedA.data(),
                                                     numVec,
                                                     S.data(),
                                                     numVec,
                                                     eigenValuesMemSpace.data(),
                                                     *X.getLinAlgOpContext());

          eigenValuesMemSpace.template copyTo<utils::MemorySpace::HOST>(
            eigenValues.data(), numVec, 0, 0);
        }

      if (lapackReturn.err == LapackErrorCode::FAILED_GENERALIZED_EIGENPROBLEM)
        {
          err        = EigenSolverErrorCode::LAPACK_ERROR;
          retunValue = EigenSolverErrorMsg::isSuccessAndMsg(err);
          retunValue.msg += lapackReturn.msg;
        }
      else
        {
          err        = EigenSolverErrorCode::SUCCESS;
          retunValue = EigenSolverErrorMsg::isSuccessAndMsg(err);
        }

      return retunValue;
    }
  } // end of namespace linearAlgebra
} // end of namespace dftefe
