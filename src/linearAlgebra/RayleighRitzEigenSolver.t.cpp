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
      solve(const OpContext &                                 A,
            const MultiVector<ValueTypeOperand, memorySpace> &X,
            std::vector<RealType> &                           eigenValues,
            MultiVector<ValueType, memorySpace> &             eigenVectors,
            bool computeEigenVectors)
    {
      EigenSolverError retunValue;

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

      utils::MemoryStorage<ValueType, utils::MemorySpace::HOST> XprojectedAhost(
        XprojectedA.size());
      utils::MemoryTransfer<utils::MemorySpace::HOST, memorySpace>::copy(
        XprojectedA.size(), XprojectedAhost.data(), XprojectedA.data());

      // TODO: Copy only the real part because XprojectedA is real
      // Reason: Reduced flops.

      // MPI_AllReduce to get the XprojectedA from all procs

      int err = utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
        utils::mpi::MPIInPlace,
        XprojectedAhost.data(),
        XprojectedAhost.size(),
        utils::mpi::Types<ValueType>::getMPIDatatype(),
        utils::mpi::MPISum,
        X.getMPIPatternP2P()->mpiCommunicator());

      std::pair<bool, std::string> mpiIsSuccessAndMsg =
        utils::mpi::MPIErrIsSuccessAndMsg(err);
      utils::throwException(mpiIsSuccessAndMsg.first,
                            "MPI Error:" + mpiIsSuccessAndMsg.second);

      // Solve the standard eigenvalue problem

      lapack::heevd(lapack::Job::Vec,
                    lapack::Uplo::Lower,
                    numVec,
                    XprojectedAhost.data(),
                    numVec,
                    eigenValues.data());

      utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
        XprojectedAhost.size(), XprojectedA.data(), XprojectedAhost.data());

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
        X.data(),
        vecSize,
        (ValueType)0,
        eigenVectors.data(),
        numVec,
        *X.getLinAlgOpContext());

      return retunValue;
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    EigenSolverError
    RayleighRitzEigenSolver<ValueTypeOperator, ValueTypeOperand, memorySpace>::
      solve(const OpContext &                                 A,
            const OpContext &                                 B,
            const MultiVector<ValueTypeOperand, memorySpace> &X,
            std::vector<RealType> &                           eigenValues,
            MultiVector<ValueType, memorySpace> &             eigenVectors,
            bool computeEigenVectors)
    {
      EigenSolverError retunValue;

      size_type                           numVec  = X.getNumberComponents();
      size_type                           vecSize = X.locallyOwnedSize();
      MultiVector<ValueType, memorySpace> temp(X, (ValueType)0);

      // allocate memory for overlap matrix
      utils::MemoryStorage<ValueType, memorySpace> S(
        numVec * numVec, utils::Types<ValueType>::zero);

      std::vector<RealType> eigenValuesS(numVec);

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
        X.getMPIPatternP2P()->mpiCommunicator());

      std::pair<bool, std::string> mpiIsSuccessAndMsg =
        utils::mpi::MPIErrIsSuccessAndMsg(err);
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

      utils::MemoryStorage<ValueType, utils::MemorySpace::HOST> XprojectedAhost(
        XprojectedA.size());
      utils::MemoryTransfer<utils::MemorySpace::HOST, memorySpace>::copy(
        XprojectedA.size(), XprojectedAhost.data(), XprojectedA.data());

      // TODO: Copy only the real part because XprojectedA is real
      // Reason: Reduced flops.

      // MPI_AllReduce to get the XprojectedA from all procs

      err = utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
        utils::mpi::MPIInPlace,
        XprojectedAhost.data(),
        XprojectedAhost.size(),
        utils::mpi::Types<ValueType>::getMPIDatatype(),
        utils::mpi::MPISum,
        X.getMPIPatternP2P()->mpiCommunicator());

      mpiIsSuccessAndMsg = utils::mpi::MPIErrIsSuccessAndMsg(err);
      utils::throwException(mpiIsSuccessAndMsg.first,
                            "MPI Error:" + mpiIsSuccessAndMsg.second);

      // Solve generalized eigenvalue problem

      lapack::hegv(1,
                   lapack::Job::Vec,
                   lapack::Uplo::Lower,
                   numVec,
                   XprojectedAhost.data(),
                   numVec,
                   Shost.data(),
                   numVec,
                   eigenValues.data());

      utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
        XprojectedAhost.size(), XprojectedA.data(), XprojectedAhost.data());

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
        X.data(),
        vecSize,
        (ValueType)0,
        eigenVectors.data(),
        numVec,
        *X.getLinAlgOpContext());

      return retunValue;
    }
  } // end of namespace linearAlgebra
} // end of namespace dftefe
