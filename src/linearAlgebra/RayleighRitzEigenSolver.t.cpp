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
        const size_type eigenVectorBatchSize,
        std::shared_ptr<const utils::mpi::MPIPatternP2P<memorySpace>>
                                                      mpiPatternP2P,
        std::shared_ptr<LinAlgOpContext<memorySpace>> linAlgOpContext)
      : d_eigenVecBatchSize(eigenVectorBatchSize)
      , d_batchSizeSmall(0)
      , d_XinBatchSmall(nullptr)
      , d_XoutBatchSmall(nullptr)
    {
      d_XinBatch =
        std::make_shared<linearAlgebra::MultiVector<ValueType, memorySpace>>(
          mpiPatternP2P, linAlgOpContext, eigenVectorBatchSize, ValueType());

      d_XoutBatch =
        std::make_shared<linearAlgebra::MultiVector<ValueType, memorySpace>>(
          mpiPatternP2P, linAlgOpContext, eigenVectorBatchSize, ValueType());
    }

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
      utils::Profiler p(X.getMPIPatternP2P()->mpiCommunicator(),
                        "Rayleigh-Ritz EigenSolver");

      EigenSolverError     retunValue;
      EigenSolverErrorCode err;
      LapackError          lapackReturn;

      size_type numVec  = X.getNumberComponents();
      size_type vecSize = X.locallyOwnedSize();

      p.registerStart("Memory Storage");
      utils::MemoryStorage<ValueType, memorySpace> XprojectedA(
        numVec * numVec, utils::Types<ValueType>::zero);
      utils::MemoryStorage<ValueType, memorySpace> eigenVectorsXSubspace(
        numVec * numVec, utils::Types<ValueType>::zero);
      utils::MemoryStorage<RealType, memorySpace> eigenValuesMemSpace(numVec);
      p.registerEnd("Memory Storage");

      // Compute projected hamiltonian = X_O^H A X_O

      p.registerStart("Compute X^T H X");

      computeXTransOpX(X, XprojectedA, A);

      /**
      A.apply(X, eigenVectors, true, false);

      linearAlgebra::blasLapack::gemm<ValueType, ValueType, memorySpace>(
        linearAlgebra::blasLapack::Layout::ColMajor,
        linearAlgebra::blasLapack::Op::NoTrans,
        linearAlgebra::blasLapack::Op::ConjTrans,
        numVec,
        numVec,
        vecSize,
        (ValueType)1,
        eigenVectors.data(),
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
      DFTEFE_AssertWithMsg(mpiIsSuccessAndMsg.first,
                            "MPI Error:" + mpiIsSuccessAndMsg.second);
      **/

      p.registerEnd("Compute X^T H X");

      // Solve the standard eigenvalue problem

      if (computeEigenVectors)
        {
          p.registerStart("LAPACK Eigendecomposition");

          // lapackReturn = blasLapack::heevd<ValueType, memorySpace>(
          //   blasLapack::Job::Vec,
          //   blasLapack::Uplo::Lower,
          //   numVec,
          //   XprojectedA.data(),
          //   numVec,
          //   eigenValuesMemSpace.data(),
          //   *X.getLinAlgOpContext());

          RealType  vl     = 0;
          RealType  vu     = 0;
          size_type il     = 0;
          size_type iu     = 0;
          RealType  abstol = 0;
          size_type nfound = 0;

          lapackReturn = blasLapack::heevr<ValueType, memorySpace>(
            blasLapack::Job::Vec,
            blasLapack::Range::All,
            blasLapack::Uplo::Lower,
            numVec,
            XprojectedA.data(),
            numVec,
            vl,
            vu,
            il,
            iu,
            abstol,
            nfound,
            eigenValuesMemSpace.data(),
            eigenVectorsXSubspace.data(),
            numVec,
            *X.getLinAlgOpContext());

          eigenValuesMemSpace.template copyTo<utils::MemorySpace::HOST>(
            eigenValues.data(), numVec, 0, 0);

          p.registerEnd("LAPACK Eigendecomposition");

          // Rotation X_febasis = X_O Q.

          p.registerStart("Subspace Rotation");

          blasLapack::gemm<ValueType, ValueType, memorySpace>(
            blasLapack::Layout::ColMajor,
            blasLapack::Op::Trans,
            blasLapack::Op::NoTrans,
            numVec,
            vecSize,
            numVec,
            (ValueType)1,
            eigenVectorsXSubspace.data(),
            numVec,
            X.data(),
            numVec,
            (ValueType)0,
            eigenVectors.data(),
            numVec,
            *X.getLinAlgOpContext());

          p.registerEnd("Subspace Rotation");
        }
      else
        {
          p.registerStart("LAPACK Eigendecomposition");

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

          p.registerEnd("LAPACK Eigendecomposition");
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
      p.print();
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

      size_type numVec  = X.getNumberComponents();
      size_type vecSize = X.locallyOwnedSize();
      // MultiVector<ValueType, memorySpace> temp(X, (ValueType)0);

      // allocate memory for overlap matrix
      utils::MemoryStorage<ValueType, memorySpace> S(
        numVec * numVec, utils::Types<ValueType>::zero);

      utils::MemoryStorage<ValueType, memorySpace> XprojectedA(
        numVec * numVec, utils::Types<ValueType>::zero);

      utils::MemoryStorage<RealType, memorySpace> eigenValuesMemSpace(numVec);

      // Compute overlap matrix S = X^H B X

      B.apply(X, eigenVectors, true, false);

      linearAlgebra::blasLapack::gemm<ValueType, ValueTypeOperand, memorySpace>(
        linearAlgebra::blasLapack::Layout::ColMajor,
        linearAlgebra::blasLapack::Op::NoTrans,
        linearAlgebra::blasLapack::Op::ConjTrans,
        numVec,
        numVec,
        vecSize,
        (ValueType)1,
        eigenVectors.data(),
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
      DFTEFE_AssertWithMsg(mpiIsSuccessAndMsg.first,
                           "MPI Error:" + mpiIsSuccessAndMsg.second);

      // No orthogonalization required
      // Compute projected hamiltonian = X^H A X

      A.apply(X, eigenVectors, true, false);

      linearAlgebra::blasLapack::gemm<ValueType, ValueTypeOperand, memorySpace>(
        linearAlgebra::blasLapack::Layout::ColMajor,
        linearAlgebra::blasLapack::Op::NoTrans,
        linearAlgebra::blasLapack::Op::ConjTrans,
        numVec,
        numVec,
        vecSize,
        (ValueType)1,
        eigenVectors.data(),
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
      DFTEFE_AssertWithMsg(mpiIsSuccessAndMsg.first,
                           "MPI Error:" + mpiIsSuccessAndMsg.second);

      // Solve generalized eigenvalue problem

      if (computeEigenVectors)
        {
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

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    void
    RayleighRitzEigenSolver<ValueTypeOperator, ValueTypeOperand, memorySpace>::
      computeXTransOpX(MultiVector<ValueTypeOperand, memorySpace> &  X,
                       utils::MemoryStorage<ValueType, memorySpace> &S,
                       const OpContext &                             Op)
    {
      const utils::mpi::MPIComm comm = X.getMPIPatternP2P()->mpiCommunicator();
      LinAlgOpContext<memorySpace> linAlgOpContext = *X.getLinAlgOpContext();
      const size_type              vecSize         = X.locallyOwnedSize();
      const size_type              vecLocalSize    = X.localSize();
      const size_type              numVec          = X.getNumberComponents();
      utils::MemoryTransfer<memorySpace, memorySpace>      memoryTransfer;
      std::shared_ptr<MultiVector<ValueType, memorySpace>> subspaceBatchIn =
                                                             nullptr,
                                                           subspaceBatchOut =
                                                             nullptr;

      utils::MemoryStorage<ValueType, memorySpace> SBlock(numVec *
                                                            d_eigenVecBatchSize,
                                                          ValueType(0));
      // utils::MemoryStorage<ValueType, utils::MemorySpace::HOST>
      // SBlockHost(numVec * d_eigenVecBatchSize);

      for (size_type eigVecStartId = 0; eigVecStartId < numVec;
           eigVecStartId += d_eigenVecBatchSize)
        {
          const size_type eigVecEndId =
            std::min(eigVecStartId + d_eigenVecBatchSize, numVec);
          const size_type numEigVecInBatch = eigVecEndId - eigVecStartId;

          if (numEigVecInBatch % d_eigenVecBatchSize == 0)
            {
              for (size_type iSize = 0; iSize < vecLocalSize; iSize++)
                memoryTransfer.copy(numEigVecInBatch,
                                    d_XinBatch->data() +
                                      numEigVecInBatch * iSize,
                                    X.data() + iSize * numVec + eigVecStartId);

              subspaceBatchIn  = d_XinBatch;
              subspaceBatchOut = d_XoutBatch;
            }
          else if (numEigVecInBatch % d_eigenVecBatchSize == d_batchSizeSmall)
            {
              for (size_type iSize = 0; iSize < vecLocalSize; iSize++)
                memoryTransfer.copy(numEigVecInBatch,
                                    d_XinBatchSmall->data() +
                                      numEigVecInBatch * iSize,
                                    X.data() + iSize * numVec + eigVecStartId);

              subspaceBatchIn  = d_XinBatchSmall;
              subspaceBatchOut = d_XoutBatchSmall;
            }
          else
            {
              d_batchSizeSmall = numEigVecInBatch;

              d_XinBatchSmall = std::make_shared<
                linearAlgebra::MultiVector<ValueType, memorySpace>>(
                X.getMPIPatternP2P(),
                X.getLinAlgOpContext(),
                numEigVecInBatch,
                ValueType());

              d_XoutBatchSmall = std::make_shared<
                linearAlgebra::MultiVector<ValueType, memorySpace>>(
                X.getMPIPatternP2P(),
                X.getLinAlgOpContext(),
                numEigVecInBatch,
                ValueType());

              for (size_type iSize = 0; iSize < vecLocalSize; iSize++)
                memoryTransfer.copy(numEigVecInBatch,
                                    d_XinBatchSmall->data() +
                                      numEigVecInBatch * iSize,
                                    X.data() + iSize * numVec + eigVecStartId);

              subspaceBatchIn  = d_XinBatchSmall;
              subspaceBatchOut = d_XoutBatchSmall;
            }

          Op.apply(*subspaceBatchIn, *subspaceBatchOut, true, false);

          // Input data is read is X^T (numVec is fastest index and then
          // vecSize) Operation : S = (X)^H * ((B*X)). S^T = ((B*X)^T)*(X^T)^H

          const ValueType alpha = 1.0;
          const ValueType beta  = 0.0;

          blasLapack::gemm<ValueTypeOperand, ValueType, memorySpace>(
            blasLapack::Layout::ColMajor,
            blasLapack::Op::NoTrans,
            blasLapack::Op::ConjTrans,
            numVec - eigVecStartId,
            numEigVecInBatch,
            vecSize,
            alpha,
            X.data() + eigVecStartId,
            numVec,
            subspaceBatchOut->data(),
            numEigVecInBatch,
            beta,
            SBlock.data(),
            numVec - eigVecStartId,
            linAlgOpContext);

          // utils::MemoryTransfer<utils::MemorySpace::HOST, memorySpace>::copy(
          //   SBlock.size(), SBlockHost.data(), SBlock.data());

          int mpierr = utils::mpi::MPIAllreduce<memorySpace>(
            utils::mpi::MPIInPlace,
            SBlock /*Host*/.data(),
            (numVec - eigVecStartId) * numEigVecInBatch,
            utils::mpi::Types<ValueType>::getMPIDatatype(),
            utils::mpi::MPISum,
            comm);

          std::pair<bool, std::string> mpiIsSuccessAndMsg =
            utils::mpi::MPIErrIsSuccessAndMsg(mpierr);
          DFTEFE_AssertWithMsg(mpiIsSuccessAndMsg.first,
                               "MPI Error:" + mpiIsSuccessAndMsg.second);

          // Copying only the lower triangular part to projected matrix
          for (size_type iSize = 0; iSize < numEigVecInBatch; iSize++)
            memoryTransfer.copy(numVec - eigVecStartId - iSize,
                                S.data() + (eigVecStartId + iSize) * numVec +
                                  (eigVecStartId + iSize),
                                SBlock.data() +
                                  iSize * (numVec - eigVecStartId) + iSize);

          for (size_type iSize = 0; iSize < vecLocalSize; iSize++)
            memoryTransfer.copy(numEigVecInBatch,
                                X.data() + iSize * numVec + eigVecStartId,
                                subspaceBatchIn->data() +
                                  numEigVecInBatch * iSize);
        }
    }

  } // end of namespace linearAlgebra
} // end of namespace dftefe
