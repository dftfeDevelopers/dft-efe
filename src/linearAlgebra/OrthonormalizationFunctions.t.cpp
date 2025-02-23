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
#include <limits.h>
#include <linearAlgebra/BlasLapackTypedef.h>
#include <linearAlgebra/BlasLapack.h>
#include <utils/DataTypeOverloads.h>

namespace dftefe
{
  namespace linearAlgebra
  {
    namespace OrthonormalizationFunctionsInternal
    {
      template <typename ValueTypeOperator,
                typename ValueTypeOperand,
                utils::MemorySpace memorySpace>
      double
      doesOrthogonalizationPreserveSubspace(
        MultiVector<ValueTypeOperand, memorySpace> &X,
        MultiVector<
          blasLapack::scalar_type<ValueTypeOperator, ValueTypeOperand>,
          memorySpace> &orthogonalizedX,
        const OperatorContext<ValueTypeOperator, ValueTypeOperand, memorySpace>
          &B)
      {
        using ValueType =
          blasLapack::scalar_type<ValueTypeOperator, ValueTypeOperand>;
        LinAlgOpContext<memorySpace> linAlgOpContext = *X.getLinAlgOpContext();
        const size_type              vecSize         = X.locallyOwnedSize();
        const size_type              numVec          = X.getNumberComponents();

        linearAlgebra::MultiVector<ValueType, memorySpace> X0X0HBX(X, 0.0),
          residual(X, 0.0);
        utils::MemoryStorage<ValueType, memorySpace> X0HBX(numVec * numVec);

        B.apply(X, residual, true, false);
        linearAlgebra::blasLapack::
          gemm<ValueTypeOperand, ValueType, memorySpace>(
            linearAlgebra::blasLapack::Layout::ColMajor,
            linearAlgebra::blasLapack::Op::NoTrans,
            linearAlgebra::blasLapack::Op::ConjTrans,
            numVec,
            numVec,
            vecSize,
            1,
            residual.data(),
            numVec,
            orthogonalizedX.data(),
            numVec,
            0,
            X0HBX.data(),
            numVec,
            linAlgOpContext);

        // MPI_AllReduce to get the S from all procs
        // mpi_inplace
        int err = utils::mpi::MPIAllreduce<memorySpace>(
          utils::mpi::MPIInPlace,
          X0HBX.data(),
          X0HBX.size(),
          utils::mpi::Types<ValueType>::getMPIDatatype(),
          utils::mpi::MPISum,
          utils::mpi::MPICommWorld);

        std::pair<bool, std::string> mpiIsSuccessAndMsg =
          utils::mpi::MPIErrIsSuccessAndMsg(err);
        utils::throwException(mpiIsSuccessAndMsg.first,
                              "MPI Error:" + mpiIsSuccessAndMsg.second);

        linearAlgebra::blasLapack::gemm<ValueType, ValueType, memorySpace>(
          linearAlgebra::blasLapack::Layout::ColMajor,
          linearAlgebra::blasLapack::Op::NoTrans,
          linearAlgebra::blasLapack::Op::NoTrans,
          numVec,
          vecSize,
          numVec,
          1,
          X0HBX.data(),
          numVec,
          orthogonalizedX.data(),
          numVec,
          0,
          X0X0HBX.data(),
          numVec,
          linAlgOpContext);

        X0X0HBX.updateGhostValues();

        ValueType ones = (ValueType)1.0, nOnes = (ValueType)-1.0;
        add(ones, X, nOnes, X0X0HBX, residual);

        std::vector<double> norm = residual.lInfNorms();
        double              max  = *std::max_element(norm.begin(), norm.end());
        return max;
      }
    } // namespace OrthonormalizationFunctionsInternal

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    OrthonormalizationError
    OrthonormalizationFunctions<ValueTypeOperator,
                                ValueTypeOperand,
                                memorySpace>::
      CholeskyGramSchmidt(MultiVector<ValueTypeOperand, memorySpace> &X,
                          MultiVector<ValueType, memorySpace> &orthogonalizedX,
                          const OpContext &                    B)
    {
      OrthonormalizationErrorCode err;
      OrthonormalizationError     retunValue;
      orthogonalizedX.setValue((ValueType)0.0);

      if (X.globalSize() < X.getNumberComponents())
        {
          err = OrthonormalizationErrorCode::NON_ORTHONORMALIZABLE_MULTIVECTOR;
          OrthonormalizationError retunValue =
            OrthonormalizationErrorMsg::isSuccessAndMsg(err);
        }
      else
        {
          const utils::mpi::MPIComm comm =
            X.getMPIPatternP2P()->mpiCommunicator();
          LinAlgOpContext<memorySpace> linAlgOpContext =
            *X.getLinAlgOpContext();
          const size_type vecSize = X.locallyOwnedSize();
          const size_type numVec  = X.getNumberComponents();

          // allocate memory for overlap matrix
          utils::MemoryStorage<ValueType, memorySpace> S(
            numVec * numVec, utils::Types<ValueType>::zero);

          // compute overlap matrix

          const ValueType alpha = 1.0;
          const ValueType beta  = 0.0;

          MultiVector<ValueType, memorySpace> temp(X, (ValueType)0);

          B.apply(X, temp, true, false);

          // Input data is read is X^T (numVec is fastest index and then
          // vecSize) Operation : S^T = ((B*X)^T)*(X^T)^H

          blasLapack::gemm<ValueTypeOperand, ValueType, memorySpace>(
            blasLapack::Layout::ColMajor,
            blasLapack::Op::NoTrans,
            blasLapack::Op::ConjTrans,
            numVec,
            numVec,
            vecSize,
            alpha,
            temp.data(),
            numVec,
            X.data(),
            numVec,
            beta,
            S.data(),
            numVec,
            linAlgOpContext);

          // TODO: Copy only the real part because S is real and then do
          // cholesky Reason: Reduced flops.

          // MPI_AllReduce to get the S from all procs

          int mpierr = utils::mpi::MPIAllreduce<memorySpace>(
            utils::mpi::MPIInPlace,
            S.data(),
            S.size(),
            utils::mpi::Types<ValueType>::getMPIDatatype(),
            utils::mpi::MPISum,
            comm);

          std::pair<bool, std::string> mpiIsSuccessAndMsg =
            utils::mpi::MPIErrIsSuccessAndMsg(mpierr);
          utils::throwException(mpiIsSuccessAndMsg.first,
                                "MPI Error:" + mpiIsSuccessAndMsg.second);

          // cholesky factorization of overlap matrix
          // Operation = S^T = L^C*L^T = (L^C)*(L^C)^H ; Out: L^C

          LapackError lapackReturn1 = blasLapack::potrf<ValueType, memorySpace>(
            blasLapack::Uplo::Lower, numVec, S.data(), numVec, linAlgOpContext);

          // Compute LInv^C

          LapackError lapackReturn2 =
            blasLapack::trtri<ValueType, memorySpace>(blasLapack::Uplo::Lower,
                                                      blasLapack::Diag::NonUnit,
                                                      numVec,
                                                      S.data(),
                                                      numVec,
                                                      linAlgOpContext);

          // complete the lower triangular part
          // (LInv^C)

          utils::MemoryStorage<ValueType, utils::MemorySpace::HOST> Shost(
            S.size());
          utils::MemoryTransfer<utils::MemorySpace::HOST, memorySpace>::copy(
            S.size(), Shost.data(), S.data());

          for (size_type i = 0; i < numVec; i++) // column
            {
              for (size_type j = 0; j < numVec; j++) // row
                {
                  if (i < j) // if colid < rowid i.e. upper tri
                    {
                      *(Shost.data() + j * numVec + i) = (ValueType)0.0;
                    }
                }
            }

          utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
            Shost.size(), S.data(), Shost.data());

          // compute orthogonalizedX
          // XOrth^T = (LInv^C)^T*X^T
          // Out data as XOrtho^T

          blasLapack::gemm<ValueType, ValueTypeOperand, memorySpace>(
            blasLapack::Layout::ColMajor,
            blasLapack::Op::NoTrans,
            blasLapack::Op::NoTrans,
            numVec,
            vecSize,
            numVec,
            alpha,
            S.data(),
            numVec,
            X.data(),
            numVec,
            beta,
            orthogonalizedX.data(),
            numVec,
            linAlgOpContext);

          if (lapackReturn1.err ==
              LapackErrorCode::FAILED_CHOLESKY_FACTORIZATION)
            {
              err        = OrthonormalizationErrorCode::LAPACK_ERROR;
              retunValue = OrthonormalizationErrorMsg::isSuccessAndMsg(err);
              retunValue.msg += lapackReturn1.msg;
            }
          else if (lapackReturn2.err ==
                   LapackErrorCode::FAILED_TRIA_MATRIX_INVERSE)
            {
              err        = OrthonormalizationErrorCode::LAPACK_ERROR;
              retunValue = OrthonormalizationErrorMsg::isSuccessAndMsg(err);
              retunValue.msg += lapackReturn2.msg;
            }
          else
            {
              err        = OrthonormalizationErrorCode::SUCCESS;
              retunValue = OrthonormalizationErrorMsg::isSuccessAndMsg(err);
            }
        }
      return retunValue;
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    OrthonormalizationError
    OrthonormalizationFunctions<ValueTypeOperator,
                                ValueTypeOperand,
                                memorySpace>::
      MultipassLowdin(MultiVector<ValueTypeOperand, memorySpace> &X,
                      size_type                                   maxPass,
                      RealType                             shiftTolerance,
                      RealType                             identityTolerance,
                      MultiVector<ValueType, memorySpace> &orthogonalizedX,
                      const OpContext &                    B)
    {
      OrthonormalizationErrorCode err;
      OrthonormalizationError     retunValue;
      LapackError                 lapackReturn;

      /*
      //--------------------------DEBUG ONLY------------------------------
      linearAlgebra::MultiVector<ValueType, memorySpace> copyX(X, 0.0);
      copyX = X;
      //--------------------------DEBUG ONLY------------------------------
      */

      const utils::mpi::MPIComm comm = X.getMPIPatternP2P()->mpiCommunicator();
      LinAlgOpContext<memorySpace> linAlgOpContext = *X.getLinAlgOpContext();
      const size_type              vecSize         = X.locallyOwnedSize();
      const size_type              numVec          = X.getNumberComponents();

      orthogonalizedX.setValue((ValueType)0.0);
      /*RealType u = std::numeric_limits<RealType>::epsilon();*/

      // allocate memory for overlap matrix
      utils::MemoryStorage<ValueType, memorySpace> S(
        numVec * numVec, utils::Types<ValueType>::zero);
      utils::MemoryStorage<ValueType, memorySpace> eigenVectorsS(
        numVec * numVec, utils::Types<ValueType>::zero);
      utils::MemoryStorage<ValueType, memorySpace> scratch(
        numVec * numVec, utils::Types<ValueType>::zero);
      utils::MemoryStorage<RealType, memorySpace> sqrtInvShiftedEigenValMatrix(
        numVec * numVec, utils::Types<RealType>::zero);
      MultiVector<ValueType, memorySpace> temp(X, (ValueType)0);

      // compute overlap matrix

      const ValueType alpha = 1.0;
      const ValueType beta  = 0.0;

      if (X.globalSize() < X.getNumberComponents())
        {
          err = OrthonormalizationErrorCode::NON_ORTHONORMALIZABLE_MULTIVECTOR;
        }
      else
        {
          RealType eigenValueMin = (RealType)0;
          RealType orthoErr      = (RealType)0;

          /* Do cholesky factorization until Frobenius(||Y^T B Y -
          I||)/root(size of I) < identityTolerance
          *  To reduce computation one does not compute the above but get a
          rough
          * estimate by doing prefactor * e_machine/root(e_min) <
          identityTolerance*root(size of I) */

          size_type iPass = 0;

          while (iPass <= maxPass)
            {
              /* Get S = X^T B X */

              // Input data is read is X^T
              // Operation : S^T = ((B*X)^T)*(X^T)^H

              B.apply(X, temp, true, false);

              blasLapack::gemm<ValueTypeOperand, ValueType, memorySpace>(
                blasLapack::Layout::ColMajor,
                blasLapack::Op::NoTrans,
                blasLapack::Op::ConjTrans,
                numVec,
                numVec,
                vecSize,
                alpha,
                temp.data(),
                numVec,
                X.data(),
                numVec,
                beta,
                S.data(),
                numVec,
                linAlgOpContext);

              int mpierr = utils::mpi::MPIAllreduce<memorySpace>(
                utils::mpi::MPIInPlace,
                S.data(),
                S.size(),
                utils::mpi::Types<ValueType>::getMPIDatatype(),
                utils::mpi::MPISum,
                comm);

              std::pair<bool, std::string> mpiIsSuccessAndMsg =
                utils::mpi::MPIErrIsSuccessAndMsg(mpierr);
              utils::throwException(mpiIsSuccessAndMsg.first,
                                    "MPI Error:" + mpiIsSuccessAndMsg.second);

              utils::MemoryStorage<ValueType, utils::MemorySpace::HOST> Shost(
                S.size());
              utils::MemoryTransfer<utils::MemorySpace::HOST,
                                    memorySpace>::copy(S.size(),
                                                       Shost.data(),
                                                       S.data());

              ValueType orthoErrValueType = 0;
              // calculate frobenus norm |S - I|
              for (size_type i = 0; i < numVec; i++)
                {
                  for (size_type j = 0; j < numVec; j++)
                    {
                      if (i != j)
                        orthoErrValueType += *(Shost.data() + i * numVec + j) *
                                             utils::conjugate<ValueType>(*(
                                               Shost.data() + i * numVec + j));
                      else
                        orthoErrValueType +=
                          (*(Shost.data() + i * numVec + j) - (ValueType)1.0) *
                          utils::conjugate<ValueType>(
                            *(Shost.data() + i * numVec + j) - (ValueType)1.0);
                    }
                }
              orthoErr =
                std::sqrt(utils::realPart<ValueType>(orthoErrValueType));
              if (orthoErr < identityTolerance * std::sqrt(numVec))
                break;

              eigenVectorsS = S;

              /* do a eigendecomposition and get min eigenvalue and get shift*/

              utils::MemoryStorage<RealType, memorySpace> eigenValuesSmemory(
                numVec);

              lapackReturn = blasLapack::heevd<ValueType, memorySpace>(
                blasLapack::Job::Vec,
                blasLapack::Uplo::Lower,
                numVec,
                eigenVectorsS.data(),
                numVec,
                eigenValuesSmemory.data(),
                linAlgOpContext);

              if (!lapackReturn.isSuccess)
                {
                  err        = OrthonormalizationErrorCode::LAPACK_ERROR;
                  retunValue = OrthonormalizationErrorMsg::isSuccessAndMsg(err);
                  retunValue.msg += lapackReturn.msg;
                  break;
                }

              std::vector<RealType> eigenValuesS(numVec);
              utils::MemoryTransfer<utils::MemorySpace::HOST, memorySpace>::
                copy(eigenValuesSmemory.size(),
                     eigenValuesS.data(),
                     eigenValuesSmemory.data());

              eigenValueMin = eigenValuesS[0];

              RealType shift = (RealType)0;
              if (eigenValueMin > shiftTolerance)
                shift = (RealType)0;
              else
                shift = shiftTolerance - eigenValueMin;

              /* Shift by D<-D+shift and do D^(-1/2)*/

              std::vector<RealType> sqrtInvShiftedEigenValMatrixSTL(numVec *
                                                                    numVec);

              for (size_type i = 0; i < numVec; i++)
                sqrtInvShiftedEigenValMatrixSTL[i * numVec + i] =
                  (RealType)(1.0 / std::sqrt(eigenValuesS[i] + shift));

              utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::
                copy(sqrtInvShiftedEigenValMatrixSTL.size(),
                     sqrtInvShiftedEigenValMatrix.data(),
                     sqrtInvShiftedEigenValMatrixSTL.data());

              /* Do Y = XVD^(-1/2)V^H */
              // S = VD^(-1/2)
              blasLapack::gemm<ValueType, RealType, memorySpace>(
                blasLapack::Layout::ColMajor,
                blasLapack::Op::NoTrans,
                blasLapack::Op::NoTrans,
                numVec,
                numVec,
                numVec,
                alpha,
                eigenVectorsS.data(),
                numVec,
                sqrtInvShiftedEigenValMatrix.data(),
                numVec,
                beta,
                S.data(),
                numVec,
                linAlgOpContext);

              // S = VD^(-1/2)V^H
              blasLapack::gemm<ValueType, ValueType, memorySpace>(
                blasLapack::Layout::ColMajor,
                blasLapack::Op::NoTrans,
                blasLapack::Op::ConjTrans,
                numVec,
                numVec,
                numVec,
                alpha,
                S.data(),
                numVec,
                eigenVectorsS.data(),
                numVec,
                beta,
                scratch.data(),
                numVec,
                linAlgOpContext);

              blasLapack::gemm<ValueType, ValueType, memorySpace>(
                blasLapack::Layout::ColMajor,
                blasLapack::Op::NoTrans,
                blasLapack::Op::NoTrans,
                numVec,
                vecSize,
                numVec,
                alpha,
                scratch.data(),
                numVec,
                X.data(),
                numVec,
                beta,
                orthogonalizedX.data(),
                numVec,
                linAlgOpContext);

              X = orthogonalizedX;
              iPass++;
            }

          /**
          //--------------------------DEBUG ONLY------------------------------
          double norm = OrthonormalizationFunctionsInternal::
            doesOrthogonalizationPreserveSubspace<ValueTypeOperator,
                                                  ValueTypeOperand,
                                                  memorySpace>(copyX,
                                                               orthogonalizedX,
                                                               B);
          //--------------------------DEBUG ONLY------------------------------
          **/

          if (!(err == OrthonormalizationErrorCode::LAPACK_ERROR))
            {
              if (iPass > maxPass)
                {
                  err        = OrthonormalizationErrorCode::MAX_PASS_EXCEEDED;
                  retunValue = OrthonormalizationErrorMsg::isSuccessAndMsg(err);
                }
              else
                {
                  err        = OrthonormalizationErrorCode::SUCCESS;
                  retunValue = OrthonormalizationErrorMsg::isSuccessAndMsg(err);
                  retunValue.msg += "Maximum number of Lowdin passes are " +
                                    std::to_string(iPass) + ".";
                  /*
                  //--------------------------DEBUG
                  ONLY------------------------------ std::stringstream ss; ss <<
                  norm; retunValue.msg += " Max LInf norm |(I-QQ^HM)U|: " +
                  ss.str();
                  //--------------------------DEBUG
                  ONLY------------------------------
                  */
                }
            }
        }

      return retunValue;
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    OrthonormalizationError
    OrthonormalizationFunctions<ValueTypeOperator,
                                ValueTypeOperand,
                                memorySpace>::
      ModifiedGramSchmidt(MultiVector<ValueTypeOperand, memorySpace> &X,
                          MultiVector<ValueType, memorySpace> &orthogonalizedX,
                          const OpContext &                    B)
    {
      // Naive Modified Gram Schmidt implementation
      // https://arnold.hosted.uark.edu/NLA/Pages/CGSMGS.pdf

      OrthonormalizationErrorCode err;
      OrthonormalizationError     retunValue;

      LinAlgOpContext<memorySpace> linAlgOpContext = *X.getLinAlgOpContext();
      const size_type              vecSize         = X.locallyOwnedSize();
      const size_type              numVec          = X.getNumberComponents();

      orthogonalizedX.setValue((ValueType)0.0);
      utils::MemoryTransfer<memorySpace, memorySpace> memoryTransfer;
      ValueType alpha, nAlpha, one = (ValueType)1.0;

      Vector<ValueType, memorySpace> temp(X.getMPIPatternP2P(),
                                          X.getLinAlgOpContext(),
                                          (ValueType)0),
        w(temp, (ValueType)0), q(temp, (ValueType)0);

      if (X.globalSize() < X.getNumberComponents())
        {
          err = OrthonormalizationErrorCode::NON_ORTHONORMALIZABLE_MULTIVECTOR;
          OrthonormalizationError retunValue =
            OrthonormalizationErrorMsg::isSuccessAndMsg(err);
        }
      else
        {
          for (size_type iVec = 0; iVec < numVec; iVec++)
            {
              for (size_type iSize = 0; iSize < vecSize; iSize++)
                memoryTransfer.copy(1,
                                    w.data() + iSize,
                                    X.data() + iSize * numVec + iVec);
              for (size_type jVec = 0; jVec < iVec; jVec++)
                {
                  for (size_type iSize = 0; iSize < vecSize; iSize++)
                    memoryTransfer.copy(1,
                                        q.data() + iSize,
                                        orthogonalizedX.data() +
                                          iSize * numVec + jVec);
                  B.apply(w, temp, true, false);
                  dot<ValueType, ValueType, memorySpace>(
                    q,
                    temp,
                    alpha,
                    blasLapack::ScalarOp::Conj,
                    blasLapack::ScalarOp::Identity);
                  nAlpha = (ValueType)(-1.0) * (ValueType)alpha;
                  blasLapack::axpby(vecSize,
                                    one,
                                    w.data(),
                                    nAlpha,
                                    q.data(),
                                    w.data(),
                                    linAlgOpContext);
                }
              B.apply(w, temp, true, false);
              dot<ValueType, ValueType, memorySpace>(
                w,
                temp,
                alpha,
                blasLapack::ScalarOp::Conj,
                blasLapack::ScalarOp::Identity);
              alpha = std::sqrt(alpha);
              blasLapack::ascale<ValueType, ValueType, memorySpace>(
                vecSize,
                (ValueType)(1.0 / alpha),
                w.data(),
                q.data(),
                linAlgOpContext);
              for (size_type iSize = 0; iSize < vecSize; iSize++)
                memoryTransfer.copy(1,
                                    orthogonalizedX.data() + iSize * numVec +
                                      iVec,
                                    q.data() + iSize);
            }
          err        = OrthonormalizationErrorCode::SUCCESS;
          retunValue = OrthonormalizationErrorMsg::isSuccessAndMsg(err);
        }
      return retunValue;
    }
  } // end of namespace linearAlgebra
} // end of namespace dftefe
