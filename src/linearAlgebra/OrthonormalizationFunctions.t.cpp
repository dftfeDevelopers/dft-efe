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

namespace dftefe
{
  namespace linearAlgebra
  {
    namespace OrthonormalizationFunctionsInternal
    {
      template <typename ValueType, utils::MemorySpace memorySpace>
      void
      CholeskyGramSchmidtImpl(const size_type               vecSize,
                              const size_type               numVec,
                              const ValueType *             X,
                              ValueType *                   orthogonalizedX,
                              LinAlgOpContext<memorySpace> &linAlgOpContext,
                              const utils::mpi::MPIComm &   comm)
      {
        // allocate memory for overlap matrix
        utils::MemoryStorage<ValueType, memorySpace> S(
          numVec * numVec, utils::Types<ValueType>::zero);

        // compute overlap matrix

        const ValueType alpha = 1.0;
        const ValueType beta  = 0.0;

        // Input data is read is X^T (numVec is fastest index and then vecSize)
        // Operation : S^T = (X^T)*(X^T)^H

        blasLapack::gemm<ValueType, ValueType, memorySpace>(
          blasLapack::Layout::ColMajor,
          blasLapack::Op::NoTrans,
          blasLapack::Op::ConjTrans,
          numVec,
          numVec,
          vecSize,
          alpha,
          X,
          numVec,
          X,
          numVec,
          beta,
          S.data(),
          numVec,
          linAlgOpContext);

        utils::MemoryStorage<ValueType, utils::MemorySpace::HOST> Shost(
          S.size());
        utils::MemoryTransfer<utils::MemorySpace::HOST, memorySpace>::copy(
          S.size(), Shost.data(), S.data());

        // MPI_AllReduce to get the S from all procs

        int err = utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
          utils::mpi::MPIInPlace,
          Shost.data(),
          Shost.size(),
          utils::mpi::Types<ValueType>::getMPIDatatype(),
          utils::mpi::MPISum,
          comm);

        std::pair<bool, std::string> mpiIsSuccessAndMsg =
          utils::mpi::MPIErrIsSuccessAndMsg(err);
        utils::throwException(mpiIsSuccessAndMsg.first,
                              "MPI Error:" + mpiIsSuccessAndMsg.second);

        // cholesky factorization of overlap matrix
        // Operation = S^T = L^C*L^T = (L^C)*(L^C)^H ; Out: L^C

        lapack::potrf(lapack::Uplo::Lower, numVec, Shost.data(), numVec);

        // Compute LInv^C

        lapack::trtri(lapack::Uplo::Lower,
                      lapack::Diag::NonUnit,
                      numVec,
                      Shost.data(),
                      numVec);

        // complete the upper triangular part
        // (LInv^C)^T ; Out: LInv^H

        for (size_type i = 0; i < numVec; i++) // column
          {
            for (size_type j = 0; j < numVec; j++) // row
              {
                if (i < j) // if colid < rowid i.e. lower tri
                  {
                    *(Shost.data() + j * numVec + i) =
                      *(Shost.data() + i * numVec + j);
                    *(Shost.data() + i * numVec + j) = (ValueType)0.0;
                  }
              }
          }

        // std::cout << "LInvhostafter: \n";
        // for (size_type i = 0; i < numVec; i++)
        //   {
        //     std::cout << "[";
        //     for (size_type j = 0; j < numVec; j++)
        //       {
        //         std::cout << *(Shost.data() + j * numVec + i) << ",";
        //       }
        //     std::cout << "]\n";
        //   }

        utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
          Shost.size(), S.data(), Shost.data());

        // compute orthogonalizedX
        // XOrth^T = (LInv^H)^T*X^T
        // Out data as XOrtho^T

        blasLapack::gemm<ValueType, ValueType, memorySpace>(
          blasLapack::Layout::ColMajor,
          blasLapack::Op::Trans,
          blasLapack::Op::NoTrans,
          numVec,
          vecSize,
          numVec,
          alpha,
          S.data(),
          numVec,
          X,
          numVec,
          beta,
          orthogonalizedX,
          numVec,
          linAlgOpContext);
      }
    } // namespace OrthonormalizationFunctionsInternal

    template <typename ValueType, utils::MemorySpace memorySpace>
    OrthonormalizationError
    OrthonormalizationFunctions<ValueType, memorySpace>::CholeskyGramSchmidt(
      const MultiVector<ValueType, memorySpace> &X,
      MultiVector<ValueType, memorySpace> &      orthogonalizedX)
    {
      OrthonormalizationErrorCode err;
      orthogonalizedX.setValue((ValueType)0.0);

      if (X.globalSize() < X.getNumberComponents())
        {
          err = OrthonormalizationErrorCode::NON_ORTHONORMALIZABLE_MULTIVECTOR;
        }
      else
        {
          OrthonormalizationFunctionsInternal::CholeskyGramSchmidtImpl<
            ValueType,
            memorySpace>(X.locallyOwnedSize(),
                         X.getNumberComponents(),
                         X.data(),
                         orthogonalizedX.data(),
                         *X.getLinAlgOpContext(),
                         X.getMPIPatternP2P()->mpiCommunicator());

          err = OrthonormalizationErrorCode::SUCCESS;
        }

      OrthonormalizationError retunValue =
        OrthonormalizationErrorMsg::isSuccessAndMsg(err);

      return retunValue;
    }

  } // end of namespace linearAlgebra
} // end of namespace dftefe
