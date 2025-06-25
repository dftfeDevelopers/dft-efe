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
#include <linearAlgebra/Defaults.h>
#include <utils/ConditionalOStream.h>

namespace dftefe
{
  namespace linearAlgebra
  {
    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    ChebyshevFilteredEigenSolver<ValueTypeOperator,
                                 ValueTypeOperand,
                                 memorySpace>::
      ChebyshevFilteredEigenSolver(
        const double                                wantedSpectrumLowerBound,
        const double                                wantedSpectrumUpperBound,
        const double                                unWantedSpectrumUpperBound,
        const double                                polynomialDegree,
        const double                                illConditionTolerance,
        MultiVector<ValueTypeOperand, memorySpace> &eigenSubspaceGuess,
        bool                                        isResidualChebyshevFilter,
        const size_type                             eigenVectorBatchSize,
        OrthogonalizationType                       orthoType,
        bool                                        storeIntermediateSubspaces)
      : d_p(eigenSubspaceGuess.getMPIPatternP2P()->mpiCommunicator(), "CHFSI")
      , d_isResidualChebyFilter(isResidualChebyshevFilter)
      , d_storeIntermediateSubspaces(storeIntermediateSubspaces)
      , d_eigenVecBatchSize(eigenVectorBatchSize)
      , d_eigVecBatchSmall(nullptr)
      , d_eigVecBatch(nullptr)
      , d_subspaceBatchIn(nullptr)
      , d_subspaceBatchOut(nullptr)
      , d_filSubspaceBatchSmall(nullptr)
      , d_filSubspaceBatch(nullptr)
      , d_filteredSubspace(nullptr)
      , d_filteredSubspaceOrtho(nullptr)
      , d_batchSizeSmall(0)
      , d_mpiPatternP2P(eigenSubspaceGuess.getMPIPatternP2P())
      , d_printL2Norms(false)
      , d_orthoType(orthoType)
    {
      if (d_storeIntermediateSubspaces)
        {
          d_filteredSubspaceOrtho =
            std::make_shared<MultiVector<ValueType, memorySpace>>(
              eigenSubspaceGuess, (ValueType)0);
          d_filteredSubspace =
            std::make_shared<MultiVector<ValueType, memorySpace>>(
              eigenSubspaceGuess, (ValueType)0);
        }

      d_eigVecBatch =
        std::make_shared<linearAlgebra::MultiVector<ValueType, memorySpace>>(
          d_mpiPatternP2P,
          eigenSubspaceGuess.getLinAlgOpContext(),
          eigenVectorBatchSize,
          ValueType());

      d_filSubspaceBatch =
        std::make_shared<linearAlgebra::MultiVector<ValueType, memorySpace>>(
          d_mpiPatternP2P,
          eigenSubspaceGuess.getLinAlgOpContext(),
          eigenVectorBatchSize,
          ValueType());

      reinit(wantedSpectrumLowerBound,
             wantedSpectrumUpperBound,
             unWantedSpectrumUpperBound,
             polynomialDegree,
             illConditionTolerance,
             eigenSubspaceGuess);
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    void
    ChebyshevFilteredEigenSolver<ValueTypeOperator,
                                 ValueTypeOperand,
                                 memorySpace>::
      reinit(const double wantedSpectrumLowerBound,
             const double wantedSpectrumUpperBound,
             const double unWantedSpectrumUpperBound,
             const double polynomialDegree,
             const double illConditionTolerance,
             MultiVector<ValueTypeOperand, memorySpace> &eigenSubspaceGuess)
    {
      d_eigenSubspaceGuess         = &eigenSubspaceGuess;
      d_polynomialDegree           = polynomialDegree;
      d_wantedSpectrumLowerBound   = wantedSpectrumLowerBound;
      d_wantedSpectrumUpperBound   = wantedSpectrumUpperBound;
      d_unWantedSpectrumUpperBound = unWantedSpectrumUpperBound;

      d_rr = std::make_shared<
        RayleighRitzEigenSolver<ValueTypeOperator, ValueType, memorySpace>>();

      /*create filtered subspace vec after calling isCompatible*/

      if (d_storeIntermediateSubspaces)
        {
          if (!d_filteredSubspace->isCompatible(eigenSubspaceGuess))
            {
              d_filteredSubspaceOrtho =
                std::make_shared<MultiVector<ValueType, memorySpace>>(
                  eigenSubspaceGuess, (ValueType)0);
              d_filteredSubspace =
                std::make_shared<MultiVector<ValueType, memorySpace>>(
                  eigenSubspaceGuess, (ValueType)0);
            }
        }

      if (!d_mpiPatternP2P->isCompatible(
            *eigenSubspaceGuess.getMPIPatternP2P()))
        {
          d_mpiPatternP2P = eigenSubspaceGuess.getMPIPatternP2P();
          d_eigVecBatch   = std::make_shared<
            linearAlgebra::MultiVector<ValueType, memorySpace>>(
            d_mpiPatternP2P,
            eigenSubspaceGuess.getLinAlgOpContext(),
            d_eigenVecBatchSize,
            ValueType());

          d_filSubspaceBatch = std::make_shared<
            linearAlgebra::MultiVector<ValueType, memorySpace>>(
            d_mpiPatternP2P,
            eigenSubspaceGuess.getLinAlgOpContext(),
            d_eigenVecBatchSize,
            ValueType());
        }
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    EigenSolverError
    ChebyshevFilteredEigenSolver<ValueTypeOperator,
                                 ValueTypeOperand,
                                 memorySpace>::
      solve(const OpContext &                    A,
            std::vector<RealType> &              eigenValues,
            MultiVector<ValueType, memorySpace> &eigenVectors, /*in/out*/
            bool                                 computeEigenVectors,
            const OpContext &                    B,
            const OpContext &                    BInv)
    {
      d_p.reset();
      EigenSolverError        retunValue;
      EigenSolverErrorCode    err;
      OrthonormalizationError orthoerr;
      EigenSolverError        rrerr;

      // [CF] Chebyshev filtering of \psi

      int rank;
      utils::mpi::MPICommRank(d_mpiPatternP2P->mpiCommunicator(), &rank);
      utils::ConditionalOStream rootCout(std::cout);
      rootCout.setCondition(rank == 0);

      if (d_printL2Norms)
        {
          rootCout << "d_eigenSubspaceGuess l2norms CHFSI: ";
          for (auto &i : d_eigenSubspaceGuess->l2Norms())
            rootCout << i << "\t";
          rootCout << "\n";
        }

      d_p.registerStart("Chebyshev Filter");

      size_type numEigenVectors   = eigenVectors.getNumberComponents();
      size_type eigenVecLocalSize = eigenVectors.localSize();
      utils::MemoryTransfer<memorySpace, memorySpace> memoryTransfer;

      for (size_type eigVecStartId = 0; eigVecStartId < numEigenVectors;
           eigVecStartId += d_eigenVecBatchSize)
        {
          const size_type eigVecEndId =
            std::min(eigVecStartId + d_eigenVecBatchSize, numEigenVectors);
          const size_type numEigVecInBatch = eigVecEndId - eigVecStartId;

          std::vector<RealType> eigenValBatch(numEigVecInBatch, 0);

          std::copy(eigenValues.data() + eigVecStartId,
                    eigenValues.data() + eigVecEndId,
                    eigenValBatch.begin());

          if (numEigVecInBatch % d_eigenVecBatchSize == 0)
            {
              for (size_type iSize = 0; iSize < eigenVecLocalSize; iSize++)
                memoryTransfer.copy(numEigVecInBatch,
                                    d_eigVecBatch->data() +
                                      numEigVecInBatch * iSize,
                                    d_eigenSubspaceGuess->data() +
                                      iSize * numEigenVectors + eigVecStartId);

              d_subspaceBatchIn  = d_eigVecBatch;
              d_subspaceBatchOut = d_filSubspaceBatch;
            }
          else if (numEigVecInBatch % d_eigenVecBatchSize == d_batchSizeSmall)
            {
              for (size_type iSize = 0; iSize < eigenVecLocalSize; iSize++)
                memoryTransfer.copy(numEigVecInBatch,
                                    d_eigVecBatchSmall->data() +
                                      numEigVecInBatch * iSize,
                                    d_eigenSubspaceGuess->data() +
                                      iSize * numEigenVectors + eigVecStartId);

              d_subspaceBatchIn  = d_eigVecBatchSmall;
              d_subspaceBatchOut = d_filSubspaceBatchSmall;
            }
          else
            {
              d_batchSizeSmall = numEigVecInBatch;

              d_eigVecBatchSmall = std::make_shared<
                linearAlgebra::MultiVector<ValueType, memorySpace>>(
                d_mpiPatternP2P,
                eigenVectors.getLinAlgOpContext(),
                numEigVecInBatch,
                ValueType());

              d_filSubspaceBatchSmall = std::make_shared<
                linearAlgebra::MultiVector<ValueType, memorySpace>>(
                d_mpiPatternP2P,
                eigenVectors.getLinAlgOpContext(),
                numEigVecInBatch,
                ValueType());

              for (size_type iSize = 0; iSize < eigenVecLocalSize; iSize++)
                memoryTransfer.copy(numEigVecInBatch,
                                    d_eigVecBatchSmall->data() +
                                      numEigVecInBatch * iSize,
                                    d_eigenSubspaceGuess->data() +
                                      iSize * numEigenVectors + eigVecStartId);

              d_subspaceBatchIn  = d_eigVecBatchSmall;
              d_subspaceBatchOut = d_filSubspaceBatchSmall;
            }
          if (d_isResidualChebyFilter)
            ResidualChebyshevFilterGEP<ValueTypeOperator,
                                       ValueTypeOperand,
                                       memorySpace>(
              A,
              B,
              BInv,
              eigenValBatch,
              *d_subspaceBatchIn, /*scratch1*/
              d_polynomialDegree,
              d_wantedSpectrumLowerBound,
              d_wantedSpectrumUpperBound,
              d_unWantedSpectrumUpperBound,
              *d_subspaceBatchOut); /*scratch2*/
          else
            ChebyshevFilter<ValueTypeOperator, ValueTypeOperand, memorySpace>(
              A,
              BInv,
              *d_subspaceBatchIn, /*scratch1*/
              d_polynomialDegree,
              d_wantedSpectrumLowerBound,
              d_wantedSpectrumUpperBound,
              d_unWantedSpectrumUpperBound,
              *d_subspaceBatchOut); /*scratch2*/

          for (size_type iSize = 0; iSize < eigenVecLocalSize; iSize++)
            memoryTransfer.copy(numEigVecInBatch,
                                eigenVectors.data() + iSize * numEigenVectors +
                                  eigVecStartId,
                                d_subspaceBatchOut->data() +
                                  numEigVecInBatch * iSize);
        }

      if (d_storeIntermediateSubspaces && d_printL2Norms)
        {
          *d_filteredSubspace = eigenVectors;
          rootCout << "d_filteredSubspace l2norms CHFSI: ";
          for (auto &i : d_filteredSubspace->l2Norms())
            rootCout << i << "\t";
          rootCout << "\n";
        }

      d_p.registerEnd("Chebyshev Filter");
      d_p.registerStart("OrthoNormalization");

      // B orthogonalization required of X -> X_O : /*scratch2->eigenvector*/

      if (d_orthoType == OrthogonalizationType::CHOLESKY_GRAMSCHMIDT)
        {
          orthoerr = OrthonormalizationFunctions<
            ValueTypeOperator,
            ValueTypeOperand,
            memorySpace>::CholeskyGramSchmidt(eigenVectors,
                                              *d_eigenSubspaceGuess,
                                              B);
        }
      else if (d_orthoType == OrthogonalizationType::MULTIPASS_LOWDIN)
        {
          orthoerr = linearAlgebra::
            OrthonormalizationFunctions<ValueType, ValueType, memorySpace>::
              MultipassLowdin(
                eigenVectors, /*in/out, eigenvector*/
                linearAlgebra::MultiPassLowdinDefaults::MAX_PASS,
                linearAlgebra::MultiPassLowdinDefaults::SHIFT_TOL,
                linearAlgebra::MultiPassLowdinDefaults::IDENTITY_TOL,
                *d_eigenSubspaceGuess, // go away
                B);
        }
      else
        {
          utils::throwException(false, "Orthogonalization type not present");
        }

      // orthoerr = linearAlgebra::OrthonormalizationFunctions<
      //   ValueType,
      //   ValueType,
      //   memorySpace>::ModifiedGramSchmidt(eigenVectors,
      //                                     *d_eigenSubspaceGuess,
      //                                     B);

      if (d_storeIntermediateSubspaces && d_printL2Norms)
        {
          *d_filteredSubspaceOrtho = *d_eigenSubspaceGuess;
          rootCout << "d_filteredSubspaceOrtho l2norms CHFSI: ";
          for (auto &i : d_filteredSubspaceOrtho->l2Norms())
            rootCout << i << "\t";
          rootCout << "\n";
        }
      d_p.registerEnd("OrthoNormalization");

      // [RR] Perform the Rayleigh–Ritz procedure for filteredSubspaceOrtho

      d_p.registerStart("RR Step");
      rrerr = d_rr->solve(A,
                          *d_eigenSubspaceGuess, // go away
                          eigenValues,
                          eigenVectors, /*in/out*/
                          computeEigenVectors);
      d_p.registerEnd("RR Step");
      d_p.print();

      if (d_printL2Norms)
        {
          rootCout << "eigenVectors l2norms CHFSI: ";
          for (auto &i : eigenVectors.l2Norms())
            rootCout << i << "\t";
          rootCout << "\n";
        }

      /* Using GHEP with B = Identity in orthogonalization
       * does not work. Prob due to no distribute C2P
       * in IdenstiyOperator. */
      // rrerr = d_rr->solve(A,
      //                     B,
      //                     *d_eigenSubspaceGuess,//go away
      //                     eigenValues,
      //                     eigenVectors,/*in/out*/
      //                     computeEigenVectors);

      if (!orthoerr.isSuccess)
        {
          err        = EigenSolverErrorCode::CHFSI_ORTHONORMALIZATION_ERROR;
          retunValue = EigenSolverErrorMsg::isSuccessAndMsg(err);
          retunValue.msg += orthoerr.msg;
        }
      else if (!rrerr.isSuccess)
        {
          err        = EigenSolverErrorCode::CHFSI_RAYLEIGH_RITZ_ERROR;
          retunValue = EigenSolverErrorMsg::isSuccessAndMsg(err);
          retunValue.msg += rrerr.msg;
        }
      else
        {
          err        = EigenSolverErrorCode::SUCCESS;
          retunValue = EigenSolverErrorMsg::isSuccessAndMsg(err);
          retunValue.msg += orthoerr.msg;
        }

      return retunValue;
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    MultiVector<blasLapack::scalar_type<ValueTypeOperator, ValueTypeOperand>,
                memorySpace> &
    ChebyshevFilteredEigenSolver<ValueTypeOperator,
                                 ValueTypeOperand,
                                 memorySpace>::getFilteredSubspace()
    {
      if (d_storeIntermediateSubspaces == true)
        return *d_filteredSubspace;
      else
        {
          utils::throwException(
            false,
            "storeIntermediateSubspaces is false in CHFSI class. Cannot return the filtered Subspace.");
          return *d_filteredSubspace;
        }
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    MultiVector<blasLapack::scalar_type<ValueTypeOperator, ValueTypeOperand>,
                memorySpace> &
    ChebyshevFilteredEigenSolver<
      ValueTypeOperator,
      ValueTypeOperand,
      memorySpace>::getOrthogonalizedFilteredSubspace()
    {
      if (d_storeIntermediateSubspaces == true)
        return *d_filteredSubspaceOrtho;
      else
        {
          utils::throwException(
            false,
            "storeIntermediateSubspaces is false in CHFSI class. Cannot return the filtered Subspace Ortho.");
          return *d_filteredSubspaceOrtho;
        }
    }

  } // end of namespace linearAlgebra
} // end of namespace dftefe
