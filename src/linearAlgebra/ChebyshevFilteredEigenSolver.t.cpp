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
#include <linearAlgebra/MultiVectorOps.h>
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
        const double wantedSpectrumLowerBound,
        const double wantedSpectrumUpperBound,
        const double unWantedSpectrumUpperBound,
        const double polynomialDegree,
        const double illConditionTolerance,
        std::shared_ptr<const utils::mpi::MPIPatternP2P<memorySpace>>
                                                      mpiPatternP2P,
        std::shared_ptr<LinAlgOpContext<memorySpace>> linAlgOpContext,
        const ElpaScalapackManager &                  elpaScala,
        bool                                          isResidualChebyshevFilter,
        const size_type                               eigenVectorBatchSize,
        bool                                          isGHEP,
        OrthogonalizationType                         orthoType,
        bool storeIntermediateSubspaces,
        std::shared_ptr<MultivectorScratch<ValueType, memorySpace>> scratch)
      : d_p(mpiPatternP2P->mpiCommunicator(), "CHFSI")
      , d_pTotal(mpiPatternP2P->mpiCommunicator(), "CHFSI Solve Time")
      , d_isResidualChebyFilter(isResidualChebyshevFilter)
      , d_storeIntermediateSubspaces(storeIntermediateSubspaces)
      , d_eigenVecBatchSize(eigenVectorBatchSize)
      , d_XinBatchSmall(nullptr)
      , d_XinBatch(nullptr)
      , d_XoutBatchSmall(nullptr)
      , d_XoutBatch(nullptr)
      , d_filteredSubspace(nullptr)
      , d_filteredSubspaceOrtho(nullptr)
      , d_batchSizeSmall(0)
      , d_mpiPatternP2P(mpiPatternP2P)
      , d_printL2Norms(false)
      , d_orthoType(orthoType)
      , d_elpaScala(&elpaScala)
      , d_isGHEP(isGHEP)
      , d_scratch(scratch)
    {
      if (d_storeIntermediateSubspaces && d_printL2Norms)
        {
          d_filteredSubspaceOrtho =
            std::make_shared<MultiVector<ValueType, memorySpace>>(
              mpiPatternP2P, linAlgOpContext, 1, (ValueType)0);
          d_filteredSubspace =
            std::make_shared<MultiVector<ValueType, memorySpace>>(
              mpiPatternP2P, linAlgOpContext, 1, (ValueType)0);
        }

      const bool useScratch =
        scratch != nullptr && scratch->hasXinBatch() &&
        scratch->hasXoutBatch() &&
        scratch->getXinBatchSize() == eigenVectorBatchSize &&
        scratch->getXoutBatchSize() == eigenVectorBatchSize;

      if (useScratch)
        {
          d_XinBatch  = scratch->getXinBatch();
          d_XoutBatch = scratch->getXoutBatch();
        }
      else
        {
          d_XinBatch = std::make_shared<
            linearAlgebra::MultiVector<ValueType, memorySpace>>(
            d_mpiPatternP2P,
            linAlgOpContext,
            eigenVectorBatchSize,
            ValueType());

          d_XoutBatch = std::make_shared<
            linearAlgebra::MultiVector<ValueType, memorySpace>>(
            d_mpiPatternP2P,
            linAlgOpContext,
            eigenVectorBatchSize,
            ValueType());
        }

      d_chfsiScratch1 =
        std::make_shared<linearAlgebra::MultiVector<ValueType, memorySpace>>(
          d_mpiPatternP2P, linAlgOpContext, d_eigenVecBatchSize, ValueType());
      d_chfsiScratch2 =
        std::make_shared<linearAlgebra::MultiVector<ValueType, memorySpace>>(
          d_mpiPatternP2P, linAlgOpContext, d_eigenVecBatchSize, ValueType());
      if (d_isResidualChebyFilter)
        {
          d_chfsiResidualScratch1 = std::make_shared<
            linearAlgebra::MultiVector<ValueType, memorySpace>>(
            d_mpiPatternP2P, linAlgOpContext, d_eigenVecBatchSize, ValueType());
          d_chfsiResidualScratch2 = std::make_shared<
            linearAlgebra::MultiVector<ValueType, memorySpace>>(
            d_mpiPatternP2P, linAlgOpContext, d_eigenVecBatchSize, ValueType());
        }
      else
        {
          d_chfsiResidualScratch1 = nullptr;
          d_chfsiResidualScratch2 = nullptr;
        }

      if (!d_isGHEP)
        d_ortho =
          std::make_shared<OrthonormalizationFunctions<ValueTypeOperator,
                                                       ValueType,
                                                       memorySpace>>(
            eigenVectorBatchSize,
            *d_elpaScala,
            d_mpiPatternP2P,
            linAlgOpContext,
            true,
            scratch);

      d_rr = std::make_shared<
        RayleighRitzEigenSolver<ValueTypeOperator, ValueType, memorySpace>>(
        eigenVectorBatchSize,
        *d_elpaScala,
        d_mpiPatternP2P,
        linAlgOpContext,
        true,
        scratch);

      reinit(wantedSpectrumLowerBound,
             wantedSpectrumUpperBound,
             unWantedSpectrumUpperBound,
             polynomialDegree,
             illConditionTolerance,
             d_mpiPatternP2P,
             linAlgOpContext);
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
             std::shared_ptr<const utils::mpi::MPIPatternP2P<memorySpace>>
                                                           mpiPatternP2P,
             std::shared_ptr<LinAlgOpContext<memorySpace>> linAlgOpContext)
    {
      // d_eigenSubspaceGuess         = &eigenSubspaceGuess;
      d_polynomialDegree           = polynomialDegree;
      d_wantedSpectrumLowerBound   = wantedSpectrumLowerBound;
      d_wantedSpectrumUpperBound   = wantedSpectrumUpperBound;
      d_unWantedSpectrumUpperBound = unWantedSpectrumUpperBound;

      if (!d_mpiPatternP2P->isCompatible(*mpiPatternP2P))
        {
          if (d_storeIntermediateSubspaces && d_printL2Norms)
            {
              d_filteredSubspaceOrtho =
                std::make_shared<MultiVector<ValueType, memorySpace>>(
                  mpiPatternP2P, linAlgOpContext, 1, (ValueType)0);
              d_filteredSubspace =
                std::make_shared<MultiVector<ValueType, memorySpace>>(
                  mpiPatternP2P, linAlgOpContext, 1, (ValueType)0);
            }

          d_mpiPatternP2P = mpiPatternP2P;
          d_XinBatch      = std::make_shared<
            linearAlgebra::MultiVector<ValueType, memorySpace>>(
            d_mpiPatternP2P, linAlgOpContext, d_eigenVecBatchSize, ValueType());

          d_XoutBatch = std::make_shared<
            linearAlgebra::MultiVector<ValueType, memorySpace>>(
            d_mpiPatternP2P, linAlgOpContext, d_eigenVecBatchSize, ValueType());

          d_chfsiScratch1 = std::make_shared<
            linearAlgebra::MultiVector<ValueType, memorySpace>>(
            d_mpiPatternP2P, linAlgOpContext, d_eigenVecBatchSize, ValueType());
          d_chfsiScratch2 = std::make_shared<
            linearAlgebra::MultiVector<ValueType, memorySpace>>(
            d_mpiPatternP2P, linAlgOpContext, d_eigenVecBatchSize, ValueType());
          if (d_isResidualChebyFilter)
            {
              d_chfsiResidualScratch1 = std::make_shared<
                linearAlgebra::MultiVector<ValueType, memorySpace>>(
                d_mpiPatternP2P,
                linAlgOpContext,
                d_eigenVecBatchSize,
                ValueType());
              d_chfsiResidualScratch2 = std::make_shared<
                linearAlgebra::MultiVector<ValueType, memorySpace>>(
                d_mpiPatternP2P,
                linAlgOpContext,
                d_eigenVecBatchSize,
                ValueType());
            }

          if (!d_isGHEP)
            d_ortho =
              std::make_shared<OrthonormalizationFunctions<ValueTypeOperator,
                                                           ValueType,
                                                           memorySpace>>(
                d_eigenVecBatchSize,
                *d_elpaScala,
                d_mpiPatternP2P,
                linAlgOpContext,
                true,
                d_scratch);

          d_rr = std::make_shared<
            RayleighRitzEigenSolver<ValueTypeOperator, ValueType, memorySpace>>(
            d_eigenVecBatchSize,
            *d_elpaScala,
            d_mpiPatternP2P,
            linAlgOpContext,
            true,
            d_scratch);
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
          rootCout << "eigenSubspaceGuess l2norms CHFSI: ";
          for (auto &i : eigenVectors.l2Norms())
            rootCout << i << "\t";
          rootCout << "\n";
        }

      d_p.registerStart("Chebyshev Filter");
      d_pTotal.registerStart("Chebyshev Filter");

      size_type numEigenVectors   = eigenVectors.getNumberComponents();
      size_type eigenVecLocalSize = eigenVectors.localSize();
      utils::MemoryTransfer<memorySpace, memorySpace> memoryTransfer;
      std::shared_ptr<MultiVector<ValueType, memorySpace>>
        subspaceBatchIn  = nullptr,
        subspaceBatchOut = nullptr, chfsiScratch1 = nullptr,
        chfsiScratch2 = nullptr, chfsiResidualScratch1 = nullptr,
        chfsiResidualScratch2 = nullptr;

      auto *Xps = static_cast<
        linearAlgebra::MultiVectorProductSpace<ValueType, memorySpace> *>(
        &eigenVectors);
      const size_type numSpaces        = Xps->numSpaces();
      const size_type numVecPerSpace   = Xps->numVectorsPerSpace();
      const size_type eigVecBatchPerSp = d_eigenVecBatchSize / numSpaces;

      if (d_scratch)
        d_scratch->acquire();

      for (size_type eigVecStartId = 0; eigVecStartId < numVecPerSpace;
           eigVecStartId += eigVecBatchPerSp)
        {
          const size_type numEigVecInBatch =
            std::min(eigVecStartId + eigVecBatchPerSp, numVecPerSpace) -
            eigVecStartId;
          const size_type numEigVecInBatchTotal = numSpaces * numEigVecInBatch;

          std::vector<RealType> eigenValBatch(numEigVecInBatchTotal, 0);
          for (size_type s = 0; s < numSpaces; ++s)
            std::copy(eigenValues.data() + s * numVecPerSpace + eigVecStartId,
                      eigenValues.data() + s * numVecPerSpace + eigVecStartId +
                        numEigVecInBatch,
                      eigenValBatch.begin() + s * numEigVecInBatch);

          if (numEigVecInBatch == eigVecBatchPerSp)
            {
              // for (size_type iSize = 0; iSize < eigenVecLocalSize; iSize++)
              //   memoryTransfer.copy(numEigVecInBatch,
              //                       d_XinBatch->data() +
              //                         numEigVecInBatch * iSize,
              //                       eigenVectors.data() +
              //                         iSize * numEigenVectors +
              //                         eigVecStartId);

              MultiVectorOps::copyToBatch(*Xps,
                                          eigVecStartId,
                                          numEigVecInBatch,
                                          *d_XinBatch,
                                          *eigenVectors.getLinAlgOpContext());

              subspaceBatchIn  = d_XinBatch;
              subspaceBatchOut = d_XoutBatch;

              chfsiScratch1         = d_chfsiScratch1;
              chfsiScratch2         = d_chfsiScratch2;
              chfsiResidualScratch1 = d_chfsiResidualScratch1;
              chfsiResidualScratch2 = d_chfsiResidualScratch2;
            }
          else if (numEigVecInBatch == d_batchSizeSmall)
            {
              MultiVectorOps::copyToBatch(*Xps,
                                          eigVecStartId,
                                          numEigVecInBatch,
                                          *d_XinBatchSmall,
                                          *eigenVectors.getLinAlgOpContext());

              subspaceBatchIn  = d_XinBatchSmall;
              subspaceBatchOut = d_XoutBatchSmall;

              chfsiScratch1         = d_chfsiScratch1Small;
              chfsiScratch2         = d_chfsiScratch2Small;
              chfsiResidualScratch1 = d_chfsiResidualScratch1Small;
              chfsiResidualScratch2 = d_chfsiResidualScratch2Small;
            }
          else
            {
              d_batchSizeSmall = numEigVecInBatch;

              const bool useSmallScratch =
                d_scratch != nullptr && d_scratch->hasXinBatchSmall() &&
                d_scratch->hasXoutBatchSmall() &&
                d_scratch->getXinBatchSmallSize() == numEigVecInBatchTotal;

              if (useSmallScratch)
                {
                  d_XinBatchSmall  = d_scratch->getXinBatchSmall();
                  d_XoutBatchSmall = d_scratch->getXoutBatchSmall();
                }
              else
                {
                  d_XinBatchSmall = std::make_shared<
                    linearAlgebra::MultiVector<ValueType, memorySpace>>(
                    d_mpiPatternP2P,
                    eigenVectors.getLinAlgOpContext(),
                    numEigVecInBatchTotal,
                    ValueType());

                  d_XoutBatchSmall = std::make_shared<
                    linearAlgebra::MultiVector<ValueType, memorySpace>>(
                    d_mpiPatternP2P,
                    eigenVectors.getLinAlgOpContext(),
                    numEigVecInBatchTotal,
                    ValueType());
                  if (d_scratch != nullptr)
                    {
                      d_scratch->setXinBatchSmall(d_XinBatchSmall);
                      d_scratch->setXoutBatchSmall(d_XoutBatchSmall);
                    }
                }

              d_chfsiScratch1Small = std::make_shared<
                linearAlgebra::MultiVector<ValueType, memorySpace>>(
                d_mpiPatternP2P,
                eigenVectors.getLinAlgOpContext(),
                numEigVecInBatchTotal,
                ValueType());
              d_chfsiScratch2Small = std::make_shared<
                linearAlgebra::MultiVector<ValueType, memorySpace>>(
                d_mpiPatternP2P,
                eigenVectors.getLinAlgOpContext(),
                numEigVecInBatchTotal,
                ValueType());
              if (d_isResidualChebyFilter)
                {
                  d_chfsiResidualScratch1Small = std::make_shared<
                    linearAlgebra::MultiVector<ValueType, memorySpace>>(
                    d_mpiPatternP2P,
                    eigenVectors.getLinAlgOpContext(),
                    numEigVecInBatchTotal,
                    ValueType());
                  d_chfsiResidualScratch2Small = std::make_shared<
                    linearAlgebra::MultiVector<ValueType, memorySpace>>(
                    d_mpiPatternP2P,
                    eigenVectors.getLinAlgOpContext(),
                    numEigVecInBatchTotal,
                    ValueType());
                }

              MultiVectorOps::copyToBatch(*Xps,
                                          eigVecStartId,
                                          numEigVecInBatch,
                                          *d_XinBatchSmall,
                                          *eigenVectors.getLinAlgOpContext());

              subspaceBatchIn  = d_XinBatchSmall;
              subspaceBatchOut = d_XoutBatchSmall;

              chfsiScratch1         = d_chfsiScratch1Small;
              chfsiScratch2         = d_chfsiScratch2Small;
              chfsiResidualScratch1 = d_chfsiResidualScratch1Small;
              chfsiResidualScratch2 = d_chfsiResidualScratch2Small;
            }
          if (d_isResidualChebyFilter)
            ResidualChebyshevFilterGEP<ValueTypeOperator,
                                       ValueTypeOperand,
                                       memorySpace>(
              A,
              B,
              BInv,
              eigenValBatch,
              *subspaceBatchIn, /*scratch1*/
              d_polynomialDegree,
              d_wantedSpectrumLowerBound,
              d_wantedSpectrumUpperBound,
              d_unWantedSpectrumUpperBound,
              *subspaceBatchOut,
              *chfsiScratch1,
              *chfsiScratch2,
              *chfsiResidualScratch1,
              *chfsiResidualScratch2); /*scratch2*/
          else
            ChebyshevFilter<ValueTypeOperator, ValueTypeOperand, memorySpace>(
              A,
              BInv,
              *subspaceBatchIn, /*scratch1*/
              d_polynomialDegree,
              d_wantedSpectrumLowerBound,
              d_wantedSpectrumUpperBound,
              d_unWantedSpectrumUpperBound,
              *subspaceBatchOut,
              *chfsiScratch1,
              *chfsiScratch2); /*scratch2*/

          utils::printCurrentMemoryUsage<memorySpace>(
            d_mpiPatternP2P->mpiCommunicator(),
            "During blocked chebyshev filtering");

          MultiVectorOps::copyFromBatch(*subspaceBatchOut,
                                        eigVecStartId,
                                        numEigVecInBatch,
                                        *Xps,
                                        *eigenVectors.getLinAlgOpContext());

          // for (size_type iSize = 0; iSize < eigenVecLocalSize; iSize++)
          //   memoryTransfer.copy(numEigVecInBatch,
          //                       eigenVectors.data() + iSize * numEigenVectors
          //                       +
          //                         eigVecStartId,
          //                       subspaceBatchOut->data() +
          //                         numEigVecInBatch * iSize);
        }

      if (d_scratch)
        d_scratch->release();

      if (d_storeIntermediateSubspaces && d_printL2Norms)
        {
          if (d_filteredSubspaceOrtho->getNumberComponents() != numEigenVectors)
            {
              d_filteredSubspaceOrtho =
                std::make_shared<MultiVector<ValueType, memorySpace>>(
                  d_mpiPatternP2P,
                  eigenVectors.getLinAlgOpContext(),
                  numEigenVectors,
                  (ValueType)0);
              d_filteredSubspace =
                std::make_shared<MultiVector<ValueType, memorySpace>>(
                  d_mpiPatternP2P,
                  eigenVectors.getLinAlgOpContext(),
                  numEigenVectors,
                  (ValueType)0);
            }
          *d_filteredSubspace = eigenVectors;
          rootCout << "d_filteredSubspace l2norms CHFSI: ";
          for (auto &i : d_filteredSubspace->l2Norms())
            rootCout << i << "\t";
          rootCout << "\n";
        }

      d_p.registerEnd("Chebyshev Filter");
      d_pTotal.registerEnd("Chebyshev Filter");

      if (!d_isGHEP)
        {
          d_p.registerStart("OrthoNormalization");
          d_pTotal.registerStart("OrthoNormalization");

          // B orthogonalization required of X -> X_O :
          // /*scratch2->eigenvector*/

          if (d_orthoType == OrthogonalizationType::CHOLESKY_GRAMSCHMIDT)
            {
              orthoerr = d_ortho->CholeskyGramSchmidt(eigenVectors, B);
            }
          else if (d_orthoType == OrthogonalizationType::MULTIPASS_CGS)
            {
              orthoerr = d_ortho->MultipassCGS(
                eigenVectors, /*in/out, eigenvector*/
                linearAlgebra::MultiPassOrthoDefaults::MAX_PASS,
                linearAlgebra::MultiPassOrthoDefaults::SHIFT_TOL,
                linearAlgebra::MultiPassOrthoDefaults::IDENTITY_TOL,
                B);
            }
          else
            {
              utils::throwException(false,
                                    "Orthogonalization type not present");
            }

          if (d_storeIntermediateSubspaces && d_printL2Norms)
            {
              *d_filteredSubspaceOrtho = eigenVectors;
              rootCout << "d_filteredSubspaceOrtho l2norms CHFSI: ";
              for (auto &i : d_filteredSubspaceOrtho->l2Norms())
                rootCout << i << "\t";
              rootCout << "\n";
            }
          d_p.registerEnd("OrthoNormalization");
          d_pTotal.registerEnd("OrthoNormalization");

          // [RR] Perform the Rayleigh–Ritz procedure for filteredSubspaceOrtho

          d_p.registerStart("RR Step");
          d_pTotal.registerStart("RR Step");
          rrerr = d_rr->solve(A,
                              eigenValues,
                              eigenVectors, /*in/out*/
                              computeEigenVectors);
          d_p.registerEnd("RR Step");
          d_pTotal.registerEnd("RR Step");
        }
      else
        {
          d_p.registerStart("RR Step");
          d_pTotal.registerStart("RR Step");
          OrthonormalizationErrorCode err1 =
            OrthonormalizationErrorCode::SUCCESS;
          orthoerr = OrthonormalizationErrorMsg::isSuccessAndMsg(err1);

          /* Using GHEP with B = Identity in orthogonalization
           * does not work. Prob due to no distribute C2P
           * in IdenstiyOperator. */
          rrerr = d_rr->solve(A,
                              B,
                              eigenValues,
                              eigenVectors, /*in/out*/
                              computeEigenVectors);
          d_p.registerEnd("RR Step");
          d_pTotal.registerEnd("RR Step");
        }

      d_p.print();

      if (d_printL2Norms)
        {
          rootCout << "eigenVectors l2norms CHFSI: ";
          for (auto &i : eigenVectors.l2Norms())
            rootCout << i << "\t";
          rootCout << "\n";
        }

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
      if (d_storeIntermediateSubspaces && d_printL2Norms)
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
      if (d_storeIntermediateSubspaces && d_printL2Norms)
        return *d_filteredSubspaceOrtho;
      else
        {
          utils::throwException(
            false,
            "storeIntermediateSubspaces is false in CHFSI class. Cannot return the filtered Subspace Ortho.");
          return *d_filteredSubspaceOrtho;
        }
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    void
    ChebyshevFilteredEigenSolver<ValueTypeOperator,
                                 ValueTypeOperand,
                                 memorySpace>::printTotalInScopeTimings()
    {
      d_pTotal.print();
    }
  } // end of namespace linearAlgebra
} // end of namespace dftefe
