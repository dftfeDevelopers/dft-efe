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
#include <linearAlgebra/MultiVectorProductSpace.h>
#include <linearAlgebra/MultiVectorProductSpaceBlocked.h>
#include <linearAlgebra/MultiVectorOps.h>
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
        const size_type             eigenVectorBatchSize,
        const ElpaScalapackManager &elpaScala,
        std::shared_ptr<const utils::mpi::MPIPatternP2P<memorySpace>>
                                                      mpiPatternP2P,
        std::shared_ptr<LinAlgOpContext<memorySpace>> linAlgOpContext,
        const bool                                    useScalpack,
        std::shared_ptr<MultivectorScratch<ValueType, memorySpace>> scratch)
      : d_eigenVecBatchSize(eigenVectorBatchSize)
      , d_batchSizeSmall(0)
      , d_XinBatchSmall(nullptr)
      , d_XoutBatchSmall(nullptr)
      , d_elpaScala(&elpaScala)
      , d_useELPA(d_elpaScala->useElpa())
      , d_useScalapack(useScalpack)
      , d_scratch(scratch)
    {
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
          d_XinBatch = std::make_shared<MultiVector<ValueType, memorySpace>>(
            mpiPatternP2P, linAlgOpContext, eigenVectorBatchSize, ValueType());

          d_XoutBatch = std::make_shared<MultiVector<ValueType, memorySpace>>(
            mpiPatternP2P, linAlgOpContext, eigenVectorBatchSize, ValueType());
        }
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    EigenSolverError
    RayleighRitzEigenSolver<ValueTypeOperator, ValueTypeOperand, memorySpace>::
      solve(const OpContext &                    A,
            std::vector<RealType> &              eigenValues,
            MultiVector<ValueType, memorySpace> &X,
            bool                                 computeEigenVectors)
    {
      EigenSolverError     retunValue;
      EigenSolverErrorCode err;

      const size_type numVec  = X.getNumberComponents();
      const size_type vecSize = X.locallyOwnedSize();

      if (d_useScalapack)
        {
          bool                         solveSuccess = true;
          utils::Profiler<memorySpace> p(
            X.getMPIPatternP2P()->mpiCommunicator(),
            "Rayleigh-Ritz EigenSolver");

          // Compute projected hamiltonian = X_O^H A X_O

          const size_type rowsBlockSize = d_elpaScala->getScalapackBlockSize();
          std::shared_ptr<const ProcessGrid> processGrid =
            d_elpaScala->getProcessGridDftefeScalaWrapper();

          if (auto *Xb = dynamic_cast<
                MultiVectorProductSpaceBlocked<ValueType, memorySpace> *>(&X))
            {
              // === BLOCKED (collinear): S independent N×N sub-problems ===
              const size_type N = Xb->numVectorsPerSpace();
              const size_type S = Xb->numSpaces();

              p.registerStart("Compute X^T H X");

              std::vector<ScaLAPACKMatrix<ValueType>> projHamParVec(
                S, ScaLAPACKMatrix<ValueType>(N, processGrid, rowsBlockSize));
              for (auto &projHamPar : projHamParVec)
                if (processGrid->is_process_active())
                  std::fill(
                    &projHamPar.local_el(0, 0),
                    &projHamPar.local_el(0, 0) +
                      projHamPar.local_m() * projHamPar.local_n(),
                    ValueType(0.0));

              MultiVectorOps::project(A,
                                      *Xb,
                                      projHamParVec,
                                      *d_elpaScala,
                                      d_XinBatch,
                                      d_XoutBatch,
                                      d_XinBatchSmall,
                                      d_XoutBatchSmall);

              p.registerEnd("Compute X^T H X");

              eigenValues.resize(S * N);

              std::vector<ScaLAPACKMatrix<ValueType>> rotMatVec(
                S, ScaLAPACKMatrix<ValueType>(N, processGrid, rowsBlockSize));

              if (d_useELPA)
                {
                  p.registerStart("ELPA eigen decomp, RR step");
                  for (size_type s = 0; s < S; ++s)
                    {
                      // Symmetrize projHamParVec[s]: full Hermitian = lower + upper
                      ScaLAPACKMatrix<ValueType> projHamParConjTrans(
                        N, processGrid, rowsBlockSize);
                      if (processGrid->is_process_active())
                        std::fill(
                          &projHamParConjTrans.local_el(0, 0),
                          &projHamParConjTrans.local_el(0, 0) +
                            projHamParConjTrans.local_m() *
                              projHamParConjTrans.local_n(),
                          ValueType(0.0));
                      projHamParConjTrans.copy_conjugate_transposed(
                        projHamParVec[s]);
                      projHamParVec[s].add(
                        projHamParConjTrans, ValueType(1.0), ValueType(1.0));
                      // halve diagonal
                      if (processGrid->is_process_active())
                        for (size_type i = 0;
                             i < projHamParVec[s].local_n();
                             ++i)
                          {
                            const size_type glob_i =
                              projHamParVec[s].global_column(i);
                            for (size_type j = 0;
                                 j < projHamParVec[s].local_m();
                                 ++j)
                              {
                                const size_type glob_j =
                                  projHamParVec[s].global_row(j);
                                if (glob_i == glob_j)
                                  projHamParVec[s].local_el(j, i) *=
                                    ValueType(0.5);
                              }
                          }

                      ScaLAPACKMatrix<ValueType> eigenVectorsPar(N,
                                                                 processGrid,
                                                                 rowsBlockSize);
                      if (processGrid->is_process_active())
                        std::fill(
                          &eigenVectorsPar.local_el(0, 0),
                          &eigenVectorsPar.local_el(0, 0) +
                            eigenVectorsPar.local_m() *
                              eigenVectorsPar.local_n(),
                          ValueType(0.0));

                      std::vector<RealType> eigenValsSpace(N, RealType(0));
                      if (processGrid->is_process_active())
                        {
                          int error;
                          elpa_eigenvectors(d_elpaScala->getElpaHandle(),
                                            &projHamParVec[s].local_el(0, 0),
                                            eigenValsSpace.data(),
                                            &eigenVectorsPar.local_el(0, 0),
                                            &error);
                          if (error != ELPA_OK)
                            solveSuccess = false;
                          for (size_type n = 0; n < N; ++n)
                            eigenValues[s * N + n] = eigenValsSpace[n];
                        }
                      utils::mpi::MPIBcast<utils::MemorySpace::HOST>(
                        &eigenValues[s * N],
                        N,
                        utils::mpi::Types<RealType>::getMPIDatatype(),
                        0,
                        X.getMPIPatternP2P()->mpiCommunicator());
                      // rotMatVec[s] = V_s^H for rotation
                      rotMatVec[s].copy_conjugate_transposed(eigenVectorsPar);
                    }
                  p.registerEnd("ELPA eigen decomp, RR step");
                }
              else
                {
                  p.registerStart("ScaLAPACK eigen decomp, RR step");
                  for (size_type s = 0; s < S; ++s)
                    {
                      ScalapackError scalapackError;
                      std::vector<RealType> eigenValsSpace =
                        projHamParVec[s].eigenpairs_hermitian_by_index_MRRR(
                          std::make_pair(0, (int)N - 1), true, scalapackError);
                      if (scalapackError.err != ScalapackErrorCode::SUCCESS)
                        solveSuccess = false;
                      for (size_type n = 0; n < N; ++n)
                        eigenValues[s * N + n] = eigenValsSpace[n];
                      // rotMatVec[s] = V_s^H
                      rotMatVec[s].copy_conjugate_transposed(projHamParVec[s]);
                    }
                  p.registerEnd("ScaLAPACK eigen decomp, RR step");
                }

              if (computeEigenVectors)
                {
                  p.registerStart("Subspace Rotation");
                  MultiVectorOps::rotate(*Xb, rotMatVec, *d_elpaScala);
                  p.registerEnd("Subspace Rotation");
                }
            }
          else
            {
              // === COUPLED (unpolarized S=1 / non-collinear S=2,
              // MultiVectorProductSpace) ===
              const size_type numVecTotal = X.getNumberComponents();

              p.registerStart("Compute X^T H X");

              ScaLAPACKMatrix<ValueType> projHamPar(numVecTotal,
                                                    processGrid,
                                                    rowsBlockSize);
              if (processGrid->is_process_active())
                std::fill(&projHamPar.local_el(0, 0),
                          &projHamPar.local_el(0, 0) +
                            projHamPar.local_m() * projHamPar.local_n(),
                          ValueType(0.0));

              // Use MultiVectorProductSpace overload if available, else fall
              // back to existing path
              if (auto *Xp = dynamic_cast<
                    MultiVectorProductSpace<ValueType, memorySpace> *>(&X))
                {
                  MultiVectorOps::project(A,
                                          *Xp,
                                          projHamPar,
                                          *d_elpaScala,
                                          d_XinBatch,
                                          d_XoutBatch,
                                          d_XinBatchSmall,
                                          d_XoutBatchSmall);
                }
              else
                {
                  computeXTransOpX(X, processGrid, projHamPar, A);
                }

              p.registerEnd("Compute X^T H X");

              eigenValues.resize(numVecTotal);
              ScaLAPACKMatrix<ValueType> rotMat(numVecTotal,
                                               processGrid,
                                               rowsBlockSize);

              if (d_useELPA)
                {
                  p.registerStart("ELPA eigen decomp, RR step");
                  ScaLAPACKMatrix<ValueType> eigenVectorsPar(numVecTotal,
                                                             processGrid,
                                                             rowsBlockSize);
                  if (processGrid->is_process_active())
                    std::fill(&eigenVectorsPar.local_el(0, 0),
                              &eigenVectorsPar.local_el(0, 0) +
                                eigenVectorsPar.local_m() *
                                  eigenVectorsPar.local_n(),
                              ValueType(0.0));

                  ScaLAPACKMatrix<ValueType> projHamParConjTrans(numVecTotal,
                                                                 processGrid,
                                                                 rowsBlockSize);
                  if (processGrid->is_process_active())
                    std::fill(&projHamParConjTrans.local_el(0, 0),
                              &projHamParConjTrans.local_el(0, 0) +
                                projHamParConjTrans.local_m() *
                                  projHamParConjTrans.local_n(),
                              ValueType(0.0));

                  projHamParConjTrans.copy_conjugate_transposed(projHamPar);
                  projHamPar.add(projHamParConjTrans,
                                 ValueType(1.0),
                                 ValueType(1.0));
                  if (processGrid->is_process_active())
                    for (size_type i = 0; i < projHamPar.local_n(); ++i)
                      {
                        const size_type glob_i = projHamPar.global_column(i);
                        for (size_type j = 0; j < projHamPar.local_m(); ++j)
                          {
                            const size_type glob_j = projHamPar.global_row(j);
                            if (glob_i == glob_j)
                              projHamPar.local_el(j, i) *= ValueType(0.5);
                          }
                      }

                  if (processGrid->is_process_active())
                    {
                      int error;
                      elpa_eigenvectors(d_elpaScala->getElpaHandle(),
                                        &projHamPar.local_el(0, 0),
                                        &eigenValues[0],
                                        &eigenVectorsPar.local_el(0, 0),
                                        &error);
                      if (error != ELPA_OK)
                        solveSuccess = false;
                    }

                  utils::mpi::MPIBcast<utils::MemorySpace::HOST>(
                    &eigenValues[0],
                    eigenValues.size(),
                    utils::mpi::Types<RealType>::getMPIDatatype(),
                    0,
                    X.getMPIPatternP2P()->mpiCommunicator());

                  eigenVectorsPar.copy_to(projHamPar);
                  rotMat.copy_conjugate_transposed(projHamPar);

                  p.registerEnd("ELPA eigen decomp, RR step");
                }
              else
                {
                  ScalapackError scalapackError;
                  p.registerStart("ScaLAPACK eigen decomp, RR step");
                  eigenValues = projHamPar.eigenpairs_hermitian_by_index_MRRR(
                    std::make_pair(0, (int)numVecTotal - 1),
                    true,
                    scalapackError);
                  if (scalapackError.err != ScalapackErrorCode::SUCCESS)
                    solveSuccess = false;
                  rotMat.copy_conjugate_transposed(projHamPar);
                  p.registerEnd("ScaLAPACK eigen decomp, RR step");
                }

              if (computeEigenVectors)
                {
                  p.registerStart("Subspace Rotation");
                  if (auto *Xp = dynamic_cast<
                        MultiVectorProductSpace<ValueType, memorySpace> *>(&X))
                    {
                      MultiVectorOps::rotate(*Xp, rotMat, *d_elpaScala);
                    }
                  else
                    {
                      elpaScalaOpInternal::subspaceRotation<ValueType,
                                                            memorySpace>(
                        X.data(),
                        vecSize,
                        numVecTotal,
                        processGrid,
                        X.getMPIPatternP2P()->mpiCommunicator(),
                        *X.getLinAlgOpContext(),
                        rotMat,
                        RayleighRitzDefaults::SUBSPACE_ROT_DOF_BATCH,
                        RayleighRitzDefaults::WAVE_FN_BATCH,
                        false,
                        false);
                    }
                  p.registerEnd("Subspace Rotation");
                }
            }

          if (!solveSuccess)
            {
              err        = EigenSolverErrorCode::ELPASCALAPACK_ERROR;
              retunValue = EigenSolverErrorMsg::isSuccessAndMsg(err);
            }
          else
            {
              err        = EigenSolverErrorCode::SUCCESS;
              retunValue = EigenSolverErrorMsg::isSuccessAndMsg(err);
            }
          p.print();
          return retunValue;
        }
      else
        {
          utils::throwException(
            dynamic_cast<
                MultiVectorProductSpace<ValueType, memorySpace> *>(&X) ==
              nullptr &&
              dynamic_cast<MultiVectorProductSpaceBlocked<ValueType,
                                                          memorySpace> *>(
                &X) == nullptr,
            "Non-ScaLAPACK Rayleigh-Ritz path does not support "
            "spin-polarized (S>1) wavefunctions.");
          MultiVector<ValueType, memorySpace> eigenVectors(X, (ValueType)0);
          // // ------- For DEBUG ---------------
          EigenSolverError                             retunValue;
          LapackError                                  lapackReturn;
          utils::MemoryStorage<ValueType, memorySpace> XprojectedA(
            numVec * numVec, utils::Types<ValueType>::zero);
          utils::MemoryStorage<ValueType, utils::MemorySpace::HOST>
            XprojectedAHost(numVec * numVec, utils::Types<ValueType>::zero);
          utils::MemoryStorage<ValueType, utils::MemorySpace::HOST>
            eigenVectorsXSubspace(numVec * numVec,
                                  utils::Types<ValueType>::zero);
          eigenValues.resize(numVec);

          computeXTransOpX(X, XprojectedA, A);

          // Solve the standard eigenvalue problem

          utils::MemoryTransfer<utils::MemorySpace::HOST, memorySpace>::copy(
            XprojectedA.size(),
            eigenVectorsXSubspace.data(),
            XprojectedA.data());

          lapackReturn = blasLapack::heevd<ValueType, utils::MemorySpace::HOST>(
            computeEigenVectors ? 'V' : 'N',
            'L',
            numVec,
            eigenVectorsXSubspace.data(),
            numVec,
            eigenValues.data(),
            *LinAlgOpContextDefaults::LINALG_OP_CONTXT_HOST);

          if (computeEigenVectors)
            {
              int mpierr = utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
                utils::mpi::MPIInPlace,
                eigenVectorsXSubspace.data(),
                eigenVectorsXSubspace.size(),
                utils::mpi::Types<ValueType>::getMPIDatatype(),
                utils::mpi::MPISum,
                X.getMPIPatternP2P()->mpiCommunicator());

              utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::
                copy(XprojectedA.size(),
                     XprojectedA.data(),
                     eigenVectorsXSubspace.data());

              blasLapack::gemm<ValueType, ValueType, memorySpace>(
                'T',
                'N',
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

              X = eigenVectors;
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
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    EigenSolverError
    RayleighRitzEigenSolver<ValueTypeOperator, ValueTypeOperand, memorySpace>::
      solve(const OpContext &                    A,
            const OpContext &                    B,
            std::vector<RealType> &              eigenValues,
            MultiVector<ValueType, memorySpace> &X,
            bool                                 computeEigenVectors)
    {
      // X memory layout (MultiVectorProductSpace[Blocked], S spaces, N orbitals each):
      //   data[ dof * (S*N) + s*N + n ]  — DOF slowest, space next, orbital fastest.
      //   Blocked path (collinear): S independent sub-problems; space-s block starts at
      //     ptr = X.data() + s*N,  column stride (lda) = S*N.
      //   Coupled path (unpolarized S=1 / non-collinear S=2): all S*N columns as one unit.
      //
      // eigenValues memory layout: space-major, size S*N.
      //   eigenValues[ s*N + n ]  — space s (0..S-1), orbital n (0..N-1).
      //   Blocked path: filled space-by-space in the ELPA/ScaLAPACK loop (eigenValues[s*N+n]).
      //   Coupled path: flat eigenValues[0..numVecTotal-1], columns of X in order.
      EigenSolverError     retunValue;
      EigenSolverErrorCode err;

      const size_type numVec  = X.getNumberComponents();
      const size_type vecSize = X.locallyOwnedSize();

      if (d_useScalapack)
        {
          bool                         solveSuccess = true;
          utils::Profiler<memorySpace> p(
            X.getMPIPatternP2P()->mpiCommunicator(),
            "Rayleigh-Ritz EigenSolver");

          // Compute projected hamiltonian = X_O^H A X_O

          const size_type rowsBlockSize = d_elpaScala->getScalapackBlockSize();
          std::shared_ptr<const ProcessGrid> processGrid =
            d_elpaScala->getProcessGridDftefeScalaWrapper();

          if (auto *Xb = dynamic_cast<
                MultiVectorProductSpaceBlocked<ValueType, memorySpace> *>(&X))
            {
              // === BLOCKED (collinear): S independent N×N sub-problems ===
              const size_type N = Xb->numVectorsPerSpace();
              const size_type S = Xb->numSpaces();

              p.registerStart("XtHX and XtOX, RR GEP step");

              std::vector<ScaLAPACKMatrix<ValueType>> projHamParVec(
                S, ScaLAPACKMatrix<ValueType>(N, processGrid, rowsBlockSize));
              std::vector<ScaLAPACKMatrix<ValueType>> overlapMatParVec(
                S, ScaLAPACKMatrix<ValueType>(N, processGrid, rowsBlockSize));
              for (size_type s = 0; s < S; ++s)
                {
                  if (processGrid->is_process_active())
                    {
                      std::fill(
                        &projHamParVec[s].local_el(0, 0),
                        &projHamParVec[s].local_el(0, 0) +
                          projHamParVec[s].local_m() * projHamParVec[s].local_n(),
                        ValueType(0.0));
                      std::fill(
                        &overlapMatParVec[s].local_el(0, 0),
                        &overlapMatParVec[s].local_el(0, 0) +
                          overlapMatParVec[s].local_m() *
                            overlapMatParVec[s].local_n(),
                        ValueType(0.0));
                    }
                }

              MultiVectorOps::project(A,
                                      *Xb,
                                      projHamParVec,
                                      *d_elpaScala,
                                      d_XinBatch,
                                      d_XoutBatch,
                                      d_XinBatchSmall,
                                      d_XoutBatchSmall);
              MultiVectorOps::project(B,
                                      *Xb,
                                      overlapMatParVec,
                                      *d_elpaScala,
                                      d_XinBatch,
                                      d_XoutBatch,
                                      d_XinBatchSmall,
                                      d_XoutBatchSmall);

              // Symmetrize projHamParVec[s] (full Hermitian = lower + upper, halve diag)
              for (size_type s = 0; s < S; ++s)
                {
                  ScaLAPACKMatrix<ValueType> projHamParConjTrans(
                    N, processGrid, rowsBlockSize);
                  if (processGrid->is_process_active())
                    std::fill(
                      &projHamParConjTrans.local_el(0, 0),
                      &projHamParConjTrans.local_el(0, 0) +
                        projHamParConjTrans.local_m() *
                          projHamParConjTrans.local_n(),
                      ValueType(0.0));
                  projHamParConjTrans.copy_conjugate_transposed(
                    projHamParVec[s]);
                  projHamParVec[s].add(
                    projHamParConjTrans, ValueType(1.0), ValueType(1.0));
                  if (processGrid->is_process_active())
                    for (size_type i = 0;
                         i < projHamParVec[s].local_n();
                         ++i)
                      {
                        const size_type glob_i =
                          projHamParVec[s].global_column(i);
                        for (size_type j = 0;
                             j < projHamParVec[s].local_m();
                             ++j)
                          {
                            const size_type glob_j =
                              projHamParVec[s].global_row(j);
                            if (glob_i == glob_j)
                              projHamParVec[s].local_el(j, i) *=
                                ValueType(0.5);
                          }
                      }
                }

              p.registerEnd("XtHX and XtOX, RR GEP step");

              eigenValues.resize(S * N);

              std::vector<ScaLAPACKMatrix<ValueType>> rotMatVec(
                S, ScaLAPACKMatrix<ValueType>(N, processGrid, rowsBlockSize));

              if (d_useELPA)
                {
                  p.registerStart("ELPA eigen decomp, RR step");
                  for (size_type s = 0; s < S; ++s)
                    {
                      ScaLAPACKMatrix<ValueType> eigenVectorsPar(N,
                                                                 processGrid,
                                                                 rowsBlockSize);
                      if (processGrid->is_process_active())
                        std::fill(
                          &eigenVectorsPar.local_el(0, 0),
                          &eigenVectorsPar.local_el(0, 0) +
                            eigenVectorsPar.local_m() *
                              eigenVectorsPar.local_n(),
                          ValueType(0.0));

                      // ELPA generalized: overlap must be upper-triangular form
                      ScaLAPACKMatrix<ValueType> overlapMatParConjTrans(
                        N, processGrid, rowsBlockSize);
                      if (processGrid->is_process_active())
                        std::fill(
                          &overlapMatParConjTrans.local_el(0, 0),
                          &overlapMatParConjTrans.local_el(0, 0) +
                            overlapMatParConjTrans.local_m() *
                              overlapMatParConjTrans.local_n(),
                          ValueType(0.0));
                      overlapMatParConjTrans.copy_conjugate_transposed(
                        overlapMatParVec[s]);

                      std::vector<RealType> eigenValsSpace(N, RealType(0));
                      if (processGrid->is_process_active())
                        {
                          int error;
                          elpa_generalized_eigenvectors(
                            d_elpaScala->getElpaHandle(),
                            &projHamParVec[s].local_el(0, 0),
                            &overlapMatParConjTrans.local_el(0, 0),
                            eigenValsSpace.data(),
                            &eigenVectorsPar.local_el(0, 0),
                            0,
                            &error);
                          if (error != ELPA_OK)
                            solveSuccess = false;
                          for (size_type n = 0; n < N; ++n)
                            eigenValues[s * N + n] = eigenValsSpace[n];
                        }
                      utils::mpi::MPIBcast<utils::MemorySpace::HOST>(
                        &eigenValues[s * N],
                        N,
                        utils::mpi::Types<RealType>::getMPIDatatype(),
                        0,
                        X.getMPIPatternP2P()->mpiCommunicator());
                      rotMatVec[s].copy_conjugate_transposed(eigenVectorsPar);
                    }
                  p.registerEnd("ELPA eigen decomp, RR step");
                }
              else
                {
                  p.registerStart("Cholesky and triangular matrix invert");
                  for (size_type s = 0; s < S; ++s)
                    {
                      LAPACKSupport::Property overlapMatPropertyPostCholesky;
                      overlapMatParVec[s].compute_cholesky_factorization();
                      overlapMatPropertyPostCholesky =
                        overlapMatParVec[s].get_property();
                      DFTEFE_AssertWithMsg(
                        overlapMatPropertyPostCholesky ==
                          LAPACKSupport::Property::lower_triangular,
                        "DFT-EFE Error: overlap matrix property after cholesky factorization incorrect");

                      ScaLAPACKMatrix<ValueType> LMatPar(
                        N,
                        processGrid,
                        rowsBlockSize,
                        LAPACKSupport::Property::lower_triangular);
                      if (processGrid->is_process_active())
                        for (size_type i = 0; i < LMatPar.local_n(); ++i)
                          {
                            const size_type glob_i = LMatPar.global_column(i);
                            for (size_type j = 0; j < LMatPar.local_m(); ++j)
                              {
                                const size_type glob_j = LMatPar.global_row(j);
                                if (glob_j < glob_i)
                                  LMatPar.local_el(j, i) = ValueType(0);
                                else
                                  LMatPar.local_el(j, i) =
                                    overlapMatParVec[s].local_el(j, i);
                              }
                          }
                      LMatPar.invert();

                      ScaLAPACKMatrix<ValueType> projHamParCopy(N,
                                                                processGrid,
                                                                rowsBlockSize);
                      LMatPar.mmult(projHamParCopy, projHamParVec[s]);
                      projHamParCopy.zmCmult(projHamParVec[s], LMatPar);

                      ScalapackError scalapackError;
                      std::vector<RealType> eigenValsSpace =
                        projHamParVec[s].eigenpairs_hermitian_by_index_MRRR(
                          std::make_pair(0, (int)N - 1), true, scalapackError);
                      if (scalapackError.err != ScalapackErrorCode::SUCCESS)
                        solveSuccess = false;
                      for (size_type n = 0; n < N; ++n)
                        eigenValues[s * N + n] = eigenValsSpace[n];

                      projHamParCopy.copy_conjugate_transposed(
                        projHamParVec[s]);
                      projHamParCopy.mmult(rotMatVec[s], LMatPar);
                    }
                  p.registerEnd("Cholesky and triangular matrix invert");
                }

              if (computeEigenVectors)
                {
                  p.registerStart(
                    "X^{T}={QConjPrime}^{C}*LConj^{-1}*X^{T}, RR step");
                  MultiVectorOps::rotate(*Xb, rotMatVec, *d_elpaScala);
                  p.registerEnd(
                    "X^{T}={QConjPrime}^{C}*LConj^{-1}*X^{T}, RR step");
                }
            }
          else
            {
              // === COUPLED (unpolarized S=1 / non-collinear S=2,
              // MultiVectorProductSpace or plain MultiVector) ===
              const size_type numVecTotal = X.getNumberComponents();

              p.registerStart("XtHX and XtOX, RR GEP step");

              ScaLAPACKMatrix<ValueType> overlapMatPar(numVecTotal,
                                                       processGrid,
                                                       rowsBlockSize);
              if (processGrid->is_process_active())
                std::fill(&overlapMatPar.local_el(0, 0),
                          &overlapMatPar.local_el(0, 0) +
                            overlapMatPar.local_m() * overlapMatPar.local_n(),
                          ValueType(0.0));

              ScaLAPACKMatrix<ValueType> projHamPar(numVecTotal,
                                                    processGrid,
                                                    rowsBlockSize);
              if (processGrid->is_process_active())
                std::fill(&projHamPar.local_el(0, 0),
                          &projHamPar.local_el(0, 0) +
                            projHamPar.local_m() * projHamPar.local_n(),
                          ValueType(0.0));

              if (auto *Xp = dynamic_cast<
                    MultiVectorProductSpace<ValueType, memorySpace> *>(&X))
                {
                  MultiVectorOps::project(A,
                                          *Xp,
                                          projHamPar,
                                          *d_elpaScala,
                                          d_XinBatch,
                                          d_XoutBatch,
                                          d_XinBatchSmall,
                                          d_XoutBatchSmall);
                  MultiVectorOps::project(B,
                                          *Xp,
                                          overlapMatPar,
                                          *d_elpaScala,
                                          d_XinBatch,
                                          d_XoutBatch,
                                          d_XinBatchSmall,
                                          d_XoutBatchSmall);
                }
              else
                {
                  computeXTransOpX(X, processGrid, projHamPar, A);
                  computeXTransOpX(X, processGrid, overlapMatPar, B);
                }

              // Construct the full HConjProj matrix
              ScaLAPACKMatrix<ValueType> projHamParConjTrans(numVecTotal,
                                                             processGrid,
                                                             rowsBlockSize);
              if (processGrid->is_process_active())
                std::fill(&projHamParConjTrans.local_el(0, 0),
                          &projHamParConjTrans.local_el(0, 0) +
                            projHamParConjTrans.local_m() *
                              projHamParConjTrans.local_n(),
                          ValueType(0.0));

              projHamParConjTrans.copy_conjugate_transposed(projHamPar);
              projHamPar.add(projHamParConjTrans, ValueType(1.0), ValueType(1.0));
              if (processGrid->is_process_active())
                for (size_type i = 0; i < projHamPar.local_n(); ++i)
                  {
                    const size_type glob_i = projHamPar.global_column(i);
                    for (size_type j = 0; j < projHamPar.local_m(); ++j)
                      {
                        const size_type glob_j = projHamPar.global_row(j);
                        if (glob_i == glob_j)
                          projHamPar.local_el(j, i) *= ValueType(0.5);
                      }
                  }

              p.registerEnd("XtHX and XtOX, RR GEP step");

              eigenValues.resize(numVecTotal);

              if (d_useELPA)
                {
                  p.registerStart("ELPA eigen decomp, RR step");
                  ScaLAPACKMatrix<ValueType> eigenVectorsPar(numVecTotal,
                                                             processGrid,
                                                             rowsBlockSize);
                  if (processGrid->is_process_active())
                    std::fill(&eigenVectorsPar.local_el(0, 0),
                              &eigenVectorsPar.local_el(0, 0) +
                                eigenVectorsPar.local_m() *
                                  eigenVectorsPar.local_n(),
                              ValueType(0.0));

                  ScaLAPACKMatrix<ValueType> overlapMatParConjTrans(numVecTotal,
                                                                    processGrid,
                                                                    rowsBlockSize);
                  if (processGrid->is_process_active())
                    std::fill(&overlapMatParConjTrans.local_el(0, 0),
                              &overlapMatParConjTrans.local_el(0, 0) +
                                overlapMatParConjTrans.local_m() *
                                  overlapMatParConjTrans.local_n(),
                              ValueType(0.0));

                  overlapMatParConjTrans.copy_conjugate_transposed(overlapMatPar);

                  if (processGrid->is_process_active())
                    {
                      int error;
                      elpa_generalized_eigenvectors(
                        d_elpaScala->getElpaHandle(),
                        &projHamPar.local_el(0, 0),
                        &overlapMatParConjTrans.local_el(0, 0),
                        &eigenValues[0],
                        &eigenVectorsPar.local_el(0, 0),
                        0,
                        &error);
                      if (error != ELPA_OK)
                        solveSuccess = false;
                    }

                  utils::mpi::MPIBcast<utils::MemorySpace::HOST>(
                    &eigenValues[0],
                    eigenValues.size(),
                    utils::mpi::Types<RealType>::getMPIDatatype(),
                    0,
                    X.getMPIPatternP2P()->mpiCommunicator());

                  projHamPar.copy_conjugate_transposed(eigenVectorsPar);
                  p.registerEnd("ELPA eigen decomp, RR step");
                }
              else
                {
                  // SConj=LConj*L^{T}
                  p.registerStart("Cholesky and triangular matrix invert");

                  LAPACKSupport::Property overlapMatPropertyPostCholesky;
                  overlapMatPar.compute_cholesky_factorization();
                  overlapMatPropertyPostCholesky = overlapMatPar.get_property();

                  DFTEFE_AssertWithMsg(
                    overlapMatPropertyPostCholesky ==
                      LAPACKSupport::Property::lower_triangular,
                    "DFT-EFE Error: overlap matrix property after cholesky factorization incorrect");

                  ScaLAPACKMatrix<ValueType> LMatPar(
                    numVecTotal,
                    processGrid,
                    rowsBlockSize,
                    LAPACKSupport::Property::lower_triangular);
                  if (processGrid->is_process_active())
                    for (size_type i = 0; i < LMatPar.local_n(); ++i)
                      {
                        const size_type glob_i = LMatPar.global_column(i);
                        for (size_type j = 0; j < LMatPar.local_m(); ++j)
                          {
                            const size_type glob_j = LMatPar.global_row(j);
                            if (glob_j < glob_i)
                              LMatPar.local_el(j, i) = ValueType(0);
                            else
                              LMatPar.local_el(j, i) =
                                overlapMatPar.local_el(j, i);
                          }
                      }

                  // compute LConj^{-1}
                  LMatPar.invert();
                  p.registerEnd("Cholesky and triangular matrix invert");

                  p.registerStart(
                    "Compute HSConjProj= Lconj^{-1}*HConjProj*(Lconj^{-1})^C, RR step");

                  ScaLAPACKMatrix<ValueType> projHamParCopy(numVecTotal,
                                                            processGrid,
                                                            rowsBlockSize);
                  LMatPar.mmult(projHamParCopy, projHamPar);
                  projHamParCopy.zmCmult(projHamPar, LMatPar);

                  ScalapackError scalapackError;
                  p.registerEnd(
                    "Compute HSConjProj= Lconj^{-1}*HConjProj*(Lconj^{-1})^C, RR step");
                  p.registerStart("ScaLAPACK eigen decomp, RR step");
                  eigenValues = projHamPar.eigenpairs_hermitian_by_index_MRRR(
                    std::make_pair(0, (int)numVecTotal - 1),
                    true,
                    scalapackError);
                  projHamParCopy.copy_conjugate_transposed(projHamPar);
                  projHamParCopy.mmult(projHamPar, LMatPar);

                  if (scalapackError.err != ScalapackErrorCode::SUCCESS)
                    solveSuccess = false;

                  p.registerEnd("ScaLAPACK eigen decomp, RR step");
                }

              if (computeEigenVectors)
                {
                  p.registerStart(
                    "X^{T}={QConjPrime}^{C}*LConj^{-1}*X^{T}, RR step");
                  if (auto *Xp = dynamic_cast<
                        MultiVectorProductSpace<ValueType, memorySpace> *>(&X))
                    {
                      MultiVectorOps::rotate(*Xp, projHamPar, *d_elpaScala);
                    }
                  else
                    {
                      elpaScalaOpInternal::subspaceRotation<ValueType,
                                                            memorySpace>(
                        X.data(),
                        vecSize,
                        numVecTotal,
                        processGrid,
                        X.getMPIPatternP2P()->mpiCommunicator(),
                        *X.getLinAlgOpContext(),
                        projHamPar,
                        RayleighRitzDefaults::SUBSPACE_ROT_DOF_BATCH,
                        RayleighRitzDefaults::WAVE_FN_BATCH,
                        false,
                        false);
                    }
                  p.registerEnd(
                    "X^{T}={QConjPrime}^{C}*LConj^{-1}*X^{T}, RR step");
                }
            }

          if (!solveSuccess)
            {
              err        = EigenSolverErrorCode::ELPASCALAPACK_ERROR;
              retunValue = EigenSolverErrorMsg::isSuccessAndMsg(err);
            }
          else
            {
              err        = EigenSolverErrorCode::SUCCESS;
              retunValue = EigenSolverErrorMsg::isSuccessAndMsg(err);
            }
          p.print();
          return retunValue;
        }
      else
        {
          // // ------- For DEBUG ---------------
          EigenSolverError retunValue;
          utils::throwException(
            false, "RR EigSolve not impl for without scalapack case.");
          return retunValue;
        }
    }

    // // ------------- DEBUG ------------------- // //
    // returns the Xtop(x) in full storage (memspace storage)
    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    void
    RayleighRitzEigenSolver<ValueTypeOperator, ValueTypeOperand, memorySpace>::
      computeXTransOpX(MultiVector<ValueTypeOperand, memorySpace> &  X,
                       utils::MemoryStorage<ValueType, memorySpace> &S,
                       const OpContext &                             Op,
                       const bool &                                  useBatched)
    {
      if (useBatched == true)
        {
          const utils::mpi::MPIComm comm =
            X.getMPIPatternP2P()->mpiCommunicator();
          LinAlgOpContext<memorySpace> linAlgOpContext =
            *X.getLinAlgOpContext();
          const size_type vecSize      = X.locallyOwnedSize();
          const size_type vecLocalSize = X.localSize();
          const size_type numVec       = X.getNumberComponents();
          utils::MemoryTransfer<memorySpace, memorySpace> memoryTransfer;
          std::shared_ptr<MultiVector<ValueType, memorySpace>>
            subspaceBatchIn  = nullptr,
            subspaceBatchOut = nullptr;

          utils::MemoryStorage<ValueType, memorySpace> SBlock(
            numVec * d_eigenVecBatchSize, ValueType(0));
          utils::MemoryStorage<ValueType, utils::MemorySpace::HOST> SBlockHost(
            numVec * d_eigenVecBatchSize);

          if (d_scratch)
            d_scratch->acquire();

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
                                        X.data() + iSize * numVec +
                                          eigVecStartId);

                  subspaceBatchIn  = d_XinBatch;
                  subspaceBatchOut = d_XoutBatch;
                }
              else if (numEigVecInBatch % d_eigenVecBatchSize ==
                       d_batchSizeSmall)
                {
                  for (size_type iSize = 0; iSize < vecLocalSize; iSize++)
                    memoryTransfer.copy(numEigVecInBatch,
                                        d_XinBatchSmall->data() +
                                          numEigVecInBatch * iSize,
                                        X.data() + iSize * numVec +
                                          eigVecStartId);

                  subspaceBatchIn  = d_XinBatchSmall;
                  subspaceBatchOut = d_XoutBatchSmall;
                }
              else
                {
                  d_batchSizeSmall = numEigVecInBatch;

                  const bool useSmallScratch =
                    d_scratch != nullptr && d_scratch->hasXinBatchSmall() &&
                    d_scratch->hasXoutBatchSmall() &&
                    d_scratch->getXinBatchSmallSize() == numEigVecInBatch;

                  if (useSmallScratch)
                    {
                      d_XinBatchSmall  = d_scratch->getXinBatchSmall();
                      d_XoutBatchSmall = d_scratch->getXoutBatchSmall();
                    }
                  else
                    {
                      d_XinBatchSmall =
                        std::make_shared<MultiVector<ValueType, memorySpace>>(
                          X.getMPIPatternP2P(),
                          X.getLinAlgOpContext(),
                          numEigVecInBatch,
                          ValueType());

                      d_XoutBatchSmall =
                        std::make_shared<MultiVector<ValueType, memorySpace>>(
                          X.getMPIPatternP2P(),
                          X.getLinAlgOpContext(),
                          numEigVecInBatch,
                          ValueType());
                      if (d_scratch != nullptr)
                        {
                          d_scratch->setXinBatchSmall(d_XinBatchSmall);
                          d_scratch->setXoutBatchSmall(d_XoutBatchSmall);
                        }
                    }

                  for (size_type iSize = 0; iSize < vecLocalSize; iSize++)
                    memoryTransfer.copy(numEigVecInBatch,
                                        d_XinBatchSmall->data() +
                                          numEigVecInBatch * iSize,
                                        X.data() + iSize * numVec +
                                          eigVecStartId);

                  subspaceBatchIn  = d_XinBatchSmall;
                  subspaceBatchOut = d_XoutBatchSmall;
                }

              Op.apply(*subspaceBatchIn, *subspaceBatchOut, true, false);

              // Input data is read is X^T (numVec is fastest index and then
              // vecSize) Operation : S = (X)^H * ((B*X)). S^T =
              // ((B*X)^T)*(X^T)^H

              const ValueType alpha = 1.0;
              const ValueType beta  = 0.0;

              blasLapack::gemm<ValueTypeOperand, ValueType, memorySpace>(
                'N',
                'C',
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

              utils::MemoryTransfer<utils::MemorySpace::HOST,
                                    memorySpace>::copy(SBlock.size(),
                                                       SBlockHost.data(),
                                                       SBlock.data());

              int mpierr = utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
                utils::mpi::MPIInPlace,
                SBlockHost.data(),
                (numVec - eigVecStartId) * numEigVecInBatch,
                utils::mpi::Types<ValueType>::getMPIDatatype(),
                utils::mpi::MPISum,
                comm);

              std::pair<bool, std::string> mpiIsSuccessAndMsg =
                utils::mpi::MPIErrIsSuccessAndMsg(mpierr);
              DFTEFE_AssertWithMsg(mpiIsSuccessAndMsg.first,
                                   "MPI Error:" + mpiIsSuccessAndMsg.second);

              utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::
                copy(SBlock.size(), SBlock.data(), SBlockHost.data());

              // Copying only the lower triangular part to projected matrix
              for (size_type iSize = 0; iSize < numEigVecInBatch; iSize++)
                memoryTransfer.copy(numVec - eigVecStartId - iSize,
                                    S.data() +
                                      (eigVecStartId + iSize) * numVec +
                                      (eigVecStartId + iSize),
                                    SBlock.data() +
                                      iSize * (numVec - eigVecStartId) + iSize);

              for (size_type iSize = 0; iSize < vecLocalSize; iSize++)
                memoryTransfer.copy(numEigVecInBatch,
                                    X.data() + iSize * numVec + eigVecStartId,
                                    subspaceBatchIn->data() +
                                      numEigVecInBatch * iSize);
            }

          if (d_scratch)
            d_scratch->release();
        }
      else
        {
          size_type numVec  = X.getNumberComponents();
          size_type vecSize = X.locallyOwnedSize();

          utils::MemoryStorage<ValueType, memorySpace> XprojectedA(
            numVec * numVec, utils::Types<ValueType>::zero);
          utils::MemoryStorage<ValueType, utils::MemorySpace::HOST>
            XprojectedAHost(numVec * numVec, utils::Types<ValueType>::zero);
          MultiVector<ValueType, memorySpace> scratch(X, (ValueType)0);

          Op.apply(X, scratch, true, false);

          blasLapack::gemm<ValueType, ValueType, memorySpace>(
            'N',
            'C',
            numVec,
            numVec,
            vecSize,
            (ValueType)1,
            scratch.data(),
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

          utils::MemoryTransfer<utils::MemorySpace::HOST, memorySpace>::copy(
            XprojectedA.size(), XprojectedAHost.data(), XprojectedA.data());

          int mpierr = utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
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

          utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
            XprojectedAHost.size(), XprojectedA.data(), XprojectedAHost.data());
        }
    }
    // // ------------- DEBUG ------------------- // //

    // returns the Xtop(x) in scalapck format (host storage)
    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    void
    RayleighRitzEigenSolver<ValueTypeOperator, ValueTypeOperand, memorySpace>::
      computeXTransOpX(MultiVector<ValueTypeOperand, memorySpace> &X,
                       const std::shared_ptr<const ProcessGrid> &  processGrid,
                       ScaLAPACKMatrix<ValueType> &overlapMatPar,
                       const OpContext &           Op)
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

      // get global to local index maps for Scalapack matrix
      std::unordered_map<size_type, size_type> globalToLocalColumnIdMap;
      std::unordered_map<size_type, size_type> globalToLocalRowIdMap;
      elpaScalaOpInternal::createGlobalToLocalIdMapsScaLAPACKMat(
        processGrid,
        overlapMatPar,
        globalToLocalRowIdMap,
        globalToLocalColumnIdMap);

      utils::MemoryStorage<ValueType, memorySpace> SBlock(numVec *
                                                            d_eigenVecBatchSize,
                                                          ValueType(0));

      utils::MemoryStorage<ValueType, utils::MemorySpace::HOST> SBlockHost(
        numVec * d_eigenVecBatchSize, ValueType(0));

      if (d_scratch)
        d_scratch->acquire();

      for (size_type eigVecStartId = 0; eigVecStartId < numVec;
           eigVecStartId += d_eigenVecBatchSize)
        {
          const size_type eigVecEndId =
            std::min(eigVecStartId + d_eigenVecBatchSize, numVec);
          const size_type numEigVecInBatch = eigVecEndId - eigVecStartId;

          if (numEigVecInBatch % d_eigenVecBatchSize == 0)
            {
              // for (size_type iSize = 0; iSize < vecLocalSize; iSize++)
              //   memoryTransfer.copy(numEigVecInBatch,
              //                       d_XinBatch->data() +
              //                         numEigVecInBatch * iSize,
              //                       X.data() + iSize * numVec +
              //                       eigVecStartId);

              blasLapack::stridedBlockCopy(vecLocalSize,
                                           numEigVecInBatch,
                                           numVec,
                                           eigVecStartId,
                                           numEigVecInBatch,
                                           0,
                                           X.data(),
                                           d_XinBatch->data(),
                                           *X.getLinAlgOpContext());

              subspaceBatchIn  = d_XinBatch;
              subspaceBatchOut = d_XoutBatch;
            }
          else if (numEigVecInBatch % d_eigenVecBatchSize == d_batchSizeSmall)
            {
              blasLapack::stridedBlockCopy(vecLocalSize,
                                           numEigVecInBatch,
                                           numVec,
                                           eigVecStartId,
                                           numEigVecInBatch,
                                           0,
                                           X.data(),
                                           d_XinBatchSmall->data(),
                                           *X.getLinAlgOpContext());

              subspaceBatchIn  = d_XinBatchSmall;
              subspaceBatchOut = d_XoutBatchSmall;
            }
          else
            {
              d_batchSizeSmall = numEigVecInBatch;

              const bool useSmallScratch =
                d_scratch != nullptr && d_scratch->hasXinBatchSmall() &&
                d_scratch->hasXoutBatchSmall() &&
                d_scratch->getXinBatchSmallSize() == numEigVecInBatch;

              if (useSmallScratch)
                {
                  d_XinBatchSmall  = d_scratch->getXinBatchSmall();
                  d_XoutBatchSmall = d_scratch->getXoutBatchSmall();
                }
              else
                {
                  d_XinBatchSmall =
                    std::make_shared<MultiVector<ValueType, memorySpace>>(
                      X.getMPIPatternP2P(),
                      X.getLinAlgOpContext(),
                      numEigVecInBatch,
                      ValueType());

                  d_XoutBatchSmall =
                    std::make_shared<MultiVector<ValueType, memorySpace>>(
                      X.getMPIPatternP2P(),
                      X.getLinAlgOpContext(),
                      numEigVecInBatch,
                      ValueType());
                  if (d_scratch != nullptr)
                    {
                      d_scratch->setXinBatchSmall(d_XinBatchSmall);
                      d_scratch->setXoutBatchSmall(d_XoutBatchSmall);
                    }
                }

              blasLapack::stridedBlockCopy(vecLocalSize,
                                           numEigVecInBatch,
                                           numVec,
                                           eigVecStartId,
                                           numEigVecInBatch,
                                           0,
                                           X.data(),
                                           d_XinBatchSmall->data(),
                                           *X.getLinAlgOpContext());

              subspaceBatchIn  = d_XinBatchSmall;
              subspaceBatchOut = d_XoutBatchSmall;
            }

          Op.apply(*subspaceBatchIn, *subspaceBatchOut, true, false);

          // Input data is read is X^T (numVec is fastest index and then
          // vecSize) Operation : S = (X)^H * ((B*X)). S^T = ((B*X)^T)*(X^T)^H

          const ValueType alpha = 1.0;
          const ValueType beta  = 0.0;

          blasLapack::gemm<ValueTypeOperand, ValueType, memorySpace>(
            'N',
            'C',
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

          utils::MemoryTransfer<utils::MemorySpace::HOST, memorySpace>::copy(
            (numVec - eigVecStartId) * numEigVecInBatch,
            SBlockHost.data(),
            SBlock.data());

          int mpierr = utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
            utils::mpi::MPIInPlace,
            SBlockHost.data(),
            (numVec - eigVecStartId) * numEigVecInBatch,
            utils::mpi::Types<ValueType>::getMPIDatatype(),
            utils::mpi::MPISum,
            comm);

          std::pair<bool, std::string> mpiIsSuccessAndMsg =
            utils::mpi::MPIErrIsSuccessAndMsg(mpierr);
          DFTEFE_AssertWithMsg(mpiIsSuccessAndMsg.first,
                               "MPI Error:" + mpiIsSuccessAndMsg.second);


          // Copying only the lower triangular part to the ScaLAPACK
          // overlap matrix
          if (processGrid->is_process_active())
            for (size_type iSize = 0; iSize < numEigVecInBatch; iSize++)
              if (globalToLocalColumnIdMap.find(iSize + eigVecStartId) !=
                  globalToLocalColumnIdMap.end())
                {
                  const size_type localColumnId =
                    globalToLocalColumnIdMap[iSize + eigVecStartId];
                  for (size_type jSize = eigVecStartId + iSize; jSize < numVec;
                       jSize++)
                    {
                      std::unordered_map<size_type, size_type>::iterator it =
                        globalToLocalRowIdMap.find(jSize);
                      if (it != globalToLocalRowIdMap.end())
                        overlapMatPar.local_el(it->second, localColumnId) = *(
                          SBlockHost.data() + iSize * (numVec - eigVecStartId) +
                          jSize - eigVecStartId);
                    }
                }

          // for (size_type iSize = 0; iSize < vecLocalSize; iSize++)
          //   memoryTransfer.copy(numEigVecInBatch,
          //                       X.data() + iSize * numVec + eigVecStartId,
          //                       subspaceBatchIn->data() +
          //                         numEigVecInBatch * iSize);

          blasLapack::stridedBlockCopy(vecLocalSize,
                                       numEigVecInBatch,
                                       numEigVecInBatch,
                                       0,
                                       numVec,
                                       eigVecStartId,
                                       subspaceBatchIn->data(),
                                       X.data(),
                                       *X.getLinAlgOpContext());
        }

      if (d_scratch)
        d_scratch->release();
    }

  } // end of namespace linearAlgebra
} // end of namespace dftefe
