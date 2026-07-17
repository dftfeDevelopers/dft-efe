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

#ifndef dftefeMultiVectorOps_h
#define dftefeMultiVectorOps_h

#include <utils/MemorySpaceType.h>
#include <utils/MemoryStorage.h>
#include <linearAlgebra/BlasLapackTypedef.h>
#include <linearAlgebra/MultiVector.h>
#include <linearAlgebra/MultiVectorProductSpace.h>
#include <linearAlgebra/MultiVectorProductSpaceBlocked.h>
#include <linearAlgebra/OperatorContext.h>
#include <linearAlgebra/ElpaScalapackManager.h>
#include <vector>

namespace dftefe
{
  namespace linearAlgebra
  {
    /**
     * @brief Stateless class of static functions for spin-aware projection
     *        and rotation of product-space multivectors.
     *
     * project() computes P = X^H Op X.
     * rotate() performs X <- X * Q in-place.
     *
     * Two overloads per function, selected at compile time by the multivector
     * type (most-derived-first):
     *
     *  MultiVectorProductSpaceBlocked  (collinear):
     *    Blocked loop over S independent N×N sub-problems.
     *    Spin-s columns accessed via ptr = X.data() + s*N, lda = S*N —
     *    no intermediate copy.
     *
     *  MultiVectorProductSpace  (unpolarized S=1 or non-collinear S=2):
     *    Single call on the full S*N columns as a coupled unit.
     *    S=1 degenerates exactly to the existing scalar path.
     *
     * scratchXin / scratchXout / scratchXinSmall / scratchXoutSmall for
     * project() are owned by the caller and persist across SCF calls to
     * avoid repeated allocation.  All other scratch (SBlock, rotation buffers)
     * is allocated internally per call, consistent with
     * ElpaScalapackOperations.
     *
     * dynamic_cast is NOT performed inside these functions.  The caller
     * (RayleighRitzEigenSolver::solve, OrthonormalizationFunctions methods)
     * performs one dynamic_cast per call, then passes the correctly-typed
     * reference here so the compiler selects the right overload.
     */
    class MultiVectorOps
    {
    public:
      // -----------------------------------------------------------------------
      // project
      // -----------------------------------------------------------------------

      /**
       * @brief Blocked overload (collinear, MultiVectorProductSpaceBlocked).
       *
       * Computes S independent overlap matrices:
       *   Ps[s] = Xs^H Op Xs   (N×N each)
       * where Xs is the spin-s block of X accessed as
       *   ptr = X.data() + s * X.numVectorsPerSpace(),  lda = X.numVectors()
       *
       * @param Op       Operator context (e.g. B metric or identity).
       * @param X        Blocked product-space multivector (M × S*N).
       * @param Ps       Output: S ScaLAPACK N×N matrices (pre-allocated by caller).
       * @param elpa     ELPA/ScaLAPACK manager (provides process grid).
       * @param scratchXin      Scratch multivector (M × batchSize), caller-owned.
       * @param scratchXout     Scratch multivector (M × batchSize), caller-owned.
       * @param scratchXinSmall Tail-batch scratch, nullptr on first call; cached.
       * @param scratchXoutSmall Tail-batch scratch, nullptr on first call; cached.
       */
      template <typename ValueTypeOperator,
                typename ValueTypeOperand,
                utils::MemorySpace memorySpace>
      static void
      project(
        const OperatorContext<ValueTypeOperator, ValueTypeOperand, memorySpace>
          &                                                               Op,
        MultiVectorProductSpaceBlocked<ValueTypeOperand, memorySpace> &   X,
        std::vector<ScaLAPACKMatrix<
          blasLapack::scalar_type<ValueTypeOperator, ValueTypeOperand>>> &Ps,
        const ElpaScalapackManager &                                      elpa,
        std::shared_ptr<MultiVector<
          blasLapack::scalar_type<ValueTypeOperator, ValueTypeOperand>,
          memorySpace>> &scratchXin,
        std::shared_ptr<MultiVector<
          blasLapack::scalar_type<ValueTypeOperator, ValueTypeOperand>,
          memorySpace>> &scratchXout,
        std::shared_ptr<MultiVector<
          blasLapack::scalar_type<ValueTypeOperator, ValueTypeOperand>,
          memorySpace>> &scratchXinSmall,
        std::shared_ptr<MultiVector<
          blasLapack::scalar_type<ValueTypeOperator, ValueTypeOperand>,
          memorySpace>> &scratchXoutSmall);

      /**
       * @brief Coupled overload (unpolarized S=1 / non-collinear S=2,
       *        MultiVectorProductSpace).
       *
       * Computes a single overlap matrix:
       *   P = X^H Op X   (S*N × S*N)
       * The full S*N columns are treated as a coupled unit.
       * S=1 degenerates to the existing N×N scalar path.
       *
       * @param Op       Operator context.
       * @param X        Coupled product-space multivector (M × S*N).
       * @param P        Output: single ScaLAPACK S*N×S*N matrix (pre-allocated).
       * @param elpa     ELPA/ScaLAPACK manager.
       * @param scratchXin      Scratch multivector (M × batchSize), caller-owned.
       * @param scratchXout     Scratch multivector (M × batchSize), caller-owned.
       * @param scratchXinSmall Tail-batch scratch, nullptr on first call; cached.
       * @param scratchXoutSmall Tail-batch scratch, nullptr on first call; cached.
       */
      template <typename ValueTypeOperator,
                typename ValueTypeOperand,
                utils::MemorySpace memorySpace>
      static void
      project(
        const OperatorContext<ValueTypeOperator, ValueTypeOperand, memorySpace>
          &                                                     Op,
        MultiVectorProductSpace<ValueTypeOperand, memorySpace> &X,
        ScaLAPACKMatrix<
          blasLapack::scalar_type<ValueTypeOperator, ValueTypeOperand>> &P,
        const ElpaScalapackManager &                                     elpa,
        std::shared_ptr<MultiVector<
          blasLapack::scalar_type<ValueTypeOperator, ValueTypeOperand>,
          memorySpace>> &scratchXin,
        std::shared_ptr<MultiVector<
          blasLapack::scalar_type<ValueTypeOperator, ValueTypeOperand>,
          memorySpace>> &scratchXout,
        std::shared_ptr<MultiVector<
          blasLapack::scalar_type<ValueTypeOperator, ValueTypeOperand>,
          memorySpace>> &scratchXinSmall,
        std::shared_ptr<MultiVector<
          blasLapack::scalar_type<ValueTypeOperator, ValueTypeOperand>,
          memorySpace>> &scratchXoutSmall);

      // -----------------------------------------------------------------------
      // rotate
      // -----------------------------------------------------------------------

      /**
       * @brief Blocked rotate overload (collinear, MultiVectorProductSpaceBlocked).
       *
       * Performs S independent in-place subspace rotations:
       *   Xs <- Xs * Qs[s]
       * using ptr = X.data() + s*N, lda = S*N.
       * scratchRotBlock (M×N) is reused across s iterations — S× smaller
       * than the coupled scratch.
       *
       * @param X    Blocked multivector, modified in place.
       * @param Qs   S rotation matrices (N×N each).
       * @param elpa ELPA/ScaLAPACK manager.
       */
      template <typename ValueType, utils::MemorySpace memorySpace>
      static void
      rotate(MultiVectorProductSpaceBlocked<ValueType, memorySpace> &X,
             const std::vector<ScaLAPACKMatrix<ValueType>> &         Qs,
             const ElpaScalapackManager &                            elpa);

      /**
       * @brief Coupled rotate overload (unpolarized / non-collinear,
       *        MultiVectorProductSpace).
       *
       * Performs a single in-place subspace rotation on all S*N columns:
       *   X <- X * Q
       * One M×S*N scratch buffer (unavoidable for in-place operation).
       * S=1 degenerates to the existing scalar rotation path.
       *
       * @param X    Coupled multivector, modified in place.
       * @param Q    Single rotation matrix (S*N × S*N).
       * @param elpa ELPA/ScaLAPACK manager.
       */
      template <typename ValueType, utils::MemorySpace memorySpace>
      static void
      rotate(MultiVectorProductSpace<ValueType, memorySpace> &X,
             const ScaLAPACKMatrix<ValueType> &               Q,
             const ElpaScalapackManager &                     elpa);

      // -----------------------------------------------------------------------
      // copyToBatch / copyFromBatch
      // -----------------------------------------------------------------------

      // Gather numVecBatch orbitals starting at srcStart for every spin channel
      // into a flat MultiVector batch (Xbatch columns: spin-0, then spin-1,
      // ...). Xbatch must have numSpaces * numVecBatch components. Two-type
      // template mirrors stridedBlockCopy<VT1,VT2>: X stores VT1, Xbatch stores
      // VT2 (typically VT1==VT2; differs inside projectImpl).

      template <typename ValueType1,
                typename ValueType2,
                utils::MemorySpace memorySpace>
      static void
      copyToBatch(const MultiVectorProductSpace<ValueType1, memorySpace> &X,
                  size_type                             srcStart,
                  size_type                             numVecBatch,
                  MultiVector<ValueType2, memorySpace> &Xbatch,
                  LinAlgOpContext<memorySpace> &        context);

      // Inverse of copyToBatch: scatter the batch back into the product-space
      // multivector at orbital positions [dstStart, dstStart+numVecBatch).

      template <typename ValueType1,
                typename ValueType2,
                utils::MemorySpace memorySpace>
      static void
      copyFromBatch(const MultiVector<ValueType1, memorySpace> &Ybatch,
                    size_type                                   dstStart,
                    size_type                                   numVecBatch,
                    MultiVectorProductSpace<ValueType2, memorySpace> &Y,
                    LinAlgOpContext<memorySpace> &                    context);

    }; // class MultiVectorOps

  } // namespace linearAlgebra
} // namespace dftefe

#include <linearAlgebra/MultiVectorOps.t.cpp>
#endif // dftefeMultiVectorOps_h
