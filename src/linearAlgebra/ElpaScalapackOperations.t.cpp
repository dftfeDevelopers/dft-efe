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

// @author Sambit Das, Aviup Sircar
//
#include "ElpaScalapackOperations.h"
#include <utils/ConditionalOStream.h>
#include "BlasLapack.h"

namespace dftefe
{
  namespace linearAlgebra
  {
    namespace elpaScalaOpInternal
    {
      template <typename T>
      void
      createGlobalToLocalIdMapsScaLAPACKMat(
        const std::shared_ptr<const ProcessGrid> &processGrid,
        const ScaLAPACKMatrix<T> &                mat,
        std::unordered_map<size_type, size_type> &globalToLocalRowIdMap,
        std::unordered_map<size_type, size_type> &globalToLocalColumnIdMap)
      {
        globalToLocalRowIdMap.clear();
        globalToLocalColumnIdMap.clear();
        if (processGrid->is_process_active())
          {
            for (size_type i = 0; i < mat.local_m(); ++i)
              globalToLocalRowIdMap[mat.global_row(i)] = i;

            for (size_type j = 0; j < mat.local_n(); ++j)
              globalToLocalColumnIdMap[mat.global_column(j)] = j;
          }
      }

      template <typename T>
      void
      scaleScaLAPACKMat(const std::shared_ptr<const ProcessGrid> &processGrid,
                        ScaLAPACKMatrix<T> &                      mat,
                        const T                                   scalar)
      {
        // if (processGrid->is_process_active())
        //   {
        //     const size_type numberComponents = mat.local_m() *
        //     mat.local_n(); const size_type inc              = 1;
        //     xscal(&numberComponents, &scalar, &mat.local_el(0, 0), &inc);
        //   }
      }

      template <typename T>
      void
      fillParallelOverlapMatrix(
        const T *                                 subspaceVectorsArray,
        const size_type                           subspaceVectorsArrayLocalSize,
        const size_type                           N,
        const std::shared_ptr<const ProcessGrid> &processGrid,
        const utils::mpi::MPIComm &               mpiComm,
        const size_type                           wfcBlockSize,
        ScaLAPACKMatrix<T> &                      overlapMatPar)
      {
        const size_type numLocalDofs = subspaceVectorsArrayLocalSize / N;

        // // band group parallelization data structures
        // const size_type numberBandGroups =
        //   utils::mpi::numMPIProcesses(interBandGroupComm);
        // const size_type bandGroupTaskId =
        //   utils::mpi::thisMPIProcess(interBandGroupComm);
        // std::vector<size_type> bandGroupLowHighPlusOneIndices;
        // dftUtils::createBandParallelizationIndices(
        //   interBandGroupComm, N, bandGroupLowHighPlusOneIndices);

        // get global to local index maps for Scalapack matrix
        std::unordered_map<size_type, size_type> globalToLocalColumnIdMap;
        std::unordered_map<size_type, size_type> globalToLocalRowIdMap;
        elpaScalaOpInternal::createGlobalToLocalIdMapsScaLAPACKMat(
          processGrid,
          overlapMatPar,
          globalToLocalRowIdMap,
          globalToLocalColumnIdMap);


        /*
         * Sc=X^{T}*Xc is done in a blocked approach for memory optimization:
         * Sum_{blocks} X^{T}*XcBlock. The result of each X^{T}*XBlock
         * has a much smaller memory compared to X^{T}*Xc.
         * X^{T} is a matrix with size number of wavefunctions times
         * number of local degrees of freedom (N x MLoc).
         * MLoc is denoted by numLocalDofs.
         * Xc denotes complex conjugate of X.
         * XcBlock is a matrix with size (MLoc x B). B is the block size.
         * A further optimization is done to reduce floating point operations:
         * As X^{T}*Xc is a Hermitian matrix, it suffices to compute only the
         * lower triangular part. To exploit this, we do X^{T}*Xc=Sum_{blocks}
         * XTrunc^{T}*XcBlock where XTrunc^{T} is a (D x MLoc) sub matrix of
         * X^{T} with the row indices ranging fromt the lowest global index of
         * XcBlock (denoted by ivec in the code) to N. D=N-ivec. The parallel
         * ScaLapack overlap matrix is directly filled from the
         * XTrunc^{T}*XcBlock result
         */
        const size_type vectorsBlockSize = wfcBlockSize;

        std::vector<T> overlapMatrixBlock(N * vectorsBlockSize, 0.0);

        for (size_type ivec = 0; ivec < N; ivec += vectorsBlockSize)
          {
            // Correct block dimensions if block "goes off edge of" the matrix
            const size_type B = std::min(vectorsBlockSize, N - ivec);
            blasLapack::Op
              transA = blasLapack::Op::NoTrans /*'N'*/,
              transB = std::is_same<T, std::complex<double>>::value ?
                         linearAlgebra::blasLapack::Op::ConjTrans /*'C'*/ :
                         linearAlgebra::blasLapack::Op::Trans /*'T'*/;
            const T scalarCoeffAlpha = 1.0, scalarCoeffBeta = 0.0;

            std::fill(overlapMatrixBlock.begin(), overlapMatrixBlock.end(), 0.);

            const size_type D = N - ivec;

            // Comptute local XTrunc^{T}*XcBlock.
            blasLapack::gemm<T, T, utils::MemorySpace::HOST>(
              blasLapack::Layout::ColMajor,
              transA,
              transB,
              D,
              B,
              numLocalDofs,
              scalarCoeffAlpha,
              subspaceVectorsArray + ivec,
              N,
              subspaceVectorsArray + ivec,
              N,
              scalarCoeffBeta,
              &overlapMatrixBlock[0],
              D,
              *LinAlgOpContextDefaults::LINALG_OP_CONTXT_HOST);

            utils::mpi::MPIBarrier(mpiComm);
            // Sum local XTrunc^{T}*XcBlock across domain decomposition
            // processors
            utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
              utils::mpi::MPIInPlace,
              &overlapMatrixBlock[0],
              D * B,
              utils::mpi::Types<T>::getMPIDatatype(),
              utils::mpi::MPISum,
              mpiComm);

            // Copying only the lower triangular part to the ScaLAPACK
            // overlap matrix
            if (processGrid->is_process_active())
              for (size_type i = 0; i < B; ++i)
                if (globalToLocalColumnIdMap.find(i + ivec) !=
                    globalToLocalColumnIdMap.end())
                  {
                    const size_type localColumnId =
                      globalToLocalColumnIdMap[i + ivec];
                    for (size_type j = ivec + i; j < N; ++j)
                      {
                        std::unordered_map<size_type, size_type>::iterator it =
                          globalToLocalRowIdMap.find(j);
                        if (it != globalToLocalRowIdMap.end())
                          overlapMatPar.local_el(it->second, localColumnId) =
                            overlapMatrixBlock[i * D + j - ivec];
                      }
                  }
          } // block loop
      }

      template <typename ValueType, typename utils::MemorySpace memorySpace>
      void
      subspaceRotation(
      ValueType *X,
      const size_type  M,
      const size_type  N,
      /*std::shared_ptr<
        linearAlgebra::BLASWrapper<memorySpace>> &BLASWrapperPtr,*/
      const std::shared_ptr<const ProcessGrid> &processGrid,
      const utils::mpi::MPIComm                 &mpiCommDomain,
      LinAlgOpContext<memorySpace>         &linAlgOpContext,
      const ScaLAPACKMatrix<ValueType> &rotationMatPar,
      const size_type                   subspaceRotDofsBlockSize,
      const size_type                   wfcBlockSize,
      const bool                       rotationMatTranspose,
      const bool                       isRotationMatLowerTria)
      {
        size_type maxNumLocalDofs = 0;
        utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
          &M,
          &maxNumLocalDofs,
          1,
          utils::mpi::Types<size_type>::getMPIDatatype(),
          utils::mpi::MPIMax,
          mpiCommDomain);

        std::unordered_map<size_type, size_type> globalToLocalColumnIdMap;
        std::unordered_map<size_type, size_type> globalToLocalRowIdMap;
        createGlobalToLocalIdMapsScaLAPACKMat(
          processGrid,
          rotationMatPar,
          globalToLocalRowIdMap,
          globalToLocalColumnIdMap);

        const size_type vectorsBlockSize = std::min(wfcBlockSize, N);
        const size_type dofsBlockSize =
          std::min(maxNumLocalDofs, subspaceRotDofsBlockSize);

        utils::MemoryStorage<ValueType, utils::MemorySpace::HOST>
          rotationMatBlockHost(vectorsBlockSize * N, ValueType(0));

        utils::MemoryStorage<ValueType, memorySpace>
          rotationMatBlock(vectorsBlockSize * N, ValueType(0));
        utils::MemoryStorage<ValueType, memorySpace>
          rotatedVectorsMatBlock(N * dofsBlockSize, ValueType(0));

        for (size_type idof = 0; idof < maxNumLocalDofs; idof += dofsBlockSize)
          {
            // Correct block dimensions if block "goes off edge of" the matrix
            size_type BDof = 0;
            if (M >= idof)
              BDof = std::min(dofsBlockSize, M - idof);

            for (size_type jvec = 0; jvec < N; jvec += vectorsBlockSize)
              {
                // Correct block dimensions if block "goes off edge of" the matrix
                const size_type BVec = std::min(vectorsBlockSize, N - jvec);

                const size_type D = isRotationMatLowerTria ? (jvec + BVec) : N;

                    rotationMatBlockHost.setZero(BVec * N , 0);

                    // Extract QBVec from parallel ScaLAPACK matrix Q
                    if (rotationMatTranspose)
                      {
                        if (processGrid->is_process_active())
                          for (size_type i = 0; i < D; ++i)
                            if (globalToLocalRowIdMap.find(i) !=
                                globalToLocalRowIdMap.end())
                              {
                                const size_type localRowId =
                                  globalToLocalRowIdMap[i];
                                for (size_type j = 0; j < BVec; ++j)
                                  {
                                    std::unordered_map<size_type,
                                                      size_type>::iterator
                                      it = globalToLocalColumnIdMap.find(
                                        j + jvec);
                                    if (it != globalToLocalColumnIdMap.end())
                                      *(rotationMatBlockHost.begin() +
                                        i * BVec + j) =
                                        rotationMatPar.local_el(localRowId,
                                                                it->second);
                                  }
                              }
                      }
                    else
                      {
                        if (processGrid->is_process_active())
                          for (size_type i = 0; i < D; ++i)
                            if (globalToLocalColumnIdMap.find(i) !=
                                globalToLocalColumnIdMap.end())
                              {
                                const size_type localColumnId =
                                  globalToLocalColumnIdMap[i];
                                for (size_type j = 0; j < BVec; ++j)
                                  {
                                    std::unordered_map<size_type,
                                                      size_type>::iterator
                                      it =
                                        globalToLocalRowIdMap.find(j + jvec);
                                    if (it != globalToLocalRowIdMap.end())
                                      *(rotationMatBlockHost.begin() +
                                        i * BVec + j) =
                                        rotationMatPar.local_el(
                                          it->second, localColumnId);
                                  }
                              }
                      }
                  

                    utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(utils::mpi::MPIInPlace,
                                  rotationMatBlockHost.begin(),
                                  BVec * D,
                                  utils::mpi::Types<ValueType>::getMPIDatatype(),
                                  utils::mpi::MPISum,
                                  mpiCommDomain);

                utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
                  BVec * D, rotationMatBlock.begin(), rotationMatBlockHost.begin());

                  const ValueType scalarCoeffAlpha = ValueType(1.0);
                  const ValueType scalarCoeffBeta =  ValueType(0);

                    if (BDof != 0)
                      {
                        blasLapack::gemm<ValueType, ValueType, memorySpace>(
                          blasLapack::Layout::ColMajor,
                          blasLapack::Op::NoTrans,
                          blasLapack::Op::NoTrans,
                          BVec,
                          BDof,
                          D,
                          scalarCoeffAlpha,
                          rotationMatBlock.begin(),
                          BVec,
                          X + idof * N,
                          N,
                          scalarCoeffBeta,
                          rotatedVectorsMatBlock.begin() + jvec,
                          N,
                          linAlgOpContext);                        
                      }
              } // block loop over vectors


            if (BDof != 0)
              {
                utils::MemoryTransfer<memorySpace, memorySpace>::copy(
                  N * BDof, X + idof * N, rotatedVectorsMatBlock.begin());
              }
          } // block loop over dofs
      }
    } // namespace elpaScalaOpInternal
  }   // namespace linearAlgebra
} // namespace dftefe
