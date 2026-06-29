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
 * License at the top level of DFT-EFE distribution.  If not, see            *
 *   <https://www.gnu.org/licenses/>.                                         *
 ******************************************************************************/

/*
 * @author Avirup Sircar
 */

#include <utils/Exceptions.h>
#include <utils/MPITypes.h>
#include <utils/MPIWrapper.h>
#include <utils/MemoryTransfer.h>
#include <linearAlgebra/BlasLapack.h>
#include <unordered_map>
#include <memory>
#include <algorithm>

namespace dftefe
{
  namespace linearAlgebra
  {
    namespace multiVectorOpsInternal
    {
      // Mirrors computeXTransOpX in RayleighRitzEigenSolver /
      // OrthonormalizationFunctions, extended for blocked-space support.
      // numVecBlock — vectors in the active sub-block (N blocked, S*N coupled).
      // lda         — stride of the full X buffer (always S*N).
      // baseColOffset — first column of the sub-block (s*N blocked, 0 coupled).
      // XinBatch, XoutBatch, SBlock replace the class-member counterparts
      // d_XinBatch / d_XoutBatch / SBlock(local) of the member functions.
      template <typename ValueTypeOperator,
                typename ValueTypeOperand,
                utils::MemorySpace memorySpace>
      static void
      projectImpl(
        MultiVectorProductSpace<ValueTypeOperand, memorySpace> &X,
        const size_type                                         numVecBlock,
        const size_type                                         lda,
        const size_type                                         baseColOffset,
        const std::shared_ptr<const ProcessGrid> &              processGrid,
        ScaLAPACKMatrix<blasLapack::scalar_type<ValueTypeOperator,
                                               ValueTypeOperand>> &
          overlapMatPar,
        const OperatorContext<ValueTypeOperator,
                              ValueTypeOperand,
                              memorySpace> &        Op,
        const size_type                             eigenVecBatchSize,
        std::shared_ptr<
          MultiVector<blasLapack::scalar_type<ValueTypeOperator,
                                             ValueTypeOperand>,
                      memorySpace>> &               XinBatch,
        std::shared_ptr<
          MultiVector<blasLapack::scalar_type<ValueTypeOperator,
                                             ValueTypeOperand>,
                      memorySpace>> &               XoutBatch,
        std::shared_ptr<
          MultiVector<blasLapack::scalar_type<ValueTypeOperator,
                                             ValueTypeOperand>,
                      memorySpace>> &               XinBatchSmall,
        std::shared_ptr<
          MultiVector<blasLapack::scalar_type<ValueTypeOperator,
                                             ValueTypeOperand>,
                      memorySpace>> &               XoutBatchSmall)
      {
        using ValueType =
          blasLapack::scalar_type<ValueTypeOperator, ValueTypeOperand>;

        const utils::mpi::MPIComm             comm = X.getMPIPatternP2P()->mpiCommunicator();
        LinAlgOpContext<memorySpace>           linAlgOpContext = *X.getLinAlgOpContext();
        const size_type                       vecSize      = X.locallyOwnedSize();
        const size_type                       vecLocalSize = X.localSize();
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

        utils::MemoryStorage<ValueType, memorySpace> SBlock(
          numVecBlock * eigenVecBatchSize, ValueType(0));
        utils::MemoryStorage<ValueType, utils::MemorySpace::HOST> SBlockHost(
          numVecBlock * eigenVecBatchSize, ValueType(0));

        for (size_type eigVecStartId = 0; eigVecStartId < numVecBlock;
             eigVecStartId += eigenVecBatchSize)
          {
            const size_type eigVecEndId =
              std::min(eigVecStartId + eigenVecBatchSize, numVecBlock);
            const size_type numEigVecInBatch = eigVecEndId - eigVecStartId;

            if (numEigVecInBatch % eigenVecBatchSize == 0)
              {
                // for (size_type iSize = 0; iSize < vecLocalSize; iSize++)
                //   memoryTransfer.copy(numEigVecInBatch,
                //                       XinBatch->data() +
                //                         numEigVecInBatch * iSize,
                //                       X.data() + iSize * lda +
                //                         baseColOffset + eigVecStartId);

                blasLapack::stridedBlockCopy(vecLocalSize,
                                             numEigVecInBatch,
                                             lda,
                                             baseColOffset + eigVecStartId,
                                             numEigVecInBatch,
                                             0,
                                             X.data(),
                                             XinBatch->data(),
                                             *X.getLinAlgOpContext());

                subspaceBatchIn  = XinBatch;
                subspaceBatchOut = XoutBatch;
              }
            else if (XinBatchSmall != nullptr &&
                     numEigVecInBatch ==
                       XinBatchSmall->getNumberComponents())
              {
                blasLapack::stridedBlockCopy(vecLocalSize,
                                             numEigVecInBatch,
                                             lda,
                                             baseColOffset + eigVecStartId,
                                             numEigVecInBatch,
                                             0,
                                             X.data(),
                                             XinBatchSmall->data(),
                                             *X.getLinAlgOpContext());

                subspaceBatchIn  = XinBatchSmall;
                subspaceBatchOut = XoutBatchSmall;
              }
            else
              {
                XinBatchSmall =
                  std::make_shared<MultiVector<ValueType, memorySpace>>(
                    X.getMPIPatternP2P(),
                    X.getLinAlgOpContext(),
                    numEigVecInBatch,
                    ValueType());

                XoutBatchSmall =
                  std::make_shared<MultiVector<ValueType, memorySpace>>(
                    X.getMPIPatternP2P(),
                    X.getLinAlgOpContext(),
                    numEigVecInBatch,
                    ValueType());

                blasLapack::stridedBlockCopy(vecLocalSize,
                                             numEigVecInBatch,
                                             lda,
                                             baseColOffset + eigVecStartId,
                                             numEigVecInBatch,
                                             0,
                                             X.data(),
                                             XinBatchSmall->data(),
                                             *X.getLinAlgOpContext());

                subspaceBatchIn  = XinBatchSmall;
                subspaceBatchOut = XoutBatchSmall;
              }

            Op.apply(*subspaceBatchIn, *subspaceBatchOut, true, false);

            // Input data is read as X^T (lda is fastest index and then
            // vecSize). Operation : S = (X)^H * ((Op*X)).
            // S^T = ((Op*X)^T)*(X^T)^H

            const ValueType alpha = 1.0;
            const ValueType beta  = 0.0;

            blasLapack::gemm<ValueTypeOperand, ValueType, memorySpace>(
              'N',
              'C',
              numVecBlock - eigVecStartId,
              numEigVecInBatch,
              vecSize,
              alpha,
              X.data() + baseColOffset + eigVecStartId,
              lda,
              subspaceBatchOut->data(),
              numEigVecInBatch,
              beta,
              SBlock.data(),
              numVecBlock - eigVecStartId,
              linAlgOpContext);

            utils::MemoryTransfer<utils::MemorySpace::HOST, memorySpace>::copy(
              (numVecBlock - eigVecStartId) * numEigVecInBatch,
              SBlockHost.data(),
              SBlock.data());

            int mpierr = utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
              utils::mpi::MPIInPlace,
              SBlockHost.data(),
              (numVecBlock - eigVecStartId) * numEigVecInBatch,
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
                    for (size_type jSize = eigVecStartId + iSize;
                         jSize < numVecBlock;
                         jSize++)
                      {
                        std::unordered_map<size_type, size_type>::iterator it =
                          globalToLocalRowIdMap.find(jSize);
                        if (it != globalToLocalRowIdMap.end())
                          overlapMatPar.local_el(it->second, localColumnId) = *(
                            SBlockHost.data() +
                            iSize * (numVecBlock - eigVecStartId) +
                            jSize - eigVecStartId);
                      }
                  }

            // for (size_type iSize = 0; iSize < vecLocalSize; iSize++)
            //   memoryTransfer.copy(numEigVecInBatch,
            //                       X.data() + iSize * lda +
            //                         baseColOffset + eigVecStartId,
            //                       subspaceBatchIn->data() +
            //                         numEigVecInBatch * iSize);

            blasLapack::stridedBlockCopy(vecLocalSize,
                                         numEigVecInBatch,
                                         numEigVecInBatch,
                                         0,
                                         lda,
                                         baseColOffset + eigVecStartId,
                                         subspaceBatchIn->data(),
                                         X.data(),
                                         *X.getLinAlgOpContext());
          }
      }


      // Mirrors elpaScalaOpInternal::subspaceRotation.
      // M / mpiCommDomain / linAlgOpContext are derived from X internally.
      // stridedBlockCopy replaces copyValueType1ArrToValueType2Arr so that
      // lda > numVecBlock is handled correctly (blocked case).
      template <typename ValueType, utils::MemorySpace memorySpace>
      static void
      rotateImpl(
        MultiVectorProductSpace<ValueType, memorySpace> &X,
        const size_type                                  numVecBlock,
        const size_type                                  lda,
        const size_type                                  baseColOffset,
        const std::shared_ptr<const ProcessGrid> &       processGrid,
        const ScaLAPACKMatrix<ValueType> &               rotationMatPar,
        const size_type                                  subspaceRotDofsBlockSize,
        const size_type                                  wfcBlockSize,
        const bool rotationMatTranspose   = false,
        const bool isRotationMatLowerTria = false,
        const bool allowFullCPUMemSubspaceRot =
          (memorySpace == utils::MemorySpace::DEVICE ? true : false))
      {
        const size_type              M              = X.locallyOwnedSize();
        const utils::mpi::MPIComm    mpiCommDomain  = X.getMPIPatternP2P()->mpiCommunicator();
        LinAlgOpContext<memorySpace> linAlgOpContext = *X.getLinAlgOpContext();

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
        elpaScalaOpInternal::createGlobalToLocalIdMapsScaLAPACKMat(
          processGrid,
          rotationMatPar,
          globalToLocalRowIdMap,
          globalToLocalColumnIdMap);

        const size_type vectorsBlockSize = std::min(wfcBlockSize, numVecBlock);
        const size_type dofsBlockSize =
          std::min(maxNumLocalDofs, subspaceRotDofsBlockSize);

        constexpr utils::MemorySpace hostMemSpace =
          (memorySpace == utils::MemorySpace::DEVICE) ?
            utils::MemorySpace::HOST_PINNED :
            utils::MemorySpace::HOST;

        utils::MemoryStorage<ValueType, hostMemSpace> rotationMatBlockHost;
        if (allowFullCPUMemSubspaceRot)
          {
            rotationMatBlockHost.resize(numVecBlock * numVecBlock, ValueType(0));
            rotationMatBlockHost.setValue(ValueType(0));
          }
        else
          {
            rotationMatBlockHost.resize(vectorsBlockSize * numVecBlock,
                                        ValueType(0));
          }

        utils::MemoryStorage<ValueType, memorySpace> rotationMatBlock(
          vectorsBlockSize * numVecBlock, ValueType(0));
        utils::MemoryStorage<ValueType, memorySpace> rotatedVectorsMatBlock(
          numVecBlock * dofsBlockSize, ValueType(0));

        utils::printCurrentMemoryUsage<memorySpace>(
          mpiCommDomain, "Inside Blocked subspace rotation");

        for (size_type idof = 0; idof < maxNumLocalDofs; idof += dofsBlockSize)
          {
            // Correct block dimensions if block "goes off edge of" the matrix
            size_type BDof = 0;
            if (M >= idof)
              BDof = std::min(dofsBlockSize, M - idof);

            for (size_type jvec = 0; jvec < numVecBlock; jvec += vectorsBlockSize)
              {
                // Correct block dimensions if block "goes off edge of" the
                // matrix
                const size_type BVec =
                  std::min(vectorsBlockSize, numVecBlock - jvec);

                const size_type D =
                  isRotationMatLowerTria ? (jvec + BVec) : numVecBlock;

                if (allowFullCPUMemSubspaceRot)
                  {
                    if (idof == 0)
                      {
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
                                        if (it !=
                                            globalToLocalColumnIdMap.end())
                                          *(rotationMatBlockHost.begin() +
                                            jvec * numVecBlock + i * BVec +
                                            j) =
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
                                          it = globalToLocalRowIdMap.find(j +
                                                                          jvec);
                                        if (it != globalToLocalRowIdMap.end())
                                          *(rotationMatBlockHost.begin() +
                                            jvec * numVecBlock + i * BVec +
                                            j) =
                                            rotationMatPar.local_el(
                                              it->second, localColumnId);
                                      }
                                  }
                          }

                        utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
                          utils::mpi::MPIInPlace,
                          rotationMatBlockHost.begin() + jvec * numVecBlock,
                          BVec * D,
                          utils::mpi::Types<ValueType>::getMPIDatatype(),
                          utils::mpi::MPISum,
                          mpiCommDomain);
                      }

                    utils::MemoryTransfer<memorySpace, hostMemSpace>::copy(
                      BVec * D,
                      rotationMatBlock.begin(),
                      rotationMatBlockHost.begin() + jvec * numVecBlock);
                  }
                else
                  {
                    rotationMatBlockHost.setZero(BVec * numVecBlock, 0);

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
                                                       size_type>::iterator it =
                                      globalToLocalColumnIdMap.find(j + jvec);
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
                                                       size_type>::iterator it =
                                      globalToLocalRowIdMap.find(j + jvec);
                                    if (it != globalToLocalRowIdMap.end())
                                      *(rotationMatBlockHost.begin() +
                                        i * BVec + j) =
                                        rotationMatPar.local_el(it->second,
                                                                localColumnId);
                                  }
                              }
                      }


                    utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
                      utils::mpi::MPIInPlace,
                      rotationMatBlockHost.begin(),
                      BVec * D,
                      utils::mpi::Types<ValueType>::getMPIDatatype(),
                      utils::mpi::MPISum,
                      mpiCommDomain);

                    utils::MemoryTransfer<memorySpace, hostMemSpace>::copy(
                      BVec * D,
                      rotationMatBlock.begin(),
                      rotationMatBlockHost.begin());
                  }

                const ValueType scalarCoeffAlpha = ValueType(1.0);
                const ValueType scalarCoeffBeta  = ValueType(0);

                if (BDof != 0)
                  {
                    blasLapack::gemm<ValueType, ValueType, memorySpace>(
                      'N',
                      'N',
                      BVec,
                      BDof,
                      D,
                      scalarCoeffAlpha,
                      rotationMatBlock.begin(),
                      BVec,
                      X.data() + idof * lda + baseColOffset,
                      lda,
                      scalarCoeffBeta,
                      rotatedVectorsMatBlock.begin() + jvec,
                      numVecBlock,
                      linAlgOpContext);
                  }
              } // block loop over vectors


            if (BDof != 0)
              {
                // blasLapack::copyValueType1ArrToValueType2Arr(
                //   numVecBlock * BDof,
                //   rotatedVectorsMatBlock.begin(),
                //   X.data() + idof * lda + baseColOffset,
                //   linAlgOpContext); // valid only when lda == numVecBlock

                blasLapack::stridedBlockCopy(BDof,
                                             numVecBlock,
                                             numVecBlock,
                                             0,
                                             lda,
                                             baseColOffset,
                                             rotatedVectorsMatBlock.begin(),
                                             X.data() + idof * lda,
                                             linAlgOpContext);
              }
          } // block loop over dofs
      }

    } // namespace multiVectorOpsInternal


    // -------------------------------------------------------------------------
    // project — Blocked (collinear, MultiVectorProductSpaceBlocked)
    //
    // Loops over S spin spaces.  For spin-s the active columns in the flat
    // data buffer are [s*N, s*N+N) with source lda = S*N.
    // Each spin produces an independent N×N ScaLAPACK matrix Ps[s].
    // -------------------------------------------------------------------------
    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    void
    MultiVectorOps::project(
      const OperatorContext<ValueTypeOperator,
                            ValueTypeOperand,
                            memorySpace> &Op,
      MultiVectorProductSpaceBlocked<ValueTypeOperand, memorySpace> &X,
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
        memorySpace>> &scratchXoutSmall)
    {
      using ValueType =
        blasLapack::scalar_type<ValueTypeOperator, ValueTypeOperand>;

      utils::throwException(
        Ps.size() == X.numSpaces(),
        "MultiVectorOps::project (Blocked): Ps.size() must equal numSpaces.");

      const size_type numVecBlock = X.numVectorsPerSpace(); // N
      const size_type lda         = X.numVectors();          // S*N

      const std::shared_ptr<const ProcessGrid> processGrid =
        elpa.getProcessGridDftefeScalaWrapper();

      for (size_type s = 0; s < X.numSpaces(); ++s)
        multiVectorOpsInternal::projectImpl<ValueTypeOperator,
                                            ValueTypeOperand,
                                            memorySpace>(
          X,
          numVecBlock,
          lda,
          s * numVecBlock,
          processGrid,
          Ps[s],
          Op,
          scratchXin->getNumberComponents(),
          scratchXin,
          scratchXout,
          scratchXinSmall,
          scratchXoutSmall);
    }


    // -------------------------------------------------------------------------
    // project — Coupled (unpolarized S=1 / non-collinear S=2,
    //           MultiVectorProductSpace)
    //
    // Treats all S*N columns as a single block.  S=1 degenerates exactly to
    // the existing single-spin N×N path (lda == numVecBlock).
    // -------------------------------------------------------------------------
    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    void
    MultiVectorOps::project(
      const OperatorContext<ValueTypeOperator,
                            ValueTypeOperand,
                            memorySpace> &Op,
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
        memorySpace>> &scratchXoutSmall)
    {
      using ValueType =
        blasLapack::scalar_type<ValueTypeOperator, ValueTypeOperand>;

      const size_type lda = X.numVectors(); // S*N; lda == numVecBlock for coupled

      const std::shared_ptr<const ProcessGrid> processGrid =
        elpa.getProcessGridDftefeScalaWrapper();

      multiVectorOpsInternal::projectImpl<ValueTypeOperator,
                                          ValueTypeOperand,
                                          memorySpace>(
        X,
        lda,
        lda,
        0,
        processGrid,
        P,
        Op,
        scratchXin->getNumberComponents(),
        scratchXin,
        scratchXout,
        scratchXinSmall,
        scratchXoutSmall);
    }


    // -------------------------------------------------------------------------
    // rotate — Blocked (collinear, MultiVectorProductSpaceBlocked)
    //
    // Loops over S spin spaces.  For spin-s the active columns in the flat
    // data buffer are [s*N, s*N+N) with source lda = S*N.
    // Each spin is rotated independently: Xs <- Xs * Qs[s].
    // -------------------------------------------------------------------------
    template <typename ValueType, utils::MemorySpace memorySpace>
    void
    MultiVectorOps::rotate(
      MultiVectorProductSpaceBlocked<ValueType, memorySpace> &X,
      const std::vector<ScaLAPACKMatrix<ValueType>> &         Qs,
      const ElpaScalapackManager &                            elpa)
    {
      utils::throwException(
        Qs.size() == X.numSpaces(),
        "MultiVectorOps::rotate (Blocked): Qs.size() must equal numSpaces.");

      const size_type numVecBlock = X.numVectorsPerSpace(); // N
      const size_type lda         = X.numVectors();          // S*N

      const std::shared_ptr<const ProcessGrid> processGrid =
        elpa.getProcessGridDftefeScalaWrapper();

      for (size_type s = 0; s < X.numSpaces(); ++s)
        multiVectorOpsInternal::rotateImpl<ValueType, memorySpace>(
          X,
          numVecBlock,         // vectors in this space block (N)
          lda,                 // stride of full buffer (S*N)
          s * numVecBlock,     // column start of space s
          processGrid,
          Qs[s],
          RayleighRitzDefaults::SUBSPACE_ROT_DOF_BATCH,
          RayleighRitzDefaults::WAVE_FN_BATCH);
    }


    // -------------------------------------------------------------------------
    // rotate — Coupled (unpolarized S=1 / non-collinear S=2,
    //           MultiVectorProductSpace)
    //
    // Rotates all S*N columns as a single unit: X <- X * Q.
    // S=1 degenerates exactly to the existing single-spin N×N rotation path.
    // -------------------------------------------------------------------------
    template <typename ValueType, utils::MemorySpace memorySpace>
    void
    MultiVectorOps::rotate(
      MultiVectorProductSpace<ValueType, memorySpace> &X,
      const ScaLAPACKMatrix<ValueType> &               Q,
      const ElpaScalapackManager &                     elpa)
    {
      const size_type lda = X.numVectors(); // S*N; lda == numVecBlock for coupled

      const std::shared_ptr<const ProcessGrid> processGrid =
        elpa.getProcessGridDftefeScalaWrapper();

      multiVectorOpsInternal::rotateImpl<ValueType, memorySpace>(
        X,
        lda,  // numVecBlock = S*N (coupled: full block)
        lda,  // lda = S*N
        0,    // baseColOffset = 0 (single coupled block)
        processGrid,
        Q,
        RayleighRitzDefaults::SUBSPACE_ROT_DOF_BATCH,
        RayleighRitzDefaults::WAVE_FN_BATCH);
    }

  } // namespace linearAlgebra
} // namespace dftefe
