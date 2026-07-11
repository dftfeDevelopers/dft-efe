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
      // spaceBlocked — true for MultiVectorProductSpaceBlocked (collinear);
      //               false for MultiVectorProductSpace (coupled/unpolarized).
      // overlapMatPars — pointer to first ScaLAPACKMatrix; size numSpaces for
      //                  blocked (Ps.data()), size 1 for coupled (&P).
      // XinBatch, XoutBatch, SBlock replace the class-member counterparts
      // d_XinBatch / d_XoutBatch / SBlock(local) of the member functions.
      template <typename ValueTypeOperator,
                typename ValueTypeOperand,
                utils::MemorySpace memorySpace>
      static void
      projectImpl(
        MultiVectorProductSpace<ValueTypeOperand, memorySpace> &X,
        const bool                                              spaceBlocked,
        const std::shared_ptr<const ProcessGrid> &              processGrid,
        ScaLAPACKMatrix<blasLapack::scalar_type<ValueTypeOperator,
                                               ValueTypeOperand>>
          *                                       overlapMatPars,
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

        const size_type numSpaces          = X.numSpaces();
        const size_type numVecPerSp        = X.numVectorsPerSpace();
        const size_type lda                = X.numVectors();
        const size_type numSpaceLoops       = spaceBlocked ? numSpaces : 1;
        const size_type eigVecBatchPerSpace = eigenVecBatchSize / numSpaces;
        const size_type numVecBlock        = spaceBlocked ? numVecPerSp : lda;

        // get global to local index maps for Scalapack matrix (one per space loop)
        std::vector<std::unordered_map<size_type, size_type>>
          globalToLocalColumnIdMaps(numSpaceLoops);
        std::vector<std::unordered_map<size_type, size_type>>
          globalToLocalRowIdMaps(numSpaceLoops);
        for (size_type s = 0; s < numSpaceLoops; ++s)
          elpaScalaOpInternal::createGlobalToLocalIdMapsScaLAPACKMat(
            processGrid,
            overlapMatPars[s],
            globalToLocalRowIdMaps[s],
            globalToLocalColumnIdMaps[s]);

        utils::MemoryStorage<ValueType, memorySpace> SBlock(
          numVecBlock *
            (spaceBlocked ? eigVecBatchPerSpace : eigenVecBatchSize),
          ValueType(0));
        utils::MemoryStorage<ValueType, utils::MemorySpace::HOST> SBlockHost(
          numVecBlock *
            (spaceBlocked ? eigVecBatchPerSpace : eigenVecBatchSize),
          ValueType(0));

        for (size_type eigVecStartId = 0; eigVecStartId < numVecPerSp;
             eigVecStartId += eigVecBatchPerSpace)
          {
            const size_type eigVecEndId =
              std::min(eigVecStartId + eigVecBatchPerSpace, numVecPerSp);
            const size_type numEigVecInBatch = eigVecEndId - eigVecStartId;
            const size_type numComponentsInBatch = numSpaces * numEigVecInBatch;

            if (numComponentsInBatch == eigenVecBatchSize)
              {
                // for (size_type iSize = 0; iSize < vecLocalSize; iSize++)
                //   memoryTransfer.copy(numEigVecInBatch,
                //                       XinBatch->data() +
                //                         numEigVecInBatch * iSize,
                //                       X.data() + iSize * lda +
                //                         baseColOffset + eigVecStartId);

                MultiVectorOps::copyToBatch<ValueTypeOperand,
                                            ValueType,
                                            memorySpace>(
                  X,
                  eigVecStartId,
                  numEigVecInBatch,
                  *XinBatch,
                  linAlgOpContext);

                subspaceBatchIn  = XinBatch;
                subspaceBatchOut = XoutBatch;
              }
            else if (XinBatchSmall != nullptr &&
                     numComponentsInBatch ==
                       XinBatchSmall->getNumberComponents())
              {
                MultiVectorOps::copyToBatch<ValueTypeOperand,
                                            ValueType,
                                            memorySpace>(
                  X,
                  eigVecStartId,
                  numEigVecInBatch,
                  *XinBatchSmall,
                  linAlgOpContext);

                subspaceBatchIn  = XinBatchSmall;
                subspaceBatchOut = XoutBatchSmall;
              }
            else
              {
                XinBatchSmall =
                  std::make_shared<MultiVector<ValueType, memorySpace>>(
                    X.getMPIPatternP2P(),
                    X.getLinAlgOpContext(),
                    numComponentsInBatch,
                    ValueType());

                XoutBatchSmall =
                  std::make_shared<MultiVector<ValueType, memorySpace>>(
                    X.getMPIPatternP2P(),
                    X.getLinAlgOpContext(),
                    numComponentsInBatch,
                    ValueType());

                MultiVectorOps::copyToBatch<ValueTypeOperand,
                                            ValueType,
                                            memorySpace>(
                  X,
                  eigVecStartId,
                  numEigVecInBatch,
                  *XinBatchSmall,
                  linAlgOpContext);

                subspaceBatchIn  = XinBatchSmall;
                subspaceBatchOut = XoutBatchSmall;
              }

            Op.apply(*subspaceBatchIn, *subspaceBatchOut, true, false);

            // Input data is read as X^T (lda is fastest index and then
            // vecSize). Operation : S = (X)^H * ((Op*X)).
            // S^T = ((Op*X)^T)*(X^T)^H

            const ValueType alpha = 1.0;
            const ValueType beta  = 0.0;

            for (size_type s = 0; s < numSpaceLoops; ++s)
              {
                const size_type baseColOffset =
                  spaceBlocked ? s * numVecPerSp : 0;
                const size_type gemmCols =
                  spaceBlocked ? numEigVecInBatch : numComponentsInBatch;

                blasLapack::gemm<ValueTypeOperand, ValueType, memorySpace>(
                  'N',
                  'C',
                  numVecBlock - eigVecStartId,
                  gemmCols,
                  vecSize,
                  alpha,
                  X.data() + baseColOffset + eigVecStartId,
                  lda,
                  subspaceBatchOut->data() +
                    (spaceBlocked ? s * numEigVecInBatch : 0),
                  numComponentsInBatch,
                  beta,
                  SBlock.data(),
                  numVecBlock - eigVecStartId,
                  linAlgOpContext);

                utils::MemoryTransfer<utils::MemorySpace::HOST, memorySpace>::
                  copy((numVecBlock - eigVecStartId) * gemmCols,
                       SBlockHost.data(),
                       SBlock.data());

                int mpierr = utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
                  utils::mpi::MPIInPlace,
                  SBlockHost.data(),
                  (numVecBlock - eigVecStartId) * gemmCols,
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
                  for (size_type iSize = 0; iSize < gemmCols; iSize++)
                    if (globalToLocalColumnIdMaps[s].find(
                          iSize + eigVecStartId) !=
                        globalToLocalColumnIdMaps[s].end())
                      {
                        const size_type localColumnId =
                          globalToLocalColumnIdMaps[s][iSize + eigVecStartId];
                        for (size_type jSize = eigVecStartId + iSize;
                             jSize < numVecBlock;
                             jSize++)
                          {
                            std::unordered_map<size_type, size_type>::iterator
                              it = globalToLocalRowIdMaps[s].find(jSize);
                            if (it != globalToLocalRowIdMaps[s].end())
                              overlapMatPars[s].local_el(
                                it->second, localColumnId) =
                                *(SBlockHost.data() +
                                  iSize * (numVecBlock - eigVecStartId) +
                                  jSize - eigVecStartId);
                          }
                      }

              } // space loop

            // for (size_type iSize = 0; iSize < vecLocalSize; iSize++)
            //   memoryTransfer.copy(numEigVecInBatch,
            //                       X.data() + iSize * lda +
            //                         baseColOffset + eigVecStartId,
            //                       subspaceBatchIn->data() +
            //                         numEigVecInBatch * iSize);

            MultiVectorOps::copyFromBatch<ValueType,
                                          ValueTypeOperand,
                                          memorySpace>(
              *subspaceBatchIn,
              eigVecStartId,
              numEigVecInBatch,
              X,
              linAlgOpContext);
          }
      }


      // Copies numVecBatch orbitals per space from copyFromVec → copyToVec.
      // Per space s: src column = srcBase + s*srcSpaceStride,
      //             dst column = dstBase + s*dstSpaceStride.
      // All numSpaces batches launched concurrently via varBatchedStridedBlockCopy.
      // Two-type template mirrors varBatchedStridedBlockCopy<VT1,VT2,MS>.
      template <typename ValueType1,
                typename ValueType2,
                utils::MemorySpace memorySpace>
      static void
      copyBatchImpl(const ValueType1 *            copyFromVec,
                    size_type                     srcLeadingDim,
                    size_type                     srcSpaceStride,
                    size_type                     srcBase,
                    ValueType2 *                  copyToVec,
                    size_type                     dstLeadingDim,
                    size_type                     dstSpaceStride,
                    size_type                     dstBase,
                    size_type                     vecSize,
                    size_type                     numSpaces,
                    size_type                     numVecBatch,
                    LinAlgOpContext<memorySpace> &context)
      {
        const size_type        numBatch = numSpaces;
        std::vector<size_type> strideSrc(numBatch, 0);
        std::vector<size_type> strideDst(numBatch, 0);
        std::vector<size_type> vecSizeArr(numBatch, vecSize);
        std::vector<size_type> numVecArr(numBatch, numVecBatch);
        std::vector<size_type> srcLeadingDimArr(numBatch, srcLeadingDim);
        std::vector<size_type> srcBlockStartIdArr(numBatch);
        std::vector<size_type> dstLeadingDimArr(numBatch, dstLeadingDim);
        std::vector<size_type> dstBlockStartIdArr(numBatch);

        for (size_type s = 0; s < numBatch; ++s)
          {
            srcBlockStartIdArr[s] = srcBase + s * srcSpaceStride;
            dstBlockStartIdArr[s] = dstBase + s * dstSpaceStride;
          }

        blasLapack::varBatchedStridedBlockCopy<ValueType1,
                                               ValueType2,
                                               memorySpace>(
          numBatch,
          strideSrc.data(),
          strideDst.data(),
          vecSizeArr.data(),
          numVecArr.data(),
          srcLeadingDimArr.data(),
          srcBlockStartIdArr.data(),
          dstLeadingDimArr.data(),
          dstBlockStartIdArr.data(),
          copyFromVec,
          copyToVec,
          context);
      }


      // Mirrors elpaScalaOpInternal::subspaceRotation.
      // spaceBlocked — true for MultiVectorProductSpaceBlocked (collinear);
      //               false for MultiVectorProductSpace (coupled/unpolarized).
      // rotationMatPars — pointer to first ScaLAPACKMatrix; size numSpaces for
      //                   blocked (Qs.data()), size 1 for coupled (&Q).
      // M / mpiCommDomain / linAlgOpContext are derived from X internally.
      // stridedBlockCopy replaces copyValueType1ArrToValueType2Arr so that
      // lda > numVecBlock is handled correctly (blocked case).
      template <typename ValueType, utils::MemorySpace memorySpace>
      static void
      rotateImpl(
        MultiVectorProductSpace<ValueType, memorySpace> &X,
        const bool                                       spaceBlocked,
        const std::shared_ptr<const ProcessGrid> &       processGrid,
        const ScaLAPACKMatrix<ValueType> *               rotationMatPars,
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

        const size_type numSpaces    = X.numSpaces();
        const size_type numVecPerSp  = X.numVectorsPerSpace();
        const size_type lda          = X.numVectors();
        const size_type numSpaceLoops = spaceBlocked ? numSpaces : 1;
        const size_type numVecBlock  = spaceBlocked ? numVecPerSp : lda;

        size_type maxNumLocalDofs = 0;
        utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
          &M,
          &maxNumLocalDofs,
          1,
          utils::mpi::Types<size_type>::getMPIDatatype(),
          utils::mpi::MPIMax,
          mpiCommDomain);

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

        for (size_type s = 0; s < numSpaceLoops; ++s)
          {
            const size_type baseColOffset = spaceBlocked ? s * numVecPerSp : 0;

            std::unordered_map<size_type, size_type> globalToLocalColumnIdMap;
            std::unordered_map<size_type, size_type> globalToLocalRowIdMap;
            elpaScalaOpInternal::createGlobalToLocalIdMapsScaLAPACKMat(
              processGrid,
              rotationMatPars[s],
              globalToLocalRowIdMap,
              globalToLocalColumnIdMap);

            if (allowFullCPUMemSubspaceRot)
              rotationMatBlockHost.setValue(ValueType(0));

            for (size_type idof = 0; idof < maxNumLocalDofs;
                 idof += dofsBlockSize)
              {
                // Correct block dimensions if block "goes off edge of" the matrix
                size_type BDof = 0;
                if (M >= idof)
                  BDof = std::min(dofsBlockSize, M - idof);

                for (size_type jvec = 0; jvec < numVecBlock;
                     jvec += vectorsBlockSize)
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
                                                               size_type>::
                                              iterator it =
                                                globalToLocalColumnIdMap.find(
                                                  j + jvec);
                                            if (it !=
                                                globalToLocalColumnIdMap.end())
                                              *(rotationMatBlockHost.begin() +
                                                jvec * numVecBlock +
                                                i * BVec + j) =
                                                rotationMatPars[s].local_el(
                                                  localRowId, it->second);
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
                                                               size_type>::
                                              iterator it =
                                                globalToLocalRowIdMap.find(
                                                  j + jvec);
                                            if (it !=
                                                globalToLocalRowIdMap.end())
                                              *(rotationMatBlockHost.begin() +
                                                jvec * numVecBlock +
                                                i * BVec + j) =
                                                rotationMatPars[s].local_el(
                                                  it->second, localColumnId);
                                          }
                                      }
                              }

                            utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
                              utils::mpi::MPIInPlace,
                              rotationMatBlockHost.begin() +
                                jvec * numVecBlock,
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
                                                           size_type>::iterator
                                          it =
                                            globalToLocalColumnIdMap.find(j +
                                                                          jvec);
                                        if (it !=
                                            globalToLocalColumnIdMap.end())
                                          *(rotationMatBlockHost.begin() +
                                            i * BVec + j) =
                                            rotationMatPars[s].local_el(
                                              localRowId, it->second);
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
                                            i * BVec + j) =
                                            rotationMatPars[s].local_el(
                                              it->second, localColumnId);
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
          } // space loop
      }

    } // namespace multiVectorOpsInternal


    // -------------------------------------------------------------------------
    // project — Blocked (collinear, MultiVectorProductSpaceBlocked)
    //
    // Loops over S spaces.  For space-s the active columns in the flat
    // data buffer are [s*N, s*N+N) with source lda = S*N.
    // Each space produces an independent N×N ScaLAPACK matrix Ps[s].
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

      const std::shared_ptr<const ProcessGrid> processGrid =
        elpa.getProcessGridDftefeScalaWrapper();

      multiVectorOpsInternal::projectImpl<ValueTypeOperator,
                                          ValueTypeOperand,
                                          memorySpace>(
        X,
        true,
        processGrid,
        Ps.data(),
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
    // the existing single-space N×N path (lda == numVecBlock).
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

      const std::shared_ptr<const ProcessGrid> processGrid =
        elpa.getProcessGridDftefeScalaWrapper();

      multiVectorOpsInternal::projectImpl<ValueTypeOperator,
                                          ValueTypeOperand,
                                          memorySpace>(
        X,
        false,
        processGrid,
        &P,
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
    // Loops over S spaces.  For space-s the active columns in the flat
    // data buffer are [s*N, s*N+N) with source lda = S*N.
    // Each space is rotated independently: Xs <- Xs * Qs[s].
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

      const std::shared_ptr<const ProcessGrid> processGrid =
        elpa.getProcessGridDftefeScalaWrapper();

      multiVectorOpsInternal::rotateImpl<ValueType, memorySpace>(
        X,
        true,
        processGrid,
        Qs.data(),
        RayleighRitzDefaults::SUBSPACE_ROT_DOF_BATCH,
        RayleighRitzDefaults::WAVE_FN_BATCH);
    }


    // -------------------------------------------------------------------------
    // rotate — Coupled (unpolarized S=1 / non-collinear S=2,
    //           MultiVectorProductSpace)
    //
    // Rotates all S*N columns as a single unit: X <- X * Q.
    // S=1 degenerates exactly to the existing single-space N×N rotation path.
    // -------------------------------------------------------------------------
    template <typename ValueType, utils::MemorySpace memorySpace>
    void
    MultiVectorOps::rotate(
      MultiVectorProductSpace<ValueType, memorySpace> &X,
      const ScaLAPACKMatrix<ValueType> &               Q,
      const ElpaScalapackManager &                     elpa)
    {
      const std::shared_ptr<const ProcessGrid> processGrid =
        elpa.getProcessGridDftefeScalaWrapper();

      multiVectorOpsInternal::rotateImpl<ValueType, memorySpace>(
        X,
        false,
        processGrid,
        &Q,
        RayleighRitzDefaults::SUBSPACE_ROT_DOF_BATCH,
        RayleighRitzDefaults::WAVE_FN_BATCH);
    }

    // -------------------------------------------------------------------------
    // copyToBatch — Coupled (unpolarized S=1 / non-collinear S=2,
    //               MultiVectorProductSpace)
    // -------------------------------------------------------------------------
    template <typename ValueType1,
              typename ValueType2,
              utils::MemorySpace memorySpace>
    void
    MultiVectorOps::copyToBatch(
      const MultiVectorProductSpace<ValueType1, memorySpace> &X,
      size_type                                                srcStart,
      size_type                                                numVecBatch,
      MultiVector<ValueType2, memorySpace> &                  Xbatch,
      LinAlgOpContext<memorySpace> &                          context)
    {
      const size_type numSpaces   = X.numSpaces();
      const size_type numVecPerSp = X.numVectorsPerSpace();
      const size_type vecSize     = X.localSize();
      multiVectorOpsInternal::copyBatchImpl<ValueType1, ValueType2, memorySpace>(
        X.data(),
        numSpaces * numVecPerSp,
        numVecPerSp,
        srcStart,
        Xbatch.data(),
        numSpaces * numVecBatch,
        numVecBatch,
        0,
        vecSize,
        numSpaces,
        numVecBatch,
        context);
    }


    // -------------------------------------------------------------------------
    // copyFromBatch — Coupled (unpolarized S=1 / non-collinear S=2,
    //                 MultiVectorProductSpace)
    // -------------------------------------------------------------------------
    template <typename ValueType1,
              typename ValueType2,
              utils::MemorySpace memorySpace>
    void
    MultiVectorOps::copyFromBatch(
      const MultiVector<ValueType1, memorySpace> &       Ybatch,
      size_type                                          dstStart,
      size_type                                          numVecBatch,
      MultiVectorProductSpace<ValueType2, memorySpace> &Y,
      LinAlgOpContext<memorySpace> &                     context)
    {
      const size_type numSpaces   = Y.numSpaces();
      const size_type numVecPerSp = Y.numVectorsPerSpace();
      const size_type vecSize     = Y.localSize();
      multiVectorOpsInternal::copyBatchImpl<ValueType1, ValueType2, memorySpace>(
        Ybatch.data(),
        numSpaces * numVecBatch,
        numVecBatch,
        0,
        Y.data(),
        numSpaces * numVecPerSp,
        numVecPerSp,
        dstStart,
        vecSize,
        numSpaces,
        numVecBatch,
        context);
    }

  } // namespace linearAlgebra
} // namespace dftefe
