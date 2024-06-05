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

#include <utils/DataTypeOverloads.h>

namespace dftefe
{
  namespace ksdft
  {
    namespace DensityCalculatorInternal
    {
      template <typename ValueType,
                typename RealType,
                utils::MemorySpace memorySpace>
      void
      computeRhoInBatch(
        const std::vector<RealType> &occupationInBatch,
        quadrature::QuadratureValuesContainer<ValueType, memorySpace>
          &psiBatchQuad,
        quadrature::QuadratureValuesContainer<RealType, memorySpace>
          &psiModSqBatchQuad,
        std::shared_ptr<const quadrature::QuadratureRuleContainer>
          quadRuleContainer,
        quadrature::QuadratureValuesContainer<RealType, memorySpace> &rhoBatch,
        size_type                                    cellBlockSize,
        size_type                                    numLocallyOwnedCells,
        linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext)
      {
        size_type numPsiInBatch = occupationInBatch.size();
        // hadamard for psi^C psi = mod psi^2
        linearAlgebra::blasLapack::
          hadamardProduct<ValueType, ValueType, memorySpace>(
            psiBatchQuad.nEntries(),
            psiBatchQuad.begin(),
            psiBatchQuad.begin(),
            linearAlgebra::blasLapack::ScalarOp::Conj,
            linearAlgebra::blasLapack::ScalarOp::Identity,
            psiBatchQuad.begin(),
            linAlgOpContext);

        /*----------- TODO : Optimize this -------------------------------*/
        // convert to psiBatchQuad to realType and multiply by 2
        for (size_type iCell = 0; iCell < psiBatchQuad.nCells(); iCell++)
          {
            for (size_type iComp = 0; iComp < numPsiInBatch; iComp++)
              {
                std::vector<ValueType> a(
                  quadRuleContainer->nCellQuadraturePoints(iCell));
                std::vector<RealType> b(
                  quadRuleContainer->nCellQuadraturePoints(iCell));
                psiBatchQuad
                  .template getCellQuadValues<utils::MemorySpace::HOST>(
                    iCell, iComp, a.data());
                for (size_type i = 0; i < b.size(); i++)
                  b[i] = 2.0 * utils::realPart<RealType>(a[i]);
                psiModSqBatchQuad
                  .template setCellQuadValues<utils::MemorySpace::HOST>(
                    iCell, iComp, b.data());
              }
          }

        // gemm for fi * mod psi^2

        size_type AStartOffset = 0;
        size_type CStartOffset = 0;
        for (size_type cellStartId = 0; cellStartId < numLocallyOwnedCells;
             cellStartId += cellBlockSize)
          {
            const size_type cellEndId =
              std::min(cellStartId + cellBlockSize, numLocallyOwnedCells);
            const size_type numCellsInBlock = cellEndId - cellStartId;

            std::vector<linearAlgebra::blasLapack::Op> transA(
              numCellsInBlock, linearAlgebra::blasLapack::Op::NoTrans);
            std::vector<linearAlgebra::blasLapack::Op> transB(
              numCellsInBlock, linearAlgebra::blasLapack::Op::NoTrans);
            std::vector<size_type> mSizesTmp(numCellsInBlock, 0);
            std::vector<size_type> nSizesTmp(numCellsInBlock, 0);
            std::vector<size_type> kSizesTmp(numCellsInBlock, 0);
            std::vector<size_type> ldaSizesTmp(numCellsInBlock, 0);
            std::vector<size_type> ldbSizesTmp(numCellsInBlock, 0);
            std::vector<size_type> ldcSizesTmp(numCellsInBlock, 0);
            std::vector<size_type> strideATmp(numCellsInBlock, 0);
            std::vector<size_type> strideBTmp(numCellsInBlock, 0);
            std::vector<size_type> strideCTmp(numCellsInBlock, 0);

            for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
              {
                const size_type cellId = cellStartId + iCell;
                mSizesTmp[iCell] =
                  quadRuleContainer->nCellQuadraturePoints(cellId);
                nSizesTmp[iCell]   = 1;
                kSizesTmp[iCell]   = numPsiInBatch;
                ldaSizesTmp[iCell] = mSizesTmp[iCell];
                ldbSizesTmp[iCell] = kSizesTmp[iCell];
                ldcSizesTmp[iCell] = mSizesTmp[iCell];
                strideBTmp[iCell]  = 0;
                strideCTmp[iCell]  = mSizesTmp[iCell] * nSizesTmp[iCell];
                strideATmp[iCell]  = mSizesTmp[iCell] * kSizesTmp[iCell];
              }

            utils::MemoryStorage<size_type, memorySpace> mSizes(
              numCellsInBlock);
            utils::MemoryStorage<size_type, memorySpace> nSizes(
              numCellsInBlock);
            utils::MemoryStorage<size_type, memorySpace> kSizes(
              numCellsInBlock);
            utils::MemoryStorage<size_type, memorySpace> ldaSizes(
              numCellsInBlock);
            utils::MemoryStorage<size_type, memorySpace> ldbSizes(
              numCellsInBlock);
            utils::MemoryStorage<size_type, memorySpace> ldcSizes(
              numCellsInBlock);
            utils::MemoryStorage<size_type, memorySpace> strideA(
              numCellsInBlock);
            utils::MemoryStorage<size_type, memorySpace> strideB(
              numCellsInBlock);
            utils::MemoryStorage<size_type, memorySpace> strideC(
              numCellsInBlock);

            utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>
              memoryTransfer;

            memoryTransfer.copy(numCellsInBlock,
                                mSizes.data(),
                                mSizesTmp.data());
            memoryTransfer.copy(numCellsInBlock,
                                nSizes.data(),
                                nSizesTmp.data());
            memoryTransfer.copy(numCellsInBlock,
                                kSizes.data(),
                                kSizesTmp.data());
            memoryTransfer.copy(numCellsInBlock,
                                ldaSizes.data(),
                                ldaSizesTmp.data());
            memoryTransfer.copy(numCellsInBlock,
                                ldbSizes.data(),
                                ldbSizesTmp.data());
            memoryTransfer.copy(numCellsInBlock,
                                ldcSizes.data(),
                                ldcSizesTmp.data());
            memoryTransfer.copy(numCellsInBlock,
                                strideA.data(),
                                strideATmp.data());
            memoryTransfer.copy(numCellsInBlock,
                                strideB.data(),
                                strideBTmp.data());
            memoryTransfer.copy(numCellsInBlock,
                                strideC.data(),
                                strideCTmp.data());

            // Create occupancy in memspace
            utils::MemoryStorage<RealType, memorySpace>
              occupationInBatchMemspace(occupationInBatch.size());
            memoryTransfer.copy(occupationInBatch.size(),
                                occupationInBatchMemspace.data(),
                                occupationInBatch.data());

            RealType alpha = 1.0;
            RealType beta  = 0.0;

            const RealType *A = psiModSqBatchQuad.begin() + AStartOffset;

            RealType *C = rhoBatch.begin() + CStartOffset;

            linearAlgebra::blasLapack::
              gemmStridedVarBatched<RealType, RealType, memorySpace>(
                linearAlgebra::blasLapack::Layout::ColMajor,
                numCellsInBlock,
                transA.data(),
                transB.data(),
                strideA.data(),
                strideB.data(),
                strideC.data(),
                mSizes.data(),
                nSizes.data(),
                kSizes.data(),
                alpha,
                A,
                ldaSizes.data(),
                occupationInBatchMemspace.data(),
                ldbSizes.data(),
                beta,
                C,
                ldcSizes.data(),
                linAlgOpContext);

            for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
              {
                AStartOffset += mSizesTmp[iCell] * kSizesTmp[iCell];
                CStartOffset += mSizesTmp[iCell] * nSizesTmp[iCell];
              }
          }
      }
    } // namespace DensityCalculatorInternal

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    DensityCalculator<ValueTypeBasisData,
                      ValueTypeBasisCoeff,
                      memorySpace,
                      dim>::
      DensityCalculator(
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
                                          feBasisDataStorage,
        const basis::FEBasisManager<ValueTypeBasisCoeff,
                                    ValueTypeBasisData,
                                    memorySpace,
                                    dim> &feBMPsi,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
                        linAlgOpContext,
        const size_type cellBlockSize,
        const size_type waveFuncBatchSize)
      : d_linAlgOpContext(linAlgOpContext)
      , d_cellBlockSize(cellBlockSize)
      , d_waveFuncBatchSize(waveFuncBatchSize)
    {
      reinit(feBasisDataStorage, feBMPsi);
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    DensityCalculator<ValueTypeBasisData,
                      ValueTypeBasisCoeff,
                      memorySpace,
                      dim>::~DensityCalculator()
    {
      if (d_psiBatchQuad != nullptr)
        {
          delete d_psiBatchQuad;
          d_psiBatchQuad = nullptr;
        }
      if (d_psiModSqBatchQuad != nullptr)
        {
          delete d_psiModSqBatchQuad;
          d_psiModSqBatchQuad = nullptr;
        }
      if (d_rhoBatch != nullptr)
        {
          delete d_rhoBatch;
          d_rhoBatch = nullptr;
        }
      if (d_psiBatch != nullptr)
        {
          delete d_psiBatch;
          d_psiBatch = nullptr;
        }
      if (d_psiBatchSmallQuad != nullptr)
        {
          delete d_psiBatchSmallQuad;
          d_psiBatchSmallQuad = nullptr;
        }
      if (d_psiModSqBatchSmallQuad != nullptr)
        {
          delete d_psiModSqBatchSmallQuad;
          d_psiModSqBatchSmallQuad = nullptr;
        }
      if (d_psiBatchSmall != nullptr)
        {
          delete d_psiBatchSmall;
          d_psiBatchSmall = nullptr;
        }
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    DensityCalculator<ValueTypeBasisData,
                      ValueTypeBasisCoeff,
                      memorySpace,
                      dim>::
      reinit(std::shared_ptr<
               const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
                                               feBasisDataStorage,
             const basis::FEBasisManager<ValueTypeBasisCoeff,
                                         ValueTypeBasisData,
                                         memorySpace,
                                         dim> &feBMPsi)
    {
      d_feBMPsi        = &feBMPsi;
      d_batchSizeSmall = UINT_MAX;

      d_quadRuleContainer = feBasisDataStorage->getQuadratureRuleContainer();

      // 4 scratch spaces ---- can be optimized ------
      d_psiBatchQuad =
        new quadrature::QuadratureValuesContainer<ValueType, memorySpace>(
          d_quadRuleContainer, d_waveFuncBatchSize);

      d_psiModSqBatchQuad =
        new quadrature::QuadratureValuesContainer<RealType, memorySpace>(
          d_quadRuleContainer, d_waveFuncBatchSize);

      d_rhoBatch =
        new quadrature::QuadratureValuesContainer<RealType, memorySpace>(
          d_quadRuleContainer, 1);

      d_psiBatch =
        new linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>(
          feBMPsi.getMPIPatternP2P(),
          d_linAlgOpContext,
          d_waveFuncBatchSize,
          ValueTypeBasisCoeff());
      //-------------------------------------------------

      d_feBasisOp =
        std::make_shared<const basis::FEBasisOperations<ValueTypeBasisCoeff,
                                                        ValueTypeBasisData,
                                                        memorySpace,
                                                        dim>>(
          feBasisDataStorage, d_cellBlockSize);
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    DensityCalculator<ValueTypeBasisData,
                      ValueTypeBasisCoeff,
                      memorySpace,
                      dim>::
      computeRho(
        const std::vector<RealType> &occupation,
        const linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
          &                                                           waveFunc,
        quadrature::QuadratureValuesContainer<RealType, memorySpace> &rho)
    {
      rho.setValue((RealType)0);
      const size_type numLocallyOwnedCells = d_feBMPsi->nLocallyOwnedCells();

      utils::MemoryTransfer<memorySpace, memorySpace> memoryTransfer;

      for (size_type psiStartId = 0;
           psiStartId < waveFunc.getNumberComponents();
           psiStartId += d_waveFuncBatchSize)
        {
          const size_type psiEndId = std::min(psiStartId + d_waveFuncBatchSize,
                                              waveFunc.getNumberComponents());
          const size_type numPsiInBatch = psiEndId - psiStartId;

          std::vector<RealType> occupationInBatch(numPsiInBatch, 0);

          std::copy(occupation.data() + psiStartId,
                    occupation.data() + psiEndId,
                    occupationInBatch.begin());

          /*
           * Use scratch space for case where "numPsiInBatch <
           * d_waveFuncBatchSize". cases : if(n % nb1 == 0), if(n % nb1 ==
           * d_nb2) else ( init nb_2 )
           */

          if (numPsiInBatch % d_waveFuncBatchSize == 0)
            {
              for (size_type iSize = 0; iSize < waveFunc.localSize(); iSize++)
                memoryTransfer.copy(numPsiInBatch,
                                    d_psiBatch->data() + numPsiInBatch * iSize,
                                    waveFunc.data() +
                                      iSize * waveFunc.getNumberComponents() +
                                      psiStartId);

              d_feBasisOp->interpolate(*d_psiBatch,
                                       *d_feBMPsi,
                                       *d_psiBatchQuad);

              DensityCalculatorInternal::
                computeRhoInBatch<ValueType, RealType, memorySpace>(
                  occupationInBatch,
                  *d_psiBatchQuad,
                  *d_psiModSqBatchQuad,
                  d_quadRuleContainer,
                  *d_rhoBatch,
                  d_cellBlockSize,
                  numLocallyOwnedCells,
                  *d_linAlgOpContext);

              // do add
              quadrature::add((RealType)1.0,
                              *d_rhoBatch,
                              (RealType)1.0,
                              rho,
                              rho,
                              *d_linAlgOpContext);
            }
          else if (numPsiInBatch % d_waveFuncBatchSize == d_batchSizeSmall)
            {
              for (size_type iSize = 0; iSize < waveFunc.localSize(); iSize++)
                memoryTransfer.copy(numPsiInBatch,
                                    d_psiBatchSmall->data() +
                                      numPsiInBatch * iSize,
                                    waveFunc.data() +
                                      iSize * waveFunc.getNumberComponents() +
                                      psiStartId);

              d_feBasisOp->interpolate(*d_psiBatchSmall,
                                       *d_feBMPsi,
                                       *d_psiBatchSmallQuad);

              DensityCalculatorInternal::
                computeRhoInBatch<ValueType, RealType, memorySpace>(
                  occupationInBatch,
                  *d_psiBatchSmallQuad,
                  *d_psiModSqBatchSmallQuad,
                  d_quadRuleContainer,
                  *d_rhoBatch,
                  d_cellBlockSize,
                  numLocallyOwnedCells,
                  *d_linAlgOpContext);

              // do add
              quadrature::add((RealType)1.0,
                              *d_rhoBatch,
                              (RealType)1.0,
                              rho,
                              rho,
                              *d_linAlgOpContext);
            }
          // for the first iteration where batch size is not wavefnBatch,
          // else is executed and d_batchSizeSmall is initialized
          else
            {
              d_batchSizeSmall = numPsiInBatch;

              d_psiBatchSmallQuad =
                new quadrature::QuadratureValuesContainer<ValueType,
                                                          memorySpace>(
                  d_quadRuleContainer, numPsiInBatch);

              d_psiModSqBatchSmallQuad =
                new quadrature::QuadratureValuesContainer<ValueType,
                                                          memorySpace>(
                  d_quadRuleContainer, numPsiInBatch);

              d_psiBatchSmall =
                new linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                               memorySpace>(
                  waveFunc.getMPIPatternP2P(),
                  d_linAlgOpContext,
                  numPsiInBatch,
                  ValueTypeBasisCoeff());

              for (size_type iSize = 0; iSize < waveFunc.localSize(); iSize++)
                memoryTransfer.copy(numPsiInBatch,
                                    d_psiBatchSmall->data() +
                                      numPsiInBatch * iSize,
                                    waveFunc.data() +
                                      iSize * waveFunc.getNumberComponents() +
                                      psiStartId);

              d_feBasisOp->interpolate(*d_psiBatchSmall,
                                       *d_feBMPsi,
                                       *d_psiBatchSmallQuad);

              DensityCalculatorInternal::
                computeRhoInBatch<ValueType, RealType, memorySpace>(
                  occupationInBatch,
                  *d_psiBatchSmallQuad,
                  *d_psiModSqBatchSmallQuad,
                  d_quadRuleContainer,
                  *d_rhoBatch,
                  d_cellBlockSize,
                  numLocallyOwnedCells,
                  *d_linAlgOpContext);

              // do add
              quadrature::add((RealType)1.0,
                              *d_rhoBatch,
                              (RealType)1.0,
                              rho,
                              rho,
                              *d_linAlgOpContext);
            }
        }
    }
  } // end of namespace ksdft
} // end of namespace dftefe
