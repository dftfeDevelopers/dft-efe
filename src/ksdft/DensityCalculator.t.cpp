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
#include <linearAlgebra/BlasLapack.h>
#include <ksdft/DensityCalculatorKernels.h>
#include <ksdft/Defaults.h>
namespace dftefe
{
  namespace ksdft
  {
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
      , d_psiBatchQuad(nullptr)
      , d_rhoBatch(nullptr)
      , d_rhoMemspace(nullptr)
      , d_gradPsiBatchQuad(nullptr)
      , d_gradRhoBatch(nullptr)
      , d_gradRhoMemspace(nullptr)
      , d_psiBatch(nullptr)
      , d_psiBatchSmall(nullptr)
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
      if (d_rhoBatch != nullptr)
        {
          delete d_rhoBatch;
          d_rhoBatch = nullptr;
        }
      if (d_rhoMemspace != nullptr)
        {
          delete d_rhoMemspace;
          d_rhoMemspace = nullptr;
        }
      if (d_gradPsiBatchQuad != nullptr)
        {
          delete d_gradPsiBatchQuad;
          d_gradPsiBatchQuad = nullptr;
        }
      if (d_gradRhoBatch != nullptr)
        {
          delete d_gradRhoBatch;
          d_gradRhoBatch = nullptr;
        }
      if (d_gradRhoMemspace != nullptr)
        {
          delete d_gradRhoMemspace;
          d_gradRhoMemspace = nullptr;
        }
      if (d_psiBatch != nullptr)
        {
          delete d_psiBatch;
          d_psiBatch = nullptr;
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
      d_feBMPsi              = &feBMPsi;
      d_batchSizeSmall       = ksdft::MaxSizeDefaults::SIZE_TYPE_MAX;
      d_quadRuleContainer    = feBasisDataStorage->getQuadratureRuleContainer();
      d_numLocallyOwnedCells = feBMPsi.nLocallyOwnedCells();

      std::vector<size_type> numCellQuad(d_numLocallyOwnedCells, 0);
      for (size_type iCell = 0; iCell < d_numLocallyOwnedCells; ++iCell)
        numCellQuad[iCell] = d_quadRuleContainer->nCellQuadraturePoints(iCell);
      const size_type maxQuadInCell =
        *std::max_element(numCellQuad.begin(), numCellQuad.end());

      d_psiBatchQuad = new dftefe::utils::MemoryStorage<ValueType, memorySpace>(
        d_waveFuncBatchSize * d_cellBlockSize * maxQuadInCell);

      d_modPsiSqBatchQuad = dftefe::utils::MemoryStorage<RealType, memorySpace>(
        d_waveFuncBatchSize * d_cellBlockSize * maxQuadInCell);

      d_occupationInBatch = dftefe::utils::MemoryStorage<RealType, memorySpace>(
        d_waveFuncBatchSize);

      d_rhoMemspace =
        new quadrature::QuadratureValuesContainer<RealType, memorySpace>(
          d_quadRuleContainer, 1);

      d_rhoBatch = new dftefe::utils::MemoryStorage<RealType, memorySpace>(
        d_cellBlockSize * maxQuadInCell);

      d_gradPsiBatchQuad =
        new dftefe::utils::MemoryStorage<ValueType, memorySpace>(
          d_waveFuncBatchSize * d_cellBlockSize * maxQuadInCell * dim);

      d_psiGradPsiBatch = dftefe::utils::MemoryStorage<RealType, memorySpace>(
        d_waveFuncBatchSize * d_cellBlockSize * maxQuadInCell * dim);

      d_gradRhoMemspace =
        new quadrature::QuadratureValuesContainer<RealType, memorySpace>(
          d_quadRuleContainer, dim);

      d_gradRhoBatch = new dftefe::utils::MemoryStorage<RealType, memorySpace>(
        d_cellBlockSize * maxQuadInCell * dim);

      d_psiBatch =
        new linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>(
          feBMPsi.getMPIPatternP2P(),
          d_linAlgOpContext,
          d_waveFuncBatchSize,
          ValueTypeBasisCoeff());

      d_feBasisOp =
        std::make_shared<basis::FEBasisOperations<ValueTypeBasisCoeff,
                                                  ValueTypeBasisData,
                                                  memorySpace,
                                                  dim>>(feBasisDataStorage,
                                                        d_cellBlockSize,
                                                        d_waveFuncBatchSize);
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
          &waveFunc,
        quadrature::QuadratureValuesContainer<RealType, memorySpaceHost> &rho,
        quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>
          &        gradRho,
        const bool computeGrad)
    {
      d_rhoMemspace->setValue((RealType)0);
      if (computeGrad)
        d_gradRhoMemspace->setValue((RealType)0);

      utils::MemoryTransfer<memorySpace, memorySpaceHost> memoryTransferM2H;
      utils::MemoryTransfer<memorySpaceHost, memorySpace> memoryTransferH2M;

      utils::MemoryStorage<RealType, memorySpace> occMemspace(
        occupation.size());
      memoryTransferH2M.copy(occupation.size(),
                             occMemspace.data(),
                             occupation.data());

      for (size_type cellStartId = 0; cellStartId < d_numLocallyOwnedCells;
           cellStartId += d_cellBlockSize)
        {
          const size_type cellEndId =
            std::min(cellStartId + d_cellBlockSize, d_numLocallyOwnedCells);
          const std::pair<size_type, size_type> cellRange(cellStartId,
                                                          cellEndId);

          size_type numQuadInBlock = 0;
          for (size_type iCell = cellStartId; iCell < cellEndId; iCell++)
            numQuadInBlock += d_quadRuleContainer->nCellQuadraturePoints(iCell);

          for (size_type psiStartId = 0;
               psiStartId < waveFunc.getNumberComponents();
               psiStartId += d_waveFuncBatchSize)
            {
              const size_type psiEndId =
                std::min(psiStartId + d_waveFuncBatchSize,
                         waveFunc.getNumberComponents());
              const size_type numPsiInBatch = psiEndId - psiStartId;

              linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
                *psiBatchInterim = nullptr;

              if (numPsiInBatch == d_waveFuncBatchSize)
                {
                  psiBatchInterim = d_psiBatch;
                }
              else if (numPsiInBatch == d_batchSizeSmall)
                {
                  psiBatchInterim = d_psiBatchSmall;
                }
              else
                {
                  d_batchSizeSmall = numPsiInBatch;
                  d_psiBatchSmall =
                    new linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                                   memorySpace>(
                      waveFunc.getMPIPatternP2P(),
                      d_linAlgOpContext,
                      numPsiInBatch,
                      ValueTypeBasisCoeff());
                  psiBatchInterim = d_psiBatchSmall;
                }

              linearAlgebra::blasLapack::copyValueType1ArrToValueType2Arr(
                numPsiInBatch,
                occMemspace.data() + psiStartId,
                d_occupationInBatch.data(),
                *waveFunc.getLinAlgOpContext());

              linearAlgebra::blasLapack::stridedBlockCopy(
                waveFunc.localSize(),
                numPsiInBatch,
                waveFunc.getNumberComponents(),
                psiStartId,
                numPsiInBatch,
                0,
                waveFunc.data(),
                psiBatchInterim->data(),
                *waveFunc.getLinAlgOpContext());

              // Basis data for cellRange is cached across psi-batch iterations
              d_feBasisOp->interpolate(*psiBatchInterim,
                                       *d_feBMPsi,
                                       cellRange,
                                       d_psiBatchQuad->data());

              DensityCalculatorKernels<ValueType, RealType, memorySpace, dim>::
                computeRhoInBatch(numPsiInBatch,
                                  cellRange,
                                  d_occupationInBatch.data(),
                                  d_psiBatchQuad->data(),
                                  d_modPsiSqBatchQuad.data(),
                                  d_quadRuleContainer,
                                  d_rhoBatch->data(),
                                  *d_linAlgOpContext);

              linearAlgebra::blasLapack::axpy<RealType, RealType, memorySpace>(
                numQuadInBlock,
                (RealType)1.0,
                d_rhoBatch->data(),
                1,
                d_rhoMemspace->begin(cellStartId),
                1,
                *d_linAlgOpContext);

              if (computeGrad)
                {
                  d_feBasisOp->interpolateWithBasisGradient(
                    *psiBatchInterim,
                    *d_feBMPsi,
                    cellRange,
                    d_gradPsiBatchQuad->data());

                  DensityCalculatorKernels<
                    ValueType,
                    RealType,
                    memorySpace,
                    dim>::computeGradRhoInBatch(numPsiInBatch,
                                                cellRange,
                                                d_occupationInBatch.data(),
                                                d_psiBatchQuad->data(),
                                                d_gradPsiBatchQuad->data(),
                                                d_psiGradPsiBatch.data(),
                                                d_quadRuleContainer,
                                                d_gradRhoBatch->data(),
                                                *d_linAlgOpContext);

                  linearAlgebra::blasLapack::
                    axpy<RealType, RealType, memorySpace>(
                      numQuadInBlock * dim,
                      (RealType)1.0,
                      d_gradRhoBatch->data(),
                      1,
                      d_gradRhoMemspace->begin(cellStartId),
                      1,
                      *d_linAlgOpContext);
                }
            }
        }

      memoryTransferM2H.copy(d_rhoMemspace->nEntries(),
                             rho.begin(),
                             d_rhoMemspace->begin());

      if (computeGrad)
        memoryTransferM2H.copy(d_gradRhoMemspace->nEntries(),
                               gradRho.begin(),
                               d_gradRhoMemspace->begin());
    }
  } // end of namespace ksdft
} // end of namespace dftefe
