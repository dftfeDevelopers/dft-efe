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
#include <ksdft/DensityCalculatorKernels.h>

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
      , d_psiBatch(nullptr)
      , d_psiBatchSmallQuad(nullptr)
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

      d_rhoMemspace =
        new quadrature::QuadratureValuesContainer<RealType, memorySpace>(
          d_quadRuleContainer, 1);

      d_rhoBatch =
        new quadrature::QuadratureValuesContainer<RealType, memorySpace>(
          d_quadRuleContainer, 1);

      d_psiBatch =
        new linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>(
          feBMPsi.getMPIPatternP2P(),
          d_linAlgOpContext,
          d_waveFuncBatchSize,
          ValueTypeBasisCoeff());

      if constexpr (memorySpace == utils::MemorySpace::DEVICE)
        d_modPsiSqBatchQuad = quadrature::QuadratureValuesContainer<RealType, memorySpace>(
            d_quadRuleContainer, d_waveFuncBatchSize);
      //-------------------------------------------------
      // Reinit FEBasisOp with different maxcelltimesnumvecs
      // for the case waveFnInBatch<d_waveFuncBatchSize

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
          &                                                           waveFunc,
        quadrature::QuadratureValuesContainer<RealType, memorySpaceHost> &rho)
    {
      d_rhoMemspace->setValue((RealType)0);

      utils::MemoryTransfer<memorySpace, memorySpace> memoryTransfer;

      utils::MemoryTransfer<memorySpace, memorySpaceHost> memoryTransferM2H;
      utils::MemoryTransfer<memorySpaceHost, memorySpace> memoryTransferH2M;

      utils::MemoryStorage<RealType, memorySpace> occMemspace(occupation.size());
      memoryTransferH2M.copy(occupation.size(),
                          occMemspace.data(),
                          occupation.data()); 

      for (size_type psiStartId = 0;
           psiStartId < waveFunc.getNumberComponents();
           psiStartId += d_waveFuncBatchSize)
        {
          const size_type psiEndId = std::min(psiStartId + d_waveFuncBatchSize,
                                              waveFunc.getNumberComponents());
          const size_type numPsiInBatch = psiEndId - psiStartId;

          utils::MemoryStorage<RealType, memorySpace> occupationInBatch(numPsiInBatch, 0);

          std::copy(occMemspace.data() + psiStartId,
                    occMemspace.data() + psiEndId,
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

              d_feBasisOp->reinit(d_cellBlockSize, d_waveFuncBatchSize);
              d_feBasisOp->interpolate(*d_psiBatch,
                                       *d_feBMPsi,
                                       *d_psiBatchQuad);

              DensityCalculatorKernels<ValueType, RealType, memorySpace>::computeRhoInBatch(
                  occupationInBatch,
                  *d_psiBatchQuad,
                  d_modPsiSqBatchQuad,
                  d_quadRuleContainer,
                  *d_rhoBatch,
                  *d_linAlgOpContext);

              // do add
              quadrature::add((RealType)1.0,
                              *d_rhoBatch,
                              (RealType)1.0,
                              *d_rhoMemspace,
                              *d_rhoMemspace,
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

              d_feBasisOp->reinit(d_cellBlockSize, d_batchSizeSmall);
              d_feBasisOp->interpolate(*d_psiBatchSmall,
                                       *d_feBMPsi,
                                       *d_psiBatchSmallQuad);

              DensityCalculatorKernels<ValueType, RealType, memorySpace>::computeRhoInBatch(
                  occupationInBatch,
                  *d_psiBatchSmallQuad,
                  d_modPsiSqBatchSmallQuad,
                  d_quadRuleContainer,
                  *d_rhoBatch,
                  *d_linAlgOpContext);

              // do add
              quadrature::add((RealType)1.0,
                              *d_rhoBatch,
                              (RealType)1.0,
                              *d_rhoMemspace,
                              *d_rhoMemspace,
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

              d_psiBatchSmall =
                new linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                               memorySpace>(
                  waveFunc.getMPIPatternP2P(),
                  d_linAlgOpContext,
                  numPsiInBatch,
                  ValueTypeBasisCoeff());

              if constexpr (memorySpace == utils::MemorySpace::DEVICE)
                d_modPsiSqBatchQuad = quadrature::QuadratureValuesContainer<RealType, memorySpace>(
                    d_quadRuleContainer, numPsiInBatch);

              for (size_type iSize = 0; iSize < waveFunc.localSize(); iSize++)
                memoryTransfer.copy(numPsiInBatch,
                                    d_psiBatchSmall->data() +
                                      numPsiInBatch * iSize,
                                    waveFunc.data() +
                                      iSize * waveFunc.getNumberComponents() +
                                      psiStartId);

              d_feBasisOp->reinit(d_cellBlockSize, d_batchSizeSmall);
              d_feBasisOp->interpolate(*d_psiBatchSmall,
                                       *d_feBMPsi,
                                       *d_psiBatchSmallQuad);

              DensityCalculatorKernels<ValueType, RealType, memorySpace>::computeRhoInBatch(
                  occupationInBatch,
                  *d_psiBatchSmallQuad,
                  d_modPsiSqBatchSmallQuad,
                  d_quadRuleContainer,
                  *d_rhoBatch,
                  *d_linAlgOpContext);

              // do add
              quadrature::add((RealType)1.0,
                              *d_rhoBatch,
                              (RealType)1.0,
                              *d_rhoMemspace,
                              *d_rhoMemspace,
                              *d_linAlgOpContext);
            }
        }
      memoryTransferM2H.copy(d_rhoMemspace->nEntries(),
                            rho.data(),
                            d_rhoMemspace->data()); 
    }
  } // end of namespace ksdft
} // end of namespace dftefe
