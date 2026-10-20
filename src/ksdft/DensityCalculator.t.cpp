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
#include <linearAlgebra/MultiVectorOps.h>
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
        const std::map<
          basis::kpoint::KPointId,
          std::shared_ptr<
            const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>>
          &feBasisDataStorage,
        const std::map<
          basis::kpoint::KPointId,
          std::shared_ptr<const basis::FEBasisManager<ValueTypeBasisCoeff,
                                                      ValueTypeBasisData,
                                                      memorySpace,
                                                      dim>>> &feBMPsi,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
                                                            linAlgOpContext,
        const size_type                                     cellBlockSize,
        const size_type                                     waveFuncBatchSize,
        const std::map<basis::kpoint::KPointId, basis::kpoint::KPointData> &kPoints)
      : d_linAlgOpContext(linAlgOpContext)
      , d_cellBlockSize(cellBlockSize)
      , d_waveFuncBatchSize(waveFuncBatchSize)
      , d_psiBatchQuad(nullptr)
      , d_rhoBatch(nullptr)
      , d_gradPsiBatchQuad(nullptr)
      , d_gradRhoBatch(nullptr)
      , d_psiBatch(nullptr)
      , d_psiBatchSmall(nullptr)
    {
      // ascending key order, so the local index order is deterministic
      utils::throwException<utils::InvalidArgument>(
        !kPoints.empty(),
        "DensityCalculator: the k-point map is empty. A non-periodic caller "
        "passes basis::kpoint::gammaIfEmpty({}, dim).");
      d_kPointIds.resize(kPoints.size(), (basis::kpoint::KPointId)0);
      d_kPointWeights.resize(kPoints.size(), 0.0);
      size_type iKPtLocal = 0;
      for (const auto &kv : kPoints)
        {
          d_kPointIds[iKPtLocal]     = kv.first;
          d_kPointWeights[iKPtLocal] = kv.second.weight;
          iKPtLocal++;
        }

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
      d_rhoMemspace.clear();
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
      d_gradRhoMemspace.clear();
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
      reinit(
        const std::map<
          basis::kpoint::KPointId,
          std::shared_ptr<
            const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>>
          &feBasisDataStorage,
        const std::map<
          basis::kpoint::KPointId,
          std::shared_ptr<const basis::FEBasisManager<ValueTypeBasisCoeff,
                                                      ValueTypeBasisData,
                                                      memorySpace,
                                                      dim>>> &feBMPsi)
    {
      const size_type nKLocal = d_kPointIds.size();
      utils::throwException<utils::InvalidArgument>(
        feBasisDataStorage.size() == nKLocal && feBMPsi.size() == nKLocal,
        "DensityCalculator::reinit: the basis-storage and basis-manager maps "
        "do not match the k-point set this object was built with.");
      d_feBasisDataStorage.resize(nKLocal, nullptr);
      d_feBMPsi.resize(nKLocal, nullptr);
      size_type iKPtLocal = 0;
      for (const auto &kv : feBasisDataStorage)
        {
          utils::throwException<utils::InvalidArgument>(
            kv.first == d_kPointIds[iKPtLocal],
            "DensityCalculator::reinit: the basis-storage map keys differ "
            "from the k-point map keys.");
          d_feBasisDataStorage[iKPtLocal] = kv.second;
          iKPtLocal++;
        }
      iKPtLocal = 0;
      for (const auto &kv : feBMPsi)
        {
          utils::throwException<utils::InvalidArgument>(
            kv.first == d_kPointIds[iKPtLocal],
            "DensityCalculator::reinit: the basis-manager map keys differ "
            "from the k-point map keys.");
          d_feBMPsi[iKPtLocal] = kv.second;
          iKPtLocal++;
        }

      // rho is summed over k on ONE quadrature grid, and one batch buffer built
      // from slot 0 serves every k, so every k must share both with slot 0
      for (iKPtLocal = 1; iKPtLocal < nKLocal; ++iKPtLocal)
        {
          utils::throwException<utils::InvalidArgument>(
            d_feBasisDataStorage[iKPtLocal]->getQuadratureRuleContainer() ==
              d_feBasisDataStorage[0]->getQuadratureRuleContainer(),
            "DensityCalculator::reinit: the per-k basis storages do not share "
            "one QuadratureRuleContainer.");
          utils::throwException<utils::InvalidArgument>(
            d_feBMPsi[iKPtLocal]->getMPIPatternP2P()->isCompatible(
              *d_feBMPsi[0]->getMPIPatternP2P()),
            "DensityCalculator::reinit: the per-k basis managers do not share "
            "one local dof layout.");
        }

      d_batchSizeSmall = ksdft::MaxSizeDefaults::SIZE_TYPE_MAX;
      d_quadRuleContainer =
        d_feBasisDataStorage[0]->getQuadratureRuleContainer();
      d_numLocallyOwnedCells = d_feBMPsi[0]->nLocallyOwnedCells();

      std::vector<size_type> numCellQuad(d_numLocallyOwnedCells, 0);
      for (size_type iCell = 0; iCell < d_numLocallyOwnedCells; ++iCell)
        numCellQuad[iCell] = d_quadRuleContainer->nCellQuadraturePoints(iCell);

      size_type maxQuadPerBatch = 0;
      for (size_type s = 0; s < d_numLocallyOwnedCells; s += d_cellBlockSize)
        {
          const size_type e =
            std::min(s + d_cellBlockSize, d_numLocallyOwnedCells);
          size_type sum = 0;
          for (size_type iCell = s; iCell < e; ++iCell)
            sum += numCellQuad[iCell];
          maxQuadPerBatch = std::max(maxQuadPerBatch, sum);
        }

      if (d_psiBatchQuad == nullptr ||
          d_psiBatchQuad->size() < d_waveFuncBatchSize * maxQuadPerBatch)
        {
          delete d_psiBatchQuad;
          d_psiBatchQuad =
            new dftefe::utils::MemoryStorage<ValueType, memorySpace>(
              d_waveFuncBatchSize * maxQuadPerBatch);
        }

      if (d_modPsiSqBatchQuad.size() < d_waveFuncBatchSize * maxQuadPerBatch)
        d_modPsiSqBatchQuad =
          dftefe::utils::MemoryStorage<RealType, memorySpace>(
            d_waveFuncBatchSize * maxQuadPerBatch);

      if (d_occupationInBatch.size() < d_waveFuncBatchSize)
        d_occupationInBatch =
          dftefe::utils::MemoryStorage<RealType, memorySpace>(
            d_waveFuncBatchSize);

      d_rhoMemspace.clear();

      if (d_rhoBatch == nullptr || d_rhoBatch->size() < 4 * maxQuadPerBatch)
        {
          delete d_rhoBatch;
          d_rhoBatch = new dftefe::utils::MemoryStorage<RealType, memorySpace>(
            4 * maxQuadPerBatch);
        }

      if (d_gradPsiBatchQuad == nullptr ||
          d_gradPsiBatchQuad->size() <
            d_waveFuncBatchSize * maxQuadPerBatch * dim)
        {
          delete d_gradPsiBatchQuad;
          d_gradPsiBatchQuad =
            new dftefe::utils::MemoryStorage<ValueType, memorySpace>(
              d_waveFuncBatchSize * maxQuadPerBatch * dim);
        }

      if (d_psiGradPsiBatch.size() <
          d_waveFuncBatchSize * maxQuadPerBatch * dim)
        d_psiGradPsiBatch = dftefe::utils::MemoryStorage<RealType, memorySpace>(
          d_waveFuncBatchSize * maxQuadPerBatch * dim);

      d_gradRhoMemspace.clear();

      if (d_gradRhoBatch == nullptr ||
          d_gradRhoBatch->size() < 4 * maxQuadPerBatch * dim)
        {
          delete d_gradRhoBatch;
          d_gradRhoBatch =
            new dftefe::utils::MemoryStorage<RealType, memorySpace>(
              4 * maxQuadPerBatch * dim);
        }

      d_psiBatch =
        new linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>(
          d_feBMPsi[0]->getMPIPatternP2P(),
          d_linAlgOpContext,
          d_waveFuncBatchSize,
          ValueTypeBasisCoeff());

      // one FEBasisOperations per DISTINCT storage: k sharing a basis alias
      // one object, so CFE builds it once
      d_feBasisOp.resize(nKLocal, nullptr);
      for (iKPtLocal = 0; iKPtLocal < nKLocal; ++iKPtLocal)
        {
          d_feBasisOp[iKPtLocal] = nullptr;
          for (size_type jKPtLocal = 0; jKPtLocal < iKPtLocal; ++jKPtLocal)
            if (d_feBasisDataStorage[jKPtLocal] ==
                d_feBasisDataStorage[iKPtLocal])
              {
                d_feBasisOp[iKPtLocal] = d_feBasisOp[jKPtLocal];
                break;
              }
          if (d_feBasisOp[iKPtLocal] == nullptr)
            d_feBasisOp[iKPtLocal] =
              std::make_shared<basis::FEBasisOperations<ValueTypeBasisCoeff,
                                                        ValueTypeBasisData,
                                                        memorySpace,
                                                        dim>>(
                d_feBasisDataStorage[iKPtLocal],
                d_cellBlockSize,
                d_waveFuncBatchSize);
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
      computeRho(
        const SpectralRep<ValueType, memorySpace> &spectral,
        std::vector<
          quadrature::QuadratureValuesContainer<RealType, memorySpaceHost> *>
          &rho,
        std::vector<
          quadrature::QuadratureValuesContainer<RealType, memorySpaceHost> *>
          &            gradRho,
        const bool     computeGrad,
        const SpinMode spinMode)
    {
      const size_type nKLocal   = d_kPointIds.size();
      const size_type numSpaces = (spinMode == SpinMode::Unpolarized) ? 1 : 2;
      const size_type ncomp     = (spinMode == SpinMode::Unpolarized) ? 1 :
                                  (spinMode == SpinMode::Collinear)   ? 2 :
                                                                        4;
      const size_type batchN    = d_waveFuncBatchSize / numSpaces;

      // Lazy-allocate per-component device QVC containers if needed
      if (d_rhoMemspace.size() != ncomp)
        {
          d_rhoMemspace.resize(ncomp);
          for (size_type ic = 0; ic < ncomp; ++ic)
            d_rhoMemspace[ic].reinit(d_quadRuleContainer, 1);
        }
      if (computeGrad && d_gradRhoMemspace.size() != ncomp)
        {
          d_gradRhoMemspace.resize(ncomp);
          for (size_type ic = 0; ic < ncomp; ++ic)
            d_gradRhoMemspace[ic].reinit(d_quadRuleContainer, dim);
        }

      for (size_type ic = 0; ic < ncomp; ++ic)
        d_rhoMemspace[ic].setValue((RealType)0);
      if (computeGrad)
        for (size_type ic = 0; ic < ncomp; ++ic)
          d_gradRhoMemspace[ic].setValue((RealType)0);

      utils::MemoryTransfer<memorySpace, memorySpaceHost> memoryTransferM2H;
      utils::MemoryTransfer<memorySpaceHost, memorySpace> memoryTransferH2M;

      // rho = sum_k w_k sum_n f_nk |psi_nk|^2
      for (size_type iKPtLocal = 0; iKPtLocal < nKLocal; ++iKPtLocal)
        {
          const std::vector<double> &occupationK =
            spectral.occupancies.at(d_kPointIds[iKPtLocal]);
          const linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
            &waveFuncK            = *spectral.ksOrbs.at(d_kPointIds[iKPtLocal]);
          const RealType  kWeight = (RealType)d_kPointWeights[iKPtLocal];
          const size_type numVecPerSpace =
            waveFuncK.getNumberComponents() / numSpaces;

          utils::MemoryStorage<RealType, memorySpace> occMemspace(
            occupationK.size());
          memoryTransferH2M.copy(occupationK.size(),
                                 occMemspace.data(),
                                 occupationK.data());

          const auto &wavePs = static_cast<
            const linearAlgebra::MultiVectorProductSpace<ValueTypeBasisCoeff,
                                                         memorySpace> &>(
            waveFuncK);

          for (size_type cellStartId = 0; cellStartId < d_numLocallyOwnedCells;
               cellStartId += d_cellBlockSize)
            {
              const size_type cellEndId =
                std::min(cellStartId + d_cellBlockSize, d_numLocallyOwnedCells);
              const std::pair<size_type, size_type> cellRange(cellStartId,
                                                              cellEndId);

              size_type numQuadInBlock = 0;
              for (size_type iCell = cellStartId; iCell < cellEndId; iCell++)
                numQuadInBlock +=
                  d_quadRuleContainer->nCellQuadraturePoints(iCell);

              for (size_type psiStartId = 0; psiStartId < numVecPerSpace;
                   psiStartId += batchN)
                {
                  const size_type numPsiInBatch =
                    std::min(batchN, numVecPerSpace - psiStartId);
                  const size_type numPsiInBatchTotal =
                    numSpaces * numPsiInBatch;

                  linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
                    *psiBatchInterim = nullptr;

                  if (numPsiInBatchTotal == d_waveFuncBatchSize)
                    {
                      psiBatchInterim = d_psiBatch;
                    }
                  else if (numPsiInBatchTotal == d_batchSizeSmall)
                    {
                      psiBatchInterim = d_psiBatchSmall;
                    }
                  else
                    {
                      d_batchSizeSmall = numPsiInBatchTotal;
                      delete d_psiBatchSmall;
                      d_psiBatchSmall =
                        new linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                                       memorySpace>(
                          waveFuncK.getMPIPatternP2P(),
                          d_linAlgOpContext,
                          numPsiInBatchTotal,
                          ValueTypeBasisCoeff());
                      psiBatchInterim = d_psiBatchSmall;
                    }

                  // Assemble occupation batch: spin-major layout in
                  // d_occupationInBatch
                  for (size_type s = 0; s < numSpaces; ++s)
                    linearAlgebra::blasLapack::copyValueType1ArrToValueType2Arr(
                      numPsiInBatch,
                      occMemspace.data() + s * numVecPerSpace + psiStartId,
                      d_occupationInBatch.data() + s * numPsiInBatch,
                      *waveFuncK.getLinAlgOpContext());

                  // Spin-aware gather into flat batch buffer
                  linearAlgebra::MultiVectorOps::copyToBatch(
                    wavePs,
                    psiStartId,
                    numPsiInBatch,
                    *psiBatchInterim,
                    *d_linAlgOpContext);

                  // Basis data for cellRange is cached across psi-batch
                  // iterations
                  d_feBasisOp[iKPtLocal]->interpolate(*psiBatchInterim,
                                                      *d_feBMPsi[iKPtLocal],
                                                      cellRange,
                                                      d_psiBatchQuad->data());

                  DensityCalculatorKernels<
                    ValueType,
                    RealType,
                    memorySpace,
                    dim>::computeRhoInBatch(numPsiInBatchTotal,
                                            cellRange,
                                            d_occupationInBatch.data(),
                                            d_psiBatchQuad->data(),
                                            d_modPsiSqBatchQuad.data(),
                                            d_quadRuleContainer,
                                            d_rhoBatch->data(),
                                            *d_linAlgOpContext,
                                            spinMode);

                  for (size_type ic = 0; ic < ncomp; ++ic)
                    linearAlgebra::blasLapack::
                      axpy<RealType, RealType, memorySpace>(
                        numQuadInBlock,
                        kWeight,
                        d_rhoBatch->data() + ic * numQuadInBlock,
                        1,
                        d_rhoMemspace[ic].begin(cellStartId),
                        1,
                        *d_linAlgOpContext);

                  if (computeGrad)
                    {
                      d_feBasisOp[iKPtLocal]->interpolateWithBasisGradient(
                        *psiBatchInterim,
                        *d_feBMPsi[iKPtLocal],
                        cellRange,
                        d_gradPsiBatchQuad->data());

                      DensityCalculatorKernels<
                        ValueType,
                        RealType,
                        memorySpace,
                        dim>::computeGradRhoInBatch(numPsiInBatchTotal,
                                                    cellRange,
                                                    d_occupationInBatch.data(),
                                                    d_psiBatchQuad->data(),
                                                    d_gradPsiBatchQuad->data(),
                                                    d_psiGradPsiBatch.data(),
                                                    d_quadRuleContainer,
                                                    d_gradRhoBatch->data(),
                                                    *d_linAlgOpContext,
                                                    spinMode);

                      for (size_type ic = 0; ic < ncomp; ++ic)
                        linearAlgebra::blasLapack::
                          axpy<RealType, RealType, memorySpace>(
                            numQuadInBlock * dim,
                            kWeight,
                            d_gradRhoBatch->data() + ic * numQuadInBlock * dim,
                            1,
                            d_gradRhoMemspace[ic].begin(cellStartId),
                            1,
                            *d_linAlgOpContext);
                    }
                }
            }
        } // k loop

      for (size_type ic = 0; ic < ncomp; ++ic)
        memoryTransferM2H.copy(d_rhoMemspace[ic].nEntries(),
                               rho[ic]->begin(),
                               d_rhoMemspace[ic].begin());

      if (computeGrad)
        for (size_type ic = 0; ic < ncomp; ++ic)
          memoryTransferM2H.copy(d_gradRhoMemspace[ic].nEntries(),
                                 gradRho[ic]->begin(),
                                 d_gradRhoMemspace[ic].begin());
    }
  } // end of namespace ksdft
} // end of namespace dftefe
