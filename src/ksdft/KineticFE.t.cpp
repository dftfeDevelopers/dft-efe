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
#include <utils/Exceptions.h>
#include <utils/MemoryTransfer.h>
#include <algorithm>
#include <linearAlgebra/BlasLapackTypedef.h>
#include <linearAlgebra/MultiVectorProductSpace.h>
#include <linearAlgebra/MultiVectorOps.h>

namespace dftefe
{
  namespace ksdft
  {
    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    KineticFE<ValueTypeBasisData, ValueTypeBasisCoeff, memorySpace, dim>::
      KineticFE(
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
                                                      dim>>> &feBasisManager,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
                                                            linAlgOpContext,
        const size_type                                     maxCellBlock,
        const size_type                                     waveFuncBatchSize,
        SpinMode                                            spinMode,
        const std::map<basis::kpoint::KPointId, basis::kpoint::KPointData> &kPoints)
      : d_currentKPointIndex(0)
      , d_maxCellBlock(maxCellBlock)
      , d_linAlgOpContext(linAlgOpContext)
      , d_waveFuncBatchSize(waveFuncBatchSize)
      , d_mpiPatternP2P(nullptr)
      , d_spinMode(spinMode)
      , d_S((spinMode == SpinMode::Unpolarized) ? 1 : 2)
      , d_layout((spinMode == SpinMode::NonCollinear) ?
                   SpinStorageLayout::SpinFastest :
                   SpinStorageLayout::DofFastest)
      , d_basisOverlapSize(0)
    {
      utils::throwException<utils::InvalidArgument>(
        !kPoints.empty(),
        "KineticFE: the k-point map is empty. A non-periodic caller passes "
        "basis::kpoint::gammaIfEmpty({}, dim).");
      const size_type nKLocal = kPoints.size();
      d_kPointIds.resize(nKLocal, (basis::kpoint::KPointId)0);
      d_kPointCoordinates.resize(nKLocal, utils::Point((size_type)dim, 0.0));
      d_kPointWeights.resize(nKLocal, 1.0);

      // ascending key order, so the local index order is deterministic
      size_type iKPtLocal = 0;
      for (const auto &kv : kPoints)
        {
          d_kPointIds[iKPtLocal]         = kv.first;
          d_kPointCoordinates[iKPtLocal] = kv.second.coord;
          d_kPointWeights[iKPtLocal]     = kv.second.weight;
          iKPtLocal++;
        }

      utils::throwException<utils::InvalidArgument>(
        feBasisDataStorage.size() == kPoints.size() &&
          feBasisManager.size() == kPoints.size(),
        "KineticFE: the k-point, basis-storage and basis-manager maps must "
        "have the same key set.");

      d_feBasisManager.resize(nKLocal, nullptr);
      iKPtLocal = 0;
      for (const auto &kv : feBasisManager)
        {
          utils::throwException<utils::InvalidArgument>(
            kv.first == d_kPointIds[iKPtLocal],
            "KineticFE: the basis-manager map keys differ from the k-point "
            "map keys.");
          d_feBasisManager[iKPtLocal] = kv.second;
          iKPtLocal++;
        }

      reinit(feBasisDataStorage);
    }

    // template <typename ValueTypeBasisData,
    //           typename ValueTypeBasisCoeff,
    //           utils::MemorySpace memorySpace,
    //           size_type          dim>
    // KineticFE<ValueTypeBasisData, ValueTypeBasisCoeff, memorySpace, dim>::
    //   ~KineticFE()
    // {
    //   if (d_gradPsi != nullptr)
    //     {
    //       delete d_gradPsi;
    //       d_gradPsi = nullptr;
    //     }
    // }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    KineticFE<ValueTypeBasisData, ValueTypeBasisCoeff, memorySpace, dim>::
      reinit(
        const std::map<
          basis::kpoint::KPointId,
          std::shared_ptr<
            const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>>
          &feBasisDataStorage)
    {
      const size_type nKLocal = d_kPointIds.size();
      utils::throwException<utils::InvalidArgument>(
        feBasisDataStorage.size() == nKLocal,
        "KineticFE::reinit: the basis-storage map does not match the k-point "
        "set this object was built with.");

      d_feBasisDataStorage.resize(nKLocal, nullptr);
      d_feBasisOp.resize(nKLocal, nullptr);
      d_laplaceOp.resize(nKLocal, nullptr);
      d_cellWiseStorageKineticEnergy.resize(nKLocal, nullptr);

      size_type iKPtLocal = 0;
      for (const auto &kv : feBasisDataStorage)
        {
          utils::throwException<utils::InvalidArgument>(
            kv.first == d_kPointIds[iKPtLocal],
            "KineticFE::reinit: the basis-storage map keys differ from the "
            "k-point map keys.");
          d_feBasisDataStorage[iKPtLocal] = kv.second;
          iKPtLocal++;
        }

      // one FEBasisOperations per DISTINCT storage: k sharing a basis alias
      // one object, so CFE builds it once
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
                d_feBasisDataStorage[iKPtLocal], d_maxCellBlock);
        }

      // structural queries must agree across k; slot 0 sets them and the
      // rest are checked against it
      d_basisOverlapSize = 0;
      for (iKPtLocal = 0; iKPtLocal < nKLocal; ++iKPtLocal)
        {
          auto feBDH = std::dynamic_pointer_cast<
            const basis::
              FEBasisDofHandler<ValueTypeBasisCoeff, memorySpace, dim>>(
            d_feBasisDataStorage[iKPtLocal]->getBasisDofHandler());
          utils::throwException(
            feBDH != nullptr,
            "Could not cast BasisDofHandler to FEBasisDofHandler in KineticFE::reinit");

          const size_type nCells = feBDH->nLocallyOwnedCells();
          if (iKPtLocal == 0)
            {
              d_numCellDofs.resize(nCells);
              for (size_type c = 0; c < nCells; ++c)
                {
                  d_numCellDofs[c] = feBDH->nCellDofs(c);
                  d_basisOverlapSize += d_numCellDofs[c] * d_numCellDofs[c];
                }
            }
          else
            {
              utils::throwException<utils::InvalidArgument>(
                nCells == d_numCellDofs.size(),
                "KineticFE::reinit: the per-k basis-storage slots disagree on "
                "the number of locally owned cells.");
              for (size_type c = 0; c < nCells; ++c)
                utils::throwException<utils::InvalidArgument>(
                  feBDH->nCellDofs(c) == d_numCellDofs[c],
                  "KineticFE::reinit: the per-k basis-storage slots disagree "
                  "on the number of dofs in a cell.");
            }
        }

      // T(k) = 1/2 S + 1/2 |k|^2 M - i k.A, single-sided. The basis spans the
      // periodic u of psi = e^{i k.r} u, hence the -i.
      Storage              halfStiffness(0);
      Storage              massMatrix(0);
      std::vector<Storage> crossTermMatrix(dim, Storage(0));
      for (iKPtLocal = 0; iKPtLocal < nKLocal; ++iKPtLocal)
        {
          const utils::Point &kPoint = d_kPointCoordinates[iKPtLocal];

          bool isGammaPoint = true;
          for (size_type iDim = 0; iDim < dim; iDim++)
            if (kPoint[iDim] != 0.0)
              isGammaPoint = false;

          // S, M and A_d depend on k only through the basis: rebuilt only when
          // the storage changes, so once for CFE and once per k for EFE
          if (iKPtLocal == 0 || d_feBasisDataStorage[iKPtLocal] !=
                                  d_feBasisDataStorage[iKPtLocal - 1])
            {
              // S = \int conj(grad N_i).grad N_j, kept as 1/2 S
              halfStiffness = Storage(0);
              d_feBasisOp[iKPtLocal]->computeFEMatrices(
                basis::realspace::LinearLocalOp::GRAD,
                basis::realspace::VectorMathOp::DOT,
                basis::realspace::LinearLocalOp::GRAD,
                halfStiffness,
                *d_linAlgOpContext);

              linearAlgebra::blasLapack::ascale(halfStiffness.size(),
                                                (ValueTypeBasisData)0.5,
                                                halfStiffness.data(),
                                                halfStiffness.data(),
                                                *d_linAlgOpContext);

              // M and A_d of this storage are built on its first non-Gamma k
              massMatrix = Storage(0);
              for (size_type iDim = 0; iDim < dim; iDim++)
                crossTermMatrix[iDim] = Storage(0);
            }

          // T(k) starts as 1/2 S, copied except on the last k of this storage:
          // nothing reads 1/2 S after that, so it is moved (EFE never copies)
          if (iKPtLocal + 1 == nKLocal || d_feBasisDataStorage[iKPtLocal + 1] !=
                                            d_feBasisDataStorage[iKPtLocal])
            d_cellWiseStorageKineticEnergy[iKPtLocal] =
              std::make_shared<Storage>(std::move(halfStiffness));
          else
            d_cellWiseStorageKineticEnergy[iKPtLocal] =
              std::make_shared<Storage>(halfStiffness);

          // the other two terms carry an i, so they are compiled out of a
          // real build; both vanish identically at Gamma
          if constexpr (!std::is_same<ValueType,
                                      linearAlgebra::blasLapack::real_type<
                                        ValueType>>::value)
            {
              if (isGammaPoint)
                continue;

              if (massMatrix.size() == 0)
                {
                  std::shared_ptr<const quadrature::QuadratureRuleContainer>
                    quadRuleContainer = d_feBasisDataStorage[iKPtLocal]
                                          ->getQuadratureRuleContainer();

                  // M = \int conj(N_i) N_j; the f-weighted overload with
                  // f == 1, as overload 3 is GRAD/DOT/GRAD only
                  std::vector<
                    quadrature::QuadratureValuesContainer<ValueType,
                                                          memorySpace>>
                    ones(1,
                         quadrature::QuadratureValuesContainer<ValueType,
                                                               memorySpace>(
                           quadRuleContainer, 1, (ValueType)1.0));
                  d_feBasisOp[iKPtLocal]->computeFEMatrices(
                    basis::realspace::LinearLocalOp::IDENTITY,
                    basis::realspace::VectorMathOp::MULT,
                    ones,
                    basis::realspace::VectorMathOp::MULT,
                    basis::realspace::LinearLocalOp::IDENTITY,
                    massMatrix,
                    *d_linAlgOpContext);

                  // A_d = \int conj(N_i) d_d N_j: the field is the unit vector
                  // e_d at every quadrature point, component fastest, built on
                  // host and moved in one transfer
                  const size_type numQuadTotal =
                    quadRuleContainer->nQuadraturePoints();
                  for (size_type iDim = 0; iDim < dim; iDim++)
                    {
                      std::vector<ValueType> fieldHost(numQuadTotal * dim,
                                                       (ValueType)0.0);
                      for (size_type iQuad = 0; iQuad < numQuadTotal; iQuad++)
                        fieldHost[iQuad * dim + iDim] = (ValueType)1.0;

                      std::vector<
                        quadrature::QuadratureValuesContainer<ValueType,
                                                              memorySpace>>
                        fieldVec(
                          1,
                          quadrature::QuadratureValuesContainer<ValueType,
                                                                memorySpace>(
                            quadRuleContainer, dim, (ValueType)0.0));
                      utils::MemoryTransfer<
                        memorySpace,
                        utils::MemorySpace::HOST>::copy(fieldHost.size(),
                                                        fieldVec[0].begin(),
                                                        fieldHost.data());

                      d_feBasisOp[iKPtLocal]->computeFEMatrices(
                        basis::realspace::LinearLocalOp::IDENTITY,
                        basis::realspace::VectorMathOp::MULT,
                        fieldVec,
                        basis::realspace::VectorMathOp::DOT,
                        basis::realspace::LinearLocalOp::GRAD,
                        crossTermMatrix[iDim],
                        *d_linAlgOpContext);
                    }
                }

              double kSquare = 0.0;
              for (size_type iDim = 0; iDim < dim; iDim++)
                kSquare += kPoint[iDim] * kPoint[iDim];
              linearAlgebra::blasLapack::
                axpy<ValueType, ValueType, memorySpace>(
                  massMatrix.size(),
                  (ValueType)(0.5 * kSquare),
                  massMatrix.data(),
                  1,
                  d_cellWiseStorageKineticEnergy[iKPtLocal]->data(),
                  1,
                  *d_linAlgOpContext);

              // -i k.A = sum_d (-i k_d) A_d
              for (size_type iDim = 0; iDim < dim; iDim++)
                linearAlgebra::blasLapack::
                  axpy<ValueType, ValueType, memorySpace>(
                    crossTermMatrix[iDim].size(),
                    ValueType(0.0, -kPoint[iDim]),
                    crossTermMatrix[iDim].data(),
                    1,
                    d_cellWiseStorageKineticEnergy[iKPtLocal]->data(),
                    1,
                    *d_linAlgOpContext);
            }
          else
            {
              utils::throwException<utils::InvalidArgument>(
                isGammaPoint,
                "KineticFE: a non-zero k needs a complex value type. A real "
                "build assembles neither |k|^2 M nor -i k.A, so T(k) would "
                "silently be the Gamma one. Gamma itself is fine here, and "
                "so is Gamma in a complex build.");
            }
        }

      // built here so getLocal and evalEnergy apply the same T(k)
      for (iKPtLocal = 0; iKPtLocal < nKLocal; ++iKPtLocal)
        d_laplaceOp[iKPtLocal] = std::make_shared<
          electrostatics::LaplaceOperatorContextFE<ValueTypeBasisData,
                                                   ValueTypeBasisCoeff,
                                                   memorySpace,
                                                   dim>>(
          *d_feBasisManager[iKPtLocal],
          *d_feBasisManager[iKPtLocal],
          d_cellWiseStorageKineticEnergy[iKPtLocal],
          d_maxCellBlock,
          d_waveFuncBatchSize);
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    KineticFE<ValueTypeBasisData, ValueTypeBasisCoeff, memorySpace, dim>::
      activateKpt(const basis::kpoint::KPointId kPointId)
    {
      // d_kPointIds is ascending (built from a std::map), so binary search
      const std::vector<basis::kpoint::KPointId>::const_iterator itKPointId =
        std::lower_bound(d_kPointIds.begin(), d_kPointIds.end(), kPointId);
      utils::throwException<utils::InvalidArgument>(
        itKPointId != d_kPointIds.end() && *itKPointId == kPointId,
        "KineticFE::activateKpt: this object does not hold that k-point.");
      d_currentKPointIndex = itKPointId - d_kPointIds.begin();
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    KineticFE<ValueTypeBasisData, ValueTypeBasisCoeff, memorySpace, dim>::
      getLocal(Storage &cellWiseStorage) const
    {
      // Zero-init spin-blocked output: S² × basisOverlapSize
      cellWiseStorage.resize(d_S * d_S * d_basisOverlapSize, (ValueType)0);

      // Broadcast the scalar kinetic matrix to both diagonal spin blocks
      HamiltonianSpinBlockCopyKernels<ValueType, memorySpace>::copyIntoBlock(
        *d_cellWiseStorageKineticEnergy[d_currentKPointIndex],
        cellWiseStorage,
        d_S,
        d_layout,
        std::vector<std::pair<size_type, size_type>>{{0, 0}},
        d_numCellDofs,
        *d_linAlgOpContext);
      if (d_S > 1)
        HamiltonianSpinBlockCopyKernels<ValueType, memorySpace>::copyIntoBlock(
          *d_cellWiseStorageKineticEnergy[d_currentKPointIndex],
          cellWiseStorage,
          d_S,
          d_layout,
          std::vector<std::pair<size_type, size_type>>{{1, 1}},
          d_numCellDofs,
          *d_linAlgOpContext);
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    KineticFE<ValueTypeBasisData, ValueTypeBasisCoeff, memorySpace, dim>::
      evalEnergy(RDM1<ValueType, memorySpace> &rdm1)
    {
      const size_type nKLocal = d_kPointIds.size();
      // the orbitals and occupancies live only on the spectral RDM1
      const RDM1Spectral<ValueType, memorySpace> *rdm1SpectralPtr =
        dynamic_cast<const RDM1Spectral<ValueType, memorySpace> *>(&rdm1);
      utils::throwException(
        rdm1SpectralPtr != nullptr,
        "KineticFE::evalEnergy: pass the spectral RDM1, which holds the "
        "orbitals and occupancies.");
      const SpectralRep<ValueType, memorySpace> &spectral =
        rdm1SpectralPtr->getSpectral();

      d_energy = (RealType)0;

      // sum_k w_k sum_n f_nk <psi|T(k)|psi>
      for (size_type iKPtLocal = 0; iKPtLocal < nKLocal; ++iKPtLocal)
        {
          const std::vector<double> &occupationK =
            spectral.occupancies.at(d_kPointIds[iKPtLocal]);
          const linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
            &waveFuncK           = *spectral.ksOrbs.at(d_kPointIds[iKPtLocal]);
          const RealType kWeight = (RealType)d_kPointWeights[iKPtLocal];

          d_feBasisOp[iKPtLocal]->reinit(d_maxCellBlock, d_waveFuncBatchSize);
          std::shared_ptr<const quadrature::QuadratureRuleContainer>
            quadRuleContainer =
              d_feBasisDataStorage[iKPtLocal]->getQuadratureRuleContainer();

          const linearAlgebra::MultiVectorProductSpace<ValueTypeBasisCoeff,
                                                       memorySpace> *Xps =
            static_cast<
              const linearAlgebra::MultiVectorProductSpace<ValueTypeBasisCoeff,
                                                           memorySpace> *>(
              &waveFuncK);

          const size_type numSpaces      = Xps->numSpaces();
          const size_type numVecPerSpace = Xps->numVectorsPerSpace();
          const size_type batchPerSpin   = d_waveFuncBatchSize / numSpaces;

          const RealType spinFactor =
            (d_spinMode == SpinMode::Unpolarized) ? (RealType)2 : (RealType)1;

          if (d_mpiPatternP2P == nullptr ||
              !d_mpiPatternP2P->isCompatible(*waveFuncK.getMPIPatternP2P()))
            {
              d_mpiPatternP2P = waveFuncK.getMPIPatternP2P();
              d_psiBatch      = std::make_shared<
                linearAlgebra::MultiVector<ValueType, memorySpace>>(
                d_mpiPatternP2P,
                waveFuncK.getLinAlgOpContext(),
                d_waveFuncBatchSize,
                ValueType());
              d_YBatch = std::make_shared<
                linearAlgebra::MultiVector<ValueType, memorySpace>>(
                d_mpiPatternP2P,
                waveFuncK.getLinAlgOpContext(),
                d_waveFuncBatchSize,
                ValueType());
            }

          const size_type smallSpin  = numVecPerSpace % batchPerSpin;
          const size_type smallTotal = numSpaces * smallSpin;
          if (numVecPerSpace > batchPerSpin && smallSpin != 0)
            {
              if (d_psiBatchSmall == nullptr ||
                  d_psiBatchSmall->getNumberComponents() != smallTotal)
                {
                  d_psiBatchSmall = std::make_shared<
                    linearAlgebra::MultiVector<ValueType, memorySpace>>(
                    d_mpiPatternP2P,
                    waveFuncK.getLinAlgOpContext(),
                    smallTotal,
                    ValueType());
                  d_YBatchSmall = std::make_shared<
                    linearAlgebra::MultiVector<ValueType, memorySpace>>(
                    d_mpiPatternP2P,
                    waveFuncK.getLinAlgOpContext(),
                    smallTotal,
                    ValueType());
                }
            }

          for (size_type psiStartId = 0; psiStartId < numVecPerSpace;
               psiStartId += batchPerSpin)
            {
              const size_type numPsiInBatch =
                std::min(psiStartId + batchPerSpin, numVecPerSpace) -
                psiStartId;
              const size_type numPsiInBatchTotal = numSpaces * numPsiInBatch;

              std::vector<RealType> occupationInBatch(numPsiInBatchTotal,
                                                      (RealType)0);
              for (size_type s = 0; s < numSpaces; ++s)
                std::copy(occupationK.begin() + s * numVecPerSpace + psiStartId,
                          occupationK.begin() + s * numVecPerSpace +
                            psiStartId + numPsiInBatch,
                          occupationInBatch.begin() + s * numPsiInBatch);

              std::vector<ValueType> dotProds(numPsiInBatchTotal);

              if (numPsiInBatch < batchPerSpin)
                {
                  linearAlgebra::MultiVectorOps::copyToBatch(
                    *Xps,
                    psiStartId,
                    numPsiInBatch,
                    *d_psiBatchSmall,
                    *waveFuncK.getLinAlgOpContext());

                  d_laplaceOp[iKPtLocal]->apply(*d_psiBatchSmall,
                                                *d_YBatchSmall,
                                                true,
                                                true);
                  linearAlgebra::dot(
                    *d_psiBatchSmall,
                    *d_YBatchSmall,
                    dotProds,
                    linearAlgebra::blasLapack::ScalarOp::Conj,
                    linearAlgebra::blasLapack::ScalarOp::Identity);
                }
              else
                {
                  linearAlgebra::MultiVectorOps::copyToBatch(
                    *Xps,
                    psiStartId,
                    numPsiInBatch,
                    *d_psiBatch,
                    *waveFuncK.getLinAlgOpContext());

                  d_laplaceOp[iKPtLocal]->apply(*d_psiBatch,
                                                *d_YBatch,
                                                true,
                                                true);
                  linearAlgebra::dot(
                    *d_psiBatch,
                    *d_YBatch,
                    dotProds,
                    linearAlgebra::blasLapack::ScalarOp::Conj,
                    linearAlgebra::blasLapack::ScalarOp::Identity);
                }

              for (size_type i = 0; i < numPsiInBatchTotal; ++i)
                d_energy += kWeight * utils::realPart(dotProds[i]) *
                            spinFactor * occupationInBatch[i];
            }
        } // k loop

      // for (size_type psiStartId = 0;
      //      psiStartId < waveFunc.getNumberComponents();
      //      psiStartId += d_waveFuncBatchSize)
      //   {
      //     const size_type psiEndId = std::min(psiStartId +
      //     d_waveFuncBatchSize,
      //                                         waveFunc.getNumberComponents());
      //     const size_type numPsiInBatch = psiEndId - psiStartId;

      //     std::vector<RealType> occupationInBatch(numPsiInBatch,
      //     (RealType)0); RealType              energyBatchSum = 0;

      //     std::copy(occupation.begin() + psiStartId,
      //               occupation.begin() + psiEndId,
      //               occupationInBatch.begin());

      //     if (d_gradPsi->getNumberComponents() != numPsiInBatch * dim)
      //       d_gradPsi->reinit(quadRuleContainer, numPsiInBatch * dim);

      //     if (numPsiInBatch < d_waveFuncBatchSize)
      //       {
      //         for (size_type iSize = 0; iSize < waveFunc.localSize();
      //         iSize++)
      //           memoryTransfer.copy(numPsiInBatch,
      //                               d_psiBatchSmall->data() +
      //                                 numPsiInBatch * iSize,
      //                               waveFunc.data() +
      //                                 iSize * waveFunc.getNumberComponents()
      //                                 + psiStartId);

      //         d_feBasisOp->interpolateWithBasisGradient(*d_psiBatchSmall,
      //                                                   feBMPsi,
      //                                                   *d_gradPsi);
      //       }
      //     else
      //       {
      //         for (size_type iSize = 0; iSize < waveFunc.localSize();
      //         iSize++)
      //           memoryTransfer.copy(numPsiInBatch,
      //                               d_psiBatch->data() + numPsiInBatch *
      //                               iSize, waveFunc.data() +
      //                                 iSize * waveFunc.getNumberComponents()
      //                                 + psiStartId);

      //         d_feBasisOp->interpolateWithBasisGradient(*d_psiBatch,
      //                                                   feBMPsi,
      //                                                   *d_gradPsi);
      //       }

      //     ValueType *gradPsiIter = d_gradPsi->begin();

      //     auto jxwStorage = d_feBasisDataStorage->getJxWInAllCells();
      //     ValueTypeBasisData *jxwStorageIter    = jxwStorage.data();
      //     size_type cumulativeQuadGradPsiInCell = 0, cumulativeQuadInCell =
      //     0;

      //     for (size_type iCell = 0; iCell < d_gradPsi->nCells(); iCell++)
      //       {
      //         size_type numQuadInCell =
      //           quadRuleContainer->nCellQuadraturePoints(iCell);
      //         for (size_type iQuad = 0; iQuad < numQuadInCell; iQuad++)
      //           {
      //             const ValueTypeBasisData jxwVal =
      //               jxwStorageIter[cumulativeQuadInCell + iQuad];
      //             for (size_type iDim = 0; iDim < dim; iDim++)
      //               {
      //                 for (size_type iComp = 0; iComp < numPsiInBatch;
      //                 iComp++)
      //                   {
      //                     const ValueType gradPsiVal =
      //                       gradPsiIter[cumulativeQuadGradPsiInCell +
      //                                   numPsiInBatch * iQuad * dim +
      //                                   iDim * numPsiInBatch + iComp];
      //                     energyBatchSum += utils::absSq(gradPsiVal) *
      //                                       occupationInBatch[iComp] *
      //                                       jxwVal;
      //                   }
      //               }
      //           }
      //         cumulativeQuadGradPsiInCell +=
      //           numQuadInCell * numPsiInBatch * dim;
      //         cumulativeQuadInCell += numQuadInCell;
      //       }

      //     int mpierr = utils::mpi::MPIAllreduce<memorySpace>(
      //       utils::mpi::MPIInPlace,
      //       &energyBatchSum,
      //       1,
      //       utils::mpi::Types<RealType>::getMPIDatatype(),
      //       utils::mpi::MPISum,
      //       waveFunc.getMPIPatternP2P()->mpiCommunicator());

      //     d_energy += (RealType)(energyBatchSum);

      //     /*No multiplication by 1/2 due to spin up and down electrons*/
      //   }
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    typename KineticFE<ValueTypeBasisData,
                       ValueTypeBasisCoeff,
                       memorySpace,
                       dim>::RealType
    KineticFE<ValueTypeBasisData, ValueTypeBasisCoeff, memorySpace, dim>::
      getEnergy() const
    {
      return d_energy;
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    KineticFE<ValueTypeBasisData, ValueTypeBasisCoeff, memorySpace, dim>::
      applyNonLocal(
        linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace> &X,
        linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace> &Y,
        bool updateGhostX,
        bool updateGhostY) const
    {
      utils::throwException(
        false, "Non-Local component not present to call in KineticFE.h");
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    bool
    KineticFE<ValueTypeBasisData, ValueTypeBasisCoeff, memorySpace, dim>::
      hasLocalComponent() const
    {
      return true;
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    bool
    KineticFE<ValueTypeBasisData, ValueTypeBasisCoeff, memorySpace, dim>::
      hasNonLocalComponent() const
    {
      return false;
    }

  } // end of namespace ksdft
} // end of namespace dftefe
