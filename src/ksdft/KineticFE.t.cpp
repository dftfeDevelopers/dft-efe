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
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
          feBasisDataStorage,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
                        linAlgOpContext,
        const size_type maxCellBlock,
        const size_type waveFuncBatchSize,
        SpinMode        spinMode)
      : d_maxCellBlock(maxCellBlock)
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
      reinit(std::shared_ptr<
             const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
               feBasisDataStorage)
    {
      d_cellWiseStorageKineticEnergy = std::make_shared<Storage>(0);
      d_feBasisDataStorage           = feBasisDataStorage;
      d_feBasisOp =
        std::make_shared<basis::FEBasisOperations<ValueTypeBasisCoeff,
                                                  ValueTypeBasisData,
                                                  memorySpace,
                                                  dim>>(feBasisDataStorage,
                                                        d_maxCellBlock);

      auto feBDH = std::dynamic_pointer_cast<
        const basis::FEBasisDofHandler<ValueTypeBasisCoeff, memorySpace, dim>>(
        feBasisDataStorage->getBasisDofHandler());
      utils::throwException(
        feBDH != nullptr,
        "Could not cast BasisDofHandler to FEBasisDofHandler in KineticFE::reinit");
      const size_type nCells = feBDH->nLocallyOwnedCells();
      d_numCellDofs.resize(nCells);
      d_basisOverlapSize = 0;
      for (size_type c = 0; c < nCells; ++c)
        {
          d_numCellDofs[c] = feBDH->nCellDofs(c);
          d_basisOverlapSize += d_numCellDofs[c] * d_numCellDofs[c];
        }

      d_feBasisOp->computeFEMatrices(basis::realspace::LinearLocalOp::GRAD,
                                     basis::realspace::VectorMathOp::DOT,
                                     basis::realspace::LinearLocalOp::GRAD,
                                     *d_cellWiseStorageKineticEnergy,
                                     *d_linAlgOpContext);

      linearAlgebra::blasLapack::ascale(d_cellWiseStorageKineticEnergy->size(),
                                        (ValueTypeBasisData)0.5,
                                        d_cellWiseStorageKineticEnergy->data(),
                                        d_cellWiseStorageKineticEnergy->data(),
                                        *d_linAlgOpContext);
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
      cellWiseStorage.resize(d_S * d_S * d_basisOverlapSize,
                             (ValueTypeBasisData)0);

      // Broadcast the scalar kinetic matrix to both diagonal spin blocks
      HamiltonianSpinBlockCopyKernels<ValueTypeBasisData, memorySpace>::
        copyIntoBlock(*d_cellWiseStorageKineticEnergy,
                      cellWiseStorage,
                      d_S,
                      d_layout,
                      std::vector<std::pair<size_type, size_type>>{{0, 0}},
                      d_numCellDofs,
                      *d_linAlgOpContext);
      if (d_S > 1)
        HamiltonianSpinBlockCopyKernels<ValueTypeBasisData, memorySpace>::
          copyIntoBlock(*d_cellWiseStorageKineticEnergy,
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
      evalEnergy(const std::vector<RealType> &                  occupation,
                 const basis::FEBasisManager<ValueTypeBasisCoeff,
                                             ValueTypeBasisData,
                                             memorySpace,
                                             dim> &             feBMPsi,
                 const linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                                  memorySpace> &waveFunc)
    {
      d_feBasisOp->reinit(d_maxCellBlock, d_waveFuncBatchSize);
      std::shared_ptr<const quadrature::QuadratureRuleContainer>
        quadRuleContainer = d_feBasisDataStorage->getQuadratureRuleContainer();

      d_energy = (RealType)0;

      const linearAlgebra::MultiVectorProductSpace<ValueTypeBasisCoeff,
                                                   memorySpace> *Xps =
        static_cast<
          const linearAlgebra::MultiVectorProductSpace<ValueTypeBasisCoeff,
                                                       memorySpace> *>(
          &waveFunc);

      const size_type numSpaces      = Xps->numSpaces();
      const size_type numVecPerSpace = Xps->numVectorsPerSpace();
      const size_type batchPerSpin   = d_waveFuncBatchSize / numSpaces;

      const RealType spinFactor =
        (d_spinMode == SpinMode::Unpolarized) ? (RealType)2 : (RealType)1;

      if (d_mpiPatternP2P == nullptr ||
          !d_mpiPatternP2P->isCompatible(*waveFunc.getMPIPatternP2P()))
        {
          d_mpiPatternP2P = waveFunc.getMPIPatternP2P();
          d_psiBatch      = std::make_shared<
            linearAlgebra::MultiVector<ValueType, memorySpace>>(
            d_mpiPatternP2P,
            waveFunc.getLinAlgOpContext(),
            d_waveFuncBatchSize,
            ValueType());
          d_YBatch = std::make_shared<
            linearAlgebra::MultiVector<ValueType, memorySpace>>(
            d_mpiPatternP2P,
            waveFunc.getLinAlgOpContext(),
            d_waveFuncBatchSize,
            ValueType());
          d_laplaceOp = std::make_shared<
            electrostatics::LaplaceOperatorContextFE<ValueTypeBasisData,
                                                     ValueTypeBasisCoeff,
                                                     memorySpace,
                                                     dim>>(
            feBMPsi,
            feBMPsi,
            d_cellWiseStorageKineticEnergy,
            d_maxCellBlock,
            d_waveFuncBatchSize);
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
                waveFunc.getLinAlgOpContext(),
                smallTotal,
                ValueType());
              d_YBatchSmall = std::make_shared<
                linearAlgebra::MultiVector<ValueType, memorySpace>>(
                d_mpiPatternP2P,
                waveFunc.getLinAlgOpContext(),
                smallTotal,
                ValueType());
            }
        }

      for (size_type psiStartId = 0; psiStartId < numVecPerSpace;
           psiStartId += batchPerSpin)
        {
          const size_type numPsiInBatch =
            std::min(psiStartId + batchPerSpin, numVecPerSpace) - psiStartId;
          const size_type numPsiInBatchTotal = numSpaces * numPsiInBatch;

          std::vector<RealType> occupationInBatch(numPsiInBatchTotal,
                                                  (RealType)0);
          for (size_type s = 0; s < numSpaces; ++s)
            std::copy(occupation.begin() + s * numVecPerSpace + psiStartId,
                      occupation.begin() + s * numVecPerSpace + psiStartId +
                        numPsiInBatch,
                      occupationInBatch.begin() + s * numPsiInBatch);

          std::vector<RealType> dotProds(numPsiInBatchTotal);

          if (numPsiInBatch < batchPerSpin)
            {
              linearAlgebra::MultiVectorOps::copyToBatch(
                *Xps,
                psiStartId,
                numPsiInBatch,
                *d_psiBatchSmall,
                *waveFunc.getLinAlgOpContext());

              d_laplaceOp->apply(*d_psiBatchSmall, *d_YBatchSmall, true, true);
              linearAlgebra::dot(*d_psiBatchSmall,
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
                *waveFunc.getLinAlgOpContext());

              d_laplaceOp->apply(*d_psiBatch, *d_YBatch, true, true);
              linearAlgebra::dot(*d_psiBatch,
                                 *d_YBatch,
                                 dotProds,
                                 linearAlgebra::blasLapack::ScalarOp::Conj,
                                 linearAlgebra::blasLapack::ScalarOp::Identity);
            }

          for (size_type i = 0; i < numPsiInBatchTotal; ++i)
            d_energy +=
              (RealType)(dotProds[i] * spinFactor * occupationInBatch[i]);
        }

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
