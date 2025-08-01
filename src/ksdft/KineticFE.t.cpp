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
        const size_type waveFuncBatchSize)
      : d_maxCellBlock(maxCellBlock)
      , d_linAlgOpContext(linAlgOpContext)
      , d_waveFuncBatchSize(waveFuncBatchSize)
      , d_mpiPatternP2P(nullptr)
    {
      reinit(feBasisDataStorage);
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    KineticFE<ValueTypeBasisData, ValueTypeBasisCoeff, memorySpace, dim>::
      ~KineticFE()
    {
      if (d_gradPsi != nullptr)
        {
          delete d_gradPsi;
          d_gradPsi = nullptr;
        }
    }

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
      d_feBasisDataStorage = feBasisDataStorage;
      d_feBasisOp =
        std::make_shared<basis::FEBasisOperations<ValueTypeBasisCoeff,
                                                  ValueTypeBasisData,
                                                  memorySpace,
                                                  dim>>(feBasisDataStorage,
                                                        d_maxCellBlock);

      d_gradPsi =
        new quadrature::QuadratureValuesContainer<ValueType, memorySpace>(
          d_feBasisDataStorage->getQuadratureRuleContainer(),
          d_waveFuncBatchSize * dim);

      d_feBasisOp->computeFEMatrices(basis::realspace::LinearLocalOp::GRAD,
                                     basis::realspace::VectorMathOp::DOT,
                                     basis::realspace::LinearLocalOp::GRAD,
                                     d_cellWiseStorageKineticEnergy,
                                     *d_linAlgOpContext);

      linearAlgebra::blasLapack::ascale(d_cellWiseStorageKineticEnergy.size(),
                                        (ValueTypeBasisData)0.5,
                                        d_cellWiseStorageKineticEnergy.data(),
                                        d_cellWiseStorageKineticEnergy.data(),
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
      cellWiseStorage = d_cellWiseStorageKineticEnergy;
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

      d_energy = 0;

      if ((d_mpiPatternP2P == nullptr) ||
          (d_mpiPatternP2P != nullptr &&
           !d_mpiPatternP2P->isCompatible(*waveFunc.getMPIPatternP2P())))
        {
          d_mpiPatternP2P = waveFunc.getMPIPatternP2P();
          d_psiBatch      = std::make_shared<
            linearAlgebra::MultiVector<ValueType, memorySpace>>(
            d_mpiPatternP2P,
            waveFunc.getLinAlgOpContext(),
            d_waveFuncBatchSize,
            ValueType());
          if (waveFunc.getNumberComponents() > d_waveFuncBatchSize)
            d_psiBatchSmall = std::make_shared<
              linearAlgebra::MultiVector<ValueType, memorySpace>>(
              d_mpiPatternP2P,
              waveFunc.getLinAlgOpContext(),
              waveFunc.getNumberComponents() % d_waveFuncBatchSize,
              ValueType());
        }

      utils::MemoryTransfer<memorySpace, memorySpace> memoryTransfer;

      for (size_type psiStartId = 0;
           psiStartId < waveFunc.getNumberComponents();
           psiStartId += d_waveFuncBatchSize)
        {
          const size_type psiEndId = std::min(psiStartId + d_waveFuncBatchSize,
                                              waveFunc.getNumberComponents());
          const size_type numPsiInBatch = psiEndId - psiStartId;

          std::vector<RealType> occupationInBatch(numPsiInBatch, (RealType)0);
          RealType              energyBatchSum = 0;

          std::copy(occupation.begin() + psiStartId,
                    occupation.begin() + psiEndId,
                    occupationInBatch.begin());

          if (d_gradPsi->getNumberComponents() != numPsiInBatch * dim)
            d_gradPsi->reinit(quadRuleContainer, numPsiInBatch * dim);

          if (numPsiInBatch < d_waveFuncBatchSize)
            {
              for (size_type iSize = 0; iSize < waveFunc.localSize(); iSize++)
                memoryTransfer.copy(numPsiInBatch,
                                    d_psiBatchSmall->data() +
                                      numPsiInBatch * iSize,
                                    waveFunc.data() +
                                      iSize * waveFunc.getNumberComponents() +
                                      psiStartId);

              d_feBasisOp->interpolateWithBasisGradient(*d_psiBatchSmall,
                                                        feBMPsi,
                                                        *d_gradPsi);
            }
          else
            {
              for (size_type iSize = 0; iSize < waveFunc.localSize(); iSize++)
                memoryTransfer.copy(numPsiInBatch,
                                    d_psiBatch->data() + numPsiInBatch * iSize,
                                    waveFunc.data() +
                                      iSize * waveFunc.getNumberComponents() +
                                      psiStartId);

              d_feBasisOp->interpolateWithBasisGradient(*d_psiBatch,
                                                        feBMPsi,
                                                        *d_gradPsi);
            }

          ValueType *gradPsiIter = d_gradPsi->begin();

          auto jxwStorage = d_feBasisDataStorage->getJxWInAllCells();
          ValueTypeBasisData *jxwStorageIter    = jxwStorage.data();
          size_type cumulativeQuadGradPsiInCell = 0, cumulativeQuadInCell = 0;

          for (size_type iCell = 0; iCell < d_gradPsi->nCells(); iCell++)
            {
              size_type numQuadInCell =
                quadRuleContainer->nCellQuadraturePoints(iCell);
              for (size_type iQuad = 0; iQuad < numQuadInCell; iQuad++)
                {
                  const ValueTypeBasisData jxwVal =
                    jxwStorageIter[cumulativeQuadInCell + iQuad];
                  for (size_type iDim = 0; iDim < dim; iDim++)
                    {
                      for (size_type iComp = 0; iComp < numPsiInBatch; iComp++)
                        {
                          const ValueType gradPsiVal =
                            gradPsiIter[cumulativeQuadGradPsiInCell +
                                        numPsiInBatch * iQuad * dim +
                                        iDim * numPsiInBatch + iComp];
                          energyBatchSum += utils::absSq(gradPsiVal) *
                                            occupationInBatch[iComp] * jxwVal;
                        }
                    }
                }
              cumulativeQuadGradPsiInCell +=
                numQuadInCell * numPsiInBatch * dim;
              cumulativeQuadInCell += numQuadInCell;
            }

          int mpierr = utils::mpi::MPIAllreduce<memorySpace>(
            utils::mpi::MPIInPlace,
            &energyBatchSum,
            1,
            utils::mpi::Types<RealType>::getMPIDatatype(),
            utils::mpi::MPISum,
            waveFunc.getMPIPatternP2P()->mpiCommunicator());

          d_energy += (RealType)(energyBatchSum);

          /*No multiplication by 1/2 due to spin up and down electrons*/
        }
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
