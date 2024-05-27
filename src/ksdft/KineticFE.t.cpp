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
        const size_type cellBlockSize)
      : d_cellBlockSize(cellBlockSize)
      , d_linAlgOpContext(linAlgOpContext)
    {
      reinit(feBasisDataStorage);
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
    KineticFE<ValueTypeBasisData, ValueTypeBasisCoeff, memorySpace, dim>::
      getLocal(Storage &cellWiseStorage) const
    {
      d_feBasisOp->computeFEMatrices(basis::realspace::LinearLocalOp::GRAD,
                                     basis::realspace::VectorMathOp::DOT,
                                     basis::realspace::LinearLocalOp::GRAD,
                                     cellWiseStorage,
                                     *d_linAlgOpContext);

      linearAlgebra::blasLapack::ascale(cellWiseStorage.size(),
                                        (ValueTypeBasisData)0.5,
                                        cellWiseStorage.data(),
                                        cellWiseStorage.data(),
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
                                                  memorySpace> &waveFunc,
                 const size_type waveFuncBatchSize)
    {
      std::shared_ptr<const quadrature::QuadratureRuleContainer>
        quadRuleContainer = d_feBasisDataStorage->getQuadratureRuleContainer();

      d_energy = 0;

      linearAlgebra::MultiVector<ValueType, memorySpace> psiBatch(
        waveFunc.getMPIPatternP2P(),
        d_linAlgOpContext,
        waveFuncBatchSize,
        ValueType());

      quadrature::QuadratureValuesContainer<ValueType, memorySpace> gradPsi(
        quadRuleContainer, waveFuncBatchSize * dim);

      quadrature::QuadratureValuesContainer<ValueType, memorySpace>
        gradPsiModSq(quadRuleContainer, waveFuncBatchSize * dim);

      quadrature::QuadratureValuesContainer<ValueType, memorySpace>
        gradPsiModSqJxW(quadRuleContainer, waveFuncBatchSize * dim);

      utils::MemoryTransfer<memorySpace, memorySpace> memoryTransfer;

      for (size_type psiStartId = 0;
           psiStartId < waveFunc.getNumberComponents();
           psiStartId += waveFuncBatchSize)
        {
          const size_type psiEndId = std::min(psiStartId + waveFuncBatchSize,
                                              waveFunc.getNumberComponents());
          const size_type numPsiInBatch = psiEndId - psiStartId;

          std::vector<RealType> occupationInBatch(numPsiInBatch, 0);

          std::copy(occupation.data() + psiStartId,
                    occupation.data() + psiEndId,
                    occupationInBatch.begin());

          if (numPsiInBatch < waveFuncBatchSize)
            {
              gradPsi.reinit(quadRuleContainer, numPsiInBatch * dim);

              gradPsiModSq.reinit(quadRuleContainer, numPsiInBatch * dim);

              gradPsiModSqJxW.reinit(quadRuleContainer, numPsiInBatch * dim);

              linearAlgebra::MultiVector<ValueType, memorySpace> psiBatchSmall(
                waveFunc.getMPIPatternP2P(),
                d_linAlgOpContext,
                numPsiInBatch,
                ValueType());

              for (size_type iSize = 0; iSize < waveFunc.localSize(); iSize++)
                memoryTransfer.copy(numPsiInBatch,
                                    psiBatchSmall.data() +
                                      numPsiInBatch * iSize,
                                    waveFunc.data() +
                                      iSize * waveFunc.getNumberComponents() +
                                      psiStartId);

              d_feBasisOp->interpolateWithBasisGradient(psiBatchSmall,
                                                        feBMPsi,
                                                        gradPsi);
            }
          else
            {
              for (size_type iSize = 0; iSize < waveFunc.localSize(); iSize++)
                memoryTransfer.copy(numPsiInBatch,
                                    psiBatch.data() + numPsiInBatch * iSize,
                                    waveFunc.data() +
                                      iSize * waveFunc.getNumberComponents() +
                                      psiStartId);

              d_feBasisOp->interpolateWithBasisGradient(psiBatch,
                                                        feBMPsi,
                                                        gradPsi);
            }

          linearAlgebra::blasLapack::
            hadamardProduct<ValueType, ValueType, memorySpace>(
              gradPsi.nEntries(),
              gradPsi.begin(),
              gradPsi.begin(),
              linearAlgebra::blasLapack::ScalarOp::Conj,
              linearAlgebra::blasLapack::ScalarOp::Identity,
              gradPsiModSq.begin(),
              *d_linAlgOpContext);

          auto jxwStorage = d_feBasisDataStorage->getJxWInAllCells();

          const size_type numLocallyOwnedCells = feBMPsi.nLocallyOwnedCells();

          std::vector<linearAlgebra::blasLapack::ScalarOp> scalarOpA(
            numLocallyOwnedCells,
            linearAlgebra::blasLapack::ScalarOp::Identity);
          std::vector<linearAlgebra::blasLapack::ScalarOp> scalarOpB(
            numLocallyOwnedCells,
            linearAlgebra::blasLapack::ScalarOp::Identity);
          std::vector<size_type> mTmp(numLocallyOwnedCells, 0);
          std::vector<size_type> nTmp(numLocallyOwnedCells, 0);
          std::vector<size_type> kTmp(numLocallyOwnedCells, 0);
          std::vector<size_type> stATmp(numLocallyOwnedCells, 0);
          std::vector<size_type> stBTmp(numLocallyOwnedCells, 0);
          std::vector<size_type> stCTmp(numLocallyOwnedCells, 0);

          for (size_type iCell = 0; iCell < numLocallyOwnedCells; iCell++)
            {
              mTmp[iCell]   = 1;
              nTmp[iCell]   = numPsiInBatch * dim;
              kTmp[iCell]   = quadRuleContainer->nCellQuadraturePoints(iCell);
              stATmp[iCell] = mTmp[iCell] * kTmp[iCell];
              stBTmp[iCell] = nTmp[iCell] * kTmp[iCell];
              stCTmp[iCell] = mTmp[iCell] * nTmp[iCell] * kTmp[iCell];
            }

          utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>
            memoryTransferHost;

          utils::MemoryStorage<size_type, memorySpace> mSize(
            numLocallyOwnedCells);
          utils::MemoryStorage<size_type, memorySpace> nSize(
            numLocallyOwnedCells);
          utils::MemoryStorage<size_type, memorySpace> kSize(
            numLocallyOwnedCells);
          utils::MemoryStorage<size_type, memorySpace> stA(
            numLocallyOwnedCells);
          utils::MemoryStorage<size_type, memorySpace> stB(
            numLocallyOwnedCells);
          utils::MemoryStorage<size_type, memorySpace> stC(
            numLocallyOwnedCells);
          memoryTransferHost.copy(numLocallyOwnedCells,
                                  mSize.data(),
                                  mTmp.data());
          memoryTransferHost.copy(numLocallyOwnedCells,
                                  nSize.data(),
                                  nTmp.data());
          memoryTransferHost.copy(numLocallyOwnedCells,
                                  kSize.data(),
                                  kTmp.data());
          memoryTransferHost.copy(numLocallyOwnedCells,
                                  stA.data(),
                                  stATmp.data());
          memoryTransferHost.copy(numLocallyOwnedCells,
                                  stB.data(),
                                  stBTmp.data());
          memoryTransferHost.copy(numLocallyOwnedCells,
                                  stC.data(),
                                  stCTmp.data());

          linearAlgebra::blasLapack::
            scaleStridedVarBatched<ValueTypeBasisData, ValueType, memorySpace>(
              numLocallyOwnedCells,
              scalarOpA.data(),
              scalarOpB.data(),
              stA.data(),
              stB.data(),
              stC.data(),
              mSize.data(),
              nSize.data(),
              kSize.data(),
              jxwStorage.data(),
              gradPsiModSq.begin(),
              gradPsiModSqJxW.begin(),
              *d_linAlgOpContext);

          std::vector<RealType> integralModGradPsiSq(numPsiInBatch),
            energy(numPsiInBatch);

          for (size_type iCell = 0; iCell < gradPsiModSqJxW.nCells(); iCell++)
            {
              for (size_type iDim = 0; iDim < dim; iDim++)
                {
                  for (size_type iComp = 0; iComp < numPsiInBatch; iComp++)
                    {
                      std::vector<ValueType> a(
                        quadRuleContainer->nCellQuadraturePoints(iCell));
                      gradPsiModSqJxW
                        .template getCellQuadValues<utils::MemorySpace::HOST>(
                          iCell, iDim * numPsiInBatch + iComp, a.data());
                      integralModGradPsiSq[iComp] +=
                        (RealType)(utils::realPart<ValueType>(
                          std::accumulate(a.begin(), a.end(), (ValueType)0)));
                    }
                }
            }

          int mpierr = utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
            utils::mpi::MPIInPlace,
            integralModGradPsiSq.data(),
            integralModGradPsiSq.size(),
            utils::mpi::Types<RealType>::getMPIDatatype(),
            utils::mpi::MPISum,
            waveFunc.getMPIPatternP2P()->mpiCommunicator());

          linearAlgebra::blasLapack::
            hadamardProduct<RealType, RealType, utils::MemorySpace::HOST>(
              integralModGradPsiSq.size(),
              integralModGradPsiSq.data(),
              occupationInBatch.data(),
              linearAlgebra::blasLapack::ScalarOp::Identity,
              linearAlgebra::blasLapack::ScalarOp::Identity,
              energy.data(),
              *d_linAlgOpContext);

          d_energy +=
            std::accumulate(energy.begin(), energy.end(), (RealType)0);

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

  } // end of namespace ksdft
} // end of namespace dftefe
