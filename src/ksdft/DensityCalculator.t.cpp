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
        std::shared_ptr<const quadrature::QuadratureRuleContainer>
          quadRuleContainer,
        quadrature::QuadratureValuesContainer<RealType, memorySpace> &rhoBatch,
        size_type                                    cellBlockSize,
        size_type                                    numLocallyOwnedCells,
        linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext)
      {
        size_type numPsiInBatch = occupationInBatch.size();
        // hadamard for psi^C psi = mod psi^2
        // linearAlgebra::blasLapack::
        //   hadamardProduct<ValueType, ValueType, memorySpace>(
        //     psiBatchQuad.nEntries(),
        //     psiBatchQuad.begin(),
        //     psiBatchQuad.begin(),
        //     linearAlgebra::blasLapack::ScalarOp::Conj,
        //     linearAlgebra::blasLapack::ScalarOp::Identity,
        //     psiBatchQuad.begin(),
        //     linAlgOpContext);

        /*----------- TODO : Optimize this -------------------------------*/
        // convert to psiBatchQuad to realType and multiply by 2
        // for (size_type iCell = 0; iCell < psiBatchQuad.nCells(); iCell++)
        //   {
        //     std::vector<ValueType> a(
        //       quadRuleContainer->nCellQuadraturePoints(iCell) *
        //       numPsiInBatch);
        //     std::vector<RealType> b(
        //       quadRuleContainer->nCellQuadraturePoints(iCell) *
        //       numPsiInBatch);
        //     psiBatchQuad.template getCellValues<utils::MemorySpace::HOST>(
        //       iCell, a.data());
        //     for (size_type i = 0; i < b.size(); i++)
        //       b[i] = 2.0 * utils::realPart<RealType>(a[i]);
        //     psiModSqBatchQuad.template
        //     setCellValues<utils::MemorySpace::HOST>(
        //       iCell, b.data());
        //   }

        ValueType *psiBatchQuadIter     = psiBatchQuad.begin();
        RealType * rhoBatchIter         = rhoBatch.begin();
        size_type  cumulativeQuadInCell = 0, cumulativeQuadPsiInCell = 0;
        for (size_type iCell = 0; iCell < psiBatchQuad.nCells(); iCell++)
          {
            size_type numQuadInCell =
              quadRuleContainer->nCellQuadraturePoints(iCell);
            for (size_type iQuad = 0; iQuad < numQuadInCell; iQuad++)
              {
                RealType b = 0;
                for (size_type i = 0; i < numPsiInBatch; i++)
                  {
                    const ValueType psi =
                      psiBatchQuadIter[cumulativeQuadPsiInCell +
                                       numPsiInBatch * iQuad + i];
                    b += 2.0 * utils::absSq(psi) * occupationInBatch[i];
                  }
                rhoBatchIter[cumulativeQuadInCell + iQuad] = b;
              }
            cumulativeQuadPsiInCell += numQuadInCell * numPsiInBatch;
            cumulativeQuadInCell += numQuadInCell;
          }

        // // gemm for fi * mod psi^2

        // utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>
        //   memoryTransfer;

        // // Create occupancy in memspace
        // utils::MemoryStorage<RealType, memorySpace>
        // occupationInBatchMemspace(
        //   occupationInBatch.size());
        // memoryTransfer.copy(occupationInBatch.size(),
        //                     occupationInBatchMemspace.data(),
        //                     occupationInBatch.data());

        // size_type AStartOffset = 0;
        // size_type CStartOffset = 0;
        // for (size_type cellStartId = 0; cellStartId < numLocallyOwnedCells;
        //      cellStartId += cellBlockSize)
        //   {
        //     const size_type cellEndId =
        //       std::min(cellStartId + cellBlockSize, numLocallyOwnedCells);
        //     const size_type numCellsInBlock = cellEndId - cellStartId;

        //     RealType alpha = 1.0;
        //     RealType beta  = 0.0;

        //     RealType *C = rhoBatch.begin() + CStartOffset;

        //     size_type n = 0;
        //     for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
        //       {
        //         n +=
        //           quadRuleContainer->nCellQuadraturePoints(cellStartId +
        //           iCell);
        //       }

        //     linearAlgebra::blasLapack::gemm<RealType, RealType, memorySpace>(
        //       'T',
        //       'N',
        //       1,
        //       n,
        //       numPsiInBatch,
        //       alpha,
        //       occupationInBatchMemspace.data(),
        //       numPsiInBatch,
        //       psiModSqBatchQuad.begin() + AStartOffset,
        //       numPsiInBatch,
        //       beta,
        //       C,
        //       1,
        //       linAlgOpContext);

        //   AStartOffset +=
        //     numPsiInBatch * n;
        //   CStartOffset += n;
        // }
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

              d_feBasisOp->reinit(d_cellBlockSize, d_waveFuncBatchSize);
              d_feBasisOp->interpolate(*d_psiBatch,
                                       *d_feBMPsi,
                                       *d_psiBatchQuad);

              DensityCalculatorInternal::
                computeRhoInBatch<ValueType, RealType, memorySpace>(
                  occupationInBatch,
                  *d_psiBatchQuad,
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

              d_feBasisOp->reinit(d_cellBlockSize, d_batchSizeSmall);
              d_feBasisOp->interpolate(*d_psiBatchSmall,
                                       *d_feBMPsi,
                                       *d_psiBatchSmallQuad);

              DensityCalculatorInternal::
                computeRhoInBatch<ValueType, RealType, memorySpace>(
                  occupationInBatch,
                  *d_psiBatchSmallQuad,
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

              d_feBasisOp->reinit(d_cellBlockSize, d_batchSizeSmall);
              d_feBasisOp->interpolate(*d_psiBatchSmall,
                                       *d_feBMPsi,
                                       *d_psiBatchSmallQuad);

              DensityCalculatorInternal::
                computeRhoInBatch<ValueType, RealType, memorySpace>(
                  occupationInBatch,
                  *d_psiBatchSmallQuad,
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
