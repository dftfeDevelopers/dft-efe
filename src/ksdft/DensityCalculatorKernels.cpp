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

#include <ksdft/DensityCalculatorKernels.h>
#include <utils/DataTypeOverloads.h>

namespace dftefe
{
  namespace ksdft
  {
    template <typename ValueType, typename RealType, utils::MemorySpace memorySpace>
    void
    DensityCalculatorKernels<ValueType, RealType, memorySpace>::computeRhoInBatch(
        const utils::MemoryStorage<RealType, memorySpace> &occupationInBatch,
        quadrature::QuadratureValuesContainer<ValueType, memorySpace>
          &psiBatchQuad,
        quadrature::QuadratureValuesContainer<RealType, memorySpace>
          &modPsiSqBatchQuad,
        std::shared_ptr<const quadrature::QuadratureRuleContainer>
          quadRuleContainer,
        quadrature::QuadratureValuesContainer<RealType, memorySpace> &rhoBatch,
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
                    b += 2.0 * utils::absSq(psi) * *(occupationInBatch.data() + i);
                  }
                rhoBatchIter[cumulativeQuadInCell + iQuad] = b;
              }
            cumulativeQuadPsiInCell += numQuadInCell * numPsiInBatch;
            cumulativeQuadInCell += numQuadInCell;
          }

        // // gemm for fi * mod psi^2

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

    template class DensityCalculatorKernels<double, double,
                                            dftefe::utils::MemorySpace::HOST>;
    template class DensityCalculatorKernels<float, double,
                                            dftefe::utils::MemorySpace::HOST>;
    template class DensityCalculatorKernels<std::complex<double>, double,
                                            dftefe::utils::MemorySpace::HOST>;
    template class DensityCalculatorKernels<std::complex<float>, double,
                                            dftefe::utils::MemorySpace::HOST>;

#ifdef DFTEFE_WITH_DEVICE
    template class DensityCalculatorKernels<
      double, double,
      dftefe::utils::MemorySpace::HOST_PINNED>;
    template class DensityCalculatorKernels<
      float, double,
      dftefe::utils::MemorySpace::HOST_PINNED>;
    template class DensityCalculatorKernels<
      std::complex<double>, double,
      dftefe::utils::MemorySpace::HOST_PINNED>;
    template class DensityCalculatorKernels<
      std::complex<float>, double,
      dftefe::utils::MemorySpace::HOST_PINNED>;
#endif

  } // end of namespace basis
} // end of namespace dftefe
