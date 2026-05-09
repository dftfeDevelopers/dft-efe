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
        const size_type batchSize,
        const std::pair<size_type, size_type> cellRange,
        const RealType* occupationInBatch,
        ValueType *psiBatchQuad,
        RealType *modPsiSqBatchQuad,
        std::shared_ptr<const quadrature::QuadratureRuleContainer>
          quadRuleContainer,
        RealType *rhoBatch,
        linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext)
    {
        size_type cumulativeQuadInCell = 0, cumulativeQuadPsiInCell = 0;
        for (size_type iCell = cellRange.first; iCell < cellRange.second; iCell++)
          {
            const size_type numQuadInCell =
              quadRuleContainer->nCellQuadraturePoints(iCell);
            for (size_type iQuad = 0; iQuad < numQuadInCell; iQuad++)
              {
                RealType b = 0;
                for (size_type i = 0; i < batchSize; i++)
                  {
                    const ValueType psi =
                      psiBatchQuad[cumulativeQuadPsiInCell +
                                   batchSize * iQuad + i];
                    const RealType absSqPsi = utils::absSq(psi);
                    modPsiSqBatchQuad[cumulativeQuadPsiInCell +
                                      batchSize * iQuad + i] = absSqPsi;
                    b += 2.0 * absSqPsi * occupationInBatch[i];
                  }
                rhoBatch[cumulativeQuadInCell + iQuad] = b;
              }
            cumulativeQuadPsiInCell += numQuadInCell * batchSize;
            cumulativeQuadInCell += numQuadInCell;
          }
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

  } // end of namespace ksdft
} // end of namespace dftefe
