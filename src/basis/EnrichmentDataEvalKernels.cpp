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

#include <basis/EnrichmentDataEvalKernels.h>

namespace dftefe
{
  namespace basis
  {
    template <utils::MemorySpace memorySpace>
    void
    EnrichmentDataEvalKernels<memorySpace>::getEnrichmentValues(
      const size_type               numEnrichmentFunc,
      const std::vector<size_type> &pointsPerEnrichId,
      const std::vector<std::shared_ptr<atoms::SphericalData>>
        &                                          sphericalDataVec,
      const double *                               points,
      const double *                               origin,
      double *                                     values,
      linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext)
    {
      size_type cumulativeValuesOffset = 0;
      size_type cumulativeCoordsOffset = 0;
      for (int i = 0; i < numEnrichmentFunc; i++)
        {
          sphericalDataVec[i]->getValue(pointsPerEnrichId[i],
                                        points + cumulativeCoordsOffset,
                                        origin + i * 3,
                                        values + cumulativeValuesOffset);
          cumulativeValuesOffset += pointsPerEnrichId[i];
          cumulativeCoordsOffset += pointsPerEnrichId[i] * 3;
        }
    }

    template <utils::MemorySpace memorySpace>
    void
    EnrichmentDataEvalKernels<memorySpace>::getEnrichmentGradients(
      const size_type               numEnrichmentFunc,
      const std::vector<size_type> &pointsPerEnrichId,
      const std::vector<std::shared_ptr<atoms::SphericalData>>
        &                                          sphericalDataVec,
      const double *                               points,
      const double *                               origin,
      double *                                     values,
      linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext)
    {
      size_type cumulativeValuesOffset = 0;
      size_type cumulativeCoordsOffset = 0;
      for (int i = 0; i < numEnrichmentFunc; i++)
        {
          sphericalDataVec[i]->getGradientValue(pointsPerEnrichId[i],
                                                points + cumulativeCoordsOffset,
                                                origin + i * 3,
                                                values +
                                                  cumulativeValuesOffset);
          cumulativeValuesOffset += pointsPerEnrichId[i] * 3;
          cumulativeCoordsOffset += pointsPerEnrichId[i] * 3;
        }
    }

    template <utils::MemorySpace memorySpace>
    void
    EnrichmentDataEvalKernels<memorySpace>::getEnrichmentValuesInCellRange(
      const double *                  quadPtsInAllCells,
      const double *                  originPtsInAllCells,
      std::pair<size_type, size_type> cellRange,
      const std::vector<size_type>    numEnrichIdsInAllCells,
      const std::vector<size_type>    numQuadPtsInAllCells,
      const atoms::SphericalDataNumerical::Func<memorySpace>
        *                                          sphericalDataFuncInAllCells,
      double *                                     output,
      linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext)
    {
      const size_type dim = 3;

      size_type cumulativeEnrich = 0;
      size_type cumulativeQuad   = 0;
      for (size_type iCell = 0; iCell < cellRange.first; iCell++)
        {
          cumulativeEnrich += numEnrichIdsInAllCells[iCell];
          cumulativeQuad += numQuadPtsInAllCells[iCell];
        }

      size_type cumulativeOutput = 0;
      for (size_type iCell = cellRange.first; iCell < cellRange.second; iCell++)
        {
          const size_type numEnrichInCell = numEnrichIdsInAllCells[iCell];
          const size_type numQuadInCell   = numQuadPtsInAllCells[iCell];

          for (size_type iThread = 0; iThread < numEnrichInCell * numQuadInCell;
               iThread++)
            {
              const size_type enrichId = iThread % numEnrichInCell;
              const size_type quadId   = iThread / numEnrichInCell;
              output[cumulativeOutput + iThread] =
                sphericalDataFuncInAllCells[cumulativeEnrich + enrichId]
                  .getValue(quadPtsInAllCells + (cumulativeQuad + quadId) * dim,
                            originPtsInAllCells +
                              (cumulativeEnrich + enrichId) * dim);
            }

          cumulativeEnrich += numEnrichInCell;
          cumulativeQuad += numQuadInCell;
          cumulativeOutput += numEnrichInCell * numQuadInCell;
        }
    }

    template <utils::MemorySpace memorySpace>
    void
    EnrichmentDataEvalKernels<memorySpace>::getEnrichmentGradientsInCellRange(
      const double *                  quadPtsInAllCells,
      const double *                  originPtsInAllCells,
      std::pair<size_type, size_type> cellRange,
      const std::vector<size_type>    numEnrichIdsInAllCells,
      const std::vector<size_type>    numQuadPtsInAllCells,
      const atoms::SphericalDataNumerical::Func<memorySpace>
        *                                          sphericalDataFuncInAllCells,
      double *                                     output,
      linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext)
    {
      const size_type dim = 3;

      size_type cumulativeEnrich = 0;
      size_type cumulativeQuad   = 0;
      for (size_type iCell = 0; iCell < cellRange.first; iCell++)
        {
          cumulativeEnrich += numEnrichIdsInAllCells[iCell];
          cumulativeQuad += numQuadPtsInAllCells[iCell];
        }

      size_type cumulativeOutput = 0;
      for (size_type iCell = cellRange.first; iCell < cellRange.second; iCell++)
        {
          const size_type numEnrichInCell = numEnrichIdsInAllCells[iCell];
          const size_type numQuadInCell   = numQuadPtsInAllCells[iCell];

          for (size_type iThread = 0; iThread < numEnrichInCell * numQuadInCell;
               iThread++)
            {
              // layout: quadId (slow) x dim x enrichId (fast)
              const size_type enrichId = iThread % numEnrichInCell;
              const size_type quadId   = iThread / numEnrichInCell;
              double          grad[3];
              sphericalDataFuncInAllCells[cumulativeEnrich + enrichId]
                .getGradientValue(quadPtsInAllCells +
                                    (cumulativeQuad + quadId) * dim,
                                  originPtsInAllCells +
                                    (cumulativeEnrich + enrichId) * dim,
                                  grad);
              output[cumulativeOutput + quadId * dim * numEnrichInCell +
                     0 * numEnrichInCell + enrichId] = grad[0];
              output[cumulativeOutput + quadId * dim * numEnrichInCell +
                     1 * numEnrichInCell + enrichId] = grad[1];
              output[cumulativeOutput + quadId * dim * numEnrichInCell +
                     2 * numEnrichInCell + enrichId] = grad[2];
            }

          cumulativeEnrich += numEnrichInCell;
          cumulativeQuad += numQuadInCell;
          cumulativeOutput += numEnrichInCell * numQuadInCell * dim;
        }
    }

    template class EnrichmentDataEvalKernels<utils::MemorySpace::HOST>;
  } // end of namespace basis
} // end of namespace dftefe
