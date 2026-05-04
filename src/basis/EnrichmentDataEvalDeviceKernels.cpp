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
#include <atoms/SphericalDataNumerical.h>

namespace dftefe
{
  namespace basis
  {
    namespace
    {
      DFTEFE_CREATE_KERNEL(
      void,
      evalEnrichmentInCell,
      {
        for(size_type iThread = globalThreadId ; iThread < numEnrichInCell * numQuadInCell ;
              iThread += nThreadsPerBlock * nThreadBlock)
        {
          size_type enrichId = iThread % numEnrichInCell;
          size_type quadId   = iThread / numEnrichInCell;

          output[iThread] = sphericalDataFunc[enrichId].getValue(quadPtsInCell + quadId * 3, origin + enrichId * 3);
        }
      },
      const double * quadPtsInCell,
      const double *origin,
      const size_type numEnrichInCell,
      const size_type numQuadInCell,
      const atoms::SphericalDataNumerical::Func<utils::MemorySpace::DEVICE> * sphericalDataFunc,
      double *output);
      DFTEFE_CREATE_KERNEL(
      void,
      evalEnrichmentGradientInCell,
      {
        for(size_type iThread = globalThreadId ; iThread < numEnrichInCell * numQuadInCell ;
              iThread += nThreadsPerBlock * nThreadBlock)
        {
          // layout: quadId (slow) x dim x enrichId (fast)
          size_type enrichId = iThread % numEnrichInCell;
          size_type quadId   = iThread / numEnrichInCell;

          double grad[3];
          sphericalDataFunc[enrichId].getGradientValue(
            quadPtsInCell + quadId * 3,
            origin + enrichId * 3,
            grad);
          output[quadId * 3 * numEnrichInCell + 0 * numEnrichInCell + enrichId] = grad[0];
          output[quadId * 3 * numEnrichInCell + 1 * numEnrichInCell + enrichId] = grad[1];
          output[quadId * 3 * numEnrichInCell + 2 * numEnrichInCell + enrichId] = grad[2];
        }
      },
      const double * quadPtsInCell,
      const double *origin,
      const size_type numEnrichInCell,
      const size_type numQuadInCell,
      const atoms::SphericalDataNumerical::Func<utils::MemorySpace::DEVICE> * sphericalDataFunc,
      double *output);
    }

    void
    EnrichmentDataEvalKernels<utils::MemorySpace::DEVICE>::getEnrichmentValuesInCellRange(
      const double * quadPtsInAllCells,
      const double * originPtsInAllCells,
      std::pair<size_type, size_type> cellRange, 
      const std::vector<size_type> numEnrichIdsInAllCells,
      const std::vector<size_type> numQuadPtsInAllCells,
      const atoms::SphericalDataNumerical::Func<utils::MemorySpace::DEVICE> *sphericalDataFuncInAllCells,
      double *output,
      linearAlgebra::LinAlgOpContext<utils::MemorySpace::DEVICE> &linAlgOpContext)
    {
      const size_type numStreams  = linAlgOpContext.numBlasStreams();
      auto *          streams     = linAlgOpContext.getBlasStreamsVec();
      size_type dim = 3;

      size_type cumulativeEnrichInCellRange = 0;
      size_type cumulativeQuadPtsInCellRange = 0;
      size_type cumulativeQuadxEnrichInCellRange = 0;
      for(int iCell = 0 ; iCell < cellRange.first ; iCell++)
      {
        const size_type numEnrichInCell = numEnrichIdsInAllCells[iCell];
        const size_type numQuadInCell = numQuadPtsInAllCells[iCell];

        cumulativeEnrichInCellRange += numEnrichInCell;
        cumulativeQuadPtsInCellRange += numQuadInCell;
        cumulativeQuadxEnrichInCellRange += numEnrichInCell * numQuadInCell;
      }

      size_type cumulativeCellWithNonZeroNumEnrich = 0;
      for(int iCell = cellRange.first ; iCell < cellRange.second ; iCell++)
      {
        const size_type numEnrichInCell = numEnrichIdsInAllCells[iCell];
        const size_type numQuadInCell = numQuadPtsInAllCells[iCell];

        if(numEnrichInCell > 0)
        {
          const size_type sid = cumulativeCellWithNonZeroNumEnrich % numStreams;
          const size_type total     = numEnrichInCell * numQuadInCell;
          const size_type blockSize = utils::DEVICE_BLOCK_SIZE;
          const size_type grid      = (total + blockSize - 1) / blockSize;

          DFTEFE_LAUNCH_KERNEL(evalEnrichmentInCell,
                              grid,
                              blockSize,
                              streams[sid],
                              quadPtsInAllCells + cumulativeQuadPtsInCellRange * dim,
                              originPtsInAllCells + cumulativeEnrichInCellRange * dim,
                              numEnrichInCell,
                              numQuadInCell,
                              sphericalDataFuncInAllCells + cumulativeEnrichInCellRange,
                              output + cumulativeQuadxEnrichInCellRange);

          cumulativeCellWithNonZeroNumEnrich += 1;
        }
        cumulativeQuadPtsInCellRange += numQuadInCell;
        cumulativeEnrichInCellRange += numEnrichInCell;
        cumulativeQuadxEnrichInCellRange += numEnrichInCell * numQuadInCell;
      }

      for (int s = 0; s < numStreams; ++s)
        utils::deviceStreamSynchronize(streams[s]);
    }

    void
    EnrichmentDataEvalKernels<utils::MemorySpace::DEVICE>::getEnrichmentGradientsInCellRange(
      const double * quadPtsInAllCells,
      const double * originPtsInAllCells,
      std::pair<size_type, size_type> cellRange,
      const std::vector<size_type> numEnrichIdsInAllCells,
      const std::vector<size_type> numQuadPtsInAllCells,
      const atoms::SphericalDataNumerical::Func<utils::MemorySpace::DEVICE> *sphericalDataFuncInAllCells,
      double *output,
      linearAlgebra::LinAlgOpContext<utils::MemorySpace::DEVICE> &linAlgOpContext)
    {
      const size_type numStreams  = linAlgOpContext.numBlasStreams();
      auto *          streams     = linAlgOpContext.getBlasStreamsVec();
      size_type dim = 3;

      size_type cumulativeEnrichInCellRange        = 0;
      size_type cumulativeQuadPtsInCellRange       = 0;
      size_type cumulativeQuadxEnrichxDimInCellRange = 0;
      for(int iCell = 0 ; iCell < cellRange.first ; iCell++)
      {
        const size_type numEnrichInCell = numEnrichIdsInAllCells[iCell];
        const size_type numQuadInCell   = numQuadPtsInAllCells[iCell];

        cumulativeEnrichInCellRange          += numEnrichInCell;
        cumulativeQuadPtsInCellRange         += numQuadInCell;
        cumulativeQuadxEnrichxDimInCellRange += numEnrichInCell * numQuadInCell * dim;
      }

      size_type cumulativeCellWithNonZeroNumEnrich = 0;
      for(int iCell = cellRange.first ; iCell < cellRange.second ; iCell++)
      {
        const size_type numEnrichInCell = numEnrichIdsInAllCells[iCell];
        const size_type numQuadInCell   = numQuadPtsInAllCells[iCell];

        if(numEnrichInCell > 0)
        {
          const size_type sid       = cumulativeCellWithNonZeroNumEnrich % numStreams;
          const size_type total     = numEnrichInCell * numQuadInCell;
          const size_type blockSize = utils::DEVICE_BLOCK_SIZE;
          const size_type grid      = (total + blockSize - 1) / blockSize;

          DFTEFE_LAUNCH_KERNEL(evalEnrichmentGradientInCell,
                              grid,
                              blockSize,
                              streams[sid],
                              quadPtsInAllCells + cumulativeQuadPtsInCellRange * dim,
                              originPtsInAllCells + cumulativeEnrichInCellRange * dim,
                              numEnrichInCell,
                              numQuadInCell,
                              sphericalDataFuncInAllCells + cumulativeEnrichInCellRange,
                              output + cumulativeQuadxEnrichxDimInCellRange);

          cumulativeCellWithNonZeroNumEnrich += 1;
        }
        cumulativeQuadPtsInCellRange         += numQuadInCell;
        cumulativeEnrichInCellRange          += numEnrichInCell;
        cumulativeQuadxEnrichxDimInCellRange += numEnrichInCell * numQuadInCell * dim;
      }

      for (int s = 0; s < numStreams; ++s)
        utils::deviceStreamSynchronize(streams[s]);
    }

    void
    EnrichmentDataEvalKernels<utils::MemorySpace::DEVICE>::getEnrichmentValues(
      const size_type  numEnrichmentFunc,
      const std::vector<size_type> &pointsPerEnrichId,
      const std::vector<std::shared_ptr<atoms::SphericalData>> &sphericalDataVec,
      const double *points,
      const double *origin,
      double * values,
      linearAlgebra::LinAlgOpContext<utils::MemorySpace::DEVICE> &linAlgOpContext)
    {
      const size_type numStreams  = linAlgOpContext.numBlasStreams();
      auto *          streams     = linAlgOpContext.getBlasStreamsVec();

      size_type cumulativeValuesOffset = 0;
      size_type cumulativeCoordsOffset = 0;
      for (int i = 0; i < numEnrichmentFunc; i++)
        {
          size_type sid = i % numStreams;
          sphericalDataVec[i]->getValueDevice(pointsPerEnrichId[i],
                                              points + cumulativeCoordsOffset,
                                              origin + i * 3,
                                              values + cumulativeValuesOffset,
                                              streams[sid]);
          cumulativeValuesOffset += pointsPerEnrichId[i];
          cumulativeCoordsOffset += pointsPerEnrichId[i] * 3;
        }
        for (int s = 0; s < numStreams; ++s)
          utils::deviceStreamSynchronize(streams[s]);
    }

    void
    EnrichmentDataEvalKernels<utils::MemorySpace::DEVICE>::getEnrichmentGradients(
      const size_type  numEnrichmentFunc,
      const std::vector<size_type> &pointsPerEnrichId,
      const std::vector<std::shared_ptr<atoms::SphericalData>> &sphericalDataVec,
      const double *points,
      const double *origin,
      double * values,
      linearAlgebra::LinAlgOpContext<utils::MemorySpace::DEVICE> &linAlgOpContext)
    {
      const size_type numStreams  = linAlgOpContext.numBlasStreams();
      auto *          streams     = linAlgOpContext.getBlasStreamsVec();

      size_type cumulativeValuesOffset = 0;
      size_type cumulativeCoordsOffset = 0;
      for (int i = 0; i < numEnrichmentFunc; i++)
        {
          size_type sid = i % numStreams;
          sphericalDataVec[i]->getGradientValueDevice(pointsPerEnrichId[i],
                                                      points + cumulativeCoordsOffset,
                                                      origin + i * 3,
                                                      values + cumulativeValuesOffset,
                                                      streams[sid]);
          cumulativeValuesOffset += pointsPerEnrichId[i] * 3;
          cumulativeCoordsOffset += pointsPerEnrichId[i] * 3;
        }
        for (int s = 0; s < numStreams; ++s)
          utils::deviceStreamSynchronize(streams[s]);
    }

  } // end of namespace basis
} // end of namespace dftefe
