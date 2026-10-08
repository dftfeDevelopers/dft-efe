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
#include <atoms/AtomSuperpositionFunctionEvalKernels.h>
#include <atoms/SphericalDataNumerical.h>
#include <utils/Exceptions.h>
#include <complex>
#include <cmath>

namespace dftefe
{
  namespace basis
  {
    namespace
    {
      // separated here, not in the kernels, whose BODY is one macro argument
      DFTEFE_FORCEINLINE DFTEFE_DEVICE void
      getZero(double *out)
      {
        *out = 0.0;
      }

      DFTEFE_FORCEINLINE DFTEFE_DEVICE void
      getZero(utils::deviceDoubleComplex *out)
      {
        *out = utils::makeComplex(0.0, 0.0);
      }

      DFTEFE_FORCEINLINE DFTEFE_DEVICE void
      getBlochPhase(double *out,
                    const double *kPointCoord,
                    const double *point,
                    const double *origin)
      {
        *out = 1.0;
      }

      DFTEFE_FORCEINLINE DFTEFE_DEVICE void
      getBlochPhase(utils::deviceDoubleComplex *out,
                    const double *             kPointCoord,
                    const double *             point,
                    const double *             origin)
      {
        double kDotR = 0.0;
        for (size_type iDim = 0; iDim < 3; iDim++)
          kDotR += kPointCoord[iDim] * (point[iDim] - origin[iDim]);
        *out = utils::makeComplex(cos(kDotR), -sin(kDotR));
      }

      // the (grad psi - i k psi) bracket; utils::add has no mixed overload
      DFTEFE_FORCEINLINE DFTEFE_DEVICE void
      getBlochGradientTerm(double *        out,
                           const double *  kPointCoord,
                           const size_type iDim,
                           const double    enrichmentValue,
                           const double    gradientValue)
      {
        *out = gradientValue;
      }

      DFTEFE_FORCEINLINE DFTEFE_DEVICE void
      getBlochGradientTerm(utils::deviceDoubleComplex *out,
                           const double *              kPointCoord,
                           const size_type             iDim,
                           const double                enrichmentValue,
                           const double                gradientValue)
      {
        *out = utils::makeComplex(gradientValue,
                                  -kPointCoord[iDim] * enrichmentValue);
      }

      template <typename ValueType>
      DFTEFE_CREATE_KERNEL(
        void,
        evalEnrichmentInCell,
        {
          for (size_type iThread = globalThreadId;
               iThread < numEnrichInCell * numQuadInCell;
               iThread += nThreadsPerBlock * nThreadBlock)
            {
              // quadId is the fast index so that enrichId, and with it the
              // origin loop bounds below, are uniform across a warp. The
              // stored layout is unchanged: quadId (slow) x enrichId (fast).
              size_type quadId   = iThread % numQuadInCell;
              size_type enrichId = iThread / numQuadInCell;

              // Without periodicity every enrichment has exactly one origin,
              // so the offsets are 0,1,2,... and this loop runs once.
              // the origin loop is the lattice sum, one phase per image
              ValueType value;
              getZero(&value);
              for (size_type iOrigin = originOffsetPerEnrich[enrichId];
                   iOrigin < originOffsetPerEnrich[enrichId + 1];
                   iOrigin++)
                {
                  ValueType phase;
                  getBlochPhase(&phase,
                                kPointCoord,
                                quadPtsInCell + quadId * 3,
                                origin + iOrigin * 3);
                  value = utils::add(
                    value,
                    utils::mult(phase,
                                sphericalDataFunc[enrichId].getValue(
                                  quadPtsInCell + quadId * 3,
                                  origin + iOrigin * 3)));
                }

              utils::copyValue(output + quadId * numEnrichInCell + enrichId,
                               value);
            }
        },
        const double *   quadPtsInCell,
        const double *   origin,
        const size_type *originOffsetPerEnrich,
        const size_type  numEnrichInCell,
        const size_type  numQuadInCell,
        const atoms::SphericalDataNumerical::Func<utils::MemorySpace::DEVICE>
          *                   sphericalDataFunc,
        ValueType *output,
        const double *kPointCoord);

      template <typename ValueType>
      DFTEFE_CREATE_KERNEL(
        void,
        evalEnrichmentGradientInCell,
        {
          for (size_type iThread = globalThreadId;
               iThread < numEnrichInCell * numQuadInCell;
               iThread += nThreadsPerBlock * nThreadBlock)
            {
              // layout: quadId (slow) x dim x enrichId (fast)
              // quadId is the fast thread index so that enrichId, and with it
              // the origin loop bounds below, are uniform across a warp.
              size_type quadId   = iThread % numQuadInCell;
              size_type enrichId = iThread / numQuadInCell;

              // Without periodicity every enrichment has exactly one origin,
              // so the offsets are 0,1,2,... and this loop runs once.
              ValueType grad[3];
              getZero(&grad[0]);
              getZero(&grad[1]);
              getZero(&grad[2]);
              for (size_type iOrigin = originOffsetPerEnrich[enrichId];
                   iOrigin < originOffsetPerEnrich[enrichId + 1];
                   iOrigin++)
                {
                  double originGrad[3];
                  sphericalDataFunc[enrichId].getGradientValue(
                    quadPtsInCell + quadId * 3,
                    origin + iOrigin * 3,
                    originGrad);
                  // the product rule needs psi, evaluated a second time
                  ValueType phase;
                  getBlochPhase(&phase,
                                kPointCoord,
                                quadPtsInCell + quadId * 3,
                                origin + iOrigin * 3);
                  const double enrichmentValue =
                    sphericalDataFunc[enrichId].getValue(
                      quadPtsInCell + quadId * 3, origin + iOrigin * 3);
                  ValueType blochGradientTerm;
                  getBlochGradientTerm(&blochGradientTerm,
                                       kPointCoord,
                                       0,
                                       enrichmentValue,
                                       originGrad[0]);
                  grad[0] =
                    utils::add(grad[0], utils::mult(phase, blochGradientTerm));
                  getBlochGradientTerm(&blochGradientTerm,
                                       kPointCoord,
                                       1,
                                       enrichmentValue,
                                       originGrad[1]);
                  grad[1] =
                    utils::add(grad[1], utils::mult(phase, blochGradientTerm));
                  getBlochGradientTerm(&blochGradientTerm,
                                       kPointCoord,
                                       2,
                                       enrichmentValue,
                                       originGrad[2]);
                  grad[2] =
                    utils::add(grad[2], utils::mult(phase, blochGradientTerm));
                }
              for (size_type iDim = 0; iDim < 3; iDim++)
                utils::copyValue(output + quadId * 3 * numEnrichInCell +
                                   iDim * numEnrichInCell + enrichId,
                                 grad[iDim]);
            }
        },
        const double *   quadPtsInCell,
        const double *   origin,
        const size_type *originOffsetPerEnrich,
        const size_type  numEnrichInCell,
        const size_type  numQuadInCell,
        const atoms::SphericalDataNumerical::Func<utils::MemorySpace::DEVICE>
          *                   sphericalDataFunc,
        ValueType *output,
        const double *kPointCoord);


      // widens the real evaluation into ValueType and applies the phase
      template <typename ValueType>
      DFTEFE_CREATE_KERNEL(
        void,
        applyBlochPhaseToEnrichment,
        {
          for (size_type iThread = globalThreadId; iThread < numPoints;
               iThread += nThreadsPerBlock * nThreadBlock)
            {
              ValueType phase;
              getBlochPhase(&phase, kPointCoord, points + iThread * 3, origin);
              utils::copyValue(output + iThread,
                               utils::mult(phase, enrichmentValues[iThread]));
            }
        },
        const double *  enrichmentValues,
        const double *  points,
        const double *  origin,
        const size_type numPoints,
        ValueType *     output,
        const double *  kPointCoord);

      template <typename ValueType>
      DFTEFE_CREATE_KERNEL(
        void,
        applyBlochPhaseToEnrichmentGradient,
        {
          for (size_type iThread = globalThreadId; iThread < numPoints;
               iThread += nThreadsPerBlock * nThreadBlock)
            {
              ValueType phase;
              getBlochPhase(&phase, kPointCoord, points + iThread * 3, origin);
              for (size_type iDim = 0; iDim < 3; iDim++)
                {
                  ValueType blochGradientTerm;
                  getBlochGradientTerm(&blochGradientTerm,
                                       kPointCoord,
                                       iDim,
                                       enrichmentValues[iThread],
                                       gradientValues[iThread * 3 + iDim]);
                  utils::copyValue(output + iThread * 3 + iDim,
                                   utils::mult(phase, blochGradientTerm));
                }
            }
        },
        const double *  gradientValues,
        const double *  enrichmentValues,
        const double *  points,
        const double *  origin,
        const size_type numPoints,
        ValueType *     output,
        const double *  kPointCoord);
    } // namespace

    template <typename ValueType>
    void
    EnrichmentDataEvalKernels<ValueType,
                              utils::MemorySpace::DEVICE>::
      getEnrichmentValuesInCellRange(
        const double *                  quadPtsInAllCells,
        const double *                  originPtsInAllCells,
        const size_type *               originOffsetPerCellEnrich,
        std::pair<size_type, size_type> cellRange,
        const std::vector<size_type>    numEnrichIdsInAllCells,
        const std::vector<size_type>    numQuadPtsInAllCells,
        const atoms::SphericalDataNumerical::Func<utils::MemorySpace::DEVICE>
          *     sphericalDataFuncInAllCells,
        ValueType *output,
        linearAlgebra::LinAlgOpContext<utils::MemorySpace::DEVICE>
          &linAlgOpContext,
        const double *kPointCoord)
    {
      const size_type numStreams = linAlgOpContext.numBlasStreams();
      auto *          streams    = linAlgOpContext.getBlasStreamsVec();
      size_type       dim        = 3;

      size_type cumulativeEnrichInCellRange      = 0;
      size_type cumulativeQuadPtsInCellRange     = 0;
      size_type cumulativeQuadxEnrichInCellRange = 0;
      for (int iCell = 0; iCell < cellRange.first; iCell++)
        {
          const size_type numEnrichInCell = numEnrichIdsInAllCells[iCell];
          const size_type numQuadInCell   = numQuadPtsInAllCells[iCell];

          cumulativeEnrichInCellRange += numEnrichInCell;
          cumulativeQuadPtsInCellRange += numQuadInCell;
        }

      size_type cumulativeCellWithNonZeroNumEnrich = 0;
      for (int iCell = cellRange.first; iCell < cellRange.second; iCell++)
        {
          const size_type numEnrichInCell = numEnrichIdsInAllCells[iCell];
          const size_type numQuadInCell   = numQuadPtsInAllCells[iCell];

          if (numEnrichInCell > 0)
            {
              const size_type sid =
                cumulativeCellWithNonZeroNumEnrich % numStreams;
              const size_type total     = numEnrichInCell * numQuadInCell;
              const size_type blockSize = utils::DEVICE_BLOCK_SIZE;
              const size_type grid      = (total + blockSize - 1) / blockSize;

              DFTEFE_LAUNCH_KERNEL(
                evalEnrichmentInCell,
                grid,
                blockSize,
                streams[sid],
                quadPtsInAllCells + cumulativeQuadPtsInCellRange * dim,
                // Origin offsets hold absolute indices, so the origin array is
                // passed whole and only the offsets are sliced to this cell.
                originPtsInAllCells,
                originOffsetPerCellEnrich + cumulativeEnrichInCellRange,
                numEnrichInCell,
                numQuadInCell,
                sphericalDataFuncInAllCells + cumulativeEnrichInCellRange,
                utils::makeDataTypeDeviceCompatible(output + cumulativeQuadxEnrichInCellRange),
                kPointCoord);

              cumulativeCellWithNonZeroNumEnrich += 1;
            }
          cumulativeQuadPtsInCellRange += numQuadInCell;
          cumulativeEnrichInCellRange += numEnrichInCell;
          cumulativeQuadxEnrichInCellRange += numEnrichInCell * numQuadInCell;
        }

      for (int s = 0; s < numStreams; ++s)
        {
          utils::deviceError_t err = utils::deviceStreamSynchronize(streams[s]);
          DEVICE_API_CHECK(err);
        }
    }

    template <typename ValueType>
    void
    EnrichmentDataEvalKernels<ValueType,
                              utils::MemorySpace::DEVICE>::
      getEnrichmentGradientsInCellRange(
        const double *                  quadPtsInAllCells,
        const double *                  originPtsInAllCells,
        const size_type *               originOffsetPerCellEnrich,
        std::pair<size_type, size_type> cellRange,
        const std::vector<size_type>    numEnrichIdsInAllCells,
        const std::vector<size_type>    numQuadPtsInAllCells,
        const atoms::SphericalDataNumerical::Func<utils::MemorySpace::DEVICE>
          *     sphericalDataFuncInAllCells,
        ValueType *output,
        linearAlgebra::LinAlgOpContext<utils::MemorySpace::DEVICE>
          &linAlgOpContext,
        const double *kPointCoord)
    {
      const size_type numStreams = linAlgOpContext.numBlasStreams();
      auto *          streams    = linAlgOpContext.getBlasStreamsVec();
      size_type       dim        = 3;

      size_type cumulativeEnrichInCellRange          = 0;
      size_type cumulativeQuadPtsInCellRange         = 0;
      size_type cumulativeQuadxEnrichxDimInCellRange = 0;
      for (int iCell = 0; iCell < cellRange.first; iCell++)
        {
          const size_type numEnrichInCell = numEnrichIdsInAllCells[iCell];
          const size_type numQuadInCell   = numQuadPtsInAllCells[iCell];

          cumulativeEnrichInCellRange += numEnrichInCell;
          cumulativeQuadPtsInCellRange += numQuadInCell;
        }

      size_type cumulativeCellWithNonZeroNumEnrich = 0;
      for (int iCell = cellRange.first; iCell < cellRange.second; iCell++)
        {
          const size_type numEnrichInCell = numEnrichIdsInAllCells[iCell];
          const size_type numQuadInCell   = numQuadPtsInAllCells[iCell];

          if (numEnrichInCell > 0)
            {
              const size_type sid =
                cumulativeCellWithNonZeroNumEnrich % numStreams;
              const size_type total     = numEnrichInCell * numQuadInCell;
              const size_type blockSize = utils::DEVICE_BLOCK_SIZE;
              const size_type grid      = (total + blockSize - 1) / blockSize;

              DFTEFE_LAUNCH_KERNEL(
                evalEnrichmentGradientInCell,
                grid,
                blockSize,
                streams[sid],
                quadPtsInAllCells + cumulativeQuadPtsInCellRange * dim,
                // Origin offsets hold absolute indices, so the origin array is
                // passed whole and only the offsets are sliced to this cell.
                originPtsInAllCells,
                originOffsetPerCellEnrich + cumulativeEnrichInCellRange,
                numEnrichInCell,
                numQuadInCell,
                sphericalDataFuncInAllCells + cumulativeEnrichInCellRange,
                utils::makeDataTypeDeviceCompatible(output + cumulativeQuadxEnrichxDimInCellRange),
                kPointCoord);

              cumulativeCellWithNonZeroNumEnrich += 1;
            }
          cumulativeQuadPtsInCellRange += numQuadInCell;
          cumulativeEnrichInCellRange += numEnrichInCell;
          cumulativeQuadxEnrichxDimInCellRange +=
            numEnrichInCell * numQuadInCell * dim;
        }

      for (int s = 0; s < numStreams; ++s)
        {
          utils::deviceError_t err = utils::deviceStreamSynchronize(streams[s]);
          DEVICE_API_CHECK(err);
        }
    }


    template <typename ValueType>
    void
    EnrichmentDataEvalKernels<ValueType,
                              utils::MemorySpace::DEVICE>::getEnrichmentValues(
      const size_type               numEnrichmentFunc,
      const std::vector<size_type> &pointsPerEnrichId,
      const std::vector<std::shared_ptr<atoms::SphericalData>>
        &           sphericalDataVec,
      const double *points,
      const double *origin,
      ValueType *   values,
      double *      scratchReal,
      linearAlgebra::LinAlgOpContext<utils::MemorySpace::DEVICE>
        &           linAlgOpContext,
      const double *kPointCoord)
    {
      const size_type numStreams = linAlgOpContext.numBlasStreams();
      auto *          streams    = linAlgOpContext.getBlasStreamsVec();

      atoms::AtomSuperpositionFunctionEvalKernels<utils::MemorySpace::DEVICE>::
        getEnrichmentValues(numEnrichmentFunc,
                            pointsPerEnrichId,
                            sphericalDataVec,
                            points,
                            origin,
                            scratchReal,
                            linAlgOpContext);

      size_type cumulativeValuesOffset = 0;
      size_type cumulativeCoordsOffset = 0;
      for (int i = 0; i < numEnrichmentFunc; i++)
        {
          const size_type sid       = i % numStreams;
          const size_type blockSize = utils::DEVICE_BLOCK_SIZE;
          const size_type grid =
            (pointsPerEnrichId[i] + blockSize - 1) / blockSize;
          DFTEFE_LAUNCH_KERNEL(
            applyBlochPhaseToEnrichment,
            grid,
            blockSize,
            streams[sid],
            scratchReal + cumulativeValuesOffset,
            points + cumulativeCoordsOffset,
            origin + i * 3,
            pointsPerEnrichId[i],
            utils::makeDataTypeDeviceCompatible(values +
                                                cumulativeValuesOffset),
            kPointCoord);
          cumulativeValuesOffset += pointsPerEnrichId[i];
          cumulativeCoordsOffset += pointsPerEnrichId[i] * 3;
        }
      for (int s = 0; s < numStreams; ++s)
        {
          utils::deviceError_t err = utils::deviceStreamSynchronize(streams[s]);
          DEVICE_API_CHECK(err);
        }
    }

    template <typename ValueType>
    void
    EnrichmentDataEvalKernels<ValueType, utils::MemorySpace::DEVICE>::
      getEnrichmentGradients(
        const size_type               numEnrichmentFunc,
        const std::vector<size_type> &pointsPerEnrichId,
        const std::vector<std::shared_ptr<atoms::SphericalData>>
          &           sphericalDataVec,
        const double *points,
        const double *origin,
        ValueType *   values,
        double *      scratchReal,
        linearAlgebra::LinAlgOpContext<utils::MemorySpace::DEVICE>
          &           linAlgOpContext,
        const double *kPointCoord)
    {
      const size_type numStreams = linAlgOpContext.numBlasStreams();
      auto *          streams    = linAlgOpContext.getBlasStreamsVec();

      size_type numValues = 0;
      for (int i = 0; i < numEnrichmentFunc; i++)
        numValues += pointsPerEnrichId[i] * 3;

      atoms::AtomSuperpositionFunctionEvalKernels<utils::MemorySpace::DEVICE>::
        getEnrichmentGradients(numEnrichmentFunc,
                               pointsPerEnrichId,
                               sphericalDataVec,
                               points,
                               origin,
                               scratchReal,
                               linAlgOpContext);

      // the -i k psi term needs a complex ValueType, so a real run skips psi
      double *enrichmentValues = scratchReal + numValues;
      if (!std::is_same<ValueType,
                        linearAlgebra::blasLapack::real_type<ValueType>>::value)
        atoms::AtomSuperpositionFunctionEvalKernels<
          utils::MemorySpace::DEVICE>::getEnrichmentValues(numEnrichmentFunc,
                                                           pointsPerEnrichId,
                                                           sphericalDataVec,
                                                           points,
                                                           origin,
                                                           enrichmentValues,
                                                           linAlgOpContext);

      size_type cumulativeValuesOffset     = 0;
      size_type cumulativeCoordsOffset     = 0;
      size_type cumulativeEnrichmentOffset = 0;
      for (int i = 0; i < numEnrichmentFunc; i++)
        {
          const size_type sid       = i % numStreams;
          const size_type blockSize = utils::DEVICE_BLOCK_SIZE;
          const size_type grid =
            (pointsPerEnrichId[i] + blockSize - 1) / blockSize;
          DFTEFE_LAUNCH_KERNEL(
            applyBlochPhaseToEnrichmentGradient,
            grid,
            blockSize,
            streams[sid],
            scratchReal + cumulativeValuesOffset,
            enrichmentValues + cumulativeEnrichmentOffset,
            points + cumulativeCoordsOffset,
            origin + i * 3,
            pointsPerEnrichId[i],
            utils::makeDataTypeDeviceCompatible(values +
                                                cumulativeValuesOffset),
            kPointCoord);
          cumulativeValuesOffset += pointsPerEnrichId[i] * 3;
          cumulativeCoordsOffset += pointsPerEnrichId[i] * 3;
          cumulativeEnrichmentOffset += pointsPerEnrichId[i];
        }
      for (int s = 0; s < numStreams; ++s)
        {
          utils::deviceError_t err = utils::deviceStreamSynchronize(streams[s]);
          DEVICE_API_CHECK(err);
        }
    }
    
    template class EnrichmentDataEvalKernels<double,
                                             utils::MemorySpace::DEVICE>;
    template class EnrichmentDataEvalKernels<std::complex<double>,
                                             utils::MemorySpace::DEVICE>;
  } // end of namespace basis
} // end of namespace dftefe
