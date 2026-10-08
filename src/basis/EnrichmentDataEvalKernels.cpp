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
#include <complex>
#include <cmath>

namespace dftefe
{
  namespace basis
  {
    namespace
    {
      // separated by overload, as in the device kernels
      void
      getBlochPhase(double *out,
                    const double *kPointCoord,
                    const double *point,
                    const double *origin)
      {
        *out = 1.0;
      }

      void
      getBlochPhase(std::complex<double> *out,
                    const double *        kPointCoord,
                    const double *        point,
                    const double *        origin)
      {
        double kDotR = 0.0;
        for (size_type iDim = 0; iDim < 3; iDim++)
          kDotR += kPointCoord[iDim] * (point[iDim] - origin[iDim]);
        *out = std::complex<double>(std::cos(kDotR), -std::sin(kDotR));
      }

      // the (grad psi - i k psi) bracket; both value types share one expression
      void
      getBlochGradientTerm(double *        out,
                           const double *  kPointCoord,
                           const size_type iDim,
                           const double    enrichmentValue,
                           const double    gradientValue)
      {
        *out = gradientValue;
      }

      void
      getBlochGradientTerm(std::complex<double> *out,
                           const double *        kPointCoord,
                           const size_type       iDim,
                           const double          enrichmentValue,
                           const double          gradientValue)
      {
        *out = std::complex<double>(gradientValue,
                                    -kPointCoord[iDim] * enrichmentValue);
      }

    } // namespace

    template <typename ValueType, utils::MemorySpace memorySpace>
    void
    EnrichmentDataEvalKernels<ValueType, memorySpace>::getEnrichmentValues(
      const size_type               numEnrichmentFunc,
      const std::vector<size_type> &pointsPerEnrichId,
      const std::vector<std::shared_ptr<atoms::SphericalData>>
        &                                          sphericalDataVec,
      const double *                               points,
      const double *                               origin,
      ValueType *                                  values,
      double *                                     scratchReal,
      linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext,
      const double *                               kPointCoord)
    {
      atoms::AtomSuperpositionFunctionEvalKernels<memorySpace>::
        getEnrichmentValues(numEnrichmentFunc,
                            pointsPerEnrichId,
                            sphericalDataVec,
                            points,
                            origin,
                            scratchReal,
                            linAlgOpContext);

      // one (enrichment, origin) pair per entry; the phase uses its own origin
      size_type cumulativeValuesOffset = 0;
      size_type cumulativeCoordsOffset = 0;
      for (int i = 0; i < numEnrichmentFunc; i++)
        {
          for (size_type iPt = 0; iPt < pointsPerEnrichId[i]; iPt++)
            {
              ValueType phase;
              getBlochPhase(&phase,
                            kPointCoord,
                            points + cumulativeCoordsOffset + iPt * 3,
                            origin + i * 3);
              values[cumulativeValuesOffset + iPt] =
                phase * scratchReal[cumulativeValuesOffset + iPt];
            }
          cumulativeValuesOffset += pointsPerEnrichId[i];
          cumulativeCoordsOffset += pointsPerEnrichId[i] * 3;
        }
    }

    template <typename ValueType, utils::MemorySpace memorySpace>
    void
    EnrichmentDataEvalKernels<ValueType, memorySpace>::getEnrichmentGradients(
      const size_type               numEnrichmentFunc,
      const std::vector<size_type> &pointsPerEnrichId,
      const std::vector<std::shared_ptr<atoms::SphericalData>>
        &                                          sphericalDataVec,
      const double *                               points,
      const double *                               origin,
      ValueType *                                  values,
      double *                                     scratchReal,
      linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext,
      const double *                               kPointCoord)
    {
      size_type numValues = 0;
      for (int i = 0; i < numEnrichmentFunc; i++)
        numValues += pointsPerEnrichId[i] * 3;

      atoms::AtomSuperpositionFunctionEvalKernels<memorySpace>::
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
        atoms::AtomSuperpositionFunctionEvalKernels<memorySpace>::
          getEnrichmentValues(numEnrichmentFunc,
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
          for (size_type iPt = 0; iPt < pointsPerEnrichId[i]; iPt++)
            {
              ValueType phase;
              getBlochPhase(&phase,
                            kPointCoord,
                            points + cumulativeCoordsOffset + iPt * 3,
                            origin + i * 3);
              const double enrichmentValue =
                enrichmentValues[cumulativeEnrichmentOffset + iPt];
              for (size_type iDim = 0; iDim < 3; iDim++)
                {
                  ValueType blochGradientTerm;
                  getBlochGradientTerm(&blochGradientTerm,
                                       kPointCoord,
                                       iDim,
                                       enrichmentValue,
                                       scratchReal[cumulativeValuesOffset +
                                                   iPt * 3 + iDim]);
                  values[cumulativeValuesOffset + iPt * 3 + iDim] =
                    phase * blochGradientTerm;
                }
            }
          cumulativeValuesOffset += pointsPerEnrichId[i] * 3;
          cumulativeCoordsOffset += pointsPerEnrichId[i] * 3;
          cumulativeEnrichmentOffset += pointsPerEnrichId[i];
        }
    }

    template <typename ValueType, utils::MemorySpace memorySpace>
    void
    EnrichmentDataEvalKernels<ValueType, memorySpace>::getEnrichmentValuesInCellRange(
      const double *                  quadPtsInAllCells,
      const double *                  originPtsInAllCells,
      const size_type *               originOffsetPerCellEnrich,
      std::pair<size_type, size_type> cellRange,
      const std::vector<size_type>    numEnrichIdsInAllCells,
      const std::vector<size_type>    numQuadPtsInAllCells,
      const atoms::SphericalDataNumerical::Func<memorySpace>
        *                                          sphericalDataFuncInAllCells,
      ValueType *                         output,
      linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext,
      const double *                               kPointCoord)
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

              // Without periodicity every enrichment has exactly one origin,
              // so the offsets are 0,1,2,... and this loop runs once.
              const size_type oBegin =
                originOffsetPerCellEnrich[cumulativeEnrich + enrichId];
              const size_type oEnd =
                originOffsetPerCellEnrich[cumulativeEnrich + enrichId + 1];

              // the origin loop is the lattice sum, one phase per image
              ValueType value = utils::Types<ValueType>::zero;
              for (size_type iOrigin = oBegin; iOrigin < oEnd; iOrigin++)
                {
                  ValueType phase;
                  getBlochPhase(&phase,
                                kPointCoord,
                                quadPtsInAllCells +
                                  (cumulativeQuad + quadId) * dim,
                                originPtsInAllCells + iOrigin * dim);
                  value +=
                    phase *
                    sphericalDataFuncInAllCells[cumulativeEnrich + enrichId]
                      .getValue(quadPtsInAllCells +
                                  (cumulativeQuad + quadId) * dim,
                                originPtsInAllCells + iOrigin * dim);
                }
              output[cumulativeOutput + iThread] = value;
            }

          cumulativeEnrich += numEnrichInCell;
          cumulativeQuad += numQuadInCell;
          cumulativeOutput += numEnrichInCell * numQuadInCell;
        }
    }

    template <typename ValueType, utils::MemorySpace memorySpace>
    void
    EnrichmentDataEvalKernels<ValueType, memorySpace>::getEnrichmentGradientsInCellRange(
      const double *                  quadPtsInAllCells,
      const double *                  originPtsInAllCells,
      const size_type *               originOffsetPerCellEnrich,
      std::pair<size_type, size_type> cellRange,
      const std::vector<size_type>    numEnrichIdsInAllCells,
      const std::vector<size_type>    numQuadPtsInAllCells,
      const atoms::SphericalDataNumerical::Func<memorySpace>
        *                                          sphericalDataFuncInAllCells,
      ValueType *                         output,
      linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext,
      const double *                               kPointCoord)
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

              // Without periodicity every enrichment has exactly one origin,
              // so the offsets are 0,1,2,... and this loop runs once.
              const size_type oBegin =
                originOffsetPerCellEnrich[cumulativeEnrich + enrichId];
              const size_type oEnd =
                originOffsetPerCellEnrich[cumulativeEnrich + enrichId + 1];

              ValueType grad[3] = {utils::Types<ValueType>::zero,
                                   utils::Types<ValueType>::zero,
                                   utils::Types<ValueType>::zero};
              for (size_type iOrigin = oBegin; iOrigin < oEnd; iOrigin++)
                {
                  double originGrad[3];
                  sphericalDataFuncInAllCells[cumulativeEnrich + enrichId]
                    .getGradientValue(quadPtsInAllCells +
                                        (cumulativeQuad + quadId) * dim,
                                      originPtsInAllCells + iOrigin * dim,
                                      originGrad);
                  // per image: e^{-ik.d} (grad psi - i k psi)
                  ValueType phase;
                  getBlochPhase(&phase,
                                kPointCoord,
                                quadPtsInAllCells +
                                  (cumulativeQuad + quadId) * dim,
                                originPtsInAllCells + iOrigin * dim);
                  const double enrichmentValue =
                    sphericalDataFuncInAllCells[cumulativeEnrich + enrichId]
                      .getValue(quadPtsInAllCells +
                                  (cumulativeQuad + quadId) * dim,
                                originPtsInAllCells + iOrigin * dim);
                  ValueType blochGradientTerm;
                  getBlochGradientTerm(&blochGradientTerm,
                                       kPointCoord,
                                       0,
                                       enrichmentValue,
                                       originGrad[0]);
                  grad[0] += phase * blochGradientTerm;
                  getBlochGradientTerm(&blochGradientTerm,
                                       kPointCoord,
                                       1,
                                       enrichmentValue,
                                       originGrad[1]);
                  grad[1] += phase * blochGradientTerm;
                  getBlochGradientTerm(&blochGradientTerm,
                                       kPointCoord,
                                       2,
                                       enrichmentValue,
                                       originGrad[2]);
                  grad[2] += phase * blochGradientTerm;
                }
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

    template class EnrichmentDataEvalKernels<double, utils::MemorySpace::HOST>;
    template class EnrichmentDataEvalKernels<std::complex<double>,
                                             utils::MemorySpace::HOST>;
  } // end of namespace basis
} // end of namespace dftefe
