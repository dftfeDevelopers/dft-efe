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
 * @author Bikash Kanungo, Vishal Subramanian, Avirup Sircar
 */


#ifdef DFTEFE_WITH_DEVICE

#  include <utils/DeviceKernelLauncherHelpers.h>
#  include <utils/DeviceDataTypeOverloads.h>
#  include <utils/DeviceTypeConfigHalfPrec.h>
#  include <utils/Exceptions.h>
#  include <complex>
#  include <algorithm>
#  include "DensityCalculatorKernels.h"

namespace dftefe
{
  namespace ksdft
  {
    namespace
    {
      template <typename ValueType, typename RealType>
      DFTEFE_CREATE_KERNEL(
        void,
        computeModPsiSqFromInterpolatedValues,
        {
          const size_type numberEntries = quadPtsInCellsBlockSize * numVectors;

          for (size_type index = globalThreadId; index < numberEntries;
              index += nThreadsPerBlock * nThreadBlock)
            {
              const double psi = psiBatchQuad[index];
              modPsiSqBatchQuadIter[index] = psi * psi;
            }
        },
        const size_type numVectors,
        const size_type quadPtsInCellsBlockSize,
        ValueType           *psiBatchQuad,
        RealType           *modPsiSqBatchQuadIter);

      template <typename RealType>
      DFTEFE_CREATE_KERNEL(
        void,
        computeModPsiSqFromInterpolatedValues,
        {
          const size_type numberEntries = quadPtsInCellsBlockSize * numVectors;

          for (size_type index = globalThreadId; index < numberEntries;
              index += nThreadsPerBlock * nThreadBlock)
            {
              const utils::deviceDoubleComplex psi = psiBatchQuad[index];
              modPsiSqBatchQuadIter[index] = (RealType)(utils::realPartDevice(psi) *
                                utils::realPartDevice(psi) +
                              utils::imagPartDevice(psi) *
                                utils::imagPartDevice(psi));
            }
        },
        const size_type numVectors,
        const size_type quadPtsInCellsBlockSize,
        dftefe::utils::deviceDoubleComplex *psiBatchQuad,
        RealType           *modPsiSqBatchQuadIter);

      template <typename RealType>
      DFTEFE_CREATE_KERNEL(
        void,
        computeModPsiSqFromInterpolatedValues,
        {
          const size_type numberEntries = quadPtsInCellsBlockSize * numVectors;

          for (size_type index = globalThreadId; index < numberEntries;
              index += nThreadsPerBlock * nThreadBlock)
            {
              const utils::deviceFloatComplex psi = psiBatchQuad[index];
              modPsiSqBatchQuadIter[index] = (RealType)(utils::realPartDevice(psi) *
                                utils::realPartDevice(psi) +
                              utils::imagPartDevice(psi) *
                                utils::imagPartDevice(psi));
            }
        },
        const size_type numVectors,
        const size_type quadPtsInCellsBlockSize,
        dftefe::utils::deviceFloatComplex *psiBatchQuad,
        RealType           *modPsiSqBatchQuadIter);
    } // namespace

    namespace
    {
      template <typename ValueType, typename RealType>
      DFTEFE_CREATE_KERNEL(
        void,
        computePsiGradPsiFromInterpolatedValues,
        {
          const size_type numberEntries = quadPtsInCellsBlockSize * dimVal * numVectors;

          for (size_type index = globalThreadId; index < numberEntries;
              index += nThreadsPerBlock * nThreadBlock)
            {
              const size_type i   = index % numVectors;
              const size_type qd  = index / numVectors;
              const size_type q   = qd / dimVal;
              const double psiVal = psiBatchQuad[q * numVectors + i];
              const double gVal   = gradPsiBatchQuad[i + numVectors * qd];
              psiGradPsiBatch[index] = (RealType)(psiVal * gVal);
            }
        },
        const size_type numVectors,
        const size_type quadPtsInCellsBlockSize,
        const size_type dimVal,
        const ValueType *psiBatchQuad,
        const ValueType *gradPsiBatchQuad,
        RealType  *psiGradPsiBatch);

      template <typename RealType>
      DFTEFE_CREATE_KERNEL(
        void,
        computePsiGradPsiFromInterpolatedValues,
        {
          const size_type numberEntries = quadPtsInCellsBlockSize * dimVal * numVectors;

          for (size_type index = globalThreadId; index < numberEntries;
              index += nThreadsPerBlock * nThreadBlock)
            {
              const size_type i  = index % numVectors;
              const size_type qd = index / numVectors;
              const size_type q  = qd / dimVal;
              const utils::deviceDoubleComplex psiVal = psiBatchQuad[q * numVectors + i];
              const utils::deviceDoubleComplex gVal   = gradPsiBatchQuad[i + numVectors * qd];
              psiGradPsiBatch[index] =
                (RealType)(utils::realPartDevice(psiVal) * utils::realPartDevice(gVal) +
                           utils::imagPartDevice(psiVal) * utils::imagPartDevice(gVal));
            }
        },
        const size_type numVectors,
        const size_type quadPtsInCellsBlockSize,
        const size_type dimVal,
        const dftefe::utils::deviceDoubleComplex *psiBatchQuad,
        const dftefe::utils::deviceDoubleComplex *gradPsiBatchQuad,
        RealType *psiGradPsiBatch);

      template <typename RealType>
      DFTEFE_CREATE_KERNEL(
        void,
        computePsiGradPsiFromInterpolatedValues,
        {
          const size_type numberEntries = quadPtsInCellsBlockSize * dimVal * numVectors;

          for (size_type index = globalThreadId; index < numberEntries;
              index += nThreadsPerBlock * nThreadBlock)
            {
              const size_type i  = index % numVectors;
              const size_type qd = index / numVectors;
              const size_type q  = qd / dimVal;
              const utils::deviceFloatComplex psiVal = psiBatchQuad[q * numVectors + i];
              const utils::deviceFloatComplex gVal   = gradPsiBatchQuad[i + numVectors * qd];
              psiGradPsiBatch[index] =
                (RealType)(utils::realPartDevice(psiVal) * utils::realPartDevice(gVal) +
                           utils::imagPartDevice(psiVal) * utils::imagPartDevice(gVal));
            }
        },
        const size_type numVectors,
        const size_type quadPtsInCellsBlockSize,
        const size_type dimVal,
        const dftefe::utils::deviceFloatComplex *psiBatchQuad,
        const dftefe::utils::deviceFloatComplex *gradPsiBatchQuad,
        RealType *psiGradPsiBatch);
    } // namespace (grad kernels)

    template <typename ValueType, typename RealType, size_type dim>
    void
    DensityCalculatorKernels<ValueType, RealType, utils::MemorySpace::DEVICE, dim>::
      computeRhoInBatch(
        const size_type batchSize,
        const std::pair<size_type, size_type> cellRange,
        const RealType* occupationInBatch,
        ValueType *psiBatchQuad,
        RealType *modPsiSqBatchQuad,
        std::shared_ptr<const quadrature::QuadratureRuleContainer>
          quadRuleContainer,
        RealType *rhoBatch,
        linearAlgebra::LinAlgOpContext<utils::MemorySpace::DEVICE> &linAlgOpContext)
    {
        size_type quadPtsInCellsBlockSize = 0;
        for (size_type iCell = cellRange.first; iCell < cellRange.second; iCell++)
          quadPtsInCellsBlockSize +=
            quadRuleContainer->nCellQuadraturePoints(iCell);

        DFTEFE_LAUNCH_KERNEL(
          computeModPsiSqFromInterpolatedValues,
          (batchSize + (utils::DEVICE_BLOCK_SIZE - 1)) /
            utils::DEVICE_BLOCK_SIZE * quadPtsInCellsBlockSize,
          utils::DEVICE_BLOCK_SIZE,
          utils::defaultStream,
          batchSize,
          quadPtsInCellsBlockSize,
          utils::makeDataTypeDeviceCompatible(psiBatchQuad),
          utils::makeDataTypeDeviceCompatible(modPsiSqBatchQuad));

        const RealType alpha = 2.0; // 2 for spin up and down
        const RealType beta  = 0.0;

        linearAlgebra::blasLapack::gemm<RealType, RealType, utils::MemorySpace::DEVICE>(
          'N',
          'N',
          1,
          quadPtsInCellsBlockSize,
          batchSize,
          alpha,
          occupationInBatch,
          1,
          modPsiSqBatchQuad,
          batchSize,
          beta,
          rhoBatch,
          1,
          linAlgOpContext);
    }

    template <typename ValueType, typename RealType, size_type dim>
    void
    DensityCalculatorKernels<ValueType, RealType, utils::MemorySpace::DEVICE, dim>::
      computeGradRhoInBatch(
        const size_type batchSize,
        const std::pair<size_type, size_type> cellRange,
        const RealType *occupationInBatch,
        const ValueType *psiBatchQuad,
        const ValueType *gradPsiBatchQuad,
        RealType *psiGradPsiBatch,
        std::shared_ptr<const quadrature::QuadratureRuleContainer>
          quadRuleContainer,
        RealType *gradRhoBatch,
        linearAlgebra::LinAlgOpContext<utils::MemorySpace::DEVICE> &linAlgOpContext)
    {
        size_type quadPtsInCellsBlockSize = 0;
        for (size_type iCell = cellRange.first; iCell < cellRange.second; iCell++)
          quadPtsInCellsBlockSize +=
            quadRuleContainer->nCellQuadraturePoints(iCell);

        const size_type totalEntries = batchSize * quadPtsInCellsBlockSize * dim;

        DFTEFE_LAUNCH_KERNEL(
          computePsiGradPsiFromInterpolatedValues,
          (totalEntries + (utils::DEVICE_BLOCK_SIZE - 1)) /
            utils::DEVICE_BLOCK_SIZE,
          utils::DEVICE_BLOCK_SIZE,
          utils::defaultStream,
          batchSize,
          quadPtsInCellsBlockSize,
          dim,
          utils::makeDataTypeDeviceCompatible(psiBatchQuad),
          utils::makeDataTypeDeviceCompatible(gradPsiBatchQuad),
          utils::makeDataTypeDeviceCompatible(psiGradPsiBatch));

        const RealType alpha = 4.0; // 2 for spin up and down, 2 for grad(|psi|^2) = 2Re(psi* grad psi)
        const RealType beta  = 0.0;

        linearAlgebra::blasLapack::gemm<RealType, RealType, utils::MemorySpace::DEVICE>(
          'N',
          'N',
          1,
          quadPtsInCellsBlockSize * dim,
          batchSize,
          alpha,
          occupationInBatch,
          1,
          psiGradPsiBatch,
          batchSize,
          beta,
          gradRhoBatch,
          1,
          linAlgOpContext);
    }

    template class DensityCalculatorKernels<double, double,
                                            dftefe::utils::MemorySpace::DEVICE, 3>;
    template class DensityCalculatorKernels<float, double,
                                            dftefe::utils::MemorySpace::DEVICE, 3>;
    template class DensityCalculatorKernels<std::complex<double>, double,
                                            dftefe::utils::MemorySpace::DEVICE, 3>;
    template class DensityCalculatorKernels<std::complex<float>, double,
                                            dftefe::utils::MemorySpace::DEVICE, 3>;
  } // namespace ksdft
} // namespace dftefe
#endif
