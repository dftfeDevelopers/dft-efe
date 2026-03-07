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
        computeRhoFromInterpolatedValues,
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
        computeRhoFromInterpolatedValues,
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
        computeRhoFromInterpolatedValues,
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

    template <typename ValueType, typename RealType>
    void
    DensityCalculatorKernels<ValueType, RealType, utils::MemorySpace::DEVICE>::
      computeRhoInBatch(
        const utils::MemoryStorage<RealType, utils::MemorySpace::DEVICE> &occupationInBatch,
        quadrature::QuadratureValuesContainer<ValueType, utils::MemorySpace::DEVICE>
          &psiBatchQuad,
        quadrature::QuadratureValuesContainer<RealType, utils::MemorySpace::DEVICE>
          &modPsiSqBatchQuad,
        std::shared_ptr<const quadrature::QuadratureRuleContainer>
          quadRuleContainer,
        quadrature::QuadratureValuesContainer<RealType, utils::MemorySpace::DEVICE> &rhoBatch,
        linearAlgebra::LinAlgOpContext<utils::MemorySpace::DEVICE> &linAlgOpContext)
    {
        ValueType *psiBatchQuadIter = psiBatchQuad.begin();
        RealType *modPsiSqBatchQuadIter = modPsiSqBatchQuad.begin();

        const size_type quadPtsInCellsBlockSize      = modPsiSqBatchQuad.nQuadraturePoints();
        const size_type numPsiInBatch    = occupationInBatch.size();

        DFTEFE_LAUNCH_KERNEL(
          computeRhoFromInterpolatedValues,
          (numPsiInBatch + (utils::DEVICE_BLOCK_SIZE - 1)) /
            utils::DEVICE_BLOCK_SIZE * quadPtsInCellsBlockSize,
          utils::DEVICE_BLOCK_SIZE,
          utils::defaultStream,
          numPsiInBatch,
          quadPtsInCellsBlockSize,
          utils::makeDataTypeDeviceCompatible(psiBatchQuadIter),
          utils::makeDataTypeDeviceCompatible(modPsiSqBatchQuadIter));

        const RealType      alpha = 2.0; // 2 for spin up and down
        const RealType      beta  = 0.0;

        linearAlgebra::blasLapack::gemm<RealType, RealType, utils::MemorySpace::DEVICE>(
          'N',
          'N',
          1,
          quadPtsInCellsBlockSize,
          numPsiInBatch,
          alpha,
          occupationInBatch.data(),
          1,
          modPsiSqBatchQuad.begin(),
          numPsiInBatch,
          beta,
          rhoBatch.begin(),
          1,
          linAlgOpContext);
    }

    template class DensityCalculatorKernels<double, double,
                                            dftefe::utils::MemorySpace::DEVICE>;
    template class DensityCalculatorKernels<float, double,
                                            dftefe::utils::MemorySpace::DEVICE>;
    template class DensityCalculatorKernels<std::complex<double>, double,
                                            dftefe::utils::MemorySpace::DEVICE>;
    template class DensityCalculatorKernels<std::complex<float>, double,
                                            dftefe::utils::MemorySpace::DEVICE>;
  } // namespace basis
} // namespace dftefe
#endif
