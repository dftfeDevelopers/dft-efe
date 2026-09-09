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

#ifndef dftefeDensityCalculatorKernels_h
#define dftefeDensityCalculatorKernels_h

#include <utils/MemorySpaceType.h>
#include <linearAlgebra/MultiVector.h>
#include <quadrature/QuadratureValuesContainer.h>
#include <ksdft/KSAttributes.h>

namespace dftefe
{
  namespace ksdft
  {
    template <typename ValueType,
              typename RealType,
              utils::MemorySpace memorySpace,
              size_type          dim>
    class DensityCalculatorKernels
    {
    public:
      static void
      computeRhoInBatch(
        const size_type                       batchSize,
        const std::pair<size_type, size_type> cellRange,
        const RealType *                      occupationInBatch,
        ValueType *                           psiBatchQuad,
        RealType *                            modPsiSqBatchQuad,
        std::shared_ptr<const quadrature::QuadratureRuleContainer>
                                                     quadRuleContainer,
        RealType *                                   rhoBatch,
        linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext,
        const SpinMode                               spinMode);

      static void
      computeGradRhoInBatch(
        const size_type                       batchSize,
        const std::pair<size_type, size_type> cellRange,
        const RealType *                      occupationInBatch,
        const ValueType *                     psiBatchQuad,
        const ValueType *                     gradPsiBatchQuad,
        RealType *                            psiGradPsiBatch,
        std::shared_ptr<const quadrature::QuadratureRuleContainer>
                                                     quadRuleContainer,
        RealType *                                   gradRhoBatch,
        linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext,
        const SpinMode                               spinMode);
    }; // end of class DensityCalculatorKernels


#ifdef DFTEFE_WITH_DEVICE
    template <typename ValueType, typename RealType, size_type dim>
    class DensityCalculatorKernels<ValueType,
                                   RealType,
                                   utils::MemorySpace::DEVICE,
                                   dim>
    {
    public:
      static void
      computeRhoInBatch(
        const size_type                       batchSize,
        const std::pair<size_type, size_type> cellRange,
        const RealType *                      occupationInBatch,
        ValueType *                           psiBatchQuad,
        RealType *                            modPsiSqBatchQuad,
        std::shared_ptr<const quadrature::QuadratureRuleContainer>
                  quadRuleContainer,
        RealType *rhoBatch,
        linearAlgebra::LinAlgOpContext<utils::MemorySpace::DEVICE>
          &            linAlgOpContext,
        const SpinMode spinMode);

      static void
      computeGradRhoInBatch(
        const size_type                       batchSize,
        const std::pair<size_type, size_type> cellRange,
        const RealType *                      occupationInBatch,
        const ValueType *                     psiBatchQuad,
        const ValueType *                     gradPsiBatchQuad,
        RealType *                            psiGradPsiBatch,
        std::shared_ptr<const quadrature::QuadratureRuleContainer>
                  quadRuleContainer,
        RealType *gradRhoBatch,
        linearAlgebra::LinAlgOpContext<utils::MemorySpace::DEVICE>
          &            linAlgOpContext,
        const SpinMode spinMode);
    }; // end of class DensityCalculatorKernels
#endif
  } // namespace ksdft
} // end of namespace dftefe
#endif // dftefeDensityCalculatorKernels_h
