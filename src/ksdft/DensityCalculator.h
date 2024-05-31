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

#ifndef dftefeDensityCalculator_h
#define dftefeDensityCalculator_h

#include <linearAlgebra/MultiVector.h>
#include <basis/FEBasisDataStorage.h>
#include <basis/FEBasisOperations.h>

namespace dftefe
{
  namespace ksdft
  {
    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    class DensityCalculator
    {
    public:
      using ValueType =
        linearAlgebra::blasLapack::scalar_type<ValueTypeBasisData,
                                               ValueTypeBasisCoeff>;

      using RealType = linearAlgebra::blasLapack::real_type<ValueType>;

    public:
      /**
       * @brief Constructor
       */

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
        const size_type waveFuncBatchSize);

      /**
       *@brief Default Destructor
       *
       */
      ~DensityCalculator();

      void
      reinit(std::shared_ptr<
               const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
                                               feBasisDataStorage,
             const basis::FEBasisManager<ValueTypeBasisCoeff,
                                         ValueTypeBasisData,
                                         memorySpace,
                                         dim> &feBMPsi);

      void
      computeRho(
        const std::vector<RealType> &occupation,
        const linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
          &                                                           waveFunc,
        quadrature::QuadratureValuesContainer<RealType, memorySpace> &rho);

    private:
      std::shared_ptr<const quadrature::QuadratureRuleContainer>
        d_quadRuleContainer;
      std::shared_ptr<const basis::FEBasisOperations<ValueTypeBasisCoeff,
                                                     ValueTypeBasisData,
                                                     memorySpace,
                                                     dim>>
                                        d_feBasisOp;
      const basis::FEBasisManager<ValueTypeBasisCoeff,
                                  ValueTypeBasisData,
                                  memorySpace,
                                  dim> *d_feBMPsi;
      const size_type                   d_cellBlockSize;
      const size_type                   d_waveFuncBatchSize;
      std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
        d_linAlgOpContext;

      quadrature::QuadratureValuesContainer<ValueType, memorySpace>
        *d_psiBatchQuad;

      quadrature::QuadratureValuesContainer<RealType, memorySpace>
        *d_psiModSqBatchQuad;

      quadrature::QuadratureValuesContainer<RealType, memorySpace> *d_rhoBatch;

      linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace> *d_psiBatch;

      quadrature::QuadratureValuesContainer<ValueType, memorySpace>
        *d_psiBatchSmallQuad;

      quadrature::QuadratureValuesContainer<RealType, memorySpace>
        *d_psiModSqBatchSmallQuad;

      linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
        *d_psiBatchSmall;

      size_type d_batchSizeSmall;

    }; // end of class DensityCalculator
  }    // end of namespace ksdft
} // end of namespace dftefe
#include <ksdft/DensityCalculator.t.cpp>
#endif // dftefeDensityCalculator_h
