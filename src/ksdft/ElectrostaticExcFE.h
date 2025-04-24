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

#ifndef dftefeElectrostaticExcFE_h
#define dftefeElectrostaticExcFE_h

#include <ksdft/ElectrostaticLocalFE.h>
#include <ksdft/ExchangeCorrelationFE.h>

namespace dftefe
{
  namespace ksdft
  {
    template <typename ValueTypeElectrostaticsCoeff,
              typename ValueTypeElectrostaticsBasis,
              typename ValueTypeWaveFunctionCoeff,
              typename ValueTypeWaveFunctionBasis,
              utils::MemorySpace memorySpace,
              size_type          dim>
    class ElectrostaticExcFE
      : public Hamiltonian<
          linearAlgebra::blasLapack::scalar_type<ValueTypeElectrostaticsBasis,
                                                 ValueTypeWaveFunctionBasis>,
          memorySpace>,
        public Energy<linearAlgebra::blasLapack::real_type<
          linearAlgebra::blasLapack::scalar_type<
            linearAlgebra::blasLapack::scalar_type<ValueTypeElectrostaticsBasis,
                                                   ValueTypeWaveFunctionBasis>,
            linearAlgebra::blasLapack::scalar_type<
              ValueTypeElectrostaticsCoeff,
              ValueTypeWaveFunctionCoeff>>>>
    {
    public:
      using ValueTypeOperator =
        linearAlgebra::blasLapack::scalar_type<ValueTypeElectrostaticsBasis,
                                               ValueTypeWaveFunctionBasis>;
      using ValueTypeOperand =
        linearAlgebra::blasLapack::scalar_type<ValueTypeElectrostaticsCoeff,
                                               ValueTypeWaveFunctionCoeff>;
      using ValueType =
        linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                               ValueTypeOperator>;
      using RealType = linearAlgebra::blasLapack::real_type<ValueType>;

      using Storage = utils::MemoryStorage<ValueType, memorySpace>;

    public:
      /**
       * @brief Constructor
       */
      ElectrostaticExcFE(
        std::shared_ptr<const ElectrostaticFE<ValueTypeElectrostaticsBasis,
                                              ValueTypeElectrostaticsCoeff,
                                              ValueTypeWaveFunctionBasis,
                                              memorySpace,
                                              dim>>       electroHamiltonian,
        std::shared_ptr<const ExchangeCorrelationFE<ValueTypeWaveFunctionBasis,
                                                    ValueTypeWaveFunctionCoeff,
                                                    memorySpace,
                                                    dim>> excHamiltonian);

      ~ElectrostaticExcFE() = default;

      void
      reinit(
        std::shared_ptr<const ElectrostaticFE<ValueTypeElectrostaticsBasis,
                                              ValueTypeElectrostaticsCoeff,
                                              ValueTypeWaveFunctionBasis,
                                              memorySpace,
                                              dim>>       electroHamiltonian,
        std::shared_ptr<const ExchangeCorrelationFE<ValueTypeWaveFunctionBasis,
                                                    ValueTypeWaveFunctionCoeff,
                                                    memorySpace,
                                                    dim>> excHamiltonian);

      void
      getLocal(Storage &cellWiseStorage) const override;

      RealType
      getEnergy() const override;

      void
      applyNonLocal(
        linearAlgebra::MultiVector<ValueTypeWaveFunctionCoeff, memorySpace> &X,
        linearAlgebra::MultiVector<ValueTypeWaveFunctionCoeff, memorySpace> &Y,
        bool updateGhostX,
        bool updateGhostY) const override;

      bool
      hasLocalComponent() const override;

      bool
      hasNonLocalComponent() const override;

    private:
      std::shared_ptr<const ElectrostaticFE<ValueTypeElectrostaticsBasis,
                                            ValueTypeElectrostaticsCoeff,
                                            ValueTypeWaveFunctionBasis,
                                            memorySpace,
                                            dim>>
        d_electroHamiltonian;
      std::shared_ptr<const ExchangeCorrelationFE<ValueTypeWaveFunctionBasis,
                                                  ValueTypeWaveFunctionCoeff,
                                                  memorySpace,
                                                  dim>>
        d_excHamiltonian;

    }; // end of class ElectrostaticExcFE
  }    // end of namespace ksdft
} // end of namespace dftefe
#include <ksdft/ElectrostaticExcFE.t.cpp>
#endif // dftefeElectrostaticExcFE_h
