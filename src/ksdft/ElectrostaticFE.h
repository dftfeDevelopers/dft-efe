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

#ifndef dftefeElectrostaticFE_h
#define dftefeElectrostaticFE_h

#include <utils/MemorySpaceType.h>
#include <linearAlgebra/MultiVector.h>
#include <ksdft/Hamiltonian.h>
#include <ksdft/Energy.h>
#include <quadrature/QuadratureValuesContainer.h>
#include <basis/FEBasisDataStorage.h>

namespace dftefe
{
  namespace ksdft
  {
    /**
     *@brief A derived class of linearAlgebra::OperatorContext to encapsulate
     * the action of a discrete operator on vectors, matrices, etc.
     *
     * @tparam ValueTypeOperator The datatype (float, double, complex<double>, etc.) for the underlying operator
     * on which the operator will act
     * @tparam memorySpace The meory sapce (HOST, DEVICE, HOST_PINNED, etc.) in which the data of the operator
     * and its operands reside
     *
     */
    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              typename ValueTypeWaveFnBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    class ElectrostaticFE
      : public Hamiltonian<
          linearAlgebra::blasLapack::scalar_type<
            linearAlgebra::blasLapack::scalar_type<ValueTypeBasisData,
                                                   ValueTypeWaveFnBasisData>,
            ValueTypeBasisCoeff>,
          memorySpace>,
        public Energy<linearAlgebra::blasLapack::real_type<
          linearAlgebra::blasLapack::scalar_type<ValueTypeBasisData,
                                                 ValueTypeBasisCoeff>>>
    {
    public:
      using ValueType =
        linearAlgebra::blasLapack::scalar_type<ValueTypeBasisData,
                                               ValueTypeBasisCoeff>;
      using Storage = utils::MemoryStorage<
        linearAlgebra::blasLapack::scalar_type<
          linearAlgebra::blasLapack::scalar_type<ValueTypeBasisData,
                                                 ValueTypeWaveFnBasisData>,
          ValueTypeBasisCoeff>,
        memorySpace>;
      using RealType = linearAlgebra::blasLapack::real_type<ValueType>;

    public:
      virtual ~ElectrostaticFE() = default;
      virtual void
      getLocal(Storage &cellWiseStorage) const = 0;
      virtual RealType
      getEnergy() const = 0;
      virtual const quadrature::QuadratureValuesContainer<ValueType,
                                                          memorySpace> &
      getFunctionalDerivative() const = 0;
      // virtual void
      // applyNonLocal(linearAlgebra::MultiVector<ValueType, memorySpace> &X, 
      //   linearAlgebra::MultiVector<ValueType, memorySpace> &Y) const = 0;
      // virtual bool
      // hasLocalComponent() const = 0;
      // virtual bool
      // hasNonLocalComponent() const = 0;

    }; // end of class ElectrostaticFE
  }    // end of namespace ksdft
} // end of namespace dftefe
#endif // dftefeElectrostaticFE_h
