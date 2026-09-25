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

#ifndef dftefeHamiltonian_h
#define dftefeHamiltonian_h

#include <utils/MemoryStorage.h>
#include <linearAlgebra/MultiVector.h>
#include <linearAlgebra/BlasLapackTypedef.h>

namespace dftefe
{
  namespace ksdft
  {
    /*
     * The cell-wise Hamiltonian and the wavefunctions it is applied to are
     * different quantities: the first follows the basis, the second the
     * wavefunction coefficients. They coincide at Gamma with a real basis but
     * not when only the coefficients are complex.
     */
    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    class Hamiltonian
    {
    public:
      // the assembled cell matrix is multiplied against the operand, and BLAS
      // has no mixed real-times-complex gemm, so it carries the union type
      using ValueType =
        linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                               ValueTypeOperand>;

      virtual ~Hamiltonian() = default;
      virtual void
      getLocal(
        utils::MemoryStorage<ValueType, memorySpace> &cellWiseStorage) const = 0;
      virtual void
      applyNonLocal(
        linearAlgebra::MultiVector<ValueTypeOperand, memorySpace> &X,
        linearAlgebra::MultiVector<ValueTypeOperand, memorySpace> &Y,
        bool updateGhostX,
        bool updateGhostY) const = 0;
      virtual bool
      hasLocalComponent() const = 0;
      virtual bool
      hasNonLocalComponent() const = 0;

    }; // end of Hamiltonian
  }    // end of namespace ksdft
} // end of namespace dftefe
#endif // dftefeHamiltonian_h
