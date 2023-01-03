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
 * @author Bikash Kanungo
 */


#include <utils/MemoryTransfer.h>
#include <utils/Exceptions.h>
#include <BlasLapack.h>
namespace dftefe
{
  namespace linearAlgebra
  {
    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    PreconditionerJacobi<ValueTypeOperator, ValueTypeOperand, memorySpace>::
      PreconditionerJacobi(
        const Vector<ValueTypeOperator, memorySpaceSrc> &diagonal)
      : d_digonalInv(diagonal)
      , d_pcType(PreconditionerType::JACOBI)

    {
      blasLapack::reciprocalX(diagonal.localSize(),
                              1.0,
                              digonal.data(),
                              d_diagonalInv.data(),
                              *(diagonal.getLinAlgOpContext()));
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    void
    PreconditionerJacobi<ValueTypeOperator, ValueTypeOperand, memorySpace>::
      apply(Vector<ValueTypeOperand, memorySpace> &x,
            Vector<ValueTypeUnion, memorySpace> &  y) const
    {
      blaslapack::hadamardProduct(d_digonalInv.localSize(),
                                  d_digonalInv.data(),
                                  x.data(),
                                  y.data(),
                                  *(d_digonalInv.getLinAlgOpContext()));
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    void
    PreconditionerJacobi<ValueTypeOperator, ValueTypeOperand, memorySpace>::
      apply(MultiVector<ValueTypeOperand, memorySpace> &X,
            MultiVector<ValueTypeUnion, memorySpace> &  Y) const
    {
      utils::throwException(
        false,
        "apply() in PreconditionerJacobi for MultiVector is not "
        "implemented yet.");
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    PreconditionerJacobi<ValueTypeOperator, ValueTypeOperand, memorySpace>
      : getPreconditionerType() const
    {
      return d_pcType;
    }

  } // end of namespace linearAlgebra
} // end of namespace dftefe
