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

#ifndef dftefeChebyshevFilter_h
#define dftefeChebyshevFilter_h

#include <utils/MemorySpaceType.h>
#include <linearAlgebra/BlasLapackTypedef.h>
#include <linearAlgebra/LinAlgOpContext.h>
#include <linearAlgebra/Vector.h>
#include <linearAlgebra/MultiVector.h>
#include <linearAlgebra/OperatorContext.h>
#include <memory>

namespace dftefe
{
  namespace linearAlgebra
  {
    /**
     *@brief A class to get chebyshevFiletered subspace "filteredSubspace" from original subspace "eigenSubspaceGuess".
     * Note both of these vectors have to be apriori allocated and both will
     *change after the function is called.
     *
     * @tparam ValueTypeOperator The datatype (float, double, complex<double>, etc.) for the underlying operator
     * @tparam ValueTypeOperand The datatype (float, double, complex<double>, etc.) of the vector, matrices, etc.
     * on which the operator will act
     * @tparam memorySpace The meory space (HOST, DEVICE, HOST_PINNED, etc.) in which the data of the operator
     * and its operands reside
     *
     */

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    void
    ChebyshevFilter(
      const OperatorContext<ValueTypeOperator, ValueTypeOperand, memorySpace>
        &A,
      const OperatorContext<ValueTypeOperator, ValueTypeOperand, memorySpace>
        &                                         BInv,
      MultiVector<ValueTypeOperand, memorySpace> &eigenSubspaceGuess,
      const size_type                             polynomialDegree,
      const double                                wantedSpectrumLowerBound,
      const double                                wantedSpectrumUpperBound,
      const double                                unWantedSpectrumUpperBound,
      MultiVector<blasLapack::scalar_type<ValueTypeOperator, ValueTypeOperand>,
                  memorySpace> &                  filteredSubspace);

  } // end of namespace linearAlgebra
} // end of namespace dftefe
#include <linearAlgebra/ChebyshevFilter.t.cpp>
#endif // dftefeChebyshevFilter_h
