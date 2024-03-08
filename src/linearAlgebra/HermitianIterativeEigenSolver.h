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

#ifndef dftefeHermitianIterativeEigenSolver_h
#define dftefeHermitianIterativeEigenSolver_h

#include <utils/TypeConfig.h>
#include <utils/MemorySpaceType.h>
#include <utils/MPITypes.h>
#include <linearAlgebra/Vector.h>
#include <linearAlgebra/MultiVector.h>
#include <linearAlgebra/IdentityOperatorContext.h>
#include <linearAlgebra/OperatorContext.h>
#include <linearAlgebra/BlasLapackTypedef.h>
#include <linearAlgebra/BlasLapack.h>
#include <linearAlgebra/LinearAlgebraTypes.h>

namespace dftefe
{
  namespace linearAlgebra
  {
    /**
     *
     * @brief An abstract class that encapsulates a generalized eigen value
     * problem solver. That is, in a discrete sense it represents the equation
     * \f$ \mathbf{Ax}=\mathbf{\lambda B x}$\f.
     *
     *
     * @tparam ValueTypeOperator The datatype (float, double, complex<double>,
     * etc.) for the operator (e.g. Matrix) associated with the linear solve
     * @tparam ValueTypeOperand The datatype (float, double, complex<double>,
     * etc.) of the vector, matrices, etc.
     * on which the operator will act
     * @tparam memorySpace The meory space (HOST, DEVICE, HOST_PINNED, etc.)
     * in which the data of the operator
     * and its operands reside
     *
     */
    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    class HermitianIterativeEigenSolver
    {
    public:
      using OpContext =
        OperatorContext<ValueTypeOperator, ValueTypeOperand, memorySpace>;
      using ValueType =
        blasLapack::scalar_type<ValueTypeOperator, ValueTypeOperand>;
      using ScalarType = blasLapack::real_type<ValueType>;

    public:
      virtual ~HermitianIterativeEigenSolver() = default;

      virtual EigenSolverError
      solve(const OpContext &                                   A,
            std::vector<ScalarType> &                           eigenValues,
            linearAlgebra::MultiVector<ValueType, memorySpace> &eigenVectors,
            bool             computeEigenVectors = false,
            const OpContext &B    = IdentityOperatorContext<ValueTypeOperator,
                                                         ValueTypeOperand,
                                                         memorySpace>(),
            const OpContext &BInv = IdentityOperatorContext<ValueTypeOperator,
                                                            ValueTypeOperand,
                                                            memorySpace>()) = 0;

    }; // end of class HermitianIterativeEigenSolver
  }    // end of namespace linearAlgebra
} // end of namespace dftefe
#endif // dftefeHermitianIterativeEigenSolver_h
