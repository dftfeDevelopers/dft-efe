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

#ifndef dftefeLinearSolverImpl_h
#define dftefeLinearSolverImpl_h

#include <utils/TypeConfig.h>
#include <utils/MemorySpaceType.h>
#include <linearAlgebra/LinearAlgebraTypes.h>
#include <linearAlgebra/LinearSolverFunction.h>

namespace dftefe
{
  namespace linearAlgebra
  {
    /**
     *
     * @brief Abstract class that implements the LinearSolver algorithm.
     *  For example, the derived classes of it, such as CGLinearSolver,
     *  GMRESLinearSolver implement the Conjugate-Gradient (CG) and
     *  Generalized Minimum Residual (GMRES) Krylov subspace based approches,
     *  respectively, to solve a linear system of equations.
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
    class LinearSolverImpl
    {
    public:
      /**
       * @brief Default Destructor
       */
      virtual ~LinearSolverImpl() = default;

      /**
       * @brief Function that initiates the linear solve
       *
       * @param[in] linearSolverFunction Reference to a LinearSolverFunction
       *  object that encapsulates the discrete partial differential equation
       *  that is being solved as a linear solve. Typically, the
       *  linearSolverFunction provides the right hand side
       *  vector (i.e., \f$\mathbf{b}$\f) and the handle to the action of the
       *  discrete operator on a Vector. It also stores the final solution
       *  \f$\mathbf{x}$\f
       *
       */
      virtual Error
      solve(
        LinearSolverFunction<ValueTypeOperator, ValueTypeOperand, memorySpace>
          &linearSolverFunction) = 0;
    }; // end of class LinearSolverImpl
  }    // end of namespace linearAlgebra
} // end of namespace dftefe
#endif // dftefeLinearSolverImpl_h
