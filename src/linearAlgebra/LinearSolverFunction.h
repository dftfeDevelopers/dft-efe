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

#ifndef dftefeLinearSolverFunction_h
#define dftefeLinearSolverFunction_h

#include <utils/TypeConfig.h>
#include <utils/MemorySpaceType.h>
#include <utils/MPITypes.h>
#include <linearAlgebra/Vector.h>
#include <linearAlgebra/MultiVector.h>
#include <linearAlgebra/OperatorContext.h>
#include <linearAlgebra/BlasLapackTypedef.h>

namespace dftefe
{
  namespace linearAlgebra
  {
    /**
     *
     * @brief An abstract class that encapsulates a linear partial differential equation
     *  (PDE). That in, in a discrete sense it represents the linear system of
     * equations: \f$ \mathbf{Ax}=\mathbf{b}$\f.
     *
     *  It provides the handle for the action of the linear operator on
     *  a given Vector, including the enforcement of appropriate boudnary
     *  conditions and other constraints. Additionally, it provides a handle
     *  to apply the preconditioner on a Vector. Finally, it also stores the
     *  solution.
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
    class LinearSolverFunction
    {
    public:
      /**
       * @brief define ValueType as the superior (bigger set) of the
       * ValueTypeOperator and ValueTypeOperand
       * (e.g., between double and complex<double>, complex<double>
       * is the bigger set)
       */
      using ValueType =
        blasLapack::scalar_type<ValueTypeOperator, ValueTypeOperand>;

    public:
      virtual ~LinearSolverFunction() = default;

      virtual const OperatorContext<ValueTypeOperator,
                                    ValueTypeOperand,
                                    memorySpace> &
      getAxContext() const = 0;

      virtual const OperatorContext<ValueTypeOperator,
                                    ValueTypeOperand,
                                    memorySpace> &
      getPCContext() const = 0;

      void
      setSolution(const Vector<ValueTypeOperand, memorySpace> &x) = 0;

      virtual Vector<ValueType, memorySpace>
      getRhs() const = 0;

      virtual Vector<ValueTypeOperand, memorySpace>
      getInitialGuess() const = 0;

      const utils::mpi::MPIComm &
      getMPIComm() const = 0;
    }; // end of class LinearSolverFunction
  }    // end of namespace linearAlgebra
} // end of namespace dftefe
#endif // dftefeLinearSolverFunction_h
