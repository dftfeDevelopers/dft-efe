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

#ifndef dftefeLinearSolver_h
#define dftefeLinearSolver_h

#include <utils/TypeConfig.h>
#include <utils/MemorySpaceType.h>
#include <linearAlgebra/LinearAlgebraTypes.h>
#include <linearAlgebra/LinearSolverFunction.h>
#include <linearAlgebra/LinearSolverImpl.h>
#include <linearAlgebra/LinearAlgebraProfiler.h>

namespace dftefe
{
  namespace linearAlgebra
  {
    /**
     *
     * @brief Class that encapsulates an iterative Krylov subspace based
     *  linear solver. It solves for \f$\mathbf{x}$\f in the linear system
     *  of equations: \f$ \mathbf{Ax}=\mathbf{b}$\f. Internally, it
     *  creates a derived object of LinearSolverImpl based on the input
     *  linearAlgebra::LinearSolverType and then delegates the
     *  linear solve task to it. In that sense, it works more as a
     *  LinearSolver factory.
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
    class LinearSolver
    {
    public:
      using LinearSolverProfiler = LinearAlgebraProfiler;

    public:
      /**
       * @brief Constructor
       *
       * @param[in] linearSolverType Type of the linear solver. For example,
       * linearAlgbera::LinearSolverType:CG,
       * linearAlgbera::LinearSolverType:GMRES, etc.
       * @param[in] maxIter Maximum number of iterations to allow the solver
       * to iterate. Generally, this determines the maximum size of the Krylov
       * subspace that will be used.
       * @param[in] absoluteTol Convergence tolerane on the absolute \f$ L_2 $\f
       * norm of the residual (i.e., on \f$||\mathbf{Ax}-\mathbf{b}||$\f)
       * @param[in] relativeTol Convergence tolerane on the relative L2 norm of
       * the residual (i.e., on \f$||\mathbf{Ax}-\mathbf{b}||/||\mathbf{b}||$\f)
       * @param[in] divergenceTol Tolerance to abort the linear solver if the
       * L2 norm of the residual exceeds it
       * (i.e., if \f$||\mathbf{Ax}-\mathbf{b}|| > divergenceTol$\f)
       *
       * @note Convergence is achieved if \f$||\mathbf{Ax}-\mathbf{b}|| < max(absoluteTol, relativeTol*||\mathbf{b}||)$\f
       *
       */
      LinearSolver(
        const LinearSolverType linearSolverType,
        const size_type        maxIter,
        const double           absoluteTol = LinearSolverDefaults::ABS_TOL,
        const double           relativeTol = LinearSolverDefaults::REL_TOL,
        const double divergenceTol = LinearSolverDefaults::DIVERGENCE_TOL,
        LinearSolverProfiler profiler = LinearSolverProfiler());

      /**
       * @brief Default Destructor
       */
      ~LinearSolver() = default;

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
      Error
      solve(
        LinearSolverFunction<ValueTypeOperator, ValueTypeOperand, memorySpace>
          &linearSolverFunction) const;

    private:
      LinearSolverType d_linearSolverType;
      std::shared_ptr<
        LinearSolverImpl<ValueTypeOperator, ValueTypeOperand, memorySpace>>
        d_linearSolverImpl;
    }; // end of class LinearSolver
  }    // end of namespace linearAlgebra
} // end of namespace dftefe
#include <linearAlgebra/LinearSolver.t.cpp>
#endif // dftefeLinearSolver_h
