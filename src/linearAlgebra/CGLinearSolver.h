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

#ifndef dftefeCGLinearSolver_h
#define dftefeCGLinearSolver_h

#include <utils/TypeConfig.h>
#include <utils/MemorySpaceType.h>
#include <linearAlgebra/LinearAlgebraTypes.h>
#include <linearAlgebra/LinearAlgebraProfiler.h>
#include <linearAlgebra/LinearSolverImpl.h>
#include <linearAlgebra/LinearSolverFunction.h>

namespace dftefe
{
  namespace linearAlgebra
  {
    /**
     *
     * @brief A class that implements the Conjugate-Gradient (CG) based Krylov
     *  subspace algorithm  to solve a linear system of
     *  (i.e., solve for \f$ \mathbf{Ax}=\mathbf{b}$\f).
     *
     * @see <em>An Introduction to the Conjugate Gradient Method Without the
     *   Agonizing Pain</em>, Jonathan Richard Shewchuk
     *   (<a
     * href="https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf">Painless
     * Conjugate-Gradient</a>)
     *
     * @see <em> Numerical Linear Algebra </em>, Trefethen, Lloyd N., and David Bau III., Vol. 50. Siam, 1997.
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
    class CGLinearSolver : public LinearSolverImpl<ValueTypeOperator,
                                                   ValueTypeOperand,
                                                   memorySpace>
    {
    public:
      /**
       * @brief Constructor
       *
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
       * @note Convergence is achieved if
       * \f$||\mathbf{Ax}-\mathbf{b}|| < max(absoluteTol,
       * relativeTol*||\mathbf{b}||)$\f
       */
      CGLinearSolver(const size_type       maxIter,
                     const double          absoluteTol,
                     const double          relativeTol,
                     const double          divergenceTol,
                     LinearAlgebraProfiler profiler = LinearAlgebraProfiler());

      /**
       * @brief Default Destructor
       */
      ~CGLinearSolver() = default;

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
      LinearSolverError
      solve(
        LinearSolverFunction<ValueTypeOperator, ValueTypeOperand, memorySpace>
          &linearSolverFunction) override;

    private:
      LinearAlgebraProfiler d_profiler;
      size_type             d_maxIter;
      double                d_absoluteTol;
      double                d_relativeTol;
      double                d_divergenceTol;
    }; // end of class CGLinearSolver
  }    // end of namespace linearAlgebra
} // end of namespace dftefe
#include <linearAlgebra/CGLinearSolver.t.cpp>
#endif // dftefeCGLinearSolver_h
