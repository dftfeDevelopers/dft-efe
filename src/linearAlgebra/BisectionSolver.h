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

#ifndef dftefeBisectionSolver_h
#define dftefeBisectionSolver_h

#include <utils/TypeConfig.h>
#include <linearAlgebra/LinearAlgebraTypes.h>
#include <linearAlgebra/BisectionSolverFunction.h>

namespace dftefe
{
  namespace linearAlgebra
  {
    /**
     *
     * @brief A class that implements the bisection method to find the root
     * of a function, given an interval known to bracket it. Unlike
     * NewtonRaphsonSolver, bisection only ever evaluates getValue() (never
     * a derivative), so it cannot diverge - each iteration halves the
     * bracket that is guaranteed to contain the root. This makes it a safe
     * way to localize a root before handing off to NewtonRaphsonSolver for
     * fast final convergence, mirroring DFT-FE's Fermi energy solver
     * (src/dft/fermiEnergy.cc), which bisects before running
     * Newton-Raphson.
     *
     * This is a minimal, textbook bisection: it trusts
     * getLowerBound()/getUpperBound() to already bracket a root and does
     * not attempt to expand or otherwise fix a bad bracket. Both bisection
     * itself and any bracket construction/expansion on top of it are only
     * correct if getValue() is monotonic between the bracket ends - a
     * property of the specific function being solved (e.g. an isolated
     * Fermi-Dirac occupancy sum is provably monotonic; an arbitrary
     * function need not be), not something this generic solver can verify.
     * Any such domain-specific bracket handling belongs in the
     * BisectionSolverFunction implementation, not here.
     *
     * @tparam ValueType The datatype (float, double, etc.)
     *
     */
    template <typename ValueType>
    class BisectionSolver
    {
    public:
      /**
       * @brief Constructor
       *
       * @param[in] maxIter Maximum number of bisection iterations.
       * @param[in] tolerance Convergence tolerance on |getValue(x)| at the
       * bisected midpoint.
       */
      BisectionSolver(const size_type maxIter, const double tolerance);

      /**
       * @brief Default Destructor
       */
      ~BisectionSolver() = default;

      /**
       * @brief Function that initiates the bisection solve.
       *
       * @param[in] bisectionSolverFunction
       *
       */
      BisectionError
      solve(BisectionSolverFunction<ValueType> &bisectionSolverFunction);

    private:
      size_type d_maxIter;
      double    d_tolerance;
      bool      d_isSolved;
    }; // end of class BisectionSolver
  }    // end of namespace linearAlgebra
} // end of namespace dftefe
#include <linearAlgebra/BisectionSolver.t.cpp>
#endif // dftefeBisectionSolver_h
