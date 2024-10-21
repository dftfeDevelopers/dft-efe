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

#ifndef dftefeNewtonRaphsonSolver_h
#define dftefeNewtonRaphsonSolver_h

#include <utils/TypeConfig.h>
#include <linearAlgebra/LinearAlgebraTypes.h>
#include <linearAlgebra/NewtonRaphsonSolverFunction.h>

namespace dftefe
{
  namespace linearAlgebra
  {
    /**
     *
     * @brief A class that implements the Newton-Raphson solver to find root of a function.
     *
     * @tparam ValueType The datatype (float, double, complex<double>,
     * etc.)
     *
     */
    template <typename ValueType>
    class NewtonRaphsonSolver
    {
    public:
      /**
       * @brief Constructor
       *
       * @param[in] maxIter Maximum number of iterations to allow the solver
       * to iterate.
       * @param[in] tolerance Convergence tolerane on the root
       * @param[in] forceTolerance See if derivative is cose to zero (extremum).
       *
       *
       */
      NewtonRaphsonSolver(const size_type maxIter,
                          const double    tolerance,
                          const double    forceTolerance);

      /**
       * @brief Default Destructor
       */
      ~NewtonRaphsonSolver() = default;

      /**
       * @brief Function that initiates the NR solve
       *
       * @param[in] newtonRaphsonFunction
       *
       */
      NewtonRaphsonError
      solve(
        NewtonRaphsonSolverFunction<ValueType> &newtonRaphsonSolverFunction);

      ValueType
      getResidual();

    private:
      size_type d_maxIter;
      double    d_tolerance;
      double    d_forceTolerance;
      ValueType d_residual;
      bool      d_isSolved;
    }; // end of class NewtonRaphsonSolver
  }    // end of namespace linearAlgebra
} // end of namespace dftefe
#include <linearAlgebra/NewtonRaphsonSolver.t.cpp>
#endif // dftefeNewtonRaphsonSolver_h
