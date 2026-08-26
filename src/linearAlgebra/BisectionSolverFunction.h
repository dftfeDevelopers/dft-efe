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

#ifndef dftefeBisectionSolverFunction_h
#define dftefeBisectionSolverFunction_h

#include <utils/TypeConfig.h>
#include <linearAlgebra/LinearAlgebraTypes.h>

namespace dftefe
{
  namespace linearAlgebra
  {
    /**
     * @brief Interface a class must implement to have its root found by
     * BisectionSolver. Unlike NewtonRaphsonSolverFunction, no derivative
     * (getForce) is required - bisection only ever evaluates getValue(),
     * which is what makes it immune to sharply-peaked or vanishing
     * derivatives that can send Newton-Raphson to +-infinity.
     */
    template <typename ValueType>
    class BisectionSolverFunction
    {
    public:
      virtual ~BisectionSolverFunction() = default;

      virtual const ValueType
      getValue(ValueType &x) const = 0;

      virtual void
      setSolution(const ValueType &x) = 0;

      virtual void
      getSolution(ValueType &solution) = 0;

      /**
       * @brief Lower end of an interval known to bracket the root, i.e.
       * getValue(getLowerBound()) and getValue(getUpperBound()) must have
       * opposite signs (or one of them exactly zero).
       */
      virtual const ValueType
      getLowerBound() const = 0;

      /**
       * @brief Upper end of the bracketing interval. See getLowerBound().
       */
      virtual const ValueType
      getUpperBound() const = 0;

    }; // end of class BisectionSolverFunction
  }    // end of namespace linearAlgebra
} // end of namespace dftefe
#endif // dftefeBisectionSolverFunction_h
