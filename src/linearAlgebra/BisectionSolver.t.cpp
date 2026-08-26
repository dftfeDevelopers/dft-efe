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

#include <utils/DataTypeOverloads.h>
#include <utils/Exceptions.h>
#include <iostream>

namespace dftefe
{
  namespace linearAlgebra
  {
    template <typename ValueType>
    BisectionSolver<ValueType>::BisectionSolver(const size_type maxIter,
                                                const double    tolerance)
      : d_maxIter(maxIter)
      , d_tolerance(tolerance)
      , d_isSolved(false)
    {}

    template <typename ValueType>
    BisectionError
    BisectionSolver<ValueType>::solve(
      BisectionSolverFunction<ValueType> &bisectionSolverFunction)
    {
      // NOTE: this is deliberately a minimal, textbook bisection - it
      // trusts getLowerBound()/getUpperBound() to already bracket a root
      // (getValue() has opposite signs at the two ends) and does not
      // verify or attempt to fix a bad bracket. Bisection is only correct
      // for a getValue() that is monotonic between the bracket ends; that
      // is a property of the specific function being solved (e.g.
      // FractionalOccupancyFunction, a sum of Fermi-Dirac sigmoids, is
      // provably monotonic and validates its own bracket before this
      // solver ever runs), not something this generic solver can verify
      // or assume for an arbitrary BisectionSolverFunction. Any
      // domain-specific bracket construction/validation belongs in the
      // BisectionSolverFunction implementation, not here.
      BisectionError retunValue;
      d_isSolved = true;

      ValueType xLeft  = bisectionSolverFunction.getLowerBound();
      ValueType xRight = bisectionSolverFunction.getUpperBound();
      ValueType yLeft  = bisectionSolverFunction.getValue(xLeft);
      ValueType x      = xLeft;

      BisectionErrorCode err  = BisectionErrorCode::OTHER_ERROR;
      size_type          iter = 0;

      for (; iter < d_maxIter; ++iter)
        {
          x              = (ValueType)0.5 * (xLeft + xRight);
          ValueType yMid = bisectionSolverFunction.getValue(x);

          if (yMid * yLeft > (ValueType)0)
            {
              xLeft = x;
              yLeft = yMid;
            }
          else
            xRight = x;

          if (utils::abs_(yMid) <= d_tolerance || iter == d_maxIter - 1)
            {
              err = BisectionErrorCode::SUCCESS;
              break;
            }
        }

      bisectionSolverFunction.setSolution(x);

      if (err != BisectionErrorCode::SUCCESS)
        err = BisectionErrorCode::FAILED_TO_CONVERGE;

      retunValue = BisectionErrorMsg::isSuccessAndMsg(err);

      if (retunValue.isSuccess)
        retunValue.msg += "Bisection solve converged in maximum " +
                          std::to_string(iter) + " iterations.";

      return retunValue;
    }
  } // end of namespace linearAlgebra
} // end of namespace dftefe
