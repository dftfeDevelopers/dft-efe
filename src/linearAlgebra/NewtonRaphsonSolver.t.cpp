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

#include <linearAlgebra/BlasLapackTypedef.h>
#include <linearAlgebra/BlasLapack.h>
#include <utils/DataTypeOverloads.h>
#include <iostream>
namespace dftefe
{
  namespace linearAlgebra
  {
    template <typename ValueType>
    NewtonRaphsonSolver<ValueType>::NewtonRaphsonSolver(
      const size_type maxIter,
      const double    tolerance,
      const double    forceTolerance)
      : d_maxIter(maxIter)
      , d_tolerance(tolerance)
      , d_forceTolerance(forceTolerance)
    {}


    template <typename ValueType>
    NewtonRaphsonError
    NewtonRaphsonSolver<ValueType>::solve(
      NewtonRaphsonSolverFunction<ValueType> &newtonRaphsonSolverFunction)
    {
      NewtonRaphsonError retunValue;

      ValueType x = newtonRaphsonSolverFunction.getInitialGuess();
      ValueType xConverged;

      //
      // NR loop
      //
      NewtonRaphsonErrorCode err  = NewtonRaphsonErrorCode::OTHER_ERROR;
      size_type              iter = 0;
      bool                   isForceTolErr = false;

      for (; iter <= d_maxIter; ++iter)
        {
          if (utils::abs_(newtonRaphsonSolverFunction.getForce(x)) <
              d_forceTolerance)
            {
              err           = NewtonRaphsonErrorCode::FORCE_TOLERANCE_ERR;
              isForceTolErr = true;
            }

          ValueType x1 = x - newtonRaphsonSolverFunction.getValue(x) /
                               newtonRaphsonSolverFunction.getForce(x);

          if (utils::abs_(x1 - x) < d_tolerance)
            {
              err        = NewtonRaphsonErrorCode::SUCCESS;
              xConverged = x1;
              break;
            }

          x = x1; // Update x0 for the next iteration
        }

      newtonRaphsonSolverFunction.setSolution(xConverged);

      if (iter > d_maxIter && !isForceTolErr)
        {
          err = NewtonRaphsonErrorCode::FAILED_TO_CONVERGE;
        }

      std::string msg = "";
      retunValue      = NewtonRaphsonErrorMsg::isSuccessAndMsg(err);

      if (iter > d_maxIter && isForceTolErr)
        {
          retunValue.msg += "Failed to converge.";
        }

      if (retunValue.isSuccess)
        {
          msg = "Newton Raphson solve converged in maximum " +
                std::to_string(iter) + " iterations.";
          retunValue.msg += msg;
        }
      else
        msg = retunValue.msg;

      return retunValue;
    }
  } // end of namespace linearAlgebra
} // end of namespace dftefe
