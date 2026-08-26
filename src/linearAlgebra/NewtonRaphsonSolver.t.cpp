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
#include <utils/Exceptions.h>
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
      , d_residual((ValueType)0)
      , d_isSolved(false)
    {}


    template <typename ValueType>
    NewtonRaphsonError
    NewtonRaphsonSolver<ValueType>::solve(
      NewtonRaphsonSolverFunction<ValueType> &newtonRaphsonSolverFunction)
    {
      NewtonRaphsonError retunValue;
      d_isSolved           = true;
      ValueType x          = newtonRaphsonSolverFunction.getInitialGuess();
      ValueType xConverged = x;

      //
      // NR loop
      //
      NewtonRaphsonErrorCode err  = NewtonRaphsonErrorCode::OTHER_ERROR;
      size_type              iter = 0;
      bool                   isForceTolErr = false;
      std::string            forceTolDetail;

      for (; iter <= d_maxIter; ++iter)
        {
          if (utils::abs_(newtonRaphsonSolverFunction.getForce(x)) <
              d_forceTolerance)
            {
              // The derivative has (numerically) vanished - dividing by it
              // below would produce +-inf, and the following iteration's
              // "inf - inf" would poison x to NaN. A NaN residual can never
              // satisfy the convergence check, which would silently run the
              // loop out to d_maxIter (confirmed: x collapses to -nan within
              // 1e5 iterations and stays -nan for the rest of the 2e7-
              // iteration cap - what looked like a hang). Report the
              // failure back to the caller instead of limping on with a
              // poisoned x - it can decide how severely to react.
              err            = NewtonRaphsonErrorCode::FORCE_TOLERANCE_ERR;
              isForceTolErr  = true;
              xConverged     = x;
              forceTolDetail = "force magnitude fell below tolerance (" +
                               std::to_string(d_forceTolerance) +
                               ") at x = " + std::to_string(x) + " after " +
                               std::to_string(iter) + " iterations.";
              break;
            }

          ValueType x1 = x - newtonRaphsonSolverFunction.getValue(x) /
                               newtonRaphsonSolverFunction.getForce(x);

          d_residual = utils::abs_(x1 - x);

          // Near a root where the derivative is small but not small
          // enough to trip the force-tolerance check above (a sparse
          // spectrum has stretches of low local "density of states"), the
          // step size |x1-x| = |getValue(x)/force| can stay above
          // d_tolerance indefinitely (a stable finite cycle) even once
          // getValue(x) itself is already at numerical zero - confirmed:
          // observed value ~5.7e-14 with residual ~8.1e-7 oscillating
          // forever, never satisfying the step-size check alone. So also
          // accept convergence on the function value residual, matching
          // DFT-FE's Fermi energy solver (src/dft/fermiEnergy.cc), which
          // checks the value residual R rather than a step size.
          ValueType valueResidual =
            utils::abs_(newtonRaphsonSolverFunction.getValue(x1));

          if (d_residual < d_tolerance || valueResidual < d_tolerance)
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

      if (isForceTolErr)
        {
          retunValue.msg += forceTolDetail;
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

    template <typename ValueType>
    ValueType
    NewtonRaphsonSolver<ValueType>::getResidual()
    {
      ValueType retVal = (ValueType)0;
      if (d_isSolved)
        retVal = d_residual;
      else
        utils::throwException(
          false,
          "Cannot call getResidual() before calling solve() in NewtonRaphsonSolver.");
      return retVal;
    }
  } // end of namespace linearAlgebra
} // end of namespace dftefe
