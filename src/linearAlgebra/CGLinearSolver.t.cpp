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

#include <linearAlgebra/BlasLapackTypedef.h>
#include <linearAlgebra/BlasLapack.h>

namespace dftefe
{
  namespace linearAlgebra
  {
    namespace CGLinearSolverInternal
    {
      template <typename T>
      std::string
      toStringWithPrecision(const T a_value, const int n)
      {
        std::ostringstream out;
        out.precision(n);
        out << std::fixed << a_value;
        return out.str();
      }
    } // namespace CGLinearSolverInternal

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    CGLinearSolver<ValueTypeOperator, ValueTypeOperand, memorySpace>::
      CGLinearSolver(const size_type                        maxIter,
                     const double                           absoluteTol,
                     const double                           relativeTol,
                     const double                           divergenceTol,
                     LinearSolver::LinearSolverPrintControl printControl)
      : d_maxIter(maxIter)
      , d_absoluteTol(absoluteTol)
      , d_relativeTol(relativeTol)
      , d_divergenceTol(divergenceTol)
      , d_printControl(printControl)
    {}


    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    Error
    CGLinearSolver<ValueTypeOperator, ValueTypeOperand, memorySpace>::solve(
      LinearSolverFunction<ValueTypeOperator, ValueTypeOperand, memorySpace>
        &linearSolverFunction) const
    {
      auto mpiComm = linearSolverFunction.getMPIComm();

      // register the start of the algorithm
      d_printControl.registerStart(mpiComm);

      using ValueType =
        blasLapack::scalar_type<ValueTypeOperator, ValueTypeOperand>;

      Vector<ValueType, memorySpace> b     = linearSolverFunction.getRhs();
      const double                   bNorm = b.l2Norm();

      Vector<ValueTypeOperand, memorySpace> x =
        linearSolverFunction.getInitialGuess();

      // get handle to Ax
      const OperatorContext<ValueTypeOperator, ValueTypeOperand, memorySpace>
        &AxContext = linearSolverFunction.getAxContext();

      // get handle to the preconditioner
      const OperatorContext<ValueTypeOperator, ValueTypeOperand, memorySpace>
        &pcContext = linearSolverFunction.getPCContext();

      //
      // Notations used
      // x = approx. solution
      // r = residual = Ax - b
      // z = preconditioned residual = PC(r)
      // p = search direction
      // w = Ap
      // alpha = step length
      // beta = improvement relative to previous step
      //

      //
      // r = b - Ax
      //
      Vector<ValueType, memorySpace> r(b, 0.0);
      //
      // @note: w is meant for storing Ap (p = search direction).
      //  However, for the 0-th iteration we use it to store Ax
      //  to avoid allocating memory for another Vector
      Vector<ValueType, memorySpace> w(b, 0.0);
      AxContext.apply(x, w);
      add((ValueType)1.0, b, (ValueType)-1.0, w, r);

      // ||r|| (L2 norm of r)
      double rNorm = r.l2Norm();

      //
      // z = preconditioned r
      //
      Vector<ValueType, memorySpace> z(r, 0.0);
      pcContext.apply(r, z);

      //
      // p = z
      //
      Vector<ValueType, memorySpace> p = z;

      //
      // CG loop
      //
      size_type precision = printControl.getPrecision();
      Error     err       = Error::OTHER_ERROR;
      size_type iter      = 0 for (; iter < d_maxIter; ++iter)
      {
        // register start of the iteration
        printControl.registerIterStart(iter);

        if (rNorm < std::max(d_absoluteTol, bNorm * d_relativeTol))
          {
            err = Error::SUCCESS;
            break;
          }
        else
          (rNorm > d_divergenceTol)
          {
            err = Error::RESIDUAL_DIVERGENCE;
            break;
          }

        // w = Ap
        AxContext.apply(p, w);

        // z^Hr (dot product of z-conjugate and r)
        ValueType zDotr = z.dot(r);

        // p^Hw (dot product of p-conjugate and w)
        ValueType pDotw = p.dot(w);

        ValueType alpha = zDotr / pDotw;

        // x = x + alpha*p
        add((ValueType)1.0, x, alpha, p, x);

        // r = r - alpha*w
        add((ValueType)1.0, r, -alpha, w, r);

        // z = preconditioned r
        pcContext.apply(r, z);

        // updated z^Hr (dot product of new z-conjugate and r)
        ValueType zDotrNew = z.dot(r);

        ValueType beta = zDotrNew / zDotr;

        // p = z + beta*p
        add((ValueType)1.0, z, beta, p, p);

        rNorm = r.l2Norm();

        std::string msg = "CGLinearSolver[" + std::to_string(iter) + "]";
        msg += " Abs. residual: " +
               CGLinearSolverInternal::toStringWithPrecision(rNorm, precision);
        msg += " Rel. residual: " +
               CGLinearSolverInternal::toStringWithPrecision(rNorm / bNorm,
                                                             precision);

        // register end of the iteration
        printControl.registerIterEnd(msg);

        iter++;
      }

      linearSolverFunction.setSolution(x);

      if (iter >= d_maxIter)
        {
          err = Error::FAILED_TO_CONVERGE;
        }

      std::string                  msg = "";
      std::pair<bool, std::string> successAndMsg =
        ErrorMsg::isSuccessAndMsg(err);

      if (successAndMsg.first)
        {
          msg = "CGLinear solve converged in " << iter << "iterations"
        }
      else
        msg = successAndMsg.second;

      // register end of CG
      printControl.registerEnd(msg);

      return err;
    }

  } // end of namespace linearAlgebra
} // end of namespace dftefe
