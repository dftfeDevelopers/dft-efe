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
#include <utils/DataTypeOverloads.h>

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
      CGLinearSolver(const size_type       maxIter,
                     const double          absoluteTol,
                     const double          relativeTol,
                     const double          divergenceTol,
                     LinearAlgebraProfiler profiler)
      : d_maxIter(maxIter)
      , d_absoluteTol(absoluteTol)
      , d_relativeTol(relativeTol)
      , d_divergenceTol(divergenceTol)
      , d_profiler(profiler)
    {}


    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    Error
    CGLinearSolver<ValueTypeOperator, ValueTypeOperand, memorySpace>::solve(
      LinearSolverFunction<ValueTypeOperator, ValueTypeOperand, memorySpace>
        &linearSolverFunction)
    {
      auto mpiComm = linearSolverFunction.getMPIComm();

      // register the start of the algorithm
      d_profiler.registerStart(mpiComm);

      using ValueType =
        blasLapack::scalar_type<ValueTypeOperator, ValueTypeOperand>;

      MultiVector<ValueType, memorySpace> b;
      b = (linearSolverFunction.getRhs());

      std::vector<double> bNorm(0);
      bNorm = b.l2Norms();

      size_type numComponents = b.getNumberComponents();

      MultiVector<ValueTypeOperand, memorySpace> x;
      x = (linearSolverFunction.getInitialGuess());
      MultiVector<ValueTypeOperand, memorySpace> xConverged;
      xConverged = (linearSolverFunction.getInitialGuess());

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

      MultiVector<ValueType, memorySpace> r(b, 0.0);
      MultiVector<ValueType, memorySpace> w(b, 0.0);
      MultiVector<ValueType, memorySpace> z(b, 0.0);
      MultiVector<ValueType, memorySpace> p(b, 0.0);


      //
      // CG loop
      //
      std::vector<double> rNorm(numComponents , 0.0);
      size_type precision = d_profiler.getPrecision();
      Error     err       = Error::OTHER_ERROR;
      size_type iter      = 0;
      size_type i = 0,j = 0;
      std::vector<bool> convergeFlag(numComponents, false);
      bool divergeFlag = false;
      bool allConverged = false;
      std::vector<ValueType> ones(numComponents, (ValueType)1.0);
      std::vector<ValueType> nOnes(numComponents, (ValueType)-1.0);

      for (; iter <= d_maxIter; ++iter)
      {
        // register start of the iteration
        d_profiler.registerIterStart(iter);

        if (iter == 0)
        {
          //
          // @note: w is meant for storing Ap (p = search direction).
          // However, for the 0-th iteration we use it to store Ax
          // to avoid allocating memory for another Vector
          AxContext.apply(x, w);
          add(ones, b, nOnes, w, r); //TODO

          //
          // z = preconditioned r
          //
          pcContext.apply(r, z);

          //
          // p = z
          //
          p = z;
        }

        else
        {
          // w = Ap
          AxContext.apply(p, w);

          // z^Hr (dot product of z-conjugate and r)
          std::vector<ValueType> zDotr(0);
          dot(z,
              r,
              zDotr,
              blasLapack::ScalarOp::Conj,
              blasLapack::ScalarOp::Identity);

          // p^Hw (dot product of p-conjugate and w)
          std::vector<ValueType> pDotw(0);
          dot(p,
              w,
              pDotw,
              blasLapack::ScalarOp::Conj,
              blasLapack::ScalarOp::Identity);

          //ValueType alpha = zDotr / pDotw;
          std::vector<ValueType> alpha(0), nAlpha(0);
          for (i = 0 ; i < numComponents ; i++)
          {
            alpha.push_back(zDotr[i] / pDotw[i]);
            nAlpha.push_back(-zDotr[i] / pDotw[i]);
          }

          // x = x + alpha*p
          add(ones, x, alpha, p, x); //TODO

          // r = r - alpha*w
          add(ones, r, nAlpha, w, r); //TODO

          // z = preconditioned r
          pcContext.apply(r, z);

          // updated z^Hr (dot product of new z-conjugate and r)
          std::vector<ValueType> zDotrNew(0);
          dot(z,
              r,
              zDotrNew,
              blasLapack::ScalarOp::Conj,
              blasLapack::ScalarOp::Identity);

          //ValueType beta = zDotrNew / zDotr;
          std::vector<ValueType> beta(0);
          for (i = 0 ; i < numComponents ; i++)
            beta.push_back(zDotrNew[i] / zDotr[i]);

          // p = z + beta*p
          add(ones, z, beta, p, p); //TODO
        }

        std::vector<double> rNorm(0);
        rNorm = r.l2Norms();
        //rNorm = r.l2Norm();

        std::string msg;
        for ( i = 0 ; i < numComponents ; i++ )
        {
          if (rNorm[i] < std::max(d_absoluteTol, bNorm[i] * d_relativeTol) && convergeFlag[i]==false)
          {
            err = Error::SUCCESS;
            convergeFlag[i] = true;
            //copy x to xConverged of the ith comp vector
            for(j = 0 ; j < x.locallyOwnedSize() ; j++)
            *(xConverged.data() + j*numComponents + i) = *(x.data() + j*numComponents + i);
          }

          if (rNorm[i] > d_divergenceTol && divergeFlag==false)
          {
            err = Error::RESIDUAL_DIVERGENCE;
            divergeFlag = true;
          }

        msg = "CGLinearSolver[" + std::to_string(iter) + "]";
        msg +=
          " Abs. residual: " +
          CGLinearSolverInternal::toStringWithPrecision(rNorm[i], precision);
        msg += " Rel. residual: " +
                CGLinearSolverInternal::toStringWithPrecision(rNorm[i] / bNorm[i],
                                                              precision);
        msg += "For Vector Component No: " + i;
        }

        if ( std::all_of(convergeFlag.begin(), convergeFlag.end(), [](bool boolVal){return boolVal;}) )
            allConverged = true;

        if(divergeFlag || allConverged)
          break;

        // register end of the iteration
        d_profiler.registerIterEnd(msg);
      }

      linearSolverFunction.setSolution(xConverged);

      if (iter > d_maxIter)
        {
          err = Error::FAILED_TO_CONVERGE;
        }

      std::string                  msg = "";
      std::pair<bool, std::string> successAndMsg =
        ErrorMsg::isSuccessAndMsg(err);

      if (successAndMsg.first)
        {
          msg = "CGLinear solve converged in maximum " + std::to_string(iter) +
                " iterations";
        }
      else
        msg = successAndMsg.second;

      // register end of CG
      d_profiler.registerEnd(msg);

      return err;
    }

    //    template <typename ValueTypeOperator,
    //              typename ValueTypeOperand,
    //              utils::MemorySpace memorySpace>
    //    Error
    //    CGLinearSolver<ValueTypeOperator, ValueTypeOperand,
    //    memorySpace>::solve(
    //      LinearSolverFunction<ValueTypeOperator, ValueTypeOperand,
    //      memorySpace>
    //        &linearSolverFunction)
    //    {
    //      auto mpiComm = linearSolverFunction.getMPIComm();
    //
    //      // register the start of the algorithm
    //      d_profiler.registerStart(mpiComm);
    //
    //      using ValueType =
    //        blasLapack::scalar_type<ValueTypeOperator, ValueTypeOperand>;
    //
    //      Vector<ValueType, memorySpace> b     =
    //      linearSolverFunction.getRhs(); const double                   bNorm
    //      = b.l2Norm();
    //
    //      Vector<ValueTypeOperand, memorySpace> x =
    //        linearSolverFunction.getInitialGuess();
    //
    //      // get handle to Ax
    //      const OperatorContext<ValueTypeOperator, ValueTypeOperand,
    //      memorySpace>
    //        &AxContext = linearSolverFunction.getAxContext();
    //
    //      // get handle to the preconditioner
    //      const OperatorContext<ValueTypeOperator, ValueTypeOperand,
    //      memorySpace>
    //        &pcContext = linearSolverFunction.getPCContext();
    //
    //      //
    //      // Notations used
    //      // x = approx. solution
    //      // r = residual = Ax - b
    //      // z = preconditioned residual = PC(r)
    //      // p = search direction
    //      // w = Ap
    //      // alpha = step length
    //      // beta = improvement relative to previous step
    //      //
    //
    //      Vector<ValueType, memorySpace> r(b, 0.0);
    //      Vector<ValueType, memorySpace> w(b, 0.0);
    //      Vector<ValueType, memorySpace> z(b, 0.0);
    //      Vector<ValueType, memorySpace> p(b, 0.0);
    //
    //      ValueType dpi = 0.0, a = 1.0, beta, betaold = 1.0, t = 0, dpiold;
    //      double dp = 0.0;
    //
    //      //
    //      // @note: w is meant for storing Ap (p = search direction).
    //      // However, for the 0-th iteration we use it to store Ax
    //      // to avoid allocating memory for another Vector
    //      AxContext.apply(x, w);
    //      add((ValueType)1.0, b, (ValueType)-1.0, w, r);
    //
    //      //
    //      // z = preconditioned r
    //      //
    //      pcContext.apply(r, z);
    //
    //      dp = z.l2Norm();
    //
    //      // beta = z^Hr
    //      beta = dot(z ,
    //	  r,
    //	  blasLapack::ScalarOp::Conj,
    //	  blasLapack::ScalarOp::Identity);
    //      //
    //      // CG loop
    //      //
    //      double rNorm = 0.0;
    //      size_type precision = d_profiler.getPrecision();
    //      Error     err       = Error::OTHER_ERROR;
    //      size_type iter      = 0;
    //      for (; iter <= d_maxIter; ++iter)
    //      {
    //	// register start of the iteration
    //	d_profiler.registerIterStart(iter);
    //
    //	if(iter == 0)
    //	{
    //	  // p = z
    //	  p = z;
    //
    //	  t = 0.0;
    //
    //	}
    //
    //	else
    //	{
    //	  t = beta/betaold;
    //
    //	  // p = z + t*p
    //	  add((ValueType)1.0, z, t, p, p);
    //	}
    //
    //	dpiold = dpi;
    //
    //	// w = Ap
    //	AxContext.apply(p, w);
    //
    //	// p^Hw (dot product of p-conjugate and w)
    //	dpi = dot(p,
    //	    w,
    //	    blasLapack::ScalarOp::Conj,
    //	    blasLapack::ScalarOp::Identity);
    //
    //	betaold = beta;
    //
    //	//
    //	//TODO: Check for indefinite matrix
    //	//
    //
    //	// a = beta/ (p^Hw)
    //	a = beta/dpi;
    //
    //	// x = x + a*p
    //	add((ValueType)1.0, x, a, p, x);
    //
    //	// r = r - a*w
    //	add((ValueType)1.0, r, -a, w, r);
    //
    //	// z = preconditioned r
    //	pcContext.apply(r, z);
    //
    //	dp = z.l2Norm();
    //
    //	// beta = z^Hr (dot product of z-conjugate and r)
    //	beta = dot(z,
    //	    r,
    //	    blasLapack::ScalarOp::Conj,
    //	    blasLapack::ScalarOp::Identity);
    //
    //	rNorm = r.l2Norm();
    //
    //	if(rNorm < std::max(d_absoluteTol, bNorm * d_relativeTol))
    //	{
    //	  err = Error::SUCCESS;
    //	  break;
    //	}
    //
    //	if(rNorm > d_divergenceTol)
    //	{
    //	  err = Error::RESIDUAL_DIVERGENCE;
    //	  break;
    //	}
    //
    //
    //	std::string msg = "CGLinearSolver[" + std::to_string(iter) + "]";
    //	msg +=
    //	  " Abs. residual: " +
    //	  CGLinearSolverInternal::toStringWithPrecision(rNorm, precision);
    //	msg += " Rel. residual: " +
    //	  CGLinearSolverInternal::toStringWithPrecision(rNorm / bNorm,
    //	      precision);
    //
    //	// register end of the iteration
    //	d_profiler.registerIterEnd(msg);
    //      }
    //
    //      linearSolverFunction.setSolution(x);
    //
    //      if (iter > d_maxIter)
    //      {
    //	err = Error::FAILED_TO_CONVERGE;
    //      }
    //
    //      std::string                  msg = "";
    //      std::pair<bool, std::string> successAndMsg =
    //	ErrorMsg::isSuccessAndMsg(err);
    //
    //      if (successAndMsg.first)
    //      {
    //	msg = "CGLinear solve converged in " + std::to_string(iter) +
    //	  " iterations";
    //      }
    //      else
    //	msg = successAndMsg.second;
    //
    //      // register end of CG
    //      d_profiler.registerEnd(msg);
    //
    //      return err;
    //    }
  } // end of namespace linearAlgebra
} // end of namespace dftefe
