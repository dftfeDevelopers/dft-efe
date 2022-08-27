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
    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    CGLinearSolver<ValueTypeOperator, ValueTypeOperand, memorySpace>::
      CGLinearSolver(const size_type                  maxIter,
                     const double                     absoluteTol,
                     const double                     relativeTol,
                     const utils::ConditionalOStream *coStream)
      : d_maxIter(maxIter)
      , d_absoluteTol(absoluteTol)
      , d_relativeTol(relativeTol)
      , d_coStream(coStream)
    {}


    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    SolverTypes::Error
    CGLinearSolver<ValueTypeOperator, ValueTypeOperand, memorySpace>::solve(
      LinearSolverFunction<ValueTypeOperator, ValueTypeOperand, memorySpace>
        &linearSolverFunction) const
    {
      using ValueType =
        blasLapack::scalar_type<ValueTypeOperator, ValueTypeOperand>
          Vector<ValueType, memorySpace>    b = linearSolverFunction.getRhs();
      Vector<ValueTypeOperand, memorySpace> x =
        linearSolverFunction.getInitialGuess();

      // get handle to Ax
      const OperatorContext<ValueTypeOperator, ValueTypeOperand, memorySpace>
        &AxContext = linearSolverFunction.getAxContext();

      // get handle to the preconditioner
      const OperatorContext<ValueTypeOperator, ValueTypeOperand, memorySpace>
        &pcContext = linearSolverFunction.getPCContext();

      for (size_type iter = 0; iter < d_maxIter; ++iter)
        {}
    }

  } // end of namespace linearAlgebra
} // end of namespace dftefe
