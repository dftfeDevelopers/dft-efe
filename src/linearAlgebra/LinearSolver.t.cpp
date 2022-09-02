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

#include <linearAlgebra/CGLinearSolver.h>
#include <utils/Exceptions.h>
namespace dftefe
{
  namespace linearAlgebra
  {
    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    LinearSolver<ValueTypeOperator, ValueTypeOperand, memorySpace>::
      LinearSolver(
        const LinearSolverType linearSolverType,
        const size_type        maxIter,
        const double           absoluteTol /*= LinearSolverDefaults::ABS_TOL*/,
        const double           relativeTol /*= LinearSolverDefaults::REL_TOL*/,
        const double divergenceTol /*= LinearSolverDefaults::DIVERGENCE_TOL*/,
        LinearAlgebraProfiler profiler /*= LinearSolverPrintControl()*/)
      : d_linearSolverImpl(nullptr)
    {
      if (linearSolverType == LinearSolverType::CG)
        {
          d_linearSolverImpl = std::make_shared<
            CGLinearSolver<ValueTypeOperator, ValueTypeOperand, memorySpace>>(
            maxIter, absoluteTol, relativeTol, divergenceTol, profiler);
        }
      else
        {
          utils::throwException<utils::InvalidArgument>(
            false, "Invalid LinearSolverType passed to LinearSolver");
        }
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    Error
    LinearSolver<ValueTypeOperator, ValueTypeOperand, memorySpace>::solve(
      LinearSolverFunction<ValueTypeOperator, ValueTypeOperand, memorySpace>
        &linearSolverFunction) const
    {
      return d_linearSolverImpl->solve(linearSolverFunction);
    }
  } // end of namespace linearAlgebra
} // end of namespace dftefe
