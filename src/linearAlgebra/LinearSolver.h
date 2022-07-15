
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

#ifndef dftefeLinearSolver_h
#define dftefeLinearSolver_h

#include <linearAlgebra/LinearSolverFunction.h>
#include <utils/MemorySpaceType.h>
namespace dftefe
{
  namespace linearAlgebra
  {
    /**
     * @brief An abstract class for a linear solver. The concrete implementation
     * will be provided in the derived class (e.g. CGLinearSolver,
     * GMRESLinearSolver, etc)
     *
     * @tparam ValueType datatype (float, double, complex<float>, complex<double>, etc) of the underlying matrix and vector
     * @tparam memorySpace defines the MemorySpace (i.e., HOST or
     * DEVICE) in which the underlying matrix and vector must reside.
     */

    template <typename ValueType, utils::MemorySpace memorySpace>
    class LinearSolver
    {
    public:
      enum class SolverType
      {
        CG
        // For future: GMRES, MINRES, BICG, etc
        //
      };


      enum class PCType
      {
        NONE,
        JACOBI
        // Add more sophisticated ones later
      };

      enum class ReturnType
      {
        SUCCESS,          // The linear solve was successful
        FAILURE,          // generic, no reason known
        MAX_ITER_REACHED, // Max. iteration provided by the user reached before
                          // convergence
        INDEFINITE_PC, // the preconditioner is usually assumed to be positive
                       // definite
        NAN_RES, // NaN or infinite values found in the evaluation of residual
                 // norm
      };

      virtual ~LinearSolver() = default;
      virtual ReturnType
      solve(LinearSolverFunction &function) = 0;
    };

  } // end of namespace linearAlgebra
} // end of namespace dftefe
#endif // dftefeLinearSolver_h
