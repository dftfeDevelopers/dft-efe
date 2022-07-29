
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

#ifndef dftefeCGLinearSolver_h
#  define dftefeLinearSolver_h

#  include <linearAlgebra/LinearSolverFunction.h>
#  include <linearAlgebra/LinearSolver.h>
#  include <utils/MemorySpaceType.h>
#  include <utils/TypeConfig.h>
namespace dftefe
{
  namespace linearAlgebra
  {
    /**
     * @brief An class encapsulating the linear Conjugate-Gradient (CG) solver.
     *
     * @tparam ValueType datatype (float, double, complex<float>, complex<double>, etc) of the underlying matrix and vector
     * @tparam memorySpace defines the MemorySpace (i.e., HOST or
     * DEVICE) in which the underlying matrix and vector must reside.
     */

    template <typename ValueType, utils::MemorySpace memorySpace>
    class CGLinearSolver : public LinearSolver<ValueType, memorySpace>
    {
    public:
      CGLinearSolver(const double               relTol,
                     const size_type            maxIterations,
                     const LinearSolver::PCType pcType,
                     const size_type            debugLevel)

        ~CGLinearSolver() = default;
      virtual LinearSolver::ReturnType
      solve(LinearSolverFunction &function) = 0;

    private:
      double               d_relTol;
      size_type            d_maxIterations;
      LinearSolver::PCType d_pcType;
      size_type            d_debugLevel;
    };

  } // end of namespace linearAlgebra
} // end of namespace dftefe
#endif // dftefeLinearSolver_h
