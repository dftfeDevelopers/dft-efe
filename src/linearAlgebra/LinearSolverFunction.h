
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

#ifndef dftefeLinearSolverFunction_h
#define dftefeLinearSolverFunction_h

#include <linearAlgebra/Vector.h>
#include <utils/MemorySpaceType.h>
namespace dftefe
{
  namespace linearAlgebra
  {
    /**
     * @brief An abstract class to encapsulate a linear partial differential equation (PDE)
     *
     * @tparam ValueType datatype (float, double, complex<float>, complex<double>, etc) of the underlying fields in the PDE
     * @tparam memorySpace defines the MemorySpace (i.e., HOST or
     * DEVICE) in which the underlying data (vector and matrices associated with
     * the discrete PDE) vector must reside.
     */

    template <typename ValueType, utils::MemorySpace memorySpace>
    class LinearSolverFunction
    {
      /**
       * @brief Destructor.
       */
      virtual ~LinearSolverFunction() = default;

      virtual Vector<ValueType, memorySpace>
      getRhs() = 0;

      virtual Vector<ValueType, memorySpace>
      getInitialGuess() const = 0;

      virtual Vector<ValueType, memorySpace>
      getSolution() const = 0;

      virtual void
      setSolution(const Vector<ValueType, memorySpace> &x) = 0;

      virtual void
      computeAx(const Vector<ValueType, memorySpace> &x,
                Vector<ValueType, memorySpace> &      Ax) const = 0;

      virtual void
      computeATransx(const Vector<ValueType, memorySpace> &x,
                     Vector<ValueType, memorySpace> &      ATx) const = 0;

      virtual void
      getDiagonalA(Vector<ValueType, memorySpace> &d) const = 0;
    };
  } // end of namespace linearAlgebra
} // end of namespace dftefe
#endif // dftefeLinearSolverFunction_h
