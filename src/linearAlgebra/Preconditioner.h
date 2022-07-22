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

#ifndef dftefePreconditioner_h
#define dftefePreconditioner_h

#include <linearAlgebra/Vector.h>
#include <utils/MemorySpaceType.h>
namespace dftefe
{
  namespace linearAlgebra
  {
    /**
     * @brief An abstract class for a preconditioner in a linear or non-linear solve. The concrete implementation
     * will be provided in the derived class (e.g. JacobiPreconditioner,...)
     *
     * @tparam ValueType datatype (float, double, complex<float>, complex<double>, etc) of the underlying matrix and vector
     * @tparam memorySpace defines the MemorySpace (i.e., HOST or
     * DEVICE) in which the underlying matrix and vector must reside.
     */
    template <typename ValueType, utils::MemorySpace memorySpace>
    class Preconditioner
    {
    public:
      /**
       * @brief Default destructor
       */
      virtual ~Preconditioner() = default;

      /**
       * @brief In-place apply the preconditioner on a given Vector
       * (i.e., the input vector is modified to store the output)
       *
       * @param[in] x the given input vector
       * @param[out] x the input vector is modified in-place to store the output
       */
      virtual void
      apply(Vector<ValueType, memorySpace> &x) const = 0;

      /**
       * @brief Apply the preconditioner on a given vector and return the output vector
       *
       * @param[in] x the given input vector
       * @return the vector resulting from the application of the preconditioner on the input vector x
       */
      virtual Vector<ValueType, memorySpace>
      apply(const Vector<ValueType, memorySpace> &x) const = 0;
    };
  } // end of namespace linearAlgebra
} // end of namespace dftefe
#endif // dftefePreconditioner_h
