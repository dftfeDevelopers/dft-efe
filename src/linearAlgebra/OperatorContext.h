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

#ifndef dftefeOperatorContext_h
#define dftefeOperatorContext_h

#include <utils/MemorySpaceType.h>
#include <linearAlgebra/Vector.h>
#include <linearAlgebra/MultiVector.h>
#include <linearAlgebra/BlasLapackTypedef.h>
namespace dftefe
{
  namespace linearAlgebra
  {
    /**
     *@brief Abstract class to encapsulate the action of a discrete operator on vectors, matrices, etc.
     *
     * @tparam ValueTypeOperator The datatype (float, double, complex<double>, etc.) for the underlying operator
     * @tparam ValueTypeOperand The datatype (float, double, complex<double>, etc.) of the vector, matrices, etc.
     * on which the operator will act
     * @tparam memorySpace The meory sapce (HOST, DEVICE, HOST_PINNES, etc.) in which the data of the operator
     * and its operands reside
     *
     */
    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    class OperatorContext
    {
      //
      // typedefs
      //
    public:
      //
      // alias to define the union of ValueTypeOperator and ValueTypeOperand
      // (e.g., the union of double and complex<double> is complex<double>)
      //
      using ValueTypeUnion =
        blasLapack::scalar_type<ValueTypeOperator, ValueTypeOperand>;

    public:
      /**
       *@brief Default Destructor
       *
       */
      ~OperatorContext() = default;

      /*
       * @brief Function to apply the operator on an input Vector \p x and store
       * the output in \p y. A typical use case is that the operator is a matrix
       * (\f$A$\f) and we want to evaluate \f$y=Ax$\f
       *
       * @param[in] x Input Vector
       * @param[out] y Output Vector that stores the action of the operator
       *  on \p x
       *
       * @note The input Vector \p x can be modified inside the function for
       * performance reasons. If the user needs \p x to be constant
       * (un-modified), we suggest the user to make a copy of \p x
       * prior to calling this function
       *
       */
      // virtual void
      // apply(Vector<ValueTypeOperand, memorySpace> &x,
      //       Vector<ValueTypeUnion, memorySpace> &  y) const = 0;

      /*
       * @brief Function to apply the operator on an input Vector \p X and store
       * the output in \p Y. A typical use case is that the operator is a matrix
       * (\f$A$\f) and we want to evaluate \f$Y=AX$\f
       *
       * @param[in] X Input Vector
       * @param[out] Y Output Vector that stores the action of the operator
       *  on \p X
       *
       * @note The input Vector \p X can be modified inside the function for
       * performance reasons. If the user needs \p X to be constant
       * (un-modified), we suggest the user to make a copy of \p X
       * prior to calling this function
       *
       */
      virtual void
      apply(const MultiVector<ValueTypeOperand, memorySpace> &X,
            MultiVector<ValueTypeUnion, memorySpace> &  Y) const = 0;
    };
  } // end of namespace linearAlgebra
} // end of namespace dftefe
#endif // dftefeOperatorContext_h
