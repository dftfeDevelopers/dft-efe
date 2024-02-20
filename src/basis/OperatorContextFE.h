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
     *@brief Abstract class to encapsulate the action of a discrete operator on vectors, matrices, etc. in a basis
     *
     * @tparam ValueTypeOperator The datatype (float, double, complex<double>, etc.) for the underlying operator
     * @tparam ValueTypeOperand The datatype (float, double, complex<double>, etc.) of the vector, matrices, etc.
     * on which the operator will act
     * @tparam memorySpace The memory sapce (HOST, DEVICE, HOST_PINNES, etc.) in which the data of the operator
     * and its operands reside
     *
     */
    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    class OperatorContextFE ::
      OperatorContext<ValueTypeOperator, ValueTypeOperand, memorySpace>
    {
    public:
      /**
       * @brief Constructor
       */
      OperatorContextFE(
        const basis::FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &feBasisDataStorage);

      /**
       *@brief Default Destructor
       *
       */
      ~OperatorContextFE() = default;

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
      apply(MultiVector<ValueTypeOperand, memorySpace> &X,
            MultiVector<ValueTypeUnion, memorySpace> &  Y) const = 0;

      virtual void
      getCellMatrixDataInAllCells() const = 0;

      virtual void
      getCellMatrixDataInCell() const = 0;
    };
  } // end of namespace linearAlgebra
} // end of namespace dftefe
#endif // dftefeOperatorContext_h
