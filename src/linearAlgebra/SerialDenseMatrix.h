/******************************************************************************
 * Copyright (c) 2022.                                                        *
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
 * @author Ian C. Lin, Vishal Subramanian
 */

#ifndef dftefeSerialDenseMatrix_h
#define dftefeSerialDenseMatrix_h

#include "Matrix.h"
#include <utils/TypeConfig.h>
#include <vector>
#include "BlasWrappersTypedef.h"
#include <utils/MemoryStorage.h>
#include "QueueManager.h"

namespace dftefe
{
  namespace linearAlgebra
  {
    /**
     * @brief A  Matrix class that provides an interface to the underlying
     * matrix and compatible with different MemorySpace---
     * HOST (cpu) , DEVICE (gpu), etc,.
     *
     * @tparam ValueType The underlying value type for the MemoryStorage
     *  (e.g., int, double, complex<double>, etc.)
     * @tparam memorySpace The memory space in which the MemoryStorage needs
     *  to reside
     *
     */
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    class SerialDenseMatrix : public Matrix<ValueType, memorySpace>
    {
    public:
      //
      // Pulling base class (Matrix) protected names here so to avoid full name
      // scoping inside the source file. The other work around is to use
      // this->d_m (where d_m is a protected data member of base class). This is
      // something which is peculiar to inheritance using class templates. The
      // reason why this is so is the fact that C++ does not consider base class
      // templates for name resolution (i.e., they are dependent names and
      // dependent names are not considered)
      //
      using Matrix<ValueType, memorySpace>::d_data;
      using Matrix<ValueType, memorySpace>::d_blasQueue;
      using Matrix<ValueType, memorySpace>::d_nGlobalRows;
      using Matrix<ValueType, memorySpace>::d_nGlobalCols;
      using Matrix<ValueType, memorySpace>::d_nLocalRows;
      using Matrix<ValueType, memorySpace>::d_nLocalCols;
      using Matrix<ValueType, memorySpace>::d_property;
      using Matrix<ValueType, memorySpace>::d_uplo;
      using Matrix<ValueType, memorySpace>::d_layout;

      SerialDenseMatrix() = default;

      /**
       * @brief Copy constructor for a MemoryStorage
       * @param[in] u SerialDensityMatrix object to copy from
       */
      SerialDenseMatrix(const SerialDenseMatrix &u);

      /**
       * @brief Move constructor for a matrix
       * @param[in] u SerialDensityMatrix object to move from
       */
      SerialDenseMatrix(SerialDenseMatrix &&u) noexcept;

      /**
       * @brief Copy assignment operator
       * @param[in] u const reference to SerialDenseMatrix object to copy from
       * @return reference to this object after copying data from u
       */
      SerialDenseMatrix &
      operator=(const SerialDenseMatrix &u);

      /**
       * @brief Move assignment operator
       * @param[in] u const reference to SerialDenseMatrix object to move from
       * @return reference to this object after moving data from u
       */
      SerialDenseMatrix &
      operator=(SerialDenseMatrix &&u) noexcept;

      /**
       * @brief Constructor for Matrix  with (rows,cols)
       * and initial value arguments
       * @param[in] rows Number of rows of the matrix
       * @param[in] cols Number of cols of the matrix
       * @param[in] blasQueueInput Queue handle. For Matrix objects stored in
       * HOST this is same as int. For matrix object stored in Device this is
       * blas::Queue
       * @param[in] initVal initial value of elements of the Vector
       */
      explicit SerialDenseMatrix(
        size_type                                         rows,
        size_type                                         cols,
        blasLapack::blasQueueType<memorySpace> &          blasQueueInput,
        ValueType                                         initVal = 0,
        typename Matrix<ValueType, memorySpace>::Property property =
          Matrix<ValueType, memorySpace>::Property::GENERAL,
        typename Matrix<ValueType, memorySpace>::Uplo uplo =
          Matrix<ValueType, memorySpace>::Uplo::GENERAL,
        typename Matrix<ValueType, memorySpace>::Layout layout =
          Matrix<ValueType, memorySpace>::Layout::COLMAJ);

      /**
       * @brief Destructor
       */
      ~SerialDenseMatrix();

      /**
       * @brief Returns the Frobenius norm of the matrix
       * @returns Frobenius norm of the matrix.
       */
      double
      frobeniusNorm() const;
    };
  } // namespace linearAlgebra
} // namespace dftefe

#include "SerialDenseMatrix.t.cpp"

#endif // dftefeSerialDenseMatrix_h
