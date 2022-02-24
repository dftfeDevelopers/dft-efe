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

#include "MatrixBase.h"
#include <utils/TypeConfig.h>
#include <vector>
#include "blasWrappersTypedef.h"
#include <utils/MemoryStorage.h>
#include "QueueManager.h"

namespace dftefe
{
  namespace linearAlgebra
  {
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    class SerialDenseMatrix : public MatrixBase
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


      typedef ValueType        value_type;
      typedef ValueType *      pointer;
      typedef ValueType &      reference;
      typedef const ValueType &const_reference;
      typedef ValueType *      iterator;
      typedef const ValueType *const_iterator;

    public:
      SerialDenseMatrix() = default;

      /**
       * @brief Copy constructor for a MemoryStorage
       * @param[in] u SerialDensityMatrix object to copy from
       */
      SerialDenseMatrix(const SerialDenseMatrix &u);

      // TODO Check if this is required and implement if neccessary
      //      /**
      //       * @brief Move constructor for a matrix
      //       * @param[in] u Vector object to move from
      //       */
      //      SerialDenseMatrix(SerialDenseMatrix &&u) noexcept;

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
        size_type                                rows,
        size_type                                cols,
        blasWrapper::blasQueueType<memorySapce> &blasQueueInput,
        ValueType                                initVal = 0);

      /**
       * @brief Destructor
       */
      ~SerialDenseMatrix();

      /**
       * @brief Return iterator pointing to the beginning of Matrix
       * data.
       *
       * @returns Iterator pointing to the beginning of Matrix.
       */
      iterator
      begin();

      /**
       * @brief Return iterator pointing to the beginning of Matrix
       * data.
       *
       * @returns Constant iterator pointing to the beginning of
       * Matrix.
       */
      const_iterator
      begin() const;

      /**
       * @brief Return iterator pointing to the end of Matrix data.
       *
       * @returns Iterator pointing to the end of Matrix.
       */
      iterator
      end();

      /**
       * @brief Return iterator pointing to the end of MAtrix data.
       *
       * @returns Constant iterator pointing to the end of
       * Matrix.
       */
      const_iterator
      end() const;

      //      /**
      //       * @brief Copy assignment operator
      //       * @param[in] rhs the rhs Matrix from which to copy
      //       * @returns reference to the lhs Matrix
      //       */
      //      SerialDenseMatrix &
      //      operator=(const SerialDenseMatrix &rhs);

      /**
       * @brief Return the raw pointer to the Matrix
       * @return pointer to data
       */
      ValueType *
      data() noexcept;

      /**
       * @brief Return the raw pointer to the Matrix without modifying
       * the values
       * @return pointer to const data
       */
      const ValueType *
      data() const noexcept;

      /**
       * @brief Copies the data to a Matrix object in a different memory space.
       * This provides a seamless interface to copy back and forth between
       * memory spaces , including between the same memory spaces.
       *
       * @note The destination Matrix must be pre-allocated appropriately
       *
       * @tparam memorySpaceDst memory space of the destination Matrix
       * @param[out] dstMatrix reference to the destination
       *  Matrix with the data copied into it. It must pre-allocated
       *  appropriately.
       */
      template <dftefe::utils::MemorySpace memorySpaceDst>
      void
      copyTo(MatrixBase<ValueType, memorySpaceDst> &dstMatrix) const;

      /**
       * @brief Copies data from a MemoryStorage object in a different memory space.
       * This provides a seamless interface to copy back and forth between
       * memory spaces, including between the same memory spaces.
       *
       * @note The MemoryStorage must be pre-allocated appropriately
       *
       * @tparam memorySpaceSrc memory space of the source MemoryStorage
       *  from which to copy
       * @param[in] srcMatrix reference to the source
       *  Matrix
       */
      template <dftefe::utils::MemorySpace memorySpaceSrc>
      void
      copyFrom(const MatrixBase<ValueType, memorySpaceSrc> &srcMatrix);

      /**
       * @brief Returns the Local number of rows of the Matrix
       * For A serial matrix this is same as the global number of rows
       * These functions are present to ensure compatibility with the
       * base class
       * @returns Local number of rows of the Matrix
       */
      size_type
      getLocalRows() const;

      /**
       * @brief Returns the Local number of cols of the Matrix
       * For A serial matrix this is same as the global number of cols
       * These functions are present to ensure compatibility with the
       * base class
       * @returns Local number of cols of the Matrix
       */
      size_type
      getLocalCols() const;


      /**
       * @brief Returns the Local number of (rows,cols) of the Matrix
       * For A serial matrix this is same as the global number of (rows,cols)
       * These functions are present to ensure compatibility with the
       * base class
       * @param[out] rows Local number of rows of the Matrix.
       * @param[out] cols Local number of cols of the Matrix.
       *
       */
      void
      getLocalSize(size_type &rows, size_type &cols) const;

      /**
       * @brief Returns the Global number of (rows,cols) of the Matrix
       * @param[out] rows Global number of rows of the Matrix.
       * @param[out] cols Global number of cols of the Matrix.
       *
       */
      void
      getGlobalSize(size_type &rows, size_type &cols) const;


      /**
       * @brief Returns the Global number of rows of the Matrix
       * @returns Global number of rows of the Matrix
       */
      size_type
      getGlobalRows() const;

      /**
       * @brief Returns the Global number of cols of the Matrix
       * @returns Global number of cols of the Matrix
       */
      size_type
      getGlobalCols() const;


      /**
       * @brief Returns the underlying MemoryStorage object. For Matrix object
       * stored on Host, it is same as int. For Matrix object stored on Device
       * this is same as blas::Queue
       * @returns blasWrapper::blasQueueType<memorySapce> of this class
       */
      blasWrapper::blasQueueType<memorySapce> &
      getQueue();



    private:
      size_type d_nGlobalRows = 0, d_nGlobalCols = 0;
      size_type d_nLocalRows = 0, d_nLocalCols = 0;

      blasWrapper::blasQueueType<memorySapce> d_blasQueue;

      dftefe::utils::MemoryManager<ValueType, memorySpace>::MemoryStorage
        d_data;
    };
  } // namespace linearAlgebra
} // namespace dftefe



#endif // dftefeSerialDenseMatrix_h
