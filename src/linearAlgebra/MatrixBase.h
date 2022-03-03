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
 * @author Ian C. Lin.
 */

#ifndef dftefeMatrixBase_h
#define dftefeMatrixBase_h

#include <utils/TypeConfig.h>

namespace dftefe
{
  namespace linearAlgebra
  {
    class MatrixBase
    {
    public:
      typedef ValueType        value_type;
      typedef ValueType *      pointer;
      typedef ValueType &      reference;
      typedef const ValueType &const_reference;
      typedef ValueType *      iterator;
      typedef const ValueType *const_iterator;


      MatrixBase() = default;

      /**
       * @brief Return iterator pointing to the beginning of Matrix
       * data.
       *
       * @returns Iterator pointing to the beginning of Matrix.
       */
      virtual iterator
      begin() = 0;

      /**
       * @brief Return iterator pointing to the beginning of Matrix
       * data.
       *
       * @returns Constant iterator pointing to the beginning of
       * Matrix.
       */
      virtual const_iterator
      begin() const = 0;

      /**
       * @brief Return iterator pointing to the end of Matrix data.
       *
       * @returns Iterator pointing to the end of Matrix.
       */
      virtual iterator
      end() = 0;

      /**
       * @brief Return iterator pointing to the end of MAtrix data.
       *
       * @returns Constant iterator pointing to the end of
       * Matrix.
       */
      virtual const_iterator
      end() const = 0;

      /**
       * @brief Return the raw pointer to the Matrix
       * @return pointer to data
       */
      virtual ValueType *
      data() noexcept = 0;

      /**
       * @brief Return the raw pointer to the Matrix without modifying
       * the values
       * @return pointer to const data
       */
      virtual const ValueType *
      data() const noexcept = 0;

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
      virtual void
      copyTo(MatrixBase<ValueType, memorySpaceDst> &dstMatrix) const = 0;

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
      virtual void
      copyFrom(const MatrixBase<ValueType, memorySpaceSrc> &srcMatrix) = 0;

      /**
       * @brief Returns the Local number of rows of the Matrix
       * For A serial matrix this is same as the global number of rows
       * These functions are present to ensure compatibility with the
       * base class
       * @returns Local number of rows of the Matrix
       */
      virtual size_type
      getLocalRows() const = 0;

      /**
       * @brief Returns the Local number of cols of the Matrix
       * For A serial matrix this is same as the global number of cols
       * These functions are present to ensure compatibility with the
       * base class
       * @returns Local number of cols of the Matrix
       */
      virtual size_type
      getLocalCols() const = 0;


      /**
       * @brief Returns the Local number of (rows,cols) of the Matrix
       * For A serial matrix this is same as the global number of (rows,cols)
       * These functions are present to ensure compatibility with the
       * base class
       * @param[out] rows Local number of rows of the Matrix.
       * @param[out] cols Local number of cols of the Matrix.
       *
       */
      virtual void
      getLocalSize(size_type &rows, size_type &cols) const = 0;

      /**
       * @brief Returns the Global number of (rows,cols) of the Matrix
       * @param[out] rows Global number of rows of the Matrix.
       * @param[out] cols Global number of cols of the Matrix.
       *
       */
      virtual void
      getGlobalSize(size_type &rows, size_type &cols) const = 0;


      /**
       * @brief Returns the Global number of rows of the Matrix
       * @returns Global number of rows of the Matrix
       */
      virtual size_type
      getGlobalRows() const = 0;

      /**
       * @brief Returns the Global number of cols of the Matrix
       * @returns Global number of cols of the Matrix
       */
      virtual size_type
      getGlobalCols() const = 0;

      /**
       * @brief Returns the underlying MemoryStorage object
       * @returns MemoryStorage object of this class
       */
      virtual dftefe::utils::MemoryManager<ValueType,
                                           memorySpace>::MemoryStorage &
      getDataVec() = 0;

      /**
       * @brief Returns the Queue associated with this Matrix object
       * @returns MemoryStorage object of this class
       */
      virtual blasWrapper::blasQueueType<memorySapce> &
      getQueue() = 0;
    };
  } // namespace linearAlgebra
} // namespace dftefe



#endif // dftefeMatrixBase_h
