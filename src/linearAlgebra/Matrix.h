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

#ifndef dftefeMatrix_h
#define dftefeMatrix_h

#include <utils/TypeConfig.h>
#include <vector>
#include "BlasWrappersTypedef.h"
#include <utils/MemoryStorage.h>
#include "QueueManager.h"
#include <memory>

namespace dftefe
{
  namespace linearAlgebra
  {

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    class Matrix
    {
    public:
      typedef ValueType        value_type;
      typedef ValueType *      pointer;
      typedef ValueType &      reference;
      typedef const ValueType &const_reference;
      typedef ValueType *      iterator;
      typedef const ValueType *const_iterator;


      Matrix() = default;

      virtual ~Matrix() = default;

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
      end() const = 0;

      /**
       * @brief Return the raw pointer to the Matrix
       * @return pointer to data
       */
      ValueType *
      data() ;

      /**
       * @brief Return the raw pointer to the Matrix without modifying
       * the values
       * @return pointer to const data
       */
      const ValueType *
      data() const;

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
      copyTo(Matrix<ValueType, memorySpaceDst> &dstMatrix) const ;

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
      copyFrom(const Matrix<ValueType, memorySpaceSrc> &srcMatrix);

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
       * @brief Returns the underlying MemoryStorage object
       * @returns MemoryStorage object of this class
       */
      dftefe::utils::MemoryStorage<ValueType,memorySpace> &
      getDataVec() ;

      /**
       * @brief Returns the Queue associated with this Matrix object
       * @returns MemoryStorage object of this class
       */
      blasWrapper::blasQueueType<memorySpace> &
      getQueue();


      /**
       * @brief Returns the Frobenius norm of the matrix
       * @returns Frobenius norm of the matrix.
       */
      virtual double frobeniusNorm () const = 0;


    protected:
      size_type d_nGlobalRows = 0, d_nGlobalCols = 0;
      size_type d_nLocalRows = 0, d_nLocalCols = 0;

      blasWrapper::blasQueueType<memorySpace> d_blasQueue;

      std::shared_ptr<dftefe::utils::MemoryStorage<ValueType, memorySpace>> d_data;
    };
  } // namespace linearAlgebra
} // namespace dftefe


#include "Matrix.t.cpp"
#endif // dftefeMatrix_h
