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
 * @author Ian C. Lin, Vishal subramanian.
 */

#ifndef dftefeMatrix_h
#define dftefeMatrix_h

#include <utils/TypeConfig.h>
#include <vector>
#include "BlasLapackTypedef.h"
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
      enum class Property
      {
        GENERAL,
        HERMITIAN,
        DIAGONAL,
        UPPER_TRIANGULAR,
        LOWER_TRIANGULAR,
      };

      enum class Uplo : char
      {
        GENERAL = char(blas::Uplo::General),
        UPPER   = char(blas::Uplo::Upper),
        LOWER   = char(blas::Uplo::Lower)
      };

      enum class Layout : char
      {
        COLMAJ = char(blas::Layout::ColMajor),
        ROWMAJ = char(blas::Layout::RowMajor),
      };

      using Storage    = dftefe::utils::MemoryStorage<ValueType, memorySpace>;
      using value_type = typename Storage::value_type;
      using pointer    = typename Storage::pointer;
      using reference  = typename Storage::reference;
      using const_reference = typename Storage::const_reference;
      using iterator        = typename Storage::iterator;
      using const_iterator  = typename Storage::const_iterator;


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
       * @brief Return iterator pointing to the end of Matrix data.
       *
       * @returns Constant iterator pointing to the end of
       * Matrix.
       */
      const_iterator
      end() const;

      /**
       * @brief Return the raw pointer to the Matrix
       * @return pointer to data
       */
      ValueType *
      data();

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
      copyTo(Matrix<ValueType, memorySpaceDst> &dstMatrix) const;

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
       * @brief Compound addition for elementwise addition lhs += rhs
       * @param[in] rhs the Matrix to add
       * @return the original Matrix
       * @throws exception if the sizes and type (SerialDenseMatrix or
       * DistributedDenseMatrix) are incompatible
       */
      Matrix &
      operator+=(const Matrix &rhs);


      /**
       * @brief Compound subtraction for elementwise addition lhs -= rhs
       * @param[in] rhs the Matrix to subtract
       * @return the original vector
       * @throws exception if the sizes and type (SerialDenseMatrix or
       * DistributedDenseMatrix) are incompatible
       */
      Matrix &
      operator-=(const Matrix &rhs);


      /**
       * @brief Returns a reference to the underlying storage (i.e., MemoryStorage object)
       * of the Matrix.
       *
       * @return reference to the underlying MemoryStorage.
       */
      Storage &
      getValues();

      /**
       * @brief Returns a const reference to the underlying storage (i.e., MemoryStorage object)
       * of the Matrix.
       *
       * @return const reference to the underlying MemoryStorage.
       */
      const Storage &
      getValues() const;

      /**
       * @brief Set values in the Matrix using a user provided Matrix::Storage object (i.e., MemoryStorage object).
       * The MemoryStorage may lie in a different memoryspace (say memSpace2)
       * than the Matrix's memory space (memSpace). The function internally does
       * a data transfer from memSpace2 to memSpace.
       *
       * @param[in] storage const reference to MemoryStorage object from which
       * to set values into the Matrix.
       * @throws exception if the size of the input storage is smaller than the
       * \e localSize of the Matrix
       */
      template <dftefe::utils::MemorySpace memorySpace2>
      void
      setValues(
        const typename Matrix<ValueType, memorySpace2>::Storage &storage);

      /**
       * @brief Transfer ownership of a user provided Matrix::Storage object (i.e., MemoryStorage object)
       * to the Vector. This is useful when a MemoryStorage has been already
       * been allocated and we need the Matrix to claim its ownership. This
       * avoids reallocation of memory.
       *
       * @param[in] storage unique_ptr to MemoryStorage object whose ownership
       * is to be passed to the Matrix
       *
       * @note Since we are passing the ownership of the input storage to the Matrix, the
       * storage will point to NULL after a call to this function. Accessing the
       * input storage pointer will lead to undefined behavior.
       *
       */
      void
      setStorage(std::unique_ptr<Storage> &storage);

      /**
       * @brief Returns the Queue associated with this Matrix object
       * @returns MemoryStorage object of this class
       */
      std::shared_ptr<blasLapack::BlasQueueType<memorySpace>>
      getQueue();


      /**
       * @brief Returns the Frobenius norm of the matrix
       * @returns Frobenius norm of the matrix.
       */
      virtual double
      frobeniusNorm() const = 0;


    protected:
      size_type d_nGlobalRows = 0, d_nGlobalCols = 0;
      size_type d_nLocalRows = 0, d_nLocalCols = 0;

      std::shared_ptr<blasLapack::BlasQueueType<memorySpace>> d_BlasQueue;

      Property d_property;
      Uplo     d_uplo;
      Layout   d_layout;

      std::unique_ptr<Storage> d_data;
    };
  } // namespace linearAlgebra
} // namespace dftefe


#include "Matrix.t.cpp"
#endif // dftefeMatrix_h
