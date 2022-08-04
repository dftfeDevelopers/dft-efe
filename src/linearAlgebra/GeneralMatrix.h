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

#ifndef dftefeGeneralMatrix_h
#define dftefeGeneralMatrix_h

#include <linearAlgebra/AbstractMatrix.h>

namespace dftefe
{
  namespace linearAlgebra
  {
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    class GeneralMatrix : public AbstractMatrix<ValueType, memorySpace>
    {
    public:
      GeneralMatrix(size_t   m,
                    size_t   n,
                    MPI_Comm comm,
                    size_t   p,
                    size_t   q,
                    size_t   nb = global_nb,
                    size_t   mb = global_mb);

      /**
       * @brief The value setter interface for the matrix class. This setter
       *        assumes the given pointer to ValueType to be a serially repeated
       *        matrix and copy the data to the corresponding local owned
       *        location trivially.
       * @warning There is no boundary check in this function. The user should
       *          be responsible that the size of data array should have d_m*d_n
       *          size.
       * @param data The pointer to the data to be copied from.
       */
      virtual void
      setValues(const ValueType *data) override;

//      /**
//       * @brief The value setter interface for the matrix class. This setter
//       *        assumes the given pointer to ValueType to be a serially repeated
//       *        matrix and copy the data to the corresponding local owned
//       *        sub-matrix (i1:i2-1, j1:j2-1) trivially.
//       * @warning There is no boundary check in this function. The user should
//       *          be responsible that the size of data array should have
//       *          (j2-j1)*(i2-i1) size.
//       * @param i1 the global index of the first row of the submatrix.
//       * @param i2 one passes the global index of the last row of the submatrix.
//       * @param j1 the global index of the first column of the submatrix.
//       * @param j2 one passes the global index of the last column of the
//       *           submatrix.
//       * @param data The pointer to the data to be copied from. The user should
//       *             be responsible that the size of data array should have
//       *             (j2-j1)*(i2-i1) size.
//       */
//      virtual void
//      setValues(size_t           i1,
//                size_t           i2,
//                size_t           j1,
//                size_t           j2,
//                const ValueType *data);
//
//      /**
//       * @brief The value setter interface for the matrix class. This setter
//       *        allows data to be different on each processor. The locally owned
//       *        data contains two parts: (1) values belongs to the locally owned
//       *        portion of the matrix and (2) values belong to the off-processor
//       *        portion of the matrix. This setter will distribute the
//       *        off-processor values correspondingly.
//       * @warning This routine currently is not optimized and fully
//       *          tested. Using this routing to assign values can
//       *          deteriorate the performance dramatically. This should
//       *          only be used for debugging purpose or not performance
//       *          relevant part of the code.
//       * @param i1 the global index of the first row of the submatrix.
//       * @param i2 one passes the global index of the last row of the submatrix.
//       * @param j1 the global index of the first column of the submatrix.
//       * @param j2 one passes the global index of the last column of the
//       *           submatrix.
//       * @param data The pointer to the data to be copied from. The user should
//       *             be responsible that the size of data array should have
//       *             (j2-j1)*(i2-i1) size.
//       */
//      virtual void
//      setDistributedValues(size_t           i1,
//                           size_t           i2,
//                           size_t           j1,
//                           size_t           j2,
//                           const ValueType *data) = 0;
//
//      /**
//       * @brief The value setter interface for inserting a single value.
//       * @warning This routine is sub-optimal and can seriously deteriorate the
//       *          the performance. It should be only used when necessary. Only
//       *          the processor which owns (i, j) element will be inserting
//       *          value.
//       * @param i the row index of the value.
//       * @param j the column index of the value.
//       * @param d the value.
//       */
//      virtual void
//      setValue(size_t i, size_t j, ValueType d) = 0;

      slate::Matrix<ValueType> &
      getSlateMatrix() const;

    protected:
      using AbstractMatrix<ValueType, memorySpace>::d_baseMatrix;
      slate::Matrix<ValueType> *d_matrix;
    };
  } // namespace linearAlgebra
} // namespace dftefe

#include "GeneralMatrix.t.cpp"
#endif // dftefeGeneralMatrix_h
