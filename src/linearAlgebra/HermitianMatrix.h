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

#ifndef dftefeHermitianMatrix_h
#define dftefeHermitianMatrix_h

#include <linearAlgebra/AbstractMatrix.h>

namespace dftefe
{
  namespace linearAlgebra
  {
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    class HermitianMatrix : public AbstractMatrix<ValueType, memorySpace>
    {
    public:
      HermitianMatrix(Uplo     uplo,
                      size_t   n,
                      MPI_Comm comm,
                      size_t   p,
                      size_t   q,
                      size_t   nb = global_nb);

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

      /**
       * @brief The value setter interface for the matrix class. This setter
       *        assumes the given pointer to ValueType to be a serially repeated
       *        matrix and copy the data to the corresponding local owned
       *        sub-matrix (i1:i2-1, j1:j2-1) trivially.
       * @warning There is no boundary check in this function. The user should
       *          be responsible that the size of data array should have
       *          (j2-j1)*(i2-i1) size.
       * @param i1 the global index of the first row of the submatrix.
       * @param i2 one passes the global index of the last row of the submatrix.
       * @param j1 the global index of the first column of the submatrix.
       * @param j2 one passes the global index of the last column of the
       *           submatrix.
       * @param data The pointer to the data to be copied from. The user should
       *             be responsible that the size of data array should have
       *             (j2-j1)*(i2-i1) size.
       */
      virtual void
      setValues(size_t           i1,
                size_t           i2,
                size_t           j1,
                size_t           j2,
                const ValueType *data) override;

      slate::HermitianMatrix<ValueType> &
      getSlateMatrix() const;

    protected:
      Uplo d_uplo;
      using AbstractMatrix<ValueType, memorySpace>::d_baseMatrix;
      slate::HermitianMatrix<ValueType> *d_matrix;
    };
  } // namespace linearAlgebra
} // namespace dftefe

#include "HermitianMatrix.t.cpp"
#endif // dftefeHermitianMatrix_h
