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

#include "HermitianMatrix.h"

namespace dftefe
{
  namespace linearAlgebra
  {
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    HermitianMatrix<ValueType, memorySpace>::HermitianMatrix(Uplo     uplo,
                                                             size_t   n,
                                                             MPI_Comm comm,
                                                             size_t   p,
                                                             size_t   q,
                                                             size_t   nb)
      : AbstractMatrix<ValueType, memorySpace>(n, n, comm, p, q, nb, nb)
    {
      d_matrix = new slate::HermitianMatrix<ValueType>(uplo, n, nb, p, q, comm);
      d_baseMatrix = d_matrix;
      if (memorySpace == dftefe::utils::MemorySpace::DEVICE)
        {
          d_matrix->insertLocalTiles(slate::Target::Devices);
        }
      else
        {
          d_matrix->insertLocalTiles(slate::Target::Host);
        }
      AbstractMatrix<ValueType, memorySpace>::initSlateMatrix(d_matrix);
    }
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    slate::HermitianMatrix<ValueType> &
    HermitianMatrix<ValueType, memorySpace>::getSlateMatrix() const
    {
      return *d_matrix;
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    HermitianMatrix<ValueType, memorySpace>::setValues(const ValueType *data)
    {
      AbstractMatrix<ValueType, memorySpace>::setValueSlateMatrix(d_baseMatrix,
                                                                  data);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    HermitianMatrix<ValueType, memorySpace>::setValues(size_t           i1,
                                                       size_t           i2,
                                                       size_t           j1,
                                                       size_t           j2,
                                                       const ValueType *data)
    {}
  } // namespace linearAlgebra
} // namespace dftefe
