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

#include "AbstractMatrix.h"

namespace dftefe
{
  namespace linearAlgebra
  {
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    AbstractMatrix<ValueType, memorySpace>::AbstractMatrix(size_t   m,
                                                           size_t   n,
                                                           MPI_Comm comm,
                                                           size_t   p,
                                                           size_t   q,
                                                           size_t   nb,
                                                           size_t   mb)
      : d_m(m)
      , d_n(n)
      , d_comm(comm)
      , d_p(p)
      , d_q(q)
      , d_nb(nb)
      , d_mb(mb)
    {}

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    AbstractMatrix<ValueType, memorySpace>::setValues(const ValueType *data)
    {
      setValueSlateMatrix(d_baseMatrix, data);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    AbstractMatrix<ValueType, memorySpace>::setValueSlateMatrix(
      slate::BaseMatrix<ValueType> *matrix,
      const ValueType              *data)
    {
      for (int64_t j = 0, j_offset = 0; j < matrix->nt();
           j_offset += matrix->tileNb(j++))
        {
          for (int64_t i = 0, i_offset = 0; i < matrix->mt();
               i_offset += matrix->tileMb(i++))
            {
              if (matrix->tileIsLocal(i, j))
                {
                  slate::Tile<double> T = (*matrix)(i, j);
                  // todo: check for transpose case (d_m and d_n)
                  int64_t mb = T.mb(), nb = T.nb(),
                          offset = i_offset + j_offset * d_m;
                  lapack::lacpy(lapack::MatrixType::General,
                                mb,
                                nb,
                                &data[offset],
                                d_m,
                                T.data(),
                                mb);
                }
            }
        }
    }
  } // namespace linearAlgebra
} // namespace dftefe
