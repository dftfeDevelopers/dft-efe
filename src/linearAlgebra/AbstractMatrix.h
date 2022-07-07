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

#ifndef dftefeAbstractMatrix_h
#define dftefeAbstractMatrix_h

#include <slate/slate.hh>
#include "utils/MemorySpaceType.h"

static int global_nb = 256, global_mb = 256;

namespace dftefe
{

  namespace linearAlgebra
  {

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    class AbstractMatrix
    {

    protected:
      AbstractMatrix(size_t m, size_t n, MPI_Comm comm, size_t p, size_t q, size_t nb = global_nb, size_t mb = global_mb);
      size_t d_m, d_n;
      size_t d_mb, d_nb, d_p, d_q;
      MPI_Comm d_comm;
      slate::BaseMatrix<ValueType> *d_baseMatrix = nullptr;
    };

  } // namespace linearAlgebra

} // namespace dftefe


#include "AbstractMatrix.t.cpp"
#endif // dftefeAbstractMatrix_h
