/******************************************************************************
 * Copyright (c) 2021.                                                        *
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
 * @author Vishal Subramanian.
 */

#ifndef dftefeMatrixOperations_h
#define dftefeMatrixOperations_h


#include <blas.hh>
#include "BlasWrappers.h"

namespace dftefe
{
  namespace linearAlgebra
  {
    template <typename dftefe::utils::MemorySpace memorySpace>
    class MatrixOperations
    {
      template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
      void
      matMulc( blasLayoutType layout,
               blasOperationType transA,
               blasOperationType transB,
               size_type m, size_type n, size_type k,
               ValueType alpha,
               ValueType const *dA, size_type ldda,
               ValueType const *dB, size_type lddb,
               ValueType beta,
               ValueType       *dC, size_type lddc,
               blasWrapper::blasQueueType<memorySapce> &blasQueue);
    };
  }

}

#endif // dftefeMatrixOperations_h
