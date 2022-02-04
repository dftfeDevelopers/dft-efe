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

#ifndef dftefeBlasWrappers_h
#define dftefeBlasWrappers_h

#include <blas.h>
namespace dftefe
{
  namespace linearAlgebra
  {
    typedef blas::Layout    blasLayoutType;
    typedef blas::Op blasOperationType;
    typedef blass::Queue blasQueueType;

    template<typename ValueType>
      gemm( blasLayoutType layout,
      blasOperationType transA,
    blasOperationType transB,
    size_type m, size_type n, size_type k,
           ValueType alpha,
           ValueType const *dA, size_type ldda,
           ValueType const *dB, size_type lddb,
           ValueType beta,
           ValueType       *dC, size_type lddc);

      template<typename ValueType>
      gemm(blasLayoutType layout,
           blasOperationType transA,
           blasOperationType transB,
           size_type m, size_type n, size_type k,
           ValueType alpha,
           ValueType const *dA, size_type ldda,
           ValueType const *dB, size_type lddb,
           ValueType beta,
           ValueType       *dC, size_type lddc,
           blasQueueType &queue);
  }
}

#endif // dftefeBlassWrappers_h
