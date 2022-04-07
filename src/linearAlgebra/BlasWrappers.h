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

#include <blas.hh>
#include "BlasWrappersTypedef.h"
#include <utils/TypeConfig.h>
namespace dftefe
{
  namespace linearAlgebra
  {
    template <typename ValueType>
    blasWrapper::real_type<ValueType>
    asum(size_type n, ValueType const *x, size_type incx = 1);

    template <typename ValueType>
    blasWrapper::real_type<ValueType>
    amax(size_type n, ValueType const *x, size_type incx = 1);


    template <typename ValueType1, typename ValueType2>
    void
    axpy(size_type                                        n,
         blasWrapper::scalar_type<ValueType1, ValueType2> alpha,
         ValueType1 const *                               x,
         size_type                                        incx,
         ValueType2 *                                     y,
         size_type                                        incy);


    template <typename ValueType1, typename ValueType2>
    blasWrapper::scalar_type<ValueType1, ValueType2>
    dot(size_type         n,
        ValueType1 const *x,
        size_type         incx,
        ValueType2 const *y,
        size_type         incy);


    template <typename ValueType>
    blasWrapper::real_type<ValueType>
    nrm2(size_type n, ValueType const *x, size_type incx = 1);

    template <typename ValueType,
              typename dftefe::utils::MemorySpace memorySpace>
    void
    gemm(blasWrapper::Layout                      layout,
         blasWrapper::Op                          transA,
         blasWrapper::Op                          transB,
         size_type                                m,
         size_type                                n,
         size_type                                k,
         ValueType                                alpha,
         ValueType const *                        dA,
         size_type                                ldda,
         ValueType const *                        dB,
         size_type                                lddb,
         ValueType                                beta,
         ValueType *                              dC,
         size_type                                lddc,
         blasWrapper::blasQueueType<memorySpace> &blasQueue);
  } // namespace linearAlgebra
} // namespace dftefe

#include "BlasWrappers.t.cpp"
#endif // dftefeBlasWrappers_h
