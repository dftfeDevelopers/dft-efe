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
 * @author Avirup Sircar
 */

#ifndef BLASWrapper_h
#define BLASWrapper_h

#include <cmath>
#include <linearAlgebra/LinAlgOpContext.h>
#include <utils/TypeConfig.h>
#include <linearAlgebra/BlasLapackTypedef.h>
namespace dftefe
{
  namespace linearAlgebra
  {
    namespace blasLapack
    {
      namespace blasWrapper
      {
        template <typename ValueType1,
                  typename ValueType2,
                  typename utils::MemorySpace memorySpace>
        void
        gemm(const char                                transA,
             const char                                transB,
             const size_type                           m,
             const size_type                           n,
             const size_type                           k,
             const scalar_type<ValueType1, ValueType2> alpha,
             ValueType1 const *                        A,
             const size_type                           lda,
             ValueType2 const *                        B,
             const size_type                           ldb,
             const scalar_type<ValueType1, ValueType2> beta,
             scalar_type<ValueType1, ValueType2> *     C,
             const size_type                           ldc,
             LinAlgOpContext<memorySpace> &            context);

        template <typename ValueType1,
                  typename ValueType2,
                  dftefe::utils::MemorySpace memorySpace>
        void
        gemmStridedVarBatched(const size_type                           numMats,
                              const char *                              transA,
                              const char *                              transB,
                              const size_type *                         stridea,
                              const size_type *                         strideb,
                              const size_type *                         stridec,
                              const size_type *                         m,
                              const size_type *                         n,
                              const size_type *                         k,
                              const scalar_type<ValueType1, ValueType2> alpha,
                              const ValueType1 *                        dA,
                              const size_type *                         ldda,
                              const ValueType2 *                        dB,
                              const size_type *                         lddb,
                              const scalar_type<ValueType1, ValueType2> beta,
                              scalar_type<ValueType1, ValueType2> *     dC,
                              const size_type *                         lddc,
                              LinAlgOpContext<memorySpace> &context);

        template <typename ValueType, typename utils::MemorySpace memorySpace>
        real_type<ValueType>
        asum(const size_type               n,
             ValueType const *             x,
             const size_type               incx,
             LinAlgOpContext<memorySpace> &context);

        template <typename ValueType, typename utils::MemorySpace memorySpace>
        size_type
        iamax(const size_type               n,
              ValueType const *             x,
              const size_type               incx,
              LinAlgOpContext<memorySpace> &context);

        template <typename ValueType1,
                  typename ValueType2,
                  typename utils::MemorySpace memorySpace>
        void
        axpy(const size_type                           n,
             const scalar_type<ValueType1, ValueType2> alpha,
             ValueType1 const *                        x,
             const size_type                           incx,
             ValueType2 *                              y,
             const size_type                           incy,
             LinAlgOpContext<memorySpace> &            context);

      } // namespace blasWrapper
    }   // namespace blasLapack
  }     // end of namespace linearAlgebra

} // end of namespace dftefe


#endif // BLASWrapper_h
