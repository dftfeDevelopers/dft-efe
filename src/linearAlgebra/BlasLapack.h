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
 * @author Vishal Subramanian
 */

#ifndef dftefeBlasWrappers_h
#define dftefeBlasWrappers_h

#include <linearAlgebra/BlasLapackTypedef.h>
#include <linearAlgebra/LinAlgOpContext.h>
#include <utils/TypeConfig.h>
namespace dftefe
{
  namespace linearAlgebra
  {
    namespace blasLapack
    {
      template <typename ValueType,
                typename dftefe::utils::MemorySpace memorySpace>
      real_type<ValueType>
      asum(size_type                     n,
           ValueType const *             x,
           size_type                     incx,
           LinAlgOpContext<memorySpace> &context);

      template <typename ValueType,
                typename dftefe::utils::MemorySpace memorySpace>
      real_type<ValueType>
      amax(size_type                     n,
           ValueType const *             x,
           size_type                     incx,
           LinAlgOpContext<memorySpace> &context);

      /**
       * @brief Template for computing \f$ l_{\inf} \f$ norms of all the numVec vectors in a multi Vector
       * @param[in] vecSize size of each vector
       * @param[in] numVec number of vectors in the multi Vector
       * @param[in] multiVecData multi vector data in row major format i.e.
       * vector index is the fastest index
       *
       * @return \f$ l_{\inf} \f$  norms of all the vectors
       */
      template <typename ValueType,
                typename dftefe::utils::MemorySpace memorySpace>
      std::vector<double>
      amaxsMultiVector(size_type                     vecSize,
                       size_type                     numVec,
                       ValueType const *             multiVecData,
                       LinAlgOpContext<memorySpace> &context);


      template <typename ValueType1,
                typename ValueType2,
                typename dftefe::utils::MemorySpace memorySpace>
      void
      axpy(size_type                           n,
           scalar_type<ValueType1, ValueType2> alpha,
           ValueType1 const *                  x,
           size_type                           incx,
           ValueType2 *                        y,
           size_type                           incy,
           LinAlgOpContext<memorySpace> &      context);

      // i have neglected incx & incy parameters
      /**
       * @brief Template for computing the multiplicative inverse of all the elements of x, does not check if any element is zero
       * @param[in] n size of each vector
       * @param[in] x input vector
       * @param[out] y output vector
       * @param[in] context Blas context for GPU operations
       *
       * @return \f$ l_{\inf} \f$  norms of all the vectors
       */
      template <typename ValueType1,
                typename dftefe::utils::MemorySpace memorySpace>
      void
      reciprocalX(size_type                     n,
                  ValueType1 const *            x,
                  ValueType1 *                  y,
                  LinAlgOpContext<memorySpace> &context);



      /**
       * @brief Template for performing \f$ z = \alpha x$
       * @param[in] size size of the array
       * @param[in] \f$ alpha \f$ scalar
       * @param[in] x array
       * @param[out] z array
       */
      template <typename ValueType,
                typename dftefe::utils::MemorySpace memorySpace>
      void
      ascale(size_type                     n,
             ValueType                     alpha,
             const ValueType               x,
             ValueType *                   z,
             LinAlgOpContext<memorySpace> &context);


      /**
       * @brief Template for performing \f$ z_i = x_i * y_i$
       * @param[in] size size of the array
       * @param[in] x array
       * @param[in] y array
       * @param[out] z array
       */
      template <typename ValueType,
                typename dftefe::utils::MemorySpace memorySpace>
      void
      hadamardProduct(size_type                     n,
                      const ValueType *             x,
                      const ValueType *             y,
                      ValueType *                   z,
                      LinAlgOpContext<memorySpace> &context);


      /**
       * @brief Template for performing \f$ z = \alpha x + \beta y \f$
       * @param[in] size size of the array
       * @param[in] \f$ alpha \f$ scalar
       * @param[in] x array
       * @param[in] \f$ beta \f$ scalar
       * @param[in] y array
       * @param[out] z array
       */
      template <typename ValueType,
                typename dftefe::utils::MemorySpace memorySpace>
      void
      axpby(size_type                     n,
            ValueType                     alpha,
            ValueType const *             x,
            ValueType                     beta,
            const ValueType *             y,
            ValueType *                   z,
            LinAlgOpContext<memorySpace> &context);



      template <typename ValueType1,
                typename ValueType2,
                typename dftefe::utils::MemorySpace memorySpace>
      scalar_type<ValueType1, ValueType2>
      dot(size_type                     n,
          ValueType1 const *            x,
          size_type                     incx,
          ValueType2 const *            y,
          size_type                     incy,
          LinAlgOpContext<memorySpace> &context);


      template <typename ValueType,
                typename dftefe::utils::MemorySpace memorySpace>
      real_type<ValueType>
      nrm2(size_type                     n,
           ValueType const *             x,
           size_type                     incx,
           LinAlgOpContext<memorySpace> &context);

      /**
       * @brief Template for computing \f$ l_2 \f$ norms of all the numVec vectors in a multi Vector
       * @param[in] vecSize size of each vector
       * @param[in] numVec number of vectors in the multi Vector
       * @param[in] multiVecData multi vector data in row major format i.e.
       * vector index is the fastest index
       *
       * @return \f$ l_2 \f$  norms of all the vectors
       */
      template <typename ValueType,
                typename dftefe::utils::MemorySpace memorySpace>
      std::vector<double>
      nrms2MultiVector(size_type                     vecSize,
                       size_type                     numVec,
                       ValueType const *             multiVecData,
                       LinAlgOpContext<memorySpace> &context);

      template <typename ValueType,
                typename dftefe::utils::MemorySpace memorySpace>
      void
      gemm(Layout                        layout,
           Op                            transA,
           Op                            transB,
           size_type                     m,
           size_type                     n,
           size_type                     k,
           ValueType                     alpha,
           ValueType const *             dA,
           size_type                     ldda,
           ValueType const *             dB,
           size_type                     lddb,
           ValueType                     beta,
           ValueType *                   dC,
           size_type                     lddc,
           LinAlgOpContext<memorySpace> &context);

      /**
       * @brief Variable Strided Batch GEMMM
       *
       * @note: Assumes the same alpha and beta coefficients
       * for all the matrices
       */
      template <typename ValueType,
                typename dftefe::utils::MemorySpace memorySpace>
      void
      gemmStridedVarBatched(Layout                        layout,
                            size_type                     numMats,
                            const Op *                    transA,
                            const Op *                    transB,
                            const size_type *             stridea,
                            const size_type *             strideb,
                            const size_type *             stridec,
                            const size_type *             m,
                            const size_type *             n,
                            const size_type *             k,
                            ValueType                     alpha,
                            const ValueType *             dA,
                            const size_type *             ldda,
                            const ValueType *             dB,
                            const size_type *             lddb,
                            ValueType                     beta,
                            ValueType *                   dC,
                            const size_type *             lddc,
                            LinAlgOpContext<memorySpace> &context);
    } // namespace blasLapack
  }   // namespace linearAlgebra
} // namespace dftefe

#include "BlasLapack.t.cpp"
#endif // dftefeBlasWrappers_h
