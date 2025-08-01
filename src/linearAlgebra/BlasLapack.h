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
 * @author Vishal Subramanian, Avirup Sircar
 */

#ifndef dftefeBlasWrappers_h
#define dftefeBlasWrappers_h

#include <linearAlgebra/LinearAlgebraTypes.h>
#include <linearAlgebra/BlasLapackTypedef.h>
#include <linearAlgebra/LinAlgOpContext.h>
#include <utils/TypeConfig.h>
#include <utils/ConditionalOStream.h>
#include <utils/Profiler.h>

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
       * computes \f $ y[i] = \frac{alpha}{x[i]} $ \f
       * @param[in] n size of each vector
       * @param[in] alpha scalr input for the numerator
       * @param[in] x input vector
       * @param[out] y output vector
       * @param[in] context Blas context for GPU operations
       *
       * @return norms of all the vectors
       */
      template <typename ValueType1,
                typename ValueType2,
                typename dftefe::utils::MemorySpace memorySpace>
      void
      reciprocalX(size_type                            n,
                  const ValueType1                     alpha,
                  ValueType2 const *                   x,
                  scalar_type<ValueType1, ValueType2> *y,
                  LinAlgOpContext<memorySpace> &       context);


      /**
       * @brief Template for performing \f$ z = \alpha x$
       * @param[in] size size of the array
       * @param[in] \f$ alpha \f$ scalar
       * @param[in] x array
       * @param[out] z array
       */
      template <typename ValueType1,
                typename ValueType2,
                typename dftefe::utils::MemorySpace memorySpace>
      void
      ascale(size_type                            n,
             ValueType1                           alpha,
             const ValueType2 *                   x,
             scalar_type<ValueType1, ValueType2> *z,
             LinAlgOpContext<memorySpace> &       context);


      /**
       * @brief Template for performing \f$ z_i = x_i * y_i$
       * @param[in] n size of the array
       * @param[in] x array
       * @param[in] y array
       * @param[out] z array
       */
      template <typename ValueType1,
                typename ValueType2,
                typename dftefe::utils::MemorySpace memorySpace>
      void
      hadamardProduct(size_type                            n,
                      const ValueType1 *                   x,
                      const ValueType2 *                   y,
                      scalar_type<ValueType1, ValueType2> *z,
                      LinAlgOpContext<memorySpace> &       context);

      //    /*
      //     * @brief Template for performing \f$ blockedOutput_ij = blockedInput_ij * singleVectorInput_i$
      //     * @param[in] size size of the blocked Input array
      //     * @param[in] numComponets no of componets
      //     * @param[in] blockedInput blocked array
      //     * @param[in] singleVectorInput array
      //     * @param[out] blockedOutput blocked array
      //     */
      //  template <typename ValueType1,
      //            typename ValueType2,
      //            typename dftefe::utils::MemorySpace memorySpace>
      //    void
      //    blockedHadamardProduct(const size_type                     n,
      //                    const size_type                      blockSize,
      //                    const ValueType1 *                   blockedInput,
      //                    const ValueType2 * singleVectorInput,
      //                    scalar_type<ValueType1, ValueType2> *blockedOutput,
      //                    LinAlgOpContext<memorySpace> &       context);


      /**
       * @brief Template for performing \f$ z_i = op(x_i) * op(y_i)$
       * where op represents either identity or complex conjugate
       * operation on a scalar
       * @param[in] n size of the array
       * @param[in] x array
       * @param[in] y array
       * @param[in] opx blasLapack::ScalarOp defining the operation on each
       * entry of x. The available options are
       * (a) blasLapack::ScalarOp::Identity (identity operation on a scalar),
       * and (b) blasLapack::ScalarOp::Conj (complex conjugate on a scalar)
       * @param[in] opy blasLapack::ScalarOp defining the operation on each
       * entry of y.
       * @param[out] z array
       */
      template <typename ValueType1,
                typename ValueType2,
                typename dftefe::utils::MemorySpace memorySpace>
      void
      hadamardProduct(size_type                            n,
                      const ValueType1 *                   x,
                      const ValueType2 *                   y,
                      const ScalarOp &                     opx,
                      const ScalarOp &                     opy,
                      scalar_type<ValueType1, ValueType2> *z,
                      LinAlgOpContext<memorySpace> &       context);

      /**
       * @brief Template for performing hadamard product of two columns
       * of batches of matrix A and B having num col A = m,
       * num common rows = k, num col B = n going through all the rows
       * of A and B with column B having the faster index than columnA.
       * This operation can be thought as the strided form of face-splitting
       * product between two matrices of variable strides but with common
       * rows in each stride. Also it is assumed that the matrices A and B
       * are column major. So for scalarop it represents either identity or
       * complex conjugate operation on a scalar. Size of C on output
       * will be (m*k) cols and n rows with strides.
       * @param[in] numMats number of batches
       * @param[in] scalarOpA scalar op of A
       * @param[in] scalarOpB scalar op of B
       * @param[in] stridea stride of matrix A
       * @param[in] stridea stride of matrix B
       * @param[in] stridec stride of matrix C
       * @param[in] m column of matrix A
       * @param[in] n column of matrix B
       * @param[in] k row of matrix B and A
       * @param[in] dA matrix A
       * @param[in] dB matrix B
       * @param[out] dC matrix C
       * @param[in] context memorySpace context
       */
      template <typename ValueType1,
                typename ValueType2,
                dftefe::utils::MemorySpace memorySpace>
      void
      scaleStridedVarBatched(const size_type                      numMats,
                             const Layout                         layout,
                             const ScalarOp &                     scalarOpA,
                             const ScalarOp &                     scalarOpB,
                             const size_type *                    stridea,
                             const size_type *                    strideb,
                             const size_type *                    stridec,
                             const size_type *                    m,
                             const size_type *                    n,
                             const size_type *                    k,
                             const ValueType1 *                   dA,
                             const ValueType2 *                   dB,
                             scalar_type<ValueType1, ValueType2> *dC,
                             LinAlgOpContext<memorySpace> &       context);

      /**
       * @brief Template for performing
       * In column major storage format:
       * \f$ {\bf Z}={\bf A} \odot {\bf B} = a_1 \otimes b_1
       * \quad a_2 \otimes b_2 \cdots \a_K \otimes b_K \f$, where \f${\bf A}\f$
       * is  \f$I \times K\f$ matrix, \f${\bf B}\f$ is \f$J \times K\f$, and \f$
       * {\bf Z} \f$ is \f$ (IJ)\times K \f$ matrix. \f$ a_1 \cdots \a_K \f$
       * are the columns of \f${\bf A}\f$
       * In row major storage format:
       * \f$ {\bf Z}^T={\bf A}^T \odot {\bf B}^T = a_1 \otimes b_1
       * \quad a_2 \otimes b_2 \cdots \a_K \otimes b_K \f$, where \f${\bf A}\f$
       * is  \f$K \times I\f$ matrix, \f${\bf B}\f$ is \f$K \times J\f$, and \f$
       * {\bf Z} \f$ is \f$ K\times (IJ) \f$ matrix. \f$ a_1 \cdots \a_K \f$
       * are the rows of \f${\bf A}\f$
       * @param[in] layout Layout::ColMajor or Layout::RowMajor
       * @param[in] size size I
       * @param[in] size size J
       * @param[in] size size K
       * @param[in] X array
       * @param[in] Y array
       * @param[out] Z array
       */
      template <typename ValueType1,
                typename ValueType2,
                typename dftefe::utils::MemorySpace memorySpace>
      void
      khatriRaoProduct(const Layout                         layout,
                       size_type                            sizeI,
                       size_type                            sizeJ,
                       size_type                            sizeK,
                       const ValueType1 *                   A,
                       const ValueType2 *                   B,
                       scalar_type<ValueType1, ValueType2> *Z,
                       LinAlgOpContext<memorySpace> &       context);


      /**
       * @brief Template for performing
       * In column major storage format:
       * \f$ {\bf Z}={\bf A} \odot {\bf B} = a_1 \otimes b_1
       * \quad a_2 \otimes b_2 \cdots \a_K \otimes b_K \f$, where \f${\bf A}\f$
       * is  \f$K \times I\f$ matrix, \f${\bf B}\f$ is \f$K \times J\f$, and \f$
       * {\bf Z} \f$ is \f$ K\times (IJ) \f$ matrix. \f$ a_1 \cdots \a_K \f$
       * are the rows of \f${\bf A}\f$
       * In row major storage format:
       * \f$ {\bf Z}^T={\bf A}^T \odot {\bf B}^T = a_1 \otimes b_1
       * \quad a_2 \otimes b_2 \cdots \a_K \otimes b_K \f$, where \f${\bf A}\f$
       * is  \f$I \times K\f$ matrix, \f${\bf B}\f$ is \f$J \times K\f$, and \f$
       * {\bf Z} \f$ is \f$ (IJ)\times K \f$ matrix. \f$ a_1 \cdots \a_K \f$
       * are the columns of \f${\bf A}\f$
       * @param[in] layout Layout::ColMajor or Layout::RowMajor
       * @param[in] size size I
       * @param[in] size size J
       * @param[in] size size K
       * @param[in] X array
       * @param[in] Y array
       * @param[out] Z array
       */
      template <typename ValueType1,
                typename ValueType2,
                typename dftefe::utils::MemorySpace memorySpace>
      void
      transposedKhatriRaoProduct(const Layout                         layout,
                                 size_type                            sizeI,
                                 size_type                            sizeJ,
                                 size_type                            sizeK,
                                 const ValueType1 *                   A,
                                 const ValueType2 *                   B,
                                 scalar_type<ValueType1, ValueType2> *Z,
                                 LinAlgOpContext<memorySpace> &       context);


      /**
       * @brief Template for performing \f$ z = \alpha x + \beta y \f$
       * @param[in] size size of the array
       * @param[in] \f$ alpha \f$ scalar
       * @param[in] x array
       * @param[in] \f$ beta \f$ scalar
       * @param[in] y array
       * @param[out] z array
       */
      template <typename ValueType1,
                typename ValueType2,
                typename dftefe::utils::MemorySpace memorySpace>
      void
      axpby(size_type                            n,
            scalar_type<ValueType1, ValueType2>  alpha,
            ValueType1 const *                   x,
            scalar_type<ValueType1, ValueType2>  beta,
            const ValueType2 *                   y,
            scalar_type<ValueType1, ValueType2> *z,
            LinAlgOpContext<memorySpace> &       context);

      /**
       * @brief Template for performing \f$ z = \alpha_1*\alpha x + \beta_1*\beta y \f$
       * @param[in] size size of the array
       * @param[in] \f$ alpha \f$ vector
       * @param[in] x array
       * @param[in] \f$ beta \f$ vector
       * @param[in] y array
       * @param[out] z array
       */
      template <typename ValueType1,
                typename ValueType2,
                typename dftefe::utils::MemorySpace memorySpace>
      void
      axpbyBlocked(const size_type                            n,
                   const size_type                            blockSize,
                   const scalar_type<ValueType1, ValueType2>  alpha1,
                   const scalar_type<ValueType1, ValueType2> *alpha,
                   const ValueType1 *                         x,
                   const scalar_type<ValueType1, ValueType2>  beta1,
                   const scalar_type<ValueType1, ValueType2> *beta,
                   const ValueType2 *                         y,
                   scalar_type<ValueType1, ValueType2> *      z,
                   LinAlgOpContext<memorySpace> &             context);


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


      /**
       * @brief Template for computing dot products numVec vectors in a multi Vector
       * @param[in] vecSize size of each vector
       * @param[in] numVec number of vectors in the multi Vector
       * @param[in] multiVecDataX multi vector data in row major format i.e.
       * vector index is the fastest index
       * @param[in] multiVecDataY multi vector data in row major format i.e.
       * vector index is the fastest index
       * @param[in] opX blasLapack::ScalarOp defining the operation on each
       * entry of multiVecDataX. The available options are
       * (a) blasLapack::ScalarOp::Identity (identity operation on a scalar),
       * and (b) blasLapack::ScalarOp::Conj (complex conjugate on a scalar)
       * @param[in] opY blasLapack::ScalarOp defining the operation on each
       * entry of multiVecDataY.
       * @param[out] multiVecDotProduct multi vector dot product of size numVec
       *
       */
      template <typename ValueType1,
                typename ValueType2,
                typename dftefe::utils::MemorySpace memorySpace>
      void
      dotMultiVector(size_type                            vecSize,
                     size_type                            numVec,
                     const ValueType1 *                   multiVecDataX,
                     const ValueType2 *                   multiVecDataY,
                     const ScalarOp &                     opX,
                     const ScalarOp &                     opY,
                     scalar_type<ValueType1, ValueType2> *multiVecDotProduct,
                     LinAlgOpContext<memorySpace> &       context);


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

      template <typename ValueType1,
                typename ValueType2,
                typename dftefe::utils::MemorySpace memorySpace>
      void
      gemm(Layout                               layout,
           Op                                   transA,
           Op                                   transB,
           size_type                            m,
           size_type                            n,
           size_type                            k,
           scalar_type<ValueType1, ValueType2>  alpha,
           ValueType1 const *                   dA,
           size_type                            ldda,
           ValueType2 const *                   dB,
           size_type                            lddb,
           scalar_type<ValueType1, ValueType2>  beta,
           scalar_type<ValueType1, ValueType2> *dC,
           size_type                            lddc,
           LinAlgOpContext<memorySpace> &       context);

      /**
       * @brief Variable Strided Batch GEMMM
       *
       * @note: Assumes the same alpha and beta coefficients
       * for all the matrices
       */
      template <typename ValueType1,
                typename ValueType2,
                typename dftefe::utils::MemorySpace memorySpace>
      void
      gemmStridedVarBatched(Layout                               layout,
                            size_type                            numMats,
                            const Op *                           transA,
                            const Op *                           transB,
                            const size_type *                    stridea,
                            const size_type *                    strideb,
                            const size_type *                    stridec,
                            const size_type *                    m,
                            const size_type *                    n,
                            const size_type *                    k,
                            scalar_type<ValueType1, ValueType2>  alpha,
                            const ValueType1 *                   dA,
                            const size_type *                    ldda,
                            const ValueType2 *                   dB,
                            const size_type *                    lddb,
                            scalar_type<ValueType1, ValueType2>  beta,
                            scalar_type<ValueType1, ValueType2> *dC,
                            const size_type *                    lddc,
                            LinAlgOpContext<memorySpace> &       context);


      /**
       * @brief Dense Matrix inversion
       */
      template <typename ValueType,
                typename dftefe::utils::MemorySpace memorySpace>
      LapackError
      inverse(size_type n, ValueType *A, LinAlgOpContext<memorySpace> &context);

      /**
       * @brief Triangular Matrix inversion
       */
      template <typename ValueType,
                typename dftefe::utils::MemorySpace memorySpace>
      LapackError
      trtri(Uplo                          uplo,
            Diag                          diag,
            size_type                     n,
            ValueType *                   A,
            size_type                     lda,
            LinAlgOpContext<memorySpace> &context);

      /**
       * @brief Cholesky factorization
       */
      template <typename ValueType,
                typename dftefe::utils::MemorySpace memorySpace>
      LapackError
      potrf(Uplo                          uplo,
            size_type                     n,
            ValueType *                   A,
            size_type                     lda,
            LinAlgOpContext<memorySpace> &context);

      /**
       * @brief Real Tridiagonal hermitian matrix standard eigenvalue decomposition
       */
      template <typename ValueType,
                typename dftefe::utils::MemorySpace memorySpace>
      LapackError
      steqr(Job                           jobz,
            size_type                     n,
            real_type<ValueType> *        D,
            real_type<ValueType> *        E,
            ValueType *                   Z,
            size_type                     ldz,
            LinAlgOpContext<memorySpace> &context);

      /**
       * @brief Standard hermitian eigenvalue decomposition
       */
      template <typename ValueType,
                typename dftefe::utils::MemorySpace memorySpace>
      LapackError
      heevd(Job                           jobz,
            Uplo                          uplo,
            size_type                     n,
            ValueType *                   A,
            size_type                     lda,
            real_type<ValueType> *        W,
            LinAlgOpContext<memorySpace> &context);

      /**
       * @brief Standard hermitian eigenvalue decomposition

          Parameters
          [in]	jobz

               lapack::Job::NoVec: Compute eigenvalues only;
               lapack::Job::Vec: Compute eigenvalues and eigenvectors.

          [in]	range

               lapack::Range::All: all eigenvalues will be found.
               lapack::Range::Value: all eigenvalues in the half-open interval
       (vl,vu] will be found. lapack::Range::Index: the il-th through iu-th
       eigenvalues will be found. For range = Value or Index and iu - il < n -
       1, lapack::stebz and lapack::stein are called.

          [in]	uplo

               lapack::Uplo::Upper: Upper triangle of A is stored;
               lapack::Uplo::Lower: Lower triangle of A is stored.

          [in]	n	The order of the matrix A. n >= 0.
          [in,out]	A	The n-by-n matrix A, stored in an lda-by-n array. On
       entry, the Hermitian matrix A.

               If uplo = Upper, the leading n-by-n upper triangular part of A
       contains the upper triangular part of the matrix A. If uplo = Lower, the
       leading n-by-n lower triangular part of A contains the lower triangular
       part of the matrix A. On exit, the lower triangle (if uplo=Lower) or the
       upper triangle (if uplo=Upper) of A, including the diagonal, is
       destroyed.

          [in]	lda	The leading dimension of the array A. lda >= max(1,n).
          [in]	vl	If range=Value, the lower bound of the interval to be
       searched for eigenvalues. vl < vu. Not referenced if range = All or
       Index. [in]	vu	If range=Value, the upper bound of the interval to be
       searched for eigenvalues. vl < vu. Not referenced if range = All or
       Index. [in]	il	If range=Index, the index of the smallest eigenvalue to
       be returned. 1 <= il <= iu <= n, if n > 0; il = 1 and iu = 0 if n = 0.
       Not referenced if range = All or Value.
          [in]	iu	If range=Index, the index of the largest eigenvalue to be
       returned. 1 <= il <= iu <= n, if n > 0; il = 1 and iu = 0 if n = 0. Not
       referenced if range = All or Value. [in]	abstol	The absolute error
       tolerance for the eigenvalues. An approximate eigenvalue is accepted as
       converged when it is determined to lie in an interval [a,b] of width less
       than or equal to abstol + eps * max( |a|,|b| ), where eps is the machine
       precision. If abstol is less than or equal to zero, then eps*|T| will be
       used in its place, where |T| is the 1-norm of the tridiagonal matrix
       obtained by reducing A to tridiagonal form. If high relative accuracy is
       important, set abstol to DLAMCH( 'Safe minimum' ). Doing so will
       guarantee that eigenvalues are computed to high relative accuracy when
       possible in future releases. The current code does not make any
       guarantees about high relative accuracy, but future releases will. [out]
       nfound	The total number of eigenvalues found. 0 <= nfound <= n.

               If range = All, nfound = n;
               if range = Index, nfound = iu-il+1.

          [out]	W	The vector W of length n. The first nfound elements contain
       the selected eigenvalues in ascending order. [out]	Z	The n-by-nfound
       matrix Z, stored in an ldz-by-zcol array.

               If jobz = Vec, then if successful, the first nfound columns of Z
       contain the orthonormal eigenvectors of the matrix A corresponding to the
       selected eigenvalues, with the i-th column of Z holding the eigenvector
       associated with W(i). If jobz = NoVec, then Z is not referenced. Note:
       the user must ensure that zcol >= max(1,nfound) columns are supplied in
       the array Z; if range = Value, the exact value of nfound is not known in
       advance and an upper bound must be used.

          [in]	ldz	The leading dimension of the array Z. ldz >= 1, and if jobz
       = Vec, ldz >= max(1,n). [out]isuppz	The vector isuppz of length
       2*max(1,nfound). The support of the eigenvectors in Z, i.e., the indices
       indicating the nonzero elements in Z. The i-th eigenvector is nonzero
       only in elements isuppz( 2*i-1 ) through isuppz( 2*i ). This is an output
       of lapack::stemr (tridiagonal matrix). The support of the eigenvectors of
       A is typically 1:n because of the unitary transformations applied by
       lapack::unmtr. Implemented only for range = All or Index and iu - il = n
       - 1

          Returns
          = 0: successful exit
          > 0: Internal error
       */
      template <typename ValueType,
                typename dftefe::utils::MemorySpace memorySpace>
      LapackError
      heevr(Job                           jobz,
            Range                         range,
            Uplo                          uplo,
            size_type                     n,
            ValueType *                   A,
            size_type                     lda,
            real_type<ValueType>          vl,
            real_type<ValueType>          vu,
            size_type                     il,
            size_type                     iu,
            real_type<ValueType>          abstol,
            real_type<ValueType> *        W,
            ValueType *                   Z,
            size_type                     ldz,
            LinAlgOpContext<memorySpace> &context);

      /**
       * @brief Generalized hermitian eigenvalue decomposition
       */
      template <typename ValueType,
                typename dftefe::utils::MemorySpace memorySpace>
      LapackError
      hegv(size_type                     itype,
           Job                           jobz,
           Uplo                          uplo,
           size_type                     n,
           ValueType *                   A,
           size_type                     lda,
           ValueType *                   B,
           size_type                     ldb,
           real_type<ValueType> *        W,
           LinAlgOpContext<memorySpace> &context);

      /**
       * @brief Computes the solution to a system of linear equations \(A X = B\),
       * where A is an n-by-n matrix and X and B are n-by-nrhs matrices.
       */
      template <typename ValueType,
                typename dftefe::utils::MemorySpace memorySpace>
      LapackError
      gesv(size_type                     n,
           size_type                     nrhs,
           ValueType *                   A,
           size_type                     lda,
           LapackInt *                   ipiv,
           ValueType *                   B,
           size_type                     ldb,
           LinAlgOpContext<memorySpace> &context);

    } // namespace blasLapack
  }   // namespace linearAlgebra
} // namespace dftefe

#include "BlasLapack.t.cpp"
#endif // dftefeBlasWrappers_h
