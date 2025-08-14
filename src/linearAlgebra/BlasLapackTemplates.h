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

#ifndef BlasLapackTemplates_h
#define BlasLapackTemplates_h

namespace dftefe
{
  namespace linearAlgebra
  {
  //
  // extern declarations for blas-lapack routines
  //
  extern "C"
  {
    void
    dgemv_(const char         *TRANS,
           const unsigned int *M,
           const unsigned int *N,
           const double       *alpha,
           const double       *A,
           const unsigned int *LDA,
           const double       *X,
           const unsigned int *INCX,
           const double       *beta,
           double             *C,
           const unsigned int *INCY);

    void
    sgemv_(const char         *TRANS,
           const unsigned int *M,
           const unsigned int *N,
           const float        *alpha,
           const float        *A,
           const unsigned int *LDA,
           const float        *X,
           const unsigned int *INCX,
           const float        *beta,
           float              *C,
           const unsigned int *INCY);

    void
    zgemv_(const char                 *TRANS,
           const unsigned int         *M,
           const unsigned int         *N,
           const std::complex<double> *alpha,
           const std::complex<double> *A,
           const unsigned int         *LDA,
           const std::complex<double> *X,
           const unsigned int         *INCX,
           const std::complex<double> *beta,
           std::complex<double>       *C,
           const unsigned int         *INCY);

    void
    cgemv_(const char                *TRANS,
           const unsigned int        *M,
           const unsigned int        *N,
           const std::complex<float> *alpha,
           const std::complex<float> *A,
           const unsigned int        *LDA,
           const std::complex<float> *X,
           const unsigned int        *INCX,
           const std::complex<float> *beta,
           std::complex<float>       *C,
           const unsigned int        *INCY);
    void
    dsymv_(const char         *UPLO,
           const unsigned int *N,
           const double       *alpha,
           const double       *A,
           const unsigned int *LDA,
           const double       *X,
           const unsigned int *INCX,
           const double       *beta,
           double             *C,
           const unsigned int *INCY);
    void
    dgesv_(int    *n,
           int    *nrhs,
           double *a,
           int    *lda,
           int    *ipiv,
           double *b,
           int    *ldb,
           int    *info);

    void
    zgesv_(int    *n,
           int    *nrhs,
           std::complex<double> *a,
           int    *lda,
           int    *ipiv,
           std::complex<double> *b,
           int    *ldb,
           int    *info);

    void
    dsysv_(const char *UPLO,
           const int  *n,
           const int  *nrhs,
           double     *a,
           const int  *lda,
           int        *ipiv,
           double     *b,
           const int  *ldb,
           double     *work,
           const int  *lwork,
           int        *info);

    void
    dsteqr_(const char         *jobz,
		const int  *n,
		double *D,
		double *E,
		double *Z,
		const int *lda,
		double *work,
		int        *info);

    void
    zsteqr_(const char         *jobz,
		const int  *n,
		double *D,
		double *E,
		std::complex<double> *Z,
		const int *lda,
		std::complex<double> *work,
		int        *info);

    void
    dscal_(const unsigned int *n,
           const double       *alpha,
           double             *x,
           const unsigned int *inc);
    void
    sscal_(const unsigned int *n,
           const float        *alpha,
           float              *x,
           const unsigned int *inc);
    void
    zscal_(const unsigned int         *n,
           const std::complex<double> *alpha,
           std::complex<double>       *x,
           const unsigned int         *inc);
    void
    zdscal_(const unsigned int   *n,
            const double         *alpha,
            std::complex<double> *x,
            const unsigned int   *inc);
    void
    daxpy_(const unsigned int *n,
           const double       *alpha,
           const double       *x,
           const unsigned int *incx,
           double             *y,
           const unsigned int *incy);
    void
    saxpy_(const unsigned int *n,
           const float        *alpha,
           const float        *x,
           const unsigned int *incx,
           float              *y,
           const unsigned int *incy);
    void
    dgemm_(const char         *transA,
           const char         *transB,
           const unsigned int *m,
           const unsigned int *n,
           const unsigned int *k,
           const double       *alpha,
           const double       *A,
           const unsigned int *lda,
           const double       *B,
           const unsigned int *ldb,
           const double       *beta,
           double             *C,
           const unsigned int *ldc);
    void
    sgemm_(const char         *transA,
           const char         *transB,
           const unsigned int *m,
           const unsigned int *n,
           const unsigned int *k,
           const float        *alpha,
           const float        *A,
           const unsigned int *lda,
           const float        *B,
           const unsigned int *ldb,
           const float        *beta,
           float              *C,
           const unsigned int *ldc);
    void
    dsyevd_(const char         *jobz,
            const char         *uplo,
            const unsigned int *n,
            double             *A,
            const unsigned int *lda,
            double             *w,
            double             *work,
            const unsigned int *lwork,
            int                *iwork,
            const unsigned int *liwork,
            int                *info);
    void
    dsygvx_(const int    *itype,
            const char   *jobz,
            const char   *range,
            const char   *uplo,
            const int    *n,
            double       *a,
            const int    *lda,
            double       *b,
            const int    *ldb,
            const double *vl,
            const double *vu,
            const int    *il,
            const int    *iu,
            const double *abstol,
            int          *m,
            double       *w,
            double       *z,
            const int    *ldz,
            double       *work,
            const int    *lwork,
            int          *iwork,
            int          *ifail,
            int          *info);

    void
    dsygv_(const int    *itype,
            const char   *jobz,
            const char   *uplo,
            const int    *n,
            double       *a,
            const int    *lda,
            double       *b,
            const int    *ldb,
            double       *w,
            double       *work,
            const int    *lwork,
            int          *info);

    void
    zhegv_(const int    *itype,
            const char   *jobz,
            const char   *uplo,
            const int    *n,
            std::complex<double>       *a,
            const int    *lda,
            std::complex<double>       *b,
            const int    *ldb,
            double       *w,
            std::complex<double>       *work,
            const int    *lwork,
            int          *info);
            
    void
    dsyevx_(const char   *jobz,
            const char   *range,
            const char   *uplo,
            const int    *n,
            double       *a,
            const int    *lda,
            const double *vl,
            const double *vu,
            const int    *il,
            const int    *iu,
            const double *abstol,
            int          *m,
            double       *w,
            double       *z,
            const int    *ldz,
            double       *work,
            const int    *lwork,
            int          *iwork,
            int          *ifail,
            int          *info);
    double
    dlamch_(const char *cmach);
    void
    dsyevr_(const char         *jobz,
            const char         *range,
            const char         *uplo,
            const unsigned int *n,
            double             *A,
            const unsigned int *lda,
            const double       *vl,
            const double       *vu,
            const unsigned int *il,
            const unsigned int *iu,
            const double       *abstol,
            const unsigned int *m,
            double             *w,
            double             *Z,
            const unsigned int *ldz,
            unsigned int       *isuppz,
            double             *work,
            const int          *lwork,
            int                *iwork,
            const int          *liwork,
            int                *info);
    void
    dsyrk_(const char         *uplo,
           const char         *trans,
           const unsigned int *n,
           const unsigned int *k,
           const double       *alpha,
           const double       *A,
           const unsigned int *lda,
           const double       *beta,
           double             *C,
           const unsigned int *ldc);
    void
    dsyr_(const char         *uplo,
          const unsigned int *n,
          const double       *alpha,
          const double       *X,
          const unsigned int *incx,
          double             *A,
          const unsigned int *lda);
    void
    dsyr2_(const char         *uplo,
           const unsigned int *n,
           const double       *alpha,
           const double       *x,
           const unsigned int *incx,
           const double       *y,
           const unsigned int *incy,
           double             *a,
           const unsigned int *lda);
    void
    dcopy_(const unsigned int *n,
           const double       *x,
           const unsigned int *incx,
           double             *y,
           const unsigned int *incy);
    void
    scopy_(const unsigned int *n,
           const float        *x,
           const unsigned int *incx,
           float              *y,
           const unsigned int *incy);
    void
    zgemm_(const char                 *transA,
           const char                 *transB,
           const unsigned int         *m,
           const unsigned int         *n,
           const unsigned int         *k,
           const std::complex<double> *alpha,
           const std::complex<double> *A,
           const unsigned int         *lda,
           const std::complex<double> *B,
           const unsigned int         *ldb,
           const std::complex<double> *beta,
           std::complex<double>       *C,
           const unsigned int         *ldc);
    void
    cgemm_(const char                *transA,
           const char                *transB,
           const unsigned int        *m,
           const unsigned int        *n,
           const unsigned int        *k,
           const std::complex<float> *alpha,
           const std::complex<float> *A,
           const unsigned int        *lda,
           const std::complex<float> *B,
           const unsigned int        *ldb,
           const std::complex<float> *beta,
           std::complex<float>       *C,
           const unsigned int        *ldc);
    void
    zheevd_(const char           *jobz,
            const char           *uplo,
            const unsigned int   *n,
            std::complex<double> *A,
            const unsigned int   *lda,
            double               *w,
            std::complex<double> *work,
            const unsigned int   *lwork,
            double               *rwork,
            const unsigned int   *lrwork,
            int                  *iwork,
            const unsigned int   *liwork,
            int                  *info);
    void
    zheevr_(const char           *jobz,
            const char           *range,
            const char           *uplo,
            const unsigned int   *n,
            std::complex<double> *A,
            const unsigned int   *lda,
            const double         *vl,
            const double         *vu,
            const unsigned int   *il,
            const unsigned int   *iu,
            const double         *abstol,
            const unsigned int   *m,
            double               *w,
            std::complex<double> *Z,
            const unsigned int   *ldz,
            unsigned int         *isuppz,
            std::complex<double> *work,
            const int            *lwork,
            double               *rwork,
            const int            *lrwork,
            int                  *iwork,
            const int            *liwork,
            int                  *info);
    void
    zherk_(const char                 *uplo,
           const char                 *trans,
           const unsigned int         *n,
           const unsigned int         *k,
           const double               *alpha,
           const std::complex<double> *A,
           const unsigned int         *lda,
           const double               *beta,
           std::complex<double>       *C,
           const unsigned int         *ldc);
    void
    zcopy_(const unsigned int         *n,
           const std::complex<double> *x,
           const unsigned int         *incx,
           std::complex<double>       *y,
           const unsigned int         *incy);

    void
    ccopy_(const unsigned int        *n,
           const std::complex<float> *x,
           const unsigned int        *incx,
           std::complex<float>       *y,
           const unsigned int        *incy);

    std::complex<double>
    zdotc_(const unsigned int         *N,
           const std::complex<double> *X,
           const unsigned int         *INCX,
           const std::complex<double> *Y,
           const unsigned int         *INCY);
    double
    ddot_(const unsigned int *N,
          const double       *X,
          const unsigned int *INCX,
          const double       *Y,
          const unsigned int *INCY);

    double
    dnrm2_(const unsigned int *n, const double *x, const unsigned int *incx);

    double
    dznrm2_(const unsigned int         *n,
            const std::complex<double> *x,
            const unsigned int         *incx);

    double
    sasum_(const unsigned int *n, const float *x, const unsigned int *incx);

    double
    dasum_(const unsigned int *n, const double *x, const unsigned int *incx);

    double
    scasum_(const unsigned int *n, const std::complex<float> *x, const unsigned int *incx);

    double
    dzasum_(const unsigned int *n, const std::complex<double> *x, const unsigned int *incx);

    unsigned int 
    isamax_(const unsigned int *n, const float *x, const unsigned int *incx);

    unsigned int 
    idamax_(const unsigned int *n, const double *x, const unsigned int *incx);

    unsigned int 
    icamax_(const unsigned int *n, const std::complex<float> *x, const unsigned int *incx);

    unsigned int 
    izamax_(const unsigned int *n, const std::complex<double> *x, const unsigned int *incx);

    void
    zaxpy_(const unsigned int         *n,
           const std::complex<double> *alpha,
           const std::complex<double> *x,
           const unsigned int         *incx,
           std::complex<double>       *y,
           const unsigned int         *incy);
    void
    caxpy_(const unsigned int        *n,
           const std::complex<float> *alpha,
           const std::complex<float> *x,
           const unsigned int        *incx,
           std::complex<float>       *y,
           const unsigned int        *incy);
    void
    dpotrf_(const char         *uplo,
            const unsigned int *n,
            double             *a,
            const unsigned int *lda,
            int                *info);
    void
    dpotri_(const char         *uplo,
            const unsigned int *n,
            double             *A,
            const unsigned int *lda,
            int                *info);

    void
    zpotrf_(const char           *uplo,
            const unsigned int   *n,
            std::complex<double> *a,
            const unsigned int   *lda,
            int                  *info);
    void
    dtrtri_(const char         *uplo,
            const char         *diag,
            const unsigned int *n,
            double             *a,
            const unsigned int *lda,
            int                *info);
    void
    ztrtri_(const char           *uplo,
            const char           *diag,
            const unsigned int   *n,
            std::complex<double> *a,
            const unsigned int   *lda,
            int                  *info);

    // LU decomoposition of a general matrix
    void
    dgetrf_(int *M, int *N, double *A, int *lda, int *IPIV, int *INFO);

    // generate inverse of a matrix given its LU decomposition
    void
    dgetri_(int    *N,
            double *A,
            int    *lda,
            int    *IPIV,
            double *WORK,
            int    *lwork,
            int    *INFO);
    // LU decomoposition of a general matrix
    void
    zgetrf_(int                  *M,
            int                  *N,
            std::complex<double> *A,
            int                  *lda,
            int                  *IPIV,
            int                  *INFO);

    // generate inverse of a matrix given its LU decomposition
    void
    zgetri_(int                  *N,
            std::complex<double> *A,
            int                  *lda,
            int                  *IPIV,
            std::complex<double> *WORK,
            int                  *lwork,
            int                  *INFO);
  }
  }
} // namespace dftefe
#endif