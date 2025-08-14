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
#include "LapackAPIWrapper.h"
#include "BlasLapackTemplates.h"
namespace dftefe
{
  namespace linearAlgebra
  {
    namespace blasLapack
    {
      namespace lapackWrapper
      {
        template <>
        int
        getrf<double>(const int          m,
                      const int          n,
                      double *           A,
                      const unsigned int lda,
                      int *              ipiv)
        {
          int ldaTmp = lda;
          int mTmp   = m;
          int nTmp   = n;
          int info;
          dgetrf_(&mTmp, &nTmp, A, &ldaTmp, ipiv, &info);
          return info;
        }

        template <>
        int
        getrf<std::complex<double>>(const int             m,
                                    const int             n,
                                    std::complex<double> *A,
                                    const unsigned int    lda,
                                    int *                 ipiv)
        {
          int ldaTmp = lda;
          int mTmp   = m;
          int nTmp   = n;
          int info;
          zgetrf_(&mTmp, &nTmp, A, &ldaTmp, ipiv, &info);
          return info;
        }

        template int
        getrf<double>(const int          m,
                      const int          n,
                      double *           A,
                      const unsigned int lda,
                      int *              ipiv);

        template <>
        int
        getri<double>(const int N, double *A, const unsigned int lda, int *ipiv)
        {
          int     LWORK = N * N;
          double *WORK  = new double[LWORK];
          int     INFO;

          int ldaTmp = lda;
          int nTmp   = N;
          dgetri_(&nTmp, A, &ldaTmp, ipiv, WORK, &LWORK, &INFO);

          delete[] WORK;

          return INFO;
        }

        template <>
        int
        getri<std::complex<double>>(const int             N,
                                    std::complex<double> *A,
                                    const unsigned int    lda,
                                    int *                 ipiv)
        {
          int                   LWORK = N * N;
          std::complex<double> *WORK  = new std::complex<double>[LWORK];
          int                   INFO;

          int ldaTmp = lda;
          int nTmp   = N;
          zgetri_(&nTmp, A, &ldaTmp, ipiv, WORK, &LWORK, &INFO);

          delete[] WORK;

          return INFO;
        }

        template <>
        int
        trtri<double>(const char         uplo,
                      const char         diag,
                      const unsigned int n,
                      double *           a,
                      const unsigned int lda)
        {
          int info;
          dtrtri_(&uplo, &diag, &n, a, &lda, &info);
          return info;
        }

        template <>
        int
        trtri<std::complex<double>>(const char            uplo,
                                    const char            diag,
                                    const unsigned int    n,
                                    std::complex<double> *a,
                                    const unsigned int    lda)
        {
          int info;
          ztrtri_(&uplo, &diag, &n, a, &lda, &info);
          return info;
        }

        template <>
        int
        potrf<double>(const char         uplo,
                      const unsigned int n,
                      double *           a,
                      const unsigned int lda)
        {
          int info;
          dpotrf_(&uplo, &n, a, &lda, &info);
          return info;
        }

        template <>
        int
        potrf<std::complex<double>>(const char            uplo,
                                    const unsigned int    n,
                                    std::complex<double> *a,
                                    const unsigned int    lda)
        {
          int info;
          zpotrf_(&uplo, &n, a, &lda, &info);
          return info;
        }

        template <>
        int
        steqr<double>(const char         jobz,
                      const unsigned int n,
                      real_type<double> *D,
                      real_type<double> *E,
                      double *           Z,
                      const unsigned int lda)
        {
          int                info;
          const unsigned int lwork = 2 * n - 2;
          int                nTmp  = n;

          std::vector<double> work(lwork);
          int                 ldaTmp = lda;

          dsteqr_(&jobz, &nTmp, D, E, Z, &ldaTmp, &work[0], &info);

          return info;
        }

        template <>
        int
        steqr<std::complex<double>>(const char            jobz,
                                    const unsigned int    n,
                                    real_type<double> *   D,
                                    real_type<double> *   E,
                                    std::complex<double> *Z,
                                    const unsigned int    lda)
        {
          int                info;
          const unsigned int lwork = 2 * n - 2;
          int                nTmp  = n;

          std::vector<std::complex<double>> work(lwork);
          int                               ldaTmp = lda;

          zsteqr_(&jobz, &nTmp, D, E, Z, &ldaTmp, &work[0], &info);

          return info;
        }

        template <>
        int
        heevd<double>(const char         jobz,
                      const char         uplo,
                      const unsigned int n,
                      double *           A,
                      const unsigned int lda,
                      double *           w)
        {
          int                info;
          const unsigned int lwork = 1 + 6 * n + 2 * n * n, liwork = 3 + 5 * n;
          std::vector<int>   iwork(liwork, 0);

          std::vector<double> work(lwork);

          dsyevd_(&jobz,
                  &uplo,
                  &n,
                  A,
                  &lda,
                  w,
                  &work[0],
                  &lwork,
                  &iwork[0],
                  &liwork,
                  &info);

          return info;
        }

        template <>
        int
        heevd<std::complex<double>>(const char            jobz,
                                    const char            uplo,
                                    const unsigned int    n,
                                    std::complex<double> *A,
                                    const unsigned int    lda,
                                    double *              w)
        {
          int                info;
          const unsigned int lwork = 1 + 6 * n + 2 * n * n, liwork = 3 + 5 * n;
          std::vector<int>   iwork(liwork, 0);

          const unsigned int lrwork = 1 + 5 * n + 2 * n * n;

          std::vector<double>               rwork(lrwork, 0.0);
          std::vector<std::complex<double>> work(lwork);

          zheevd_(&jobz,
                  &uplo,
                  &n,
                  A,
                  &lda,
                  w,
                  &work[0],
                  &lwork,
                  &rwork[0],
                  &lrwork,
                  &iwork[0],
                  &liwork,
                  &info);

          return info;
        }

        template <>
        int
        hegv<double>(const int          itype,
                     const char         jobz,
                     const char         uplo,
                     const unsigned int n,
                     double *           a,
                     const unsigned int lda,
                     double *           b,
                     const unsigned int ldb,
                     double *           w)
        {
          int       info;
          int       ldaTmp = lda;
          int       ldbTmp = ldb;
          int       nTmp   = n;
          const int lwork  = 3 * n - 1;

          std::vector<double> work(lwork);
          dsygv_(&itype,
                 &jobz,
                 &uplo,
                 &nTmp,
                 a,
                 &ldaTmp,
                 b,
                 &ldbTmp,
                 w,
                 &work[0],
                 &lwork,
                 &info);
          return info;
        }

        template <>
        int
        hegv<std::complex<double>>(const int             itype,
                                   const char            jobz,
                                   const char            uplo,
                                   const unsigned int    n,
                                   std::complex<double> *a,
                                   const unsigned int    lda,
                                   std::complex<double> *b,
                                   const unsigned int    ldb,
                                   double *              w)
        {
          int       info;
          int       ldaTmp = lda;
          int       ldbTmp = ldb;
          int       nTmp   = n;
          const int lwork  = 3 * n - 1;

          std::vector<std::complex<double>> work(lwork);
          zhegv_(&itype,
                 &jobz,
                 &uplo,
                 &nTmp,
                 a,
                 &ldaTmp,
                 b,
                 &ldbTmp,
                 w,
                 &work[0],
                 &lwork,
                 &info);
          return info;
        }

        template <>
        int
        gesv<double>(const unsigned int n,
                     const unsigned int nrhs,
                     double *           A,
                     const unsigned int lda,
                     int *              ipiv,
                     double *           B,
                     const unsigned int ldb)
        {
          int info;
          int nTmp    = n;
          int nrhsTmp = nrhs;
          int ldaTmp  = lda;
          int ldbTmp  = ldb;

          dgesv_(&nTmp, &nrhsTmp, A, &ldaTmp, ipiv, B, &ldbTmp, &info);
          return info;
        }

        template <>
        int
        gesv<std::complex<double>>(const unsigned int    n,
                                   const unsigned int    nrhs,
                                   std::complex<double> *A,
                                   const unsigned int    lda,
                                   int *                 ipiv,
                                   std::complex<double> *B,
                                   const unsigned int    ldb)
        {
          int info;
          int nTmp    = n;
          int nrhsTmp = nrhs;
          int ldaTmp  = lda;
          int ldbTmp  = ldb;

          zgesv_(&nTmp, &nrhsTmp, A, &ldaTmp, ipiv, B, &ldbTmp, &info);
          return info;
        }

      } // namespace lapackWrapper

    } // namespace blasLapack
  }   // End of namespace linearAlgebra
} // End of namespace dftefe
