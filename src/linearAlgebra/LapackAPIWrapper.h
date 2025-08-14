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

#ifndef LAPACKWrapper_h
#define LAPACKWrapper_h

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
      namespace lapackWrapper
      {
      template <typename ValueType>
      int
      getrf(const int m, 
            const int n, 
            ValueType *A, 
            const unsigned int lda, 
            int *ipiv);

      template <typename ValueType>
      int
      getri(const int    N,
            ValueType *A,
            const unsigned int    lda,
            int    *ipiv);

      template <typename ValueType>
      int
      trtri(const char         uplo,
            const char         diag,
            const unsigned int n,
            ValueType          *a,
            const unsigned int lda);

      template <typename ValueType>
      int
      potrf(const char           uplo,
            const unsigned int   n,
            ValueType            *a,
            const unsigned int   lda);

      template <typename ValueType>
      int
      steqr(const char                    jobz,
            const unsigned int            n,
            real_type<ValueType> *        D,
            real_type<ValueType> *        E,
            ValueType *                   Z,
            const unsigned int            lda);


      template <typename ValueType>
      int
      heevd(const char           jobz,
            const char           uplo,
            const unsigned int   n,
            ValueType            *A,
            const unsigned int   lda,
            double               *w);

      template <typename ValueType>
      int
      hegv(const int itype, 
          const char  jobz, 
          const char  uplo, 
          const unsigned int n,
          ValueType* A, 
          const unsigned int  lda,
          ValueType* B, 
          const unsigned int  ldb,
          double* w);

      template <typename ValueType>
      int
      gesv(const unsigned int  	n,
          const unsigned int  	nrhs,
          ValueType *  	A,
          const unsigned int  	lda,
          int *  	ipiv,
          ValueType *  	B,
          const unsigned int  	ldb);
      }

    }// end of LAPACKWrapper
  } // end of namespace linearAlgebra

} // end of namespace dftefe


#endif // LAPACKWrapper_h