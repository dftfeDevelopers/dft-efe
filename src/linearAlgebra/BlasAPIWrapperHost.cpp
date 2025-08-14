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
#include "BlasAPIWrapper.h"
#include "BlasLapackTemplates.h"
namespace dftefe
{
  namespace linearAlgebra
  {
    namespace blasLapack
    {
      template <typename ValueType1,
                typename ValueType2>
      void
      xgemm(const char                                   transA,
           const char                                   transB,
           const size_type                            m,
           const size_type                            n,
           const size_type                            k,
           const scalar_type<ValueType1, ValueType2>  alpha,
           ValueType1 const *                         A,
           const size_type                            lda,
           ValueType2 const *                         B,
           const size_type                            ldb,
           const scalar_type<ValueType1, ValueType2>  beta,
           scalar_type<ValueType1, ValueType2> *C,
           const size_type                            ldc,
           LinAlgOpContext<utils::MemorySpace::HOST>               &context)
      {
        utils::throwException(false, "The input valuetypes are not supported by xgemm");
      }

    template<>
    void
    xgemm<float,float,utils::MemorySpace::HOST>
          (const char                                   transA,
           const char                                   transB,
           const size_type                            m,
           const size_type                            n,
           const size_type                            k,
           const float  alpha,
           float const *                         A,
           const size_type                            lda,
           float const *                         B,
           const size_type                            ldb,
           const float  beta,
           float *C,
           const size_type                            ldc,
           LinAlgOpContext<utils::MemorySpace::HOST>   &context)
    {
      unsigned int mTmp   = m;
      unsigned int nTmp   = n;
      unsigned int kTmp   = k;
      unsigned int ldaTmp = lda;
      unsigned int ldbTmp = ldb;
      unsigned int ldcTmp = ldc;
      sgemm_(&transA,
             &transB,
             &mTmp,
             &nTmp,
             &kTmp,
             &alpha,
             A,
             &ldaTmp,
             B,
             &ldbTmp,
             &beta,
             C,
             &ldcTmp);
    }

    template<>
    void
    xgemm<double,double,utils::MemorySpace::HOST>(
          const char   transA,
          const char   transB,
          const size_type m,
          const size_type n,
          const size_type k,
          const double     alpha,
          const double     *A,
          const size_type lda,
          const double     *B,
          const size_type ldb,
          const double     beta,
          double           *C,
          const size_type ldc,
          LinAlgOpContext<utils::MemorySpace::HOST> &context)
    {
      unsigned int mTmp   = m;
      unsigned int nTmp   = n;
      unsigned int kTmp   = k;
      unsigned int ldaTmp = lda;
      unsigned int ldbTmp = ldb;
      unsigned int ldcTmp = ldc;
      dgemm_(&transA,
             &transB,
             &mTmp,
             &nTmp,
             &kTmp,
             &alpha,
             A,
             &ldaTmp,
             B,
             &ldbTmp,
             &beta,
             C,
             &ldcTmp);
    }

    template<>
    void
    xgemm<std::complex<float>,std::complex<float>,utils::MemorySpace::HOST>(
      const char                   transA,
      const char                   transB,
      const size_type          m,
      const size_type          n,
      const size_type          k,
      const std::complex<float> alpha,
      const std::complex<float> *A,
      const size_type          lda,
      const std::complex<float> *B,
      const size_type          ldb,
      const std::complex<float> beta,
      std::complex<float>       *C,
      const size_type          ldc,
      LinAlgOpContext<utils::MemorySpace::HOST> &context)
    {
      unsigned int mTmp   = m;
      unsigned int nTmp   = n;
      unsigned int kTmp   = k;
      unsigned int ldaTmp = lda;
      unsigned int ldbTmp = ldb;
      unsigned int ldcTmp = ldc;
      cgemm_(&transA,
             &transB,
             &mTmp,
             &nTmp,
             &kTmp,
             &alpha,
             A,
             &ldaTmp,
             B,
             &ldbTmp,
             &beta,
             C,
             &ldcTmp);
    }

    template<>
    void
    xgemm<std::complex<double>,std::complex<double>,utils::MemorySpace::HOST>(
      const char                    transA,
      const char                    transB,
      const size_type           m,
      const size_type           n,
      const size_type           k,
      const std::complex<double> alpha,
      const std::complex<double> *A,
      const size_type           lda,
      const std::complex<double> *B,
      const size_type           ldb,
      const std::complex<double> beta,
      std::complex<double>       *C,
      const size_type           ldc,
      LinAlgOpContext<utils::MemorySpace::HOST> &context)
    {
      unsigned int mTmp   = m;
      unsigned int nTmp   = n;
      unsigned int kTmp   = k;
      unsigned int ldaTmp = lda;
      unsigned int ldbTmp = ldb;
      unsigned int ldcTmp = ldc;
      zgemm_(&transA,
             &transB,
             &mTmp,
             &nTmp,
             &kTmp,
             &alpha,
             A,
             &ldaTmp,
             B,
             &ldbTmp,
             &beta,
             C,
             &ldcTmp);
    }


    template void
    xgemm<float,float,utils::MemorySpace::HOST>(
          const char                                   transA,
           const char                                   transB,
           const size_type                            m,
           const size_type                            n,
           const size_type                            k,
           const float  alpha,
           float const *                         A,
           const size_type                            lda,
           float const *                         B,
           const size_type                            ldb,
           const float  beta,
           float *C,
           const size_type                            ldc,
           LinAlgOpContext<utils::MemorySpace::HOST>   &context);

    template void
    xgemm<double,double,utils::MemorySpace::HOST>(
          const char   transA,
          const char   transB,
          const size_type m,
          const size_type n,
          const size_type k,
          const double     alpha,
          const double     *A,
          const size_type lda,
          const double     *B,
          const size_type ldb,
          const double     beta,
          double           *C,
          const size_type ldc,
          LinAlgOpContext<utils::MemorySpace::HOST> &context);          

    template void
    xgemm<std::complex<float>,std::complex<float>,utils::MemorySpace::HOST>(
      const char                   transA,
      const char                   transB,
      const size_type          m,
      const size_type          n,
      const size_type          k,
      const std::complex<float> alpha,
      const std::complex<float> *A,
      const size_type          lda,
      const std::complex<float> *B,
      const size_type          ldb,
      const std::complex<float> beta,
      std::complex<float>       *C,
      const size_type          ldc,
      LinAlgOpContext<utils::MemorySpace::HOST> &context);

    template void
    xgemm<std::complex<double>,std::complex<double>,utils::MemorySpace::HOST>(
      const char                    transA,
      const char                    transB,
      const size_type           m,
      const size_type           n,
      const size_type           k,
      const std::complex<double> alpha,
      const std::complex<double> *A,
      const size_type           lda,
      const std::complex<double> *B,
      const size_type           ldb,
      const std::complex<double> beta,
      std::complex<double>       *C,
      const size_type           ldc,
      LinAlgOpContext<utils::MemorySpace::HOST> &context);

      template <typename ValueType>
      real_type<ValueType>
      xasum(const size_type               n,
           ValueType const *             x,
           const size_type               incx,
           LinAlgOpContext<utils::MemorySpace::HOST> &context)
      {
        utils::throwException(false, "The input valuetypes are not supported by xgemm");
        return (real_type<ValueType>)0;
      }
      
      template <>
      real_type<float>
      xasum<float , utils::MemorySpace::HOST>
          (const size_type               n,
           float const *             x,
           const size_type               incx,
           LinAlgOpContext<utils::MemorySpace::HOST> &context)
      {
        unsigned int nTmp   = n;
        unsigned int incxTmp = incx;
        return sasum_(&nTmp,
                x,
                &incxTmp);
      }

      template <>
      real_type<double>
      xasum<double , utils::MemorySpace::HOST>
          (const size_type               n,
           double const *             x,
           const size_type               incx,
           LinAlgOpContext<utils::MemorySpace::HOST> &context)
      {
        unsigned int nTmp   = n;
        unsigned int incxTmp = incx;
        return dasum_(&nTmp,
                x,
                &incxTmp);
      }

      template <>
      real_type<std::complex<float>>
      xasum<std::complex<float> , utils::MemorySpace::HOST>
          (const size_type               n,
           std::complex<float> const *             x,
           const size_type               incx,
           LinAlgOpContext<utils::MemorySpace::HOST> &context)
      {
        unsigned int nTmp   = n;
        unsigned int incxTmp = incx;
        return scasum_(&nTmp,
                x,
                &incxTmp);
      }

      template <>
      real_type<std::complex<double>>
      xasum<std::complex<double> , utils::MemorySpace::HOST>
          (const size_type               n,
           std::complex<double> const *             x,
           const size_type               incx,
           LinAlgOpContext<utils::MemorySpace::HOST> &context)
      {
        unsigned int nTmp   = n;
        unsigned int incxTmp = incx;
        return dzasum_(&nTmp,
                x,
                &incxTmp);
      }

      template <typename ValueType>
      size_type
      xiamax(const size_type               n,
           ValueType const *             x,
           const size_type               incx,
           LinAlgOpContext<utils::MemorySpace::HOST> &context)
      {
        utils::throwException(false, "The input valuetypes are not supported by xgemm");
        return 0;
      }
      
      template <>
      size_type
      xiamax<float , utils::MemorySpace::HOST>
          (const size_type               n,
           float const *             x,
           const size_type               incx,
           LinAlgOpContext<utils::MemorySpace::HOST> &context)
      {
        unsigned int nTmp   = n;
        unsigned int incxTmp = incx;
        return isamax_(&nTmp,
                x,
                &incxTmp);
      }

      template <>
      size_type
      xiamax<double , utils::MemorySpace::HOST>
          (const size_type               n,
           double const *             x,
           const size_type               incx,
           LinAlgOpContext<utils::MemorySpace::HOST> &context)
      {
        unsigned int nTmp   = n;
        unsigned int incxTmp = incx;
        return idamax_(&nTmp,
                x,
                &incxTmp);
      }

      template <>
      size_type
      xiamax<std::complex<float> , utils::MemorySpace::HOST>
          (const size_type               n,
           std::complex<float> const *             x,
           const size_type               incx,
           LinAlgOpContext<utils::MemorySpace::HOST> &context)
      {
        unsigned int nTmp   = n;
        unsigned int incxTmp = incx;
        return icamax_(&nTmp,
                x,
                &incxTmp);
      }

      template <>
      size_type
      xiamax<std::complex<double> , utils::MemorySpace::HOST>
          (const size_type               n,
           std::complex<double> const *             x,
           const size_type               incx,
           LinAlgOpContext<utils::MemorySpace::HOST> &context)
      {
        unsigned int nTmp   = n;
        unsigned int incxTmp = incx;
        return izamax_(&nTmp,
                x,
                &incxTmp);
      }


      template <typename ValueType1,
                typename ValueType2>
      scalar_type<ValueType1, ValueType2>
      xdot(const size_type               n,
          ValueType1 const *            x,
          const size_type               incx,
          ValueType2 const *            y,
          const size_type               incy,
          LinAlgOpContext<utils::MemorySpace::HOST> &context)
      {
        utils::throwException("xdot not yet implemented in BlasWrapperAPIHost");
      }

      template <typename ValueType>
      real_type<ValueType>
      xnrm2(const size_type               n,
           ValueType const *             x,
           const size_type               incx,
           LinAlgOpContext<utils::MemorySpace::HOST> &context)
      {
        utils::throwException("xnrm2 not yet implemented in BlasWrapperAPIHost");
      }

      template <typename ValueType1,
                typename ValueType2>
      void
      xaxpy(const size_type                           n,
           const scalar_type<ValueType1, ValueType2> alpha,
           ValueType1 const *                        x,
           const size_type                           incx,
           ValueType2 *                              y,
           const size_type                           incy,
           LinAlgOpContext<utils::MemorySpace::HOST> &            context)
      {
        utils::throwException("xaxpy not yet implemented in BlasWrapperAPIHost");
      }

      template
      real_type<float>
      xasum<float , utils::MemorySpace::HOST>
          (const size_type               n,
           float const *             x,
           const size_type               incx,
           LinAlgOpContext<utils::MemorySpace::HOST> &context);

      template
      real_type<double>
      xasum<double , utils::MemorySpace::HOST>
          (const size_type               n,
           double const *             x,
           const size_type               incx,
           LinAlgOpContext<utils::MemorySpace::HOST> &context);

      template
      real_type<std::complex<float>>
      xasum<std::complex<float> , utils::MemorySpace::HOST>
          (const size_type               n,
           std::complex<float> const *             x,
           const size_type               incx,
           LinAlgOpContext<utils::MemorySpace::HOST> &context);

      template
      real_type<std::complex<double>>
      xasum<std::complex<double> , utils::MemorySpace::HOST>
          (const size_type               n,
           std::complex<double> const *             x,
           const size_type               incx,
           LinAlgOpContext<utils::MemorySpace::HOST> &context);
      
      template
      size_type
      xiamax<float , utils::MemorySpace::HOST>
          (const size_type               n,
           float const *             x,
           const size_type               incx,
           LinAlgOpContext<utils::MemorySpace::HOST> &context);

      template
      size_type
      xiamax<double , utils::MemorySpace::HOST>
          (const size_type               n,
           double const *             x,
           const size_type               incx,
           LinAlgOpContext<utils::MemorySpace::HOST> &context);

      template
      size_type
      xiamax<std::complex<float> , utils::MemorySpace::HOST>
          (const size_type               n,
           std::complex<float> const *             x,
           const size_type               incx,
           LinAlgOpContext<utils::MemorySpace::HOST> &context);

      template
      size_type
      xiamax<std::complex<double> , utils::MemorySpace::HOST>
          (const size_type               n,
           std::complex<double> const *             x,
           const size_type               incx,
           LinAlgOpContext<utils::MemorySpace::HOST> &context);

  } // namespace blasWrapper
  } // End of namespace linearAlgebra
} // End of namespace dftefe