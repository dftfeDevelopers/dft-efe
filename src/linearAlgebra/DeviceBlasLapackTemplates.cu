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
 * @author Sambit Das.
 */

#ifdef DFTEFE_WITH_DEVICE_CUDA
#  include <linearAlgebra/DeviceBlasLapackTemplates.h>
#  include <utils/DeviceDataTypeOverloads.cuh>
#  include <utils/TypeConfig.h>
#  include <utils/Exceptions.h>

namespace dftefe
{
  namespace linearAlgebra
  {
    //
    // gemm template definitions
    //

    template <>
    void
    DeviceBlasLapack<double>::gemm(deviceBlasHandleType    handle,
                                   deviceBlasOperationType transa,
                                   deviceBlasOperationType transb,
                                   int                     m,
                                   int                     n,
                                   int                     k,
                                   const double *          alpha,
                                   const double *          A,
                                   int                     lda,
                                   const double *          B,
                                   int                     ldb,
                                   const double *          beta,
                                   double *                C,
                                   int                     ldc)
    {
      cublasDgemm(
        handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }


    template <>
    void
    DeviceBlasLapack<float>::gemm(deviceBlasHandleType    handle,
                                  deviceBlasOperationType transa,
                                  deviceBlasOperationType transb,
                                  int                     m,
                                  int                     n,
                                  int                     k,
                                  const float *           alpha,
                                  const float *           A,
                                  int                     lda,
                                  const float *           B,
                                  int                     ldb,
                                  const float *           beta,
                                  float *                 C,
                                  int                     ldc)
    {
      cublasSgemm(
        handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }


    template <>
    void
    DeviceBlasLapack<std::complex<double>>::gemm(
      deviceBlasHandleType        handle,
      deviceBlasOperationType     transa,
      deviceBlasOperationType     transb,
      int                         m,
      int                         n,
      int                         k,
      const std::complex<double> *alpha,
      const std::complex<double> *A,
      int                         lda,
      const std::complex<double> *B,
      int                         ldb,
      const std::complex<double> *beta,
      std::complex<double> *      C,
      int                         ldc)
    {
      cublasZgemm(handle,
                  transa,
                  transb,
                  m,
                  n,
                  k,
                  dftefe::utils::makeDataTypeDeviceCompatible(alpha),
                  dftefe::utils::makeDataTypeDeviceCompatible(A),
                  lda,
                  dftefe::utils::makeDataTypeDeviceCompatible(B),
                  ldb,
                  dftefe::utils::makeDataTypeDeviceCompatible(beta),
                  dftefe::utils::makeDataTypeDeviceCompatible(C),
                  ldc);
    }


    template <>
    void
    DeviceBlasLapack<std::complex<float>>::gemm(
      deviceBlasHandleType       handle,
      deviceBlasOperationType    transa,
      deviceBlasOperationType    transb,
      int                        m,
      int                        n,
      int                        k,
      const std::complex<float> *alpha,
      const std::complex<float> *A,
      int                        lda,
      const std::complex<float> *B,
      int                        ldb,
      const std::complex<float> *beta,
      std::complex<float> *      C,
      int                        ldc)
    {
      cublasCgemm(handle,
                  transa,
                  transb,
                  m,
                  n,
                  k,
                  dftefe::utils::makeDataTypeDeviceCompatible(alpha),
                  dftefe::utils::makeDataTypeDeviceCompatible(A),
                  lda,
                  dftefe::utils::makeDataTypeDeviceCompatible(B),
                  ldb,
                  dftefe::utils::makeDataTypeDeviceCompatible(beta),
                  dftefe::utils::makeDataTypeDeviceCompatible(C),
                  ldc);
    }


    template <>
    void
    DeviceBlasLapack<int>::gemm(deviceBlasHandleType    handle,
                                deviceBlasOperationType transa,
                                deviceBlasOperationType transb,
                                int                     m,
                                int                     n,
                                int                     k,
                                const int *             alpha,
                                const int *             A,
                                int                     lda,
                                const int *             B,
                                int                     ldb,
                                const int *             beta,
                                int *                   C,
                                int                     ldc)
    {
      DFTEFE_AssertWithMsg(false, "Not implemented.");
    }


    template <>
    void
    DeviceBlasLapack<size_type>::gemm(deviceBlasHandleType    handle,
                                      deviceBlasOperationType transa,
                                      deviceBlasOperationType transb,
                                      int                     m,
                                      int                     n,
                                      int                     k,
                                      const size_type *       alpha,
                                      const size_type *       A,
                                      int                     lda,
                                      const size_type *       B,
                                      int                     ldb,
                                      const size_type *       beta,
                                      size_type *             C,
                                      int                     ldc)
    {
      DFTEFE_AssertWithMsg(false, "Not implemented.");
    }

    //
    // nrm2 template definitions
    //

    template <>
    void
    DeviceBlasLapack<double>::nrm2(deviceBlasHandleType handle,
                                   int                  n,
                                   const double *       x,
                                   int                  incx,
                                   double *             result)
    {
      double resultTemp;
      cublasDnrm2(handle, n, x, incx, &resultTemp);
      *result = resultTemp;
    }


    template <>
    void
    DeviceBlasLapack<float>::nrm2(deviceBlasHandleType handle,
                                  int                  n,
                                  const float *        x,
                                  int                  incx,
                                  double *             result)
    {
      float resultTemp;
      cublasSnrm2(handle, n, x, incx, &resultTemp);
      *result = resultTemp;
    }

    template <>
    void
    DeviceBlasLapack<std::complex<double>>::nrm2(deviceBlasHandleType handle,
                                                 int                  n,
                                                 const std::complex<double> *x,
                                                 int     incx,
                                                 double *result)
    {
      double resultTemp;
      cublasDznrm2(handle,
                   n,
                   dftefe::utils::makeDataTypeDeviceCompatible(x),
                   incx,
                   &resultTemp);
      *result = resultTemp;
    }


    template <>
    void
    DeviceBlasLapack<std::complex<float>>::nrm2(deviceBlasHandleType handle,
                                                int                  n,
                                                const std::complex<float> *x,
                                                int                        incx,
                                                double *result)
    {
      float resultTemp;
      cublasScnrm2(handle,
                   n,
                   dftefe::utils::makeDataTypeDeviceCompatible(x),
                   incx,
                   &resultTemp);
      *result = resultTemp;
    }

    template <>
    void
    DeviceBlasLapack<int>::nrm2(deviceBlasHandleType handle,
                                int                  n,
                                const int *          x,
                                int                  incx,
                                double *             result)
    {
      DFTEFE_AssertWithMsg(false, "Not implemented.");
    }


    template <>
    void
    DeviceBlasLapack<size_type>::nrm2(deviceBlasHandleType handle,
                                      int                  n,
                                      const size_type *    x,
                                      int                  incx,
                                      double *             result)
    {
      DFTEFE_AssertWithMsg(false, "Not implemented.");
    }

    //
    // iamax template definitions
    //

    template <>
    void
    DeviceBlasLapack<double>::iamax(deviceBlasHandleType handle,
                                    int                  n,
                                    const double *       x,
                                    int                  incx,
                                    int *                maxid)
    {
      cublasIdamax(handle, n, x, incx, maxid);
    }


    template <>
    void
    DeviceBlasLapack<float>::iamax(deviceBlasHandleType handle,
                                   int                  n,
                                   const float *        x,
                                   int                  incx,
                                   int *                maxid)
    {
      cublasIsamax(handle, n, x, incx, maxid);
    }

    template <>
    void
    DeviceBlasLapack<std::complex<double>>::iamax(deviceBlasHandleType handle,
                                                  int                  n,
                                                  const std::complex<double> *x,
                                                  int  incx,
                                                  int *maxid)
    {
      cublasIzamax(
        handle, n, dftefe::utils::makeDataTypeDeviceCompatible(x), incx, maxid);
    }


    template <>
    void
    DeviceBlasLapack<std::complex<float>>::iamax(deviceBlasHandleType handle,
                                                 int                  n,
                                                 const std::complex<float> *x,
                                                 int  incx,
                                                 int *maxid)
    {
      cublasIcamax(
        handle, n, dftefe::utils::makeDataTypeDeviceCompatible(x), incx, maxid);
    }

    template <>
    void
    DeviceBlasLapack<int>::iamax(deviceBlasHandleType handle,
                                 int                  n,
                                 const int *          x,
                                 int                  incx,
                                 int *                maxid)
    {
      DFTEFE_AssertWithMsg(false, "Not implemented.");
    }


    template <>
    void
    DeviceBlasLapack<size_type>::iamax(deviceBlasHandleType handle,
                                       int                  n,
                                       const size_type *    x,
                                       int                  incx,
                                       int *                maxid)
    {
      DFTEFE_AssertWithMsg(false, "Not implemented.");
    }
  } // namespace linearAlgebra
} // namespace dftefe
#endif // DFTEFE_WITH_DEVICE
