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
#ifdef DFTEFE_WITH_DEVICE
#  include "BlasAPIWrapper.h"
#  include "BlasLapackTemplates.h"
#  include <utils/DeviceUtils.h>
#  include <utils/DeviceTypeConfig.h>
#  include <utils/DeviceKernelLauncherHelpers.h>
#  include <utils/DeviceAPICalls.h>
#  include <utils/DeviceDataTypeOverloads.h>
#  include <utils/DeviceTypeConfigHalfPrec.h>
#  ifdef DFTEFE_WITH_DEVICE_INTEL
#    include <oneapi/mkl.hpp>
#    include <oneapi/mkl/blas.hpp>
#  endif
#  ifdef DFTEFE_WITH_DEVICE_AMD
#    define HIPBLAS_V2
#    include <rocblas.h>
#    include <hipblas.h>
#    include <hipblas/hipblas-version.h>
#  endif
#  ifdef DFTEFE_WITH_DEVICE_NVIDIA
#    include <cublas_v2.h>
#  endif

#  ifdef DFTEFE_WITH_DEVICE_NVIDIA
#    ifdef DFTEFE_WITH_64BIT_INT
#      define DFTEFE_DEVICE_BLAS_INT(type, name) cublas##type##name##_64
#    else
#      define DFTEFE_DEVICE_BLAS_INT(type, name) cublas##type##name
#    endif
#    define DFTEFE_DEVICE_BLAS(type, name) cublas##type##name
#  elif defined(DFTEFE_WITH_DEVICE_AMD)
#    ifdef DFTEFE_WITH_64BIT_INT
#      define DFTEFE_DEVICE_BLAS_INT(type, name) hipblas##type##name##_64
#    else
#      define DFTEFE_DEVICE_BLAS_INT(type, name) hipblas##type##name
#    endif
#    define DFTEFE_DEVICE_BLAS(type, name) hipblas##type##name
#  elif defined(DFTEFE_WITH_DEVICE_INTEL)
#    define DFTEFE_DEVICE_BLAS_INT(type, name) \
      oneapi::mkl::blas::column_major::name
#  else
#    error \
      "No device backend defined (DFTEFE_WITH_DEVICE_NVIDIA or DFTEFE_WITH_DEVICE_AMD or DFTEFE_WITH_DEVICE_INTEL)"
#  endif

namespace dftefe
{
  namespace linearAlgebra
  {
    namespace blasLapack
    {
      namespace blasWrapper
      {
#  if hipblasVersionMajor >= 2 || defined(DFTEFE_WITH_DEVICE_NVIDIA) || \
    defined(DFTEFE_WITH_DEVICE_INTEL)
        template <typename T>
        inline auto
        makeDataTypeDeviceBlasCompatible(T &&x)
          -> decltype(utils::makeDataTypeDeviceCompatible(std::forward<T>(x)))
        {
          return utils::makeDataTypeDeviceCompatible(std::forward<T>(x));
        }

#  else
        inline double
        makeDataTypeDeviceBlasCompatible(double a)
        {
          return a;
        }

        inline float
        makeDataTypeDeviceBlasCompatible(float a)
        {
          return a;
        }

        inline float *
        makeDataTypeDeviceBlasCompatible(float *a)
        {
          return reinterpret_cast<float *>(a);
        }

        inline const float *
        makeDataTypeDeviceBlasCompatible(const float *a)
        {
          return reinterpret_cast<const float *>(a);
        }

        inline double *
        makeDataTypeDeviceBlasCompatible(double *a)
        {
          return reinterpret_cast<double *>(a);
        }

        inline const double *
        makeDataTypeDeviceBlasCompatible(const double *a)
        {
          return reinterpret_cast<const double *>(a);
        }
        inline hipblasDoubleComplex
        makeDataTypeDeviceBlasCompatible(std::complex<double> a)
        {
          return hipblasDoubleComplex(a.real(), a.imag());
        }

        inline hipblasComplex
        makeDataTypeDeviceBlasCompatible(std::complex<float> a)
        {
          return hipblasComplex(a.real(), a.imag());
        }

        inline hipblasComplex *
        makeDataTypeDeviceBlasCompatible(std::complex<float> *a)
        {
          return reinterpret_cast<hipblasComplex *>(a);
        }

        inline const hipblasComplex *
        makeDataTypeDeviceBlasCompatible(const std::complex<float> *a)
        {
          return reinterpret_cast<const hipblasComplex *>(a);
        }

        inline hipblasDoubleComplex *
        makeDataTypeDeviceBlasCompatible(std::complex<double> *a)
        {
          return reinterpret_cast<hipblasDoubleComplex *>(a);
        }

        inline const hipblasDoubleComplex *
        makeDataTypeDeviceBlasCompatible(const std::complex<double> *a)
        {
          return reinterpret_cast<const hipblasDoubleComplex *>(a);
        }
#  endif

        template <typename ValueType1, typename ValueType2>
        void
        gemm(const char                                   transA,
             const char                                   transB,
             const size_type                              m,
             const size_type                              n,
             const size_type                              k,
             const scalar_type<ValueType1, ValueType2>    alpha,
             ValueType1 const *                           A,
             const size_type                              lda,
             ValueType2 const *                           B,
             const size_type                              ldb,
             const scalar_type<ValueType1, ValueType2>    beta,
             scalar_type<ValueType1, ValueType2> *        C,
             const size_type                              ldc,
             LinAlgOpContext<utils::MemorySpace::DEVICE> &context)
        {
          utils::throwException(
            false, "The input valuetypes are not supported by gemm");
        }

        template <>
        void
        gemm<float, float, utils::MemorySpace::DEVICE>(
          const char                                   transA,
          const char                                   transB,
          const size_type                              m,
          const size_type                              n,
          const size_type                              k,
          const float                                  alpha,
          float const *                                A,
          const size_type                              lda,
          float const *                                B,
          const size_type                              ldb,
          const float                                  beta,
          float *                                      C,
          const size_type                              ldc,
          LinAlgOpContext<utils::MemorySpace::DEVICE> &context)
        {
          utils::deviceBlasOperation_t transa, transb;
          if (transA == 'N')
            transa = utils::DEVICEBLAS_OP_N;
          else if (transA == 'T' || transA == 'C')
            transa = utils::DEVICEBLAS_OP_T;
          else
            {
              throw std::invalid_argument("Incorrect transA in gemm ");
            }
          if (transB == 'N')
            transb = utils::DEVICEBLAS_OP_N;
          else if (transB == 'T' || transB == 'C')
            transb = utils::DEVICEBLAS_OP_T;
          else
            {
              throw std::invalid_argument("Incorrect transB in gemm ");
            }
          utils::deviceBlasComputeType_t computeType =
            utils::DEVICEBLAS_COMPUTE_32F;
          if (context.getTensorOpDataType() == TensorOpDataType::TF32)
            computeType = utils::DEVICEBLAS_COMPUTE_32F_FAST_TF32;
          else if (context.getTensorOpDataType() == TensorOpDataType::BF16)
            computeType = utils::DEVICEBLAS_COMPUTE_32F_FAST_16BF;
          else if (context.getTensorOpDataType() == TensorOpDataType::FP16)
            computeType = utils::DEVICEBLAS_COMPUTE_32F_FAST_16F;
#  if defined(DFTEFE_WITH_DEVICE_LANG_CUDA) || \
    defined(DFTEFE_WITH_DEVICE_LANG_HIP)
          utils::deviceBlasStatus_t status =
            DFTEFE_DEVICE_BLAS_INT(, GemmEx)(context.getDeviceBlasHandle(),
                                             transa,
                                             transb,
                                             m,
                                             n,
                                             k,
                                             (const void *)&alpha,
                                             (const void *)A,
                                             utils::DEVICE_R_32F,
                                             lda,
                                             (const void *)B,
                                             utils::DEVICE_R_32F,
                                             ldb,
                                             (const void *)&beta,
                                             (void *)C,
                                             utils::DEVICE_R_32F,
                                             ldc,
                                             computeType,
                                             utils::DEVICEBLAS_GEMM_DEFAULT);
          DEVICEBLAS_API_CHECK(status);
#  elif defined(DFTEFE_WITH_DEVICE_LANG_SYCL)
          DEVICEBLAS_API_CHECK(
            DFTEFE_DEVICE_BLAS_INT(, gemm)(context.getDeviceBlasHandle(),
                                           transa,
                                           transb,
                                           m,
                                           n,
                                           k,
                                           &alpha,
                                           A,
                                           lda,
                                           B,
                                           ldb,
                                           &beta,
                                           C,
                                           ldc,
                                           computeType));
#  endif
        }

        template <>
        void
        gemm<double, double, utils::MemorySpace::DEVICE>(
          const char                                   transA,
          const char                                   transB,
          const size_type                              m,
          const size_type                              n,
          const size_type                              k,
          const double                                 alpha,
          const double *                               A,
          const size_type                              lda,
          const double *                               B,
          const size_type                              ldb,
          const double                                 beta,
          double *                                     C,
          const size_type                              ldc,
          LinAlgOpContext<utils::MemorySpace::DEVICE> &context)
        {
          utils::deviceBlasOperation_t transa, transb;
          if (transA == 'N')
            transa = utils::DEVICEBLAS_OP_N;
          else if (transA == 'T' || transA == 'C')
            transa = utils::DEVICEBLAS_OP_T;

          else
            {
              throw std::invalid_argument("Incorrect transA in gemm ");
            }
          if (transB == 'N')
            transb = utils::DEVICEBLAS_OP_N;
          else if (transB == 'T' || transB == 'C')
            transb = utils::DEVICEBLAS_OP_T;

          else
            {
              throw std::invalid_argument("Incorrect transB in gemm ");
            }
          DEVICEBLAS_API_CHECK(
            DFTEFE_DEVICE_BLAS_INT(D, gemm)(context.getDeviceBlasHandle(),
                                            transa,
                                            transb,
                                            m,
                                            n,
                                            k,
                                            &alpha,
                                            A,
                                            lda,
                                            B,
                                            ldb,
                                            &beta,
                                            C,
                                            ldc));
        }

        template <>
        void
        gemm<std::complex<float>,
             std::complex<float>,
             utils::MemorySpace::DEVICE>(
          const char                                   transA,
          const char                                   transB,
          const size_type                              m,
          const size_type                              n,
          const size_type                              k,
          const std::complex<float>                    alpha,
          const std::complex<float> *                  A,
          const size_type                              lda,
          const std::complex<float> *                  B,
          const size_type                              ldb,
          const std::complex<float>                    beta,
          std::complex<float> *                        C,
          const size_type                              ldc,
          LinAlgOpContext<utils::MemorySpace::DEVICE> &context)
        {
          utils::deviceBlasOperation_t transa, transb;
          if (transA == 'N')
            transa = utils::DEVICEBLAS_OP_N;
          else if (transA == 'T')
            transa = utils::DEVICEBLAS_OP_T;
          else if (transA == 'C')
            transa = utils::DEVICEBLAS_OP_C;
          else
            {
              throw std::invalid_argument("Incorrect transA in gemm ");
            }
          if (transB == 'N')
            transb = utils::DEVICEBLAS_OP_N;
          else if (transB == 'T')
            transb = utils::DEVICEBLAS_OP_T;
          else if (transB == 'C')
            transb = utils::DEVICEBLAS_OP_C;
          else
            {
              throw std::invalid_argument("Incorrect transB in gemm ");
            }
          utils::deviceBlasComputeType_t computeType =
            utils::DEVICEBLAS_COMPUTE_32F;
          if (context.getTensorOpDataType() == TensorOpDataType::TF32)
            computeType = utils::DEVICEBLAS_COMPUTE_32F_FAST_TF32;
          else if (context.getTensorOpDataType() == TensorOpDataType::BF16)
            computeType = utils::DEVICEBLAS_COMPUTE_32F_FAST_16BF;
          else if (context.getTensorOpDataType() == TensorOpDataType::FP16)
            computeType = utils::DEVICEBLAS_COMPUTE_32F_FAST_16F;
#  if defined(DFTEFE_WITH_DEVICE_LANG_CUDA) || \
    defined(DFTEFE_WITH_DEVICE_LANG_HIP)
          utils::deviceBlasStatus_t status =
            DFTEFE_DEVICE_BLAS_INT(, GemmEx)(context.getDeviceBlasHandle(),
                                             transa,
                                             transb,
                                             m,
                                             n,
                                             k,
                                             (const void *)&alpha,
                                             (const void *)A,
                                             utils::DEVICE_C_32F,
                                             lda,
                                             (const void *)B,
                                             utils::DEVICE_C_32F,
                                             ldb,
                                             (const void *)&beta,
                                             (void *)C,
                                             utils::DEVICE_C_32F,
                                             ldc,
                                             computeType,
                                             utils::DEVICEBLAS_GEMM_DEFAULT);

          DEVICEBLAS_API_CHECK(status);
#  elif defined(DFTEFE_WITH_DEVICE_LANG_SYCL)
          DEVICEBLAS_API_CHECK(
            DFTEFE_DEVICE_BLAS_INT(, gemm)(context.getDeviceBlasHandle(),
                                           transa,
                                           transb,
                                           m,
                                           n,
                                           k,
                                           &alpha,
                                           A,
                                           lda,
                                           B,
                                           ldb,
                                           &beta,
                                           C,
                                           ldc,
                                           computeType));
#  endif
        }

        template <>
        void
        gemm<std::complex<double>,
             std::complex<double>,
             utils::MemorySpace::DEVICE>(
          const char                                   transA,
          const char                                   transB,
          const size_type                              m,
          const size_type                              n,
          const size_type                              k,
          const std::complex<double>                   alpha,
          const std::complex<double> *                 A,
          const size_type                              lda,
          const std::complex<double> *                 B,
          const size_type                              ldb,
          const std::complex<double>                   beta,
          std::complex<double> *                       C,
          const size_type                              ldc,
          LinAlgOpContext<utils::MemorySpace::DEVICE> &context)
        {
          utils::deviceBlasOperation_t transa, transb;
          if (transA == 'N')
            transa = utils::DEVICEBLAS_OP_N;
          else if (transA == 'T')
            transa = utils::DEVICEBLAS_OP_T;
          else if (transA == 'C')
            transa = utils::DEVICEBLAS_OP_C;
          else
            {
              throw std::invalid_argument("Incorrect transA in gemm ");
            }
          if (transB == 'N')
            transb = utils::DEVICEBLAS_OP_N;
          else if (transB == 'T')
            transb = utils::DEVICEBLAS_OP_T;
          else if (transB == 'C')
            transb = utils::DEVICEBLAS_OP_C;
          else
            {
              throw std::invalid_argument("Incorrect transB in gemm ");
            }
          DEVICEBLAS_API_CHECK(DFTEFE_DEVICE_BLAS_INT(Z, gemm)(
            context.getDeviceBlasHandle(),
            transa,
            transb,
            m,
            n,
            k,
            makeDataTypeDeviceBlasCompatible(&alpha),
            makeDataTypeDeviceBlasCompatible(A),
            lda,
            makeDataTypeDeviceBlasCompatible(B),
            ldb,
            makeDataTypeDeviceBlasCompatible(&beta),
            makeDataTypeDeviceBlasCompatible(C),
            ldc));
        }

        template void
        gemm<float, float, utils::MemorySpace::DEVICE>(
          const char                                   transA,
          const char                                   transB,
          const size_type                              m,
          const size_type                              n,
          const size_type                              k,
          const float                                  alpha,
          float const *                                A,
          const size_type                              lda,
          float const *                                B,
          const size_type                              ldb,
          const float                                  beta,
          float *                                      C,
          const size_type                              ldc,
          LinAlgOpContext<utils::MemorySpace::DEVICE> &context);

        template void
        gemm<double, double, utils::MemorySpace::DEVICE>(
          const char                                   transA,
          const char                                   transB,
          const size_type                              m,
          const size_type                              n,
          const size_type                              k,
          const double                                 alpha,
          const double *                               A,
          const size_type                              lda,
          const double *                               B,
          const size_type                              ldb,
          const double                                 beta,
          double *                                     C,
          const size_type                              ldc,
          LinAlgOpContext<utils::MemorySpace::DEVICE> &context);

        template void
        gemm<std::complex<float>,
             std::complex<float>,
             utils::MemorySpace::DEVICE>(
          const char                                   transA,
          const char                                   transB,
          const size_type                              m,
          const size_type                              n,
          const size_type                              k,
          const std::complex<float>                    alpha,
          const std::complex<float> *                  A,
          const size_type                              lda,
          const std::complex<float> *                  B,
          const size_type                              ldb,
          const std::complex<float>                    beta,
          std::complex<float> *                        C,
          const size_type                              ldc,
          LinAlgOpContext<utils::MemorySpace::DEVICE> &context);

        template void
        gemm<std::complex<double>,
             std::complex<double>,
             utils::MemorySpace::DEVICE>(
          const char                                   transA,
          const char                                   transB,
          const size_type                              m,
          const size_type                              n,
          const size_type                              k,
          const std::complex<double>                   alpha,
          const std::complex<double> *                 A,
          const size_type                              lda,
          const std::complex<double> *                 B,
          const size_type                              ldb,
          const std::complex<double>                   beta,
          std::complex<double> *                       C,
          const size_type                              ldc,
          LinAlgOpContext<utils::MemorySpace::DEVICE> &context);

        template <typename ValueType1, typename ValueType2>
        void
        gemmStridedVarBatched(
          const size_type                              numMats,
          const char *                                 transA,
          const char *                                 transB,
          const size_type *                            stridea,
          const size_type *                            strideb,
          const size_type *                            stridec,
          const size_type *                            m,
          const size_type *                            n,
          const size_type *                            k,
          const scalar_type<ValueType1, ValueType2>    alpha,
          const ValueType1 *                           dA,
          const size_type *                            ldda,
          const ValueType2 *                           dB,
          const size_type *                            lddb,
          const scalar_type<ValueType1, ValueType2>    beta,
          scalar_type<ValueType1, ValueType2> *        dC,
          const size_type *                            lddc,
          LinAlgOpContext<utils::MemorySpace::DEVICE> &context)
        {
          utils::throwException(
            false,
            "The input valuetypes float and  complex float are not supported by gemmVarBatchedDevice");
        }

        template <>
        void
        gemmStridedVarBatched<double, double, utils::MemorySpace::DEVICE>(
          const size_type                              numMats,
          const char *                                 transA,
          const char *                                 transB,
          const size_type *                            stridea,
          const size_type *                            strideb,
          const size_type *                            stridec,
          const size_type *                            m,
          const size_type *                            n,
          const size_type *                            k,
          const double                                 alpha,
          const double *                               dA,
          const size_type *                            ldda,
          const double *                               dB,
          const size_type *                            lddb,
          const double                                 beta,
          double *                                     dC,
          const size_type *                            lddc,
          LinAlgOpContext<utils::MemorySpace::DEVICE> &context)
        {
          size_type       cumulativeA = 0;
          size_type       cumulativeB = 0;
          size_type       cumulativeC = 0;
          const size_type numStreams  = context.numBlasStreams();
          auto *          streams     = context.getBlasStreamsVec();
          auto *          handles     = context.getDeviceBlasHandlesVec();

          for (size_type ibatch = 0; ibatch < numMats; ++ibatch)
            {
              if (m[ibatch] > 0 && n[ibatch] > 0 && k[ibatch] > 0)
                {
                  size_type sid = ibatch % numStreams;

                  utils::deviceBlasOperation_t transa, transb;
                  if (transA[ibatch] == 'N')
                    transa = utils::DEVICEBLAS_OP_N;
                  else if (transA[ibatch] == 'T')
                    transa = utils::DEVICEBLAS_OP_T;
                  else if (transA[ibatch] == 'C')
                    transa = utils::DEVICEBLAS_OP_C;
                  else
                    {
                      throw std::invalid_argument("Incorrect transA in gemm ");
                    }
                  if (transB[ibatch] == 'N')
                    transb = utils::DEVICEBLAS_OP_N;
                  else if (transB[ibatch] == 'T')
                    transb = utils::DEVICEBLAS_OP_T;
                  else if (transB[ibatch] == 'C')
                    transb = utils::DEVICEBLAS_OP_C;
                  else
                    {
                      throw std::invalid_argument("Incorrect transB in gemm ");
                    }
                  DEVICEBLAS_API_CHECK(
                    DFTEFE_DEVICE_BLAS_INT(D, gemm)(handles[sid],
                                                    transa,
                                                    transb,
                                                    m[ibatch],
                                                    n[ibatch],
                                                    k[ibatch],
                                                    &alpha,
                                                    dA + cumulativeA,
                                                    ldda[ibatch],
                                                    dB + cumulativeB,
                                                    lddb[ibatch],
                                                    &beta,
                                                    dC + cumulativeC,
                                                    lddc[ibatch]));
                }

              cumulativeA += stridea[ibatch];
              cumulativeB += strideb[ibatch];
              cumulativeC += stridec[ibatch];
            }

          for (int s = 0; s < numStreams; ++s)
            utils::deviceStreamSynchronize(streams[s]);
        }

        template <>
        void
        gemmStridedVarBatched<std::complex<double>,
                              std::complex<double>,
                              utils::MemorySpace::DEVICE>(
          const size_type                              numMats,
          const char *                                 transA,
          const char *                                 transB,
          const size_type *                            stridea,
          const size_type *                            strideb,
          const size_type *                            stridec,
          const size_type *                            m,
          const size_type *                            n,
          const size_type *                            k,
          const std::complex<double>                   alpha,
          const std::complex<double> *                 dA,
          const size_type *                            ldda,
          const std::complex<double> *                 dB,
          const size_type *                            lddb,
          const std::complex<double>                   beta,
          std::complex<double> *                       dC,
          const size_type *                            lddc,
          LinAlgOpContext<utils::MemorySpace::DEVICE> &context)
        {
          size_type       cumulativeA = 0;
          size_type       cumulativeB = 0;
          size_type       cumulativeC = 0;
          const size_type numStreams  = context.numBlasStreams();
          auto *          streams     = context.getBlasStreamsVec();
          auto *          handles     = context.getDeviceBlasHandlesVec();

          for (size_type ibatch = 0; ibatch < numMats; ++ibatch)
            {
              if (m[ibatch] > 0 && n[ibatch] > 0 && k[ibatch] > 0)
                {
                  size_type sid = ibatch % numStreams;

                  utils::deviceBlasOperation_t transa, transb;
                  if (transA[ibatch] == 'N')
                    transa = utils::DEVICEBLAS_OP_N;
                  else if (transA[ibatch] == 'T')
                    transa = utils::DEVICEBLAS_OP_T;
                  else if (transA[ibatch] == 'C')
                    transa = utils::DEVICEBLAS_OP_C;
                  else
                    {
                      throw std::invalid_argument("Incorrect transA in gemm ");
                    }
                  if (transB[ibatch] == 'N')
                    transb = utils::DEVICEBLAS_OP_N;
                  else if (transB[ibatch] == 'T')
                    transb = utils::DEVICEBLAS_OP_T;
                  else if (transB[ibatch] == 'C')
                    transb = utils::DEVICEBLAS_OP_C;
                  else
                    {
                      throw std::invalid_argument("Incorrect transB in gemm ");
                    }
                  DEVICEBLAS_API_CHECK(DFTEFE_DEVICE_BLAS_INT(Z, gemm)(
                    handles[sid],
                    transa,
                    transb,
                    m[ibatch],
                    n[ibatch],
                    k[ibatch],
                    makeDataTypeDeviceBlasCompatible(&alpha),
                    makeDataTypeDeviceBlasCompatible(dA + cumulativeA),
                    ldda[ibatch],
                    makeDataTypeDeviceBlasCompatible(dB + cumulativeB),
                    lddb[ibatch],
                    makeDataTypeDeviceBlasCompatible(&beta),
                    makeDataTypeDeviceBlasCompatible(dC + cumulativeC),
                    lddc[ibatch]));
                }

              cumulativeA += stridea[ibatch];
              cumulativeB += strideb[ibatch];
              cumulativeC += stridec[ibatch];
            }

          for (int s = 0; s < numStreams; ++s)
            utils::deviceStreamSynchronize(streams[s]);
        }


        template void
        gemmStridedVarBatched<double, double, utils::MemorySpace::DEVICE>(
          const size_type                              numMats,
          const char *                                 transA,
          const char *                                 transB,
          const size_type *                            stridea,
          const size_type *                            strideb,
          const size_type *                            stridec,
          const size_type *                            m,
          const size_type *                            n,
          const size_type *                            k,
          const double                                 alpha,
          const double *                               dA,
          const size_type *                            ldda,
          const double *                               dB,
          const size_type *                            lddb,
          const double                                 beta,
          double *                                     dC,
          const size_type *                            lddc,
          LinAlgOpContext<utils::MemorySpace::DEVICE> &context);

        template void
        gemmStridedVarBatched<std::complex<double>,
                              std::complex<double>,
                              utils::MemorySpace::DEVICE>(
          const size_type                              numMats,
          const char *                                 transA,
          const char *                                 transB,
          const size_type *                            stridea,
          const size_type *                            strideb,
          const size_type *                            stridec,
          const size_type *                            m,
          const size_type *                            n,
          const size_type *                            k,
          const std::complex<double>                   alpha,
          const std::complex<double> *                 dA,
          const size_type *                            ldda,
          const std::complex<double> *                 dB,
          const size_type *                            lddb,
          const std::complex<double>                   beta,
          std::complex<double> *                       dC,
          const size_type *                            lddc,
          LinAlgOpContext<utils::MemorySpace::DEVICE> &context);

        template <typename ValueType1, typename ValueType2>
        void
        axpy(const size_type                              n,
             const scalar_type<ValueType1, ValueType2>    alpha,
             ValueType1 const *                           x,
             const size_type                              incx,
             ValueType2 *                                 y,
             const size_type                              incy,
             LinAlgOpContext<utils::MemorySpace::DEVICE> &context)
        {
          utils::throwException(
            "axpy not yet implemented in BlasWrapperAPIDEVICE");
        }

        template <>
        void
        axpy<double, double, utils::MemorySpace::DEVICE>(
          const size_type                              n,
          const scalar_type<double, double>            alpha,
          double const *                               x,
          const size_type                              incx,
          double *                                     y,
          const size_type                              incy,
          LinAlgOpContext<utils::MemorySpace::DEVICE> &context)
        {
          unsigned int nTmp    = n;
          unsigned int incxTmp = incx;
          unsigned int incyTmp = incy;
          DEVICEBLAS_API_CHECK(
            DFTEFE_DEVICE_BLAS_INT(D, axpy)(context.getDeviceBlasHandle(),
                                            nTmp,
                                            &alpha,
                                            x,
                                            incxTmp,
                                            y,
                                            incyTmp));
        }

        template <>
        void
        axpy<std::complex<double>,
             std::complex<double>,
             utils::MemorySpace::DEVICE>(
          const size_type                                               n,
          const scalar_type<std::complex<double>, std::complex<double>> alpha,
          std::complex<double> const *                                  x,
          const size_type                                               incx,
          std::complex<double> *                                        y,
          const size_type                                               incy,
          LinAlgOpContext<utils::MemorySpace::DEVICE> &                 context)
        {
          unsigned int nTmp    = n;
          unsigned int incxTmp = incx;
          unsigned int incyTmp = incy;
          DEVICEBLAS_API_CHECK(DFTEFE_DEVICE_BLAS_INT(Z, axpy)(
            context.getDeviceBlasHandle(),
            nTmp,
            makeDataTypeDeviceBlasCompatible(&alpha),
            makeDataTypeDeviceBlasCompatible(x),
            incxTmp,
            makeDataTypeDeviceBlasCompatible(y),
            incyTmp));
        }

        template <>
        void
        axpy<std::complex<float>,
             std::complex<float>,
             utils::MemorySpace::DEVICE>(
          const size_type                                             n,
          const scalar_type<std::complex<float>, std::complex<float>> alpha,
          std::complex<float> const *                                 x,
          const size_type                                             incx,
          std::complex<float> *                                       y,
          const size_type                                             incy,
          LinAlgOpContext<utils::MemorySpace::DEVICE> &               context)
        {
          unsigned int nTmp    = n;
          unsigned int incxTmp = incx;
          unsigned int incyTmp = incy;

          DEVICEBLAS_API_CHECK(DFTEFE_DEVICE_BLAS_INT(C, axpy)(
            context.getDeviceBlasHandle(),
            nTmp,
            makeDataTypeDeviceBlasCompatible(&alpha),
            makeDataTypeDeviceBlasCompatible(x),
            incxTmp,
            makeDataTypeDeviceBlasCompatible(y),
            incyTmp));
        }

        template <>
        void
        axpy<float, float, utils::MemorySpace::DEVICE>(
          const size_type                              n,
          const scalar_type<float, float>              alpha,
          float const *                                x,
          const size_type                              incx,
          float *                                      y,
          const size_type                              incy,
          LinAlgOpContext<utils::MemorySpace::DEVICE> &context)
        {
          unsigned int nTmp    = n;
          unsigned int incxTmp = incx;
          unsigned int incyTmp = incy;

          DEVICEBLAS_API_CHECK(
            DFTEFE_DEVICE_BLAS_INT(S, axpy)(context.getDeviceBlasHandle(),
                                            nTmp,
                                            &alpha,
                                            x,
                                            incxTmp,
                                            y,
                                            incyTmp));
        }

        template void
        axpy<double, double, utils::MemorySpace::DEVICE>(
          const size_type                              n,
          const scalar_type<double, double>            alpha,
          double const *                               x,
          const size_type                              incx,
          double *                                     y,
          const size_type                              incy,
          LinAlgOpContext<utils::MemorySpace::DEVICE> &context);

        template void
        axpy<std::complex<double>,
             std::complex<double>,
             utils::MemorySpace::DEVICE>(
          const size_type                                               n,
          const scalar_type<std::complex<double>, std::complex<double>> alpha,
          std::complex<double> const *                                  x,
          const size_type                                               incx,
          std::complex<double> *                                        y,
          const size_type                                               incy,
          LinAlgOpContext<utils::MemorySpace::DEVICE> &context);

        template void
        axpy<float, float, utils::MemorySpace::DEVICE>(
          const size_type                              n,
          const scalar_type<float, float>              alpha,
          float const *                                x,
          const size_type                              incx,
          float *                                      y,
          const size_type                              incy,
          LinAlgOpContext<utils::MemorySpace::DEVICE> &context);

        template void
        axpy<std::complex<float>,
             std::complex<float>,
             utils::MemorySpace::DEVICE>(
          const size_type                                             n,
          const scalar_type<std::complex<float>, std::complex<float>> alpha,
          std::complex<float> const *                                 x,
          const size_type                                             incx,
          std::complex<float> *                                       y,
          const size_type                                             incy,
          LinAlgOpContext<utils::MemorySpace::DEVICE> &               context);

      } // namespace blasWrapper

    } // namespace blasLapack
  }   // End of namespace linearAlgebra
} // End of namespace dftefe
#endif
