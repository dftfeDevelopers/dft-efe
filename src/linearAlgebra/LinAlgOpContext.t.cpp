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
 * @author Ian C. Lin, Sambit Das
 */

#include <linearAlgebra/LinAlgOpContext.h>
#ifdef DFTEFE_WITH_DEVICE
#ifdef DFTEFE_WITH_DEVICE_INTEL
#  include <oneapi/mkl.hpp>
#  include <oneapi/mkl/blas.hpp>
#endif
#ifdef DFTEFE_WITH_DEVICE_AMD
#  define HIPBLAS_V2
#  include <rocblas.h>
#  include <hipblas.h>
#  include <hipblas/hipblas-version.h>
#endif
#ifdef DFTEFE_WITH_DEVICE_NVIDIA
#  include <cublas_v2.h>
#endif

#ifdef DFTEFE_WITH_DEVICE_NVIDIA
#  ifdef DFTEFE_WITH_64BIT_INT
#    define DFTEFE_DEVICE_BLAS_INT(type, name) cublas##type##name##_64
#  else
#    define DFTEFE_DEVICE_BLAS_INT(type, name) cublas##type##name
#  endif
#  define DFTEFE_DEVICE_BLAS(type, name) cublas##type##name
#elif defined(DFTEFE_WITH_DEVICE_AMD)
#  ifdef DFTEFE_WITH_64BIT_INT
#    define DFTEFE_DEVICE_BLAS_INT(type, name) hipblas##type##name##_64
#  else
#    define DFTEFE_DEVICE_BLAS_INT(type, name) hipblas##type##name
#  endif
#  define DFTEFE_DEVICE_BLAS(type, name) hipblas##type##name
#elif defined(DFTEFE_WITH_DEVICE_INTEL)
#  define DFTEFE_DEVICE_BLAS_INT(type, name) \
    oneapi::mkl::blas::column_major::name
#else
#  error \
    "No device backend defined (DFTEFE_WITH_DEVICE_NVIDIA or DFTEFE_WITH_DEVICE_AMD)"
#endif
#endif

namespace dftefe
{
  namespace linearAlgebra
  {

    template <utils::MemorySpace memorySpace>
    LinAlgOpContext<memorySpace>::LinAlgOpContext(size_type numBlasStreams)
      : d_numBlasStreams(numBlasStreams)
    {
#ifdef DFTEFE_WITH_DEVICE_AMD
      if constexpr (memorySpace == utils::MemorySpace::DEVICE)
        rocblas_initialize();
#endif

#if defined(DFTEFE_WITH_DEVICE)
      utils::deviceBlasStatus_t status;
      d_opType = TensorOpDataType::FP32;
      d_stream = utils::defaultStream;
      status   = create(d_deviceBlasHandle);
      status   = setBlasStream(d_deviceBlasHandle, d_stream);

      d_streams.resize(d_numBlasStreams);
      d_deviceBlasHandles.resize(d_numBlasStreams);

      for (size_type i = 0; i < d_numBlasStreams; ++i)
        {
          utils::deviceError_t streamErr = utils::deviceStreamCreate(d_streams[i]);
          DEVICE_API_CHECK(streamErr);
          status = create(d_deviceBlasHandles[i]);
          status = setBlasStream(d_deviceBlasHandles[i], d_streams[i]);
        }
#endif
    }

    template <utils::MemorySpace memorySpace>
    utils::deviceBlasStatus_t
    LinAlgOpContext<memorySpace>::setBlasStream(utils::deviceStream_t &streamId)
    {
      return setBlasStream(d_deviceBlasHandle, streamId);
    }

    template <utils::MemorySpace memorySpace>
    utils::deviceBlasStatus_t
    LinAlgOpContext<memorySpace>::setBlasStream(
      utils::deviceBlasHandle_t &handleId,
      utils::deviceStream_t &    streamId)
    {
      utils::deviceBlasStatus_t status;
#if defined(DFTEFE_WITH_DEVICE_LANG_CUDA) || \
  defined(DFTEFE_WITH_DEVICE_LANG_HIP)
      status =
        DFTEFE_DEVICE_BLAS(, SetStream)(handleId, streamId);
      DEVICEBLAS_API_CHECK(status);
#elif defined(DFTEFE_WITH_DEVICE_LANG_SYCL)
      handleId = utils::queueRegistry.find(streamId)->second;
      status = utils::deviceBlasSuccess;
#endif
      return status;
    }

    template <utils::MemorySpace memorySpace>
    utils::deviceBlasStatus_t
    LinAlgOpContext<memorySpace>::create(utils::deviceBlasHandle_t &handleId)
    {
       utils::deviceBlasStatus_t status;
#if defined(DFTEFE_WITH_DEVICE_LANG_CUDA) || \
  defined(DFTEFE_WITH_DEVICE_LANG_HIP)
      status =
        DFTEFE_DEVICE_BLAS(, Create)(&handleId);
      DEVICEBLAS_API_CHECK(status);
#elif defined(DFTEFE_WITH_DEVICE_LANG_SYCL)
      handleId = utils::queueRegistry.find(utils::defaultStream)->second;
      status = utils::deviceBlasSuccess;
#endif
      return status;
    }

    template <utils::MemorySpace memorySpace>
    utils::deviceBlasStatus_t
    LinAlgOpContext<memorySpace>::destroy(utils::deviceBlasHandle_t &handleId)
    {
      utils::deviceBlasStatus_t status;
#if defined(DFTEFE_WITH_DEVICE_LANG_CUDA) || \
  defined(DFTEFE_WITH_DEVICE_LANG_HIP)
      status =
        DFTEFE_DEVICE_BLAS(, Destroy)(handleId);
      DEVICEBLAS_API_CHECK(status);
#elif defined(DFTEFE_WITH_DEVICE_LANG_SYCL)
      status = utils::deviceBlasSuccess;
#endif
      return status;
    }

  } // end of namespace linearAlgebra
} // end of namespace dftefe
