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

#ifndef dftefeLinAlgOpContext_h
#define dftefeLinAlgOpContext_h

#include <utils/MemorySpaceType.h>
#include <memory>
#include <utils/DeviceUtils.h>
#include <utils/DeviceTypeConfig.h>
#include <utils/DeviceKernelLauncherHelpers.h>
#include <utils/DeviceAPICalls.h>
#include <utils/DeviceDataTypeOverloads.h>
#include <utils/DeviceTypeConfigHalfPrec.h>
#include <linearAlgebra/BlasLapackTypedef.h>
namespace dftefe
{
  namespace linearAlgebra
  {
    enum class TensorOpDataType
    {
      FP32,
      TF32,
      BF16,
      FP16
    };

    template <utils::MemorySpace memorySpace>
    class LinAlgOpContext
    {
    public:
      LinAlgOpContext(size_type numBlasStreams = 0);

      ~LinAlgOpContext() = default;

      void
      setTensorOpDataType(TensorOpDataType opType)
      {
        d_opType = opType;
      }

      TensorOpDataType
      getTensorOpDataType()
      {
        return d_opType;
      }

      static utils::deviceBlasStatus_t
      setBlasStream(utils::deviceStream_t &streamId);

      static utils::deviceStream_t &
      getBlasStream()
      {
#if defined(DFTEFE_WITH_DEVICE_LANG_SYCL)
        enforceDefaultStreamExclusivity();
#endif
        return d_defaultStream;
      }

      static utils::deviceBlasHandle_t &
      getDeviceBlasHandle()
      {
#if defined(DFTEFE_WITH_DEVICE_LANG_SYCL)
        enforceDefaultStreamExclusivity();
#endif
        return d_deviceBlasHandle;
      }

      size_type
      numBlasStreams() const
      {
        return d_numBlasStreams;
      }

      static utils::deviceStream_t *
      getBlasStreamsVec()
      {
#if defined(DFTEFE_WITH_DEVICE_LANG_SYCL)
        enforceNonDefaultStreamExclusivity();
#endif
        return d_streams.data();
      }

      static utils::deviceBlasHandle_t *
      getDeviceBlasHandlesVec()
      {
#if defined(DFTEFE_WITH_DEVICE_LANG_SYCL)
        enforceNonDefaultStreamExclusivity();
#endif
        return d_deviceBlasHandles.data();
      }

    private:
      size_type                                            d_numBlasStreams;
      inline static std::vector<utils::deviceBlasHandle_t> d_deviceBlasHandles;
      inline static std::vector<utils::deviceStream_t>     d_streams;

      inline static utils::deviceBlasHandle_t d_deviceBlasHandle;
      inline static utils::deviceStream_t     d_defaultStream;

      /// storage for deviceblas handle
      TensorOpDataType d_opType;

#if defined(DFTEFE_WITH_DEVICE_LANG_SYCL)
      // SYCL queues (unlike CUDA/HIP streams) have no implicit ordering
      // relationship with one another, so the default-stream/non-default-
      // stream mutual exclusivity that CUDA/HIP get for free from the driver
      // (stream 0 implicitly synchronizes with every other blocking stream)
      // has to be enforced here explicitly for callers that go through
      // getBlasStream()/getDeviceBlasHandle()/getBlasStreamsVec()/
      // getDeviceBlasHandlesVec().
      enum class ActiveStreamSide
      {
        Default,
        NonDefault
      };

      inline static ActiveStreamSide d_activeStreamSide =
        ActiveStreamSide::Default;

      static void
      enforceDefaultStreamExclusivity()
      {
        if (d_activeStreamSide == ActiveStreamSide::NonDefault)
          {
            for (auto &stream : d_streams)
              utils::deviceStreamSynchronize(stream);
            d_activeStreamSide = ActiveStreamSide::Default;
          }
      }

      static void
      enforceNonDefaultStreamExclusivity()
      {
        if (d_activeStreamSide == ActiveStreamSide::Default)
          {
            utils::deviceStreamSynchronize(d_defaultStream);
            d_activeStreamSide = ActiveStreamSide::NonDefault;
          }
      }
#endif

      static utils::deviceBlasStatus_t
      setBlasStream(utils::deviceBlasHandle_t &handleId,
                    utils::deviceStream_t &    streamId);

      utils::deviceBlasStatus_t
      create(utils::deviceBlasHandle_t &handleId);

      utils::deviceBlasStatus_t
      destroy(utils::deviceBlasHandle_t &handleId);

    }; // end of LinAlgOpContext
  }    // end of namespace linearAlgebra
} // end of namespace dftefe
#include <linearAlgebra/LinAlgOpContext.t.cpp>
#endif // end of dftefeLinAlgOpContext_h
