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

      static utils::deviceBlasHandle_t &
      getDeviceBlasHandle()
      {
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
        return d_streams.data();
      }

      static utils::deviceBlasHandle_t *
      getDeviceBlasHandlesVec()
      {
        return d_deviceBlasHandles.data();
      }

    private:
#ifdef DFTEFE_WITH_DEVICE_AMD
      void
      initialize();
#endif

      size_type                                            d_numBlasStreams;
      inline static std::vector<utils::deviceBlasHandle_t> d_deviceBlasHandles;
      inline static std::vector<utils::deviceStream_t>     d_streams;

      inline static utils::deviceBlasHandle_t d_deviceBlasHandle;
      inline static utils::deviceStream_t     d_stream;

      /// storage for deviceblas handle
      TensorOpDataType d_opType;

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
