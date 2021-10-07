#include "DeviceUtils.h"

#ifdef DFTEFE_WITH_DEVICE

#  ifdef DFTEFE_WITH_DEVICE_CUDA
#    include <cuda_runtime.h>
#  endif
#  include <stdexcept>
#  include <string>

namespace dftefe
{
  void
  DeviceUtils::initialize(const int world_rank)
  {
    int n_devices = 0;
#  ifdef DFTEFE_WITH_DEVICE_CUDA
    cudaGetDeviceCount(&n_devices);
#  endif

    if (n_devices == 0)
      {
        std::string message = "Number of devices cannot be zero";
        throw std::invalid_argument(message);
      }

    int device_id = world_rank % n_devices;
#  ifdef DFTEFE_WITH_DEVICE_CUDA
    cudaSetDevice(device_id);
#  endif
  }

} // end of namespace dftefe

#endif
