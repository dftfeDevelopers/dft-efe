#include "CUDAUtils.h"

#ifdef DFTEFE_WITH_CUDA
#  include <cuda_runtime.h>
#  include <stdexcept>
#  include <string>

namespace dftefe
{
  void
  CUDAUtils::initialize(const int world_rank)
  {
    int n_devices = 0;
    cudaGetDeviceCount(&n_devices);

    if (n_devices == 0)
      {
        std::string message = "Number of devices cannot be zero";
        throw std::invalid_argument(message);
      }

    int device_id = world_rank % n_devices;
    cudaSetDevice(device_id);
  }

} // end of namespace dftefe

#endif
