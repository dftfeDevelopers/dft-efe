#ifdef DFTEFE_WITH_DEVICE
#  include "DeviceUtils.h"
#  include "DeviceAPICalls.h"
#  include <stdexcept>
#  include <string>

namespace dftefe
{
  namespace utils
  {
    void
    DeviceUtils::initialize(const int world_rank)
    {
      int n_devices = 0;
      deviceGetDeviceCount(&n_devices);

      if (n_devices == 0)
        {
          std::string message = "Number of devices cannot be zero";
          throw std::invalid_argument(message);
        }

      int device_id = world_rank % n_devices;
      deviceSetDevice(device_id);
    }
  } // namespace utils

} // end of namespace dftefe
#endif
