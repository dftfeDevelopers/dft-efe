#ifdef DFTEFE_WITH_DEVICE
#  include "DeviceUtils.h"
#  include "DeviceAPICalls.h"
#  include "DeviceDataTypeOverloads.h"
#  include "DeviceKernelLauncherHelpers.h"
#  include <stdexcept>
#  include <string>
#  include<utils/Exceptions.h>

namespace dftefe
{
  namespace utils
  {
    void
    DeviceUtils::setupDevice(const int &mpi_rank)
    {
      int n_devices = 0;
      deviceError_t err = dftefe::utils::getDeviceCount(&n_devices);
      DEVICE_API_CHECK(err);
      if (n_devices == 0)
        {
          std::string message = "Number of devices cannot be zero";
          throw std::invalid_argument(message);
        }
      // std::cout<< "Number of Devices "<<n_devices<<std::endl;
      int device_id = mpi_rank % n_devices;
      // std::cout<<"Device Id: "<<device_id<<" Task Id
      // "<<dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)<<std::endl;
      err = dftefe::utils::setDevice(device_id);
      DEVICE_API_CHECK(err);
      // dftefe::Int device = 0;
      // dftefe::utils::getDevice(&device);
      // std::cout<< "Device Id currently used is "<<device<< " for taskId:
      // "<<dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)<<std::endl;
      err = dftefe::utils::deviceReset();
      DEVICE_API_CHECK(err);
    }
  } // namespace utils

} // end of namespace dftefe
#endif
