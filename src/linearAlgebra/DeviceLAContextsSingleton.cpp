#ifdef DFTEFE_WITH_DEVICE
#  include "DeviceLAContextsSingleton.h"

namespace dftefe
{
  namespace linearAlgebra
  {
    DeviceLAContextsSingleton *
    DeviceLAContextsSingleton::getInstance()
    {
      if (d_instPtr == nullptr)
        {
          d_instPtr = new DeviceLAContextsSingleton();
        }
      return d_instPtr;
    }

    deviceBlasHandleType &
    DeviceLAContextsSingleton::getDeviceBlasHandle()
    {
      return d_blasHandle;
    }


    DeviceLAContextsSingleton *DeviceLAContextsSingleton::d_instPtr = nullptr;
  } // namespace linearAlgebra

} // end of namespace dftefe
#endif // DFTEFE_WITH_DEVICE
