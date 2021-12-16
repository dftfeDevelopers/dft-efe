#ifdef DFTEFE_WITH_DEVICE_CUDA
#  include "DeviceLAContextsSingleton.h"

namespace dftefe
{
  namespace linearAlgebra
  {
    void
    DeviceLAContextsSingleton::createDeviceBlasHandle()
    {
      cublasCreate(&d_blasHandle);
    }

    void
    DeviceLAContextsSingleton::destroyDeviceBlasHandle()
    {
      cublasDestroy(d_blasHandle);
    }
  } // namespace linearAlgebra

} // end of namespace dftefe
#endif // DFTEFE_WITH_DEVICE_CUDA
