#ifndef dftefeDeviceUtils_h
#define dftefeDeviceUtils_h


#ifdef DFTEFE_WITH_DEVICE

namespace dftefe
{
  class DeviceUtils
  {
  public:
    static void
    initialize(const int world_rank);
  };

} // namespace dftefe

#endif


#endif
