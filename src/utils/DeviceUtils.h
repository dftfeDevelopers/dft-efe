#ifdef DFTEFE_WITH_DEVICE
#  ifndef dftefeDeviceUtils_h
#define dftefeDeviceUtils_h

namespace dftefe
{
  namespace utils
  {
    class DeviceUtils
    {
    public:
      static void
      initialize(int world_rank);
    };
  } // namespace utils

} // namespace dftefe

#  endif
#endif // DFTEFE_WITH_DEVICE
