#ifndef dftefeDeviceAPICalls_H
#define dftefeDeviceAPICalls_H

#include <cstddef>

namespace dftefe
{
  namespace utils
  {
    void
    deviceMalloc(void **devPtr, size_t size);

    void
    deviceGetDeviceCount(int *count);

    void
    deviceSetDevice(int count);

    void
    deviceMemset(void *devPtr, int value, size_t count);

    void
    deviceFree(void *devPtr);
  } // namespace utils
} // namespace dftefe

#endif // dftefeDeviceAPICalls_H
