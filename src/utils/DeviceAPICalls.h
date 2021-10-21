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
    deviceMemset(void *devPtr, size_t count);

    // todo
    /**
     * @brief
     * @param devPtr
     * @param value
     * @param size
     */
    template <typename NumberType>
    void
    deviceSetValue(void *devPtr, NumberType value, size_t size);

    void
    deviceFree(void *devPtr);

    /**
     * @brief Copy array from device to host
     * @param count The memory size in bytes of the array
     */
    void
    deviceMemcpyD2H(void *dst, const void *src, size_t count);

    /**
     * @brief Copy array from device to device
     * @param count The memory size in bytes of the array
     */
    void
    deviceMemcpyD2D(void *dst, const void *src, size_t count);

    /**
     * @brief Copy array from host to device
     * @param count The memory size in bytes of the array
     */
    void
    deviceMemcpyH2D(void *dst, const void *src, size_t count);
  } // namespace utils
} // namespace dftefe

#endif // dftefeDeviceAPICalls_H
