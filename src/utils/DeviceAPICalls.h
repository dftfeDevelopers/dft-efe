#ifdef DFTEFE_WITH_DEVICE

#  ifndef dftefeDeviceAPICalls_H
#    define dftefeDeviceAPICalls_H

#    include "TypeConfig.h"

namespace dftefe
{
  namespace utils
  {
    void
    deviceGetDeviceCount(int *count);

    void
    deviceSetDevice(int count);

    void
    deviceMalloc(void **devPtr, size_type size);

    void
    deviceMemset(void *devPtr, size_type count);

    /**
     * @brief
     * @param devPtr
     * @param value
     * @param size
     */
    template <typename ValueType>
    void
    deviceSetValue(ValueType *devPtr, ValueType value, size_type size);

    void
    deviceFree(void *devPtr);

    void
    hostPinnedMalloc(void **hostPtr, size_type size);

    void
    hostPinnedFree(void *hostPtr);

    /**
     * @brief Copy array from device to host
     * @param count The memory size in bytes of the array
     */
    void
    deviceMemcpyD2H(void *dst, const void *src, size_type count);

    /**
     * @brief Copy array from device to device
     * @param count The memory size in bytes of the array
     */
    void
    deviceMemcpyD2D(void *dst, const void *src, size_type count);

    /**
     * @brief Copy array from host to device
     * @param count The memory size in bytes of the array
     */
    void
    deviceMemcpyH2D(void *dst, const void *src, size_type count);
  } // namespace utils
} // namespace dftefe

#  endif // dftefeDeviceAPICalls_H
#endif   // DFTEFE_WITH_DEVICE