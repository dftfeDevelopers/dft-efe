#ifdef DFTEFE_WITH_DEVICE_CUDA
#include "DeviceAPICalls.h"
namespace dftefe {
    void deviceMalloc(void **devPtr, size_t size) {
        cudaMalloc(devPtr, size);
    }

    void deviceMemset(void *devPtr, int value, size_t count) {
        cudaMemset(devPtr, value, count);
    }

    void deviceFree(void *devPtr) {
        cudaFree(devPtr);
    }

    void deviceGetDeviceCount(int *count) {
        cudaGetDeviceCount(count);
    }

    void deviceSetDevice(int count) {
        cudaSetDevice(count);
    }
} // namespace dftfe
#endif


