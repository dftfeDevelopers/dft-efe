#ifdef DFTEFE_WITH_DEVICE_CUDA
#  include "DeviceAPICalls.h"
#  include <stdio.h>
namespace dftefe
{
#  define CUDACHECK(cmd)                              \
    do                                                \
      {                                               \
        cudaError_t e = cmd;                          \
        if (e != cudaSuccess)                         \
          {                                           \
            printf("Failed: Cuda error %s:%d '%s'\n", \
                   __FILE__,                          \
                   __LINE__,                          \
                   cudaGetErrorString(e));            \
            exit(EXIT_FAILURE);                       \
          }                                           \
      }                                               \
    while (0)


  void
  deviceMalloc(void **devPtr, size_t size)
  {
    CUDACHECK(cudaMalloc(devPtr, size));
  }

  void
  deviceMemset(void *devPtr, int value, size_t count)
  {
    CUDACHECK(cudaMemset(devPtr, value, count));
  }

  void
  deviceFree(void *devPtr)
  {
    CUDACHECK(cudaFree(devPtr));
  }

  void
  deviceGetDeviceCount(int *count)
  {
    CUDACHECK(cudaGetDeviceCount(count));
  }

  void
  deviceSetDevice(int count)
  {
    CUDACHECK(cudaSetDevice(count));
  }
} // namespace dftefe
#endif
