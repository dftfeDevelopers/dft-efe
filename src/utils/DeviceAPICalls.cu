#ifdef DFTEFE_WITH_DEVICE_CUDA
#  include "DeviceAPICalls.h"
#  include <stdio.h>
namespace dftefe
{
  namespace utils
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

    // todo
    //  namespace {
    //    __global__ void
    //    setValueKernel(void *devPtr, int value, size_t size)
    //    {
    //
    //      const unsigned int globalThreadId =
    //        blockIdx.x * blockDim.x + threadIdx.x;
    //      for (unsigned int i = globalThreadId; i < size;
    //           i += blockDim.x * gridDim.x)
    //        {
    //          v[i] = dftefe::utils::add(dftefe::utils::mult(a, u[i]), v[i]);
    //        }
    //    }
    //  }


    void
    deviceMalloc(void **devPtr, size_t size)
    {
      CUDACHECK(cudaMalloc(devPtr, size));
    }

    void
    deviceMemset(void *devPtr, size_t count)
    {
      CUDACHECK(cudaMemset(devPtr, 0, count));
    }

    // todo
    //    void
    //    deviceSetValue(void *devPtr, int value, size_t size)
    //    {
    //
    //    }

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
    void
    deviceMemcpyD2H(void *dst, const void *src, size_t count)
    {
      CUDACHECK(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost));
    }

    void
    deviceMemcpyD2D(void *dst, const void *src, size_t count)
    {
      CUDACHECK(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToDevice));
    }
    void
    deviceMemcpyH2D(void *dst, const void *src, size_t count)
    {
      CUDACHECK(cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice));
    }
  } // namespace utils
} // namespace dftefe
#endif
