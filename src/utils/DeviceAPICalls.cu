#ifdef DFTEFE_WITH_DEVICE_CUDA
#  include <utils/DeviceAPICalls.h>
#  include <stdio.h>
#  include <vector>
#  include <utils/DeviceDataTypeOverloads.cuh>
#  include <utils/DeviceKernelLauncher.h>
#  include <utils/Exceptions.h>
namespace dftefe
{
  namespace utils
  {
    namespace
    {
      template <typename ValueType>
      __global__ void
      setValueKernel(ValueType *devPtr, ValueType value, size_type size)
      {
        const unsigned int globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        for (unsigned int i = globalThreadId; i < size;
             i += blockDim.x * gridDim.x)
          {
            devPtr[i] = value;
          }
      }
    } // namespace

    void
    deviceGetDeviceCount(int *count)
    {
      CUDA_API_CHECK(cudaGetDeviceCount(count));
    }

    void
    deviceSetDevice(int count)
    {
      CUDA_API_CHECK(cudaSetDevice(count));
    }

    void
    deviceMalloc(void **devPtr, size_type size)
    {
      CUDA_API_CHECK(cudaMalloc(devPtr, size));
    }

    void
    deviceMemset(void *devPtr, size_type count)
    {
      CUDA_API_CHECK(cudaMemset(devPtr, 0, count));
    }

    template <typename ValueType>
    void
    deviceSetValue(ValueType *devPtr, ValueType value, size_type size)
    {
      setValueKernel<<<size / dftefe::utils::BLOCK_SIZE + 1,
                       dftefe::utils::BLOCK_SIZE>>>(
        makeDataTypeDeviceCompatible(devPtr),
        makeDataTypeDeviceCompatible(value),
        size);
    }

    template void
    deviceSetValue(size_type *devPtr, size_type value, size_type size);

    template void
    deviceSetValue(int *devPtr, int value, size_type size);

    template void
    deviceSetValue(double *devPtr, double value, size_type size);

    template void
    deviceSetValue(float *devPtr, float value, size_type size);

    template void
    deviceSetValue(std::complex<float> *devPtr,
                   std::complex<float>  value,
                   size_type            size);

    template void
    deviceSetValue(std::complex<double> *devPtr,
                   std::complex<double>  value,
                   size_type             size);

    void
    deviceFree(void *devPtr)
    {
      CUDA_API_CHECK(cudaFree(devPtr));
    }

    void
    hostPinnedMalloc(void **hostPtr, size_type size)
    {
      CUDA_API_CHECK(cudaMallocHost(hostPtr, size));
    }

    void
    hostPinnedFree(void *hostPtr)
    {
      CUDA_API_CHECK(cudaFreeHost(hostPtr));
    }

    void
    deviceMemcpyD2H(void *dst, const void *src, size_type count)
    {
      CUDA_API_CHECK(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost));
    }

    void
    deviceMemcpyD2D(void *dst, const void *src, size_type count)
    {
      CUDA_API_CHECK(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToDevice));
    }
    void
    deviceMemcpyH2D(void *dst, const void *src, size_type count)
    {
      CUDA_API_CHECK(cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice));
    }
  } // namespace utils
} // namespace dftefe
#endif
