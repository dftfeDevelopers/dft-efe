#ifdef DFTEFE_WITH_DEVICE_LANG_CUDA
#  include <utils/DeviceAPICalls.h>
#  include <stdio.h>
#  include <vector>
#  include <utils/DeviceDataTypeOverloads.cu.h>
#  include <utils/DeviceKernelLauncherHelpers.h>
#  include <utils/DeviceTypeConfigHalfPrec.h>
#  include <utils/Exceptions.h>
namespace dftefe
{
  namespace utils
  {
    namespace
    {
      template <typename ValueType>
      __global__ void
      setValueKernel(ValueType *devPtr, ValueType value, std::size_t size)
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

    deviceError_t
    deviceReset()
    {
      deviceError_t err = cudaDeviceReset();
      DEVICE_API_CHECK(err);
      return err;
    }

    deviceError_t
    deviceMemGetInfo(std::size_t *free, std::size_t *total)
    {
      deviceError_t err = cudaMemGetInfo(free, total);
      DEVICE_API_CHECK(err);
      return err;
    }

    deviceError_t
    getDeviceCount(int *count)
    {
      deviceError_t err = cudaGetDeviceCount(count);
      DEVICE_API_CHECK(err);
      return err;
    }

    deviceError_t
    getDevice(int *deviceId)
    {
      deviceError_t err = cudaGetDevice(deviceId);
      DEVICE_API_CHECK(err);
      return err;
    }

    deviceError_t
    setDevice(int deviceId)
    {
      deviceError_t err = cudaSetDevice(deviceId);
      DEVICE_API_CHECK(err);
      return err;
    }

    deviceError_t
    deviceMalloc(void **devPtr, std::size_t size)
    {
      deviceError_t err = cudaMalloc(devPtr, size);
      DEVICE_API_CHECK(err);
      return err;
    }

    deviceError_t
    deviceMemset(void *devPtr, int value, std::size_t count)
    {
      deviceError_t err = cudaMemset(devPtr, value, count);
      DEVICE_API_CHECK(err);
      return err;
    }

    template <typename ValueType>
    void
    deviceSetValue(ValueType *devPtr, ValueType value, std::size_t size)
    {
      setValueKernel<<<size / dftefe::utils::DEVICE_BLOCK_SIZE + 1,
                       dftefe::utils::DEVICE_BLOCK_SIZE>>>(
        makeDataTypeDeviceCompatible(devPtr),
        makeDataTypeDeviceCompatible(value),
        size);
    }

    template void
    deviceSetValue(bool *devPtr, bool value, std::size_t size);

    template void
    deviceSetValue(int *devPtr, int value, std::size_t size);

    template void
    deviceSetValue(long int *devPtr, long int value, std::size_t size);

    template void
    deviceSetValue(unsigned int *devPtr, unsigned int value, std::size_t size);

    template void
    deviceSetValue(unsigned long int *devPtr,
                   unsigned long int  value,
                   std::size_t          size);

    template void
    deviceSetValue(double *devPtr, double value, std::size_t size);

    template void
    deviceSetValue(float *devPtr, float value, std::size_t size);

    template void
    deviceSetValue(std::complex<float> *devPtr,
                   std::complex<float>  value,
                   std::size_t            size);

    template void
    deviceSetValue(std::complex<double> *devPtr,
                   std::complex<double>  value,
                   std::size_t             size);

    template void
    deviceSetValue(uint16_t *devPtr, uint16_t value, std::size_t size);

    template void
    deviceSetValue(std::complex<uint16_t> *devPtr,
                   std::complex<uint16_t>  value,
                   std::size_t               size);

    deviceError_t
    deviceFree(void *devPtr)
    {
      deviceError_t err = cudaFree(devPtr);
      DEVICE_API_CHECK(err);
      return err;
    }

    deviceError_t
    hostPinnedMalloc(void **hostPtr, std::size_t size)
    {
      deviceError_t err = cudaMallocHost(hostPtr, size);
      DEVICE_API_CHECK(err);
      return err;
    }

    deviceError_t
    hostPinnedFree(void *hostPtr)
    {
      deviceError_t err = cudaFreeHost(hostPtr);
      DEVICE_API_CHECK(err);
      return err;
    }

    deviceError_t
    deviceMemcpyD2H(void *dst, const void *src, std::size_t count)
    {
      deviceError_t err = cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost);
      DEVICE_API_CHECK(err);
      return err;
    }

    deviceError_t
    deviceMemcpyD2D(void *dst, const void *src, std::size_t count)
    {
      deviceError_t err = cudaMemcpy(dst, src, count, cudaMemcpyDeviceToDevice);
      DEVICE_API_CHECK(err);
      return err;
    }

    deviceError_t
    deviceMemcpyH2D(void *dst, const void *src, std::size_t count)
    {
      deviceError_t err = cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice);
      DEVICE_API_CHECK(err);
      return err;
    }

    deviceError_t
    deviceMemcpyD2H_2D(void *      dst,
                       std::size_t   dpitch,
                       const void *src,
                       std::size_t   spitch,
                       std::size_t   width,
                       std::size_t   height)
    {
      deviceError_t err = cudaMemcpy2D(
        dst, dpitch, src, spitch, width, height, cudaMemcpyDeviceToHost);
      DEVICE_API_CHECK(err);
      return err;
    }


    deviceError_t
    deviceMemcpyD2D_2D(void *      dst,
                       std::size_t   dpitch,
                       const void *src,
                       std::size_t   spitch,
                       std::size_t   width,
                       std::size_t   height)
    {
      deviceError_t err = cudaMemcpy2D(
        dst, dpitch, src, spitch, width, height, cudaMemcpyDeviceToDevice);
      DEVICE_API_CHECK(err);
      return err;
    }

    deviceError_t
    deviceMemcpyH2D_2D(void *      dst,
                       std::size_t   dpitch,
                       const void *src,
                       std::size_t   spitch,
                       std::size_t   width,
                       std::size_t   height)
    {
      deviceError_t err = cudaMemcpy2D(
        dst, dpitch, src, spitch, width, height, cudaMemcpyHostToDevice);
      DEVICE_API_CHECK(err);
      return err;
    }

    deviceError_t
    deviceSynchronize()
    {
      deviceError_t err = cudaDeviceSynchronize();
      DEVICE_API_CHECK(err);
      return err;
    }

    deviceError_t
    deviceMemcpyAsyncD2H(void *         dst,
                         const void *   src,
                         std::size_t      count,
                         deviceStream_t stream)
    {
      deviceError_t err =
        cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToHost, stream);
      DEVICE_API_CHECK(err);
      return err;
    }

    deviceError_t
    deviceMemcpyAsyncD2D(void *         dst,
                         const void *   src,
                         std::size_t      count,
                         deviceStream_t stream)
    {
      deviceError_t err =
        cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, stream);
      DEVICE_API_CHECK(err);
      return err;
    }

    deviceError_t
    deviceMemcpyAsyncH2D(void *         dst,
                         const void *   src,
                         std::size_t      count,
                         deviceStream_t stream)
    {
      deviceError_t err =
        cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice, stream);
      DEVICE_API_CHECK(err);
      return err;
    }

    deviceError_t
    deviceStreamCreate(deviceStream_t &pStream, const bool nonBlocking)
    {
      if (!nonBlocking)
        {
          deviceError_t err = cudaStreamCreate(&pStream);
          DEVICE_API_CHECK(err);
          return err;
        }
      else
        {
          int priority;
          cudaDeviceGetStreamPriorityRange(NULL, &priority);
          deviceError_t err =
            cudaStreamCreateWithPriority(&pStream,
                                         cudaStreamNonBlocking,
                                         priority);
          DEVICE_API_CHECK(err);
          return err;
        }
    }

    deviceError_t
    deviceStreamDestroy(deviceStream_t &stream)
    {
      deviceError_t err = cudaStreamDestroy(stream);
      DEVICE_API_CHECK(err);
      return err;
    }

    deviceError_t
    deviceStreamSynchronize(deviceStream_t &stream)
    {
      deviceError_t err = cudaStreamSynchronize(stream);
      DEVICE_API_CHECK(err);
      return err;
    }

    deviceError_t
    deviceEventCreate(deviceEvent_t &pEvent)
    {
      deviceError_t err = cudaEventCreate(&pEvent);
      DEVICE_API_CHECK(err);
      return err;
    }

    deviceError_t
    deviceEventDestroy(deviceEvent_t &event)
    {
      deviceError_t err = cudaEventDestroy(event);
      DEVICE_API_CHECK(err);
      return err;
    }

    deviceError_t
    deviceEventRecord(deviceEvent_t &event, deviceStream_t stream)
    {
      deviceError_t err = cudaEventRecord(event, stream);
      DEVICE_API_CHECK(err);
      return err;
    }

    deviceError_t
    deviceEventSynchronize(deviceEvent_t &event)
    {
      deviceError_t err = cudaEventSynchronize(event);
      DEVICE_API_CHECK(err);
      return err;
    }

    deviceError_t
    deviceStreamWaitEvent(deviceStream_t &stream,
                          deviceEvent_t & event,
                          unsigned int    flags)
    {
      deviceError_t err = cudaStreamWaitEvent(stream, event, flags);
      DEVICE_API_CHECK(err);
      return err;
    }

    void
    printPointerLocation(const void *ptr)
    {
      cudaPointerAttributes attr;
      cudaError_t           err = cudaPointerGetAttributes(&attr, ptr);

      if (err != cudaSuccess)
        {
          std::cout << "Not a CUDA pointer (likely host memory)\n"
                    << std::flush;
          return;
        }

#  if CUDART_VERSION >= 10000
      if (attr.type == cudaMemoryTypeDevice)
        std::cout << "Device memory\n" << std::flush;
      else if (attr.type == cudaMemoryTypeHost)
        std::cout << "Host (pinned) memory\n" << std::flush;
      else
        std::cout << "Unknown CUDA memory type\n" << std::flush;
#  else
      if (attr.memoryType == cudaMemoryTypeDevice)
        std::cout << "Device memory\n" << std::flush;
      else if (attr.memoryType == cudaMemoryTypeHost)
        std::cout << "Host memory\n" << std::flush;
#  endif
    }
  } // namespace utils
} // namespace dftefe
#endif
