// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
// authors.
//
// This file is part of the DFT-FE code.
//
// The DFT-FE code is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the DFT-FE distribution.
//
// ---------------------------------------------------------------------

/*
 * @author Ian C. Lin., Sambit Das
 */
#  ifndef dftefeDeviceKernelLauncherHelpers_h
#    define dftefeDeviceKernelLauncherHelpers_h

#ifdef DFTEFE_WITH_DEVICE
#    ifdef DFTEFE_WITH_DEVICE_NVIDIA
namespace dftefe
{
  namespace utils
  {
    static const int DEVICE_WARP_SIZE      = 32;
    static const int DEVICE_MAX_BLOCK_SIZE = 1024;
    static const int DEVICE_BLOCK_SIZE     = 256;

  } // namespace utils
} // namespace dftefe

#    elif DFTEFE_WITH_DEVICE_AMD

namespace dftefe
{
  namespace utils
  {
    static const int DEVICE_WARP_SIZE      = 64;
    static const int DEVICE_MAX_BLOCK_SIZE = 1024;
    static const int DEVICE_BLOCK_SIZE     = 512;

  } // namespace utils
} // namespace dftefe

#    elif DFTEFE_WITH_DEVICE_INTEL

namespace dftefe
{
  namespace utils
  {
    static const int DEVICE_WARP_SIZE      = 32;
    static const int DEVICE_MAX_BLOCK_SIZE = 1024;
    static const int DEVICE_BLOCK_SIZE     = 256;

  } // namespace utils
} // namespace dftefe

#    endif
#    ifdef DFTEFE_WITH_DEVICE_LANG_CUDA
#      define DFTEFE_LAUNCH_KERNEL(kernel, grid, block, stream, ...) \
        do                                                           \
          {                                                          \
            kernel<<<grid, block, 0, stream>>>(__VA_ARGS__);         \
          }                                                          \
        while (0)
#    elif defined(DFTEFE_WITH_DEVICE_LANG_HIP)
#      define DFTEFE_LAUNCH_KERNEL(kernel, grid, block, stream, ...)         \
        do                                                                   \
          {                                                                  \
            hipLaunchKernelGGL(                                              \
              HIP_KERNEL_NAME(kernel), grid, block, 0, stream, __VA_ARGS__); \
          }                                                                  \
        while (0)
#    elif defined(DFTEFE_WITH_DEVICE_LANG_SYCL)
#      define DFTEFE_LAUNCH_KERNEL(kernel, grid, block, stream, ...)        \
        do                                                                  \
          {                                                                 \
            dftefe::utils::queueRegistry.find(stream)->second.parallel_for( \
              sycl::nd_range<1>((grid) * (block), block),                   \
              [=](sycl::nd_item<1> ind) { kernel(ind, __VA_ARGS__); });     \
          }                                                                 \
        while (0)
#    else
#      error \
        "No device backend defined (DFTEFE_WITH_DEVICE_LANG_CUDA or DFTEFE_WITH_DEVICE_LANG_HIP or DFTEFE_WITH_DEVICE_LANG_SYCL)"
#    endif

#    ifdef DFTEFE_WITH_DEVICE_LANG_CUDA
#      define DFTEFE_LAUNCH_KERNEL_SMEM_D(                                 \
        kernel, grid, block, smemtype, smemcount, stream, ...)             \
        do                                                                 \
          {                                                                \
            kernel<<<grid, block, smemcount * sizeof(smemtype), stream>>>( \
              __VA_ARGS__);                                                \
          }                                                                \
        while (0)
#    elif defined(DFTEFE_WITH_DEVICE_LANG_HIP)
#      define DFTEFE_LAUNCH_KERNEL_SMEM_D(                     \
        kernel, grid, block, smemtype, smemcount, stream, ...) \
        do                                                     \
          {                                                    \
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel),        \
                               grid,                           \
                               block,                          \
                               smemcount * sizeof(smemtype),   \
                               stream,                         \
                               __VA_ARGS__);                   \
          }                                                    \
        while (0)
#    elif defined(DFTEFE_WITH_DEVICE_LANG_SYCL)
#      define DFTEFE_LAUNCH_KERNEL_SMEM_D(                                   \
        kernel, grid, block, smemtype, smemcount, stream, ...)               \
        do                                                                   \
          {                                                                  \
            dftefe::utils::queueRegistry.find(stream)->second.submit(        \
              [=](sycl::handler &cgh) {                                      \
                sycl::local_accessor<smemtype, 1> SMem_acc(smemcount, cgh);  \
                cgh.parallel_for(sycl::nd_range<1>((grid) * (block), block), \
                                 [=](sycl::nd_item<1> ind) {                 \
                                   kernel(ind,                               \
                                          SMem_acc.get_pointer(),            \
                                          __VA_ARGS__);                      \
                                 });                                         \
              });                                                            \
          }                                                                  \
        while (0)
#    else
#      error \
        "No device backend defined (DFTEFE_WITH_DEVICE_LANG_CUDA or DFTEFE_WITH_DEVICE_LANG_HIP or DFTEFE_WITH_DEVICE_LANG_SYCL)"
#    endif

#    ifdef DFTEFE_WITH_DEVICE_LANG_CUDA
#      define DFTEFE_LAUNCH_KERNEL_SMEM_S(                     \
        kernel, grid, block, smemtype, smemcount, stream, ...) \
        do                                                     \
          {                                                    \
            kernel<<<grid, block, 0, stream>>>(__VA_ARGS__);   \
          }                                                    \
        while (0)
#    elif defined(DFTEFE_WITH_DEVICE_LANG_HIP)
#      define DFTEFE_LAUNCH_KERNEL_SMEM_S(                                   \
        kernel, grid, block, smemtype, smemcount, stream, ...)               \
        do                                                                   \
          {                                                                  \
            hipLaunchKernelGGL(                                              \
              HIP_KERNEL_NAME(kernel), grid, block, 0, stream, __VA_ARGS__); \
          }                                                                  \
        while (0)
#    elif defined(DFTEFE_WITH_DEVICE_LANG_SYCL)
#      define DFTEFE_LAUNCH_KERNEL_SMEM_S(                                   \
        kernel, grid, block, smemtype, smemcount, stream, ...)               \
        do                                                                   \
          {                                                                  \
            dftefe::utils::queueRegistry.find(stream)->second.submit(        \
              [=](sycl::handler &cgh) {                                      \
                sycl::local_accessor<smemtype, 1> SMem_acc(smemcount, cgh);  \
                cgh.parallel_for(sycl::nd_range<1>((grid) * (block), block), \
                                 [=](sycl::nd_item<1> ind) {                 \
                                   kernel(ind,                               \
                                          SMem_acc.get_pointer(),            \
                                          __VA_ARGS__);                      \
                                 });                                         \
              });                                                            \
          }                                                                  \
        while (0)
#    else
#      error \
        "No device backend defined (DFTEFE_WITH_DEVICE_LANG_CUDA or DFTEFE_WITH_DEVICE_LANG_HIP or DFTEFE_WITH_DEVICE_LANG_SYCL)"
#    endif


#    define DFTEFE_KERNEL_ARGUMENT(...) __VA_ARGS__


#    if defined(DFTEFE_WITH_DEVICE_LANG_CUDA) || \
      defined(DFTEFE_WITH_DEVICE_LANG_HIP)
#      define DFTEFE_CREATE_KERNEL(RET, NAME, BODY, ...) \
        __global__ RET NAME(__VA_ARGS__)                 \
        {                                                \
          const size_type globalThreadId =               \
            blockIdx.x * blockDim.x + threadIdx.x;       \
          const size_type nThreadsPerBlock = blockDim.x; \
          const size_type nThreadBlock     = gridDim.x;  \
          BODY                                           \
        }
#    elif defined(DFTEFE_WITH_DEVICE_LANG_SYCL)
#      define DFTEFE_CREATE_KERNEL(RET, NAME, BODY, ...)             \
        RET NAME(sycl::nd_item<1> ind, __VA_ARGS__)                  \
        {                                                            \
          const size_type globalThreadId   = ind.get_global_id(0);   \
          const size_type nThreadsPerBlock = ind.get_local_range(0); \
          const size_type nThreadBlock     = ind.get_group_range(0); \
          BODY                                                       \
        }
#    else
#      error \
        "No device backend defined (DFTEFE_WITH_DEVICE_LANG_CUDA or DFTEFE_WITH_DEVICE_LANG_HIP or DFTEFE_WITH_DEVICE_LANG_SYCL)"
#    endif

#    if defined(DFTEFE_WITH_DEVICE_LANG_CUDA) || \
      defined(DFTEFE_WITH_DEVICE_LANG_HIP)
#      define DFTEFE_CREATE_KERNEL_SMEM_D(SMEMTYPE, RET, NAME, BODY, ...) \
        __global__ RET NAME(__VA_ARGS__)                                  \
        {                                                                 \
          extern __shared__ SMEMTYPE smem[];                              \
          const size_type            globalThreadId =                     \
            blockIdx.x * blockDim.x + threadIdx.x;                        \
          const size_type threadId         = threadIdx.x;                 \
          const size_type blockId          = blockIdx.x;                  \
          const size_type nThreadsPerBlock = blockDim.x;                  \
          const size_type nThreadBlock     = gridDim.x;                   \
          BODY                                                            \
        }
#    elif defined(DFTEFE_WITH_DEVICE_LANG_SYCL)
#      define DFTEFE_CREATE_KERNEL_SMEM_D(SMEMTYPE, RET, NAME, BODY, ...) \
        RET NAME(sycl::nd_item<1> ind, SMEMTYPE *smem, __VA_ARGS__)       \
        {                                                                 \
          const size_type globalThreadId   = ind.get_global_id(0);        \
          const size_type threadId         = ind.get_local_id(0);         \
          const size_type blockId          = ind.get_group(0);            \
          const size_type nThreadsPerBlock = ind.get_local_range(0);      \
          const size_type nThreadBlock     = ind.get_group_range(0);      \
          BODY                                                            \
        }
#    else
#      error \
        "No device backend defined (DFTEFE_WITH_DEVICE_LANG_CUDA or DFTEFE_WITH_DEVICE_LANG_HIP or DFTEFE_WITH_DEVICE_LANG_SYCL)"
#    endif

#    if defined(DFTEFE_WITH_DEVICE_LANG_CUDA) || \
      defined(DFTEFE_WITH_DEVICE_LANG_HIP)
#      define DFTEFE_CREATE_KERNEL_SMEM_S(                \
        SMEMTYPE, SMEMCOUNT, RET, NAME, BODY, ...)        \
        __global__ RET NAME(__VA_ARGS__)                  \
        {                                                 \
          __shared__ SMEMTYPE smem[SMEMCOUNT];            \
          const size_type     globalThreadId =            \
            blockIdx.x * blockDim.x + threadIdx.x;        \
          const size_type threadId         = threadIdx.x; \
          const size_type blockId          = blockIdx.x;  \
          const size_type nThreadsPerBlock = blockDim.x;  \
          const size_type nThreadBlock     = gridDim.x;   \
          BODY                                            \
        }
#    elif defined(DFTEFE_WITH_DEVICE_LANG_SYCL)
#      define DFTEFE_CREATE_KERNEL_SMEM_S(                           \
        SMEMTYPE, SMEMCOUNT, RET, NAME, BODY, ...)                   \
        RET NAME(sycl::nd_item<1> ind, SMEMTYPE *smem, __VA_ARGS__)  \
        {                                                            \
          const size_type globalThreadId   = ind.get_global_id(0);   \
          const size_type threadId         = ind.get_local_id(0);    \
          const size_type blockId          = ind.get_group(0);       \
          const size_type nThreadsPerBlock = ind.get_local_range(0); \
          const size_type nThreadBlock     = ind.get_group_range(0); \
          BODY                                                       \
        }
#    else
#      error \
        "No device backend defined (DFTEFE_WITH_DEVICE_LANG_CUDA or DFTEFE_WITH_DEVICE_LANG_HIP or DFTEFE_WITH_DEVICE_LANG_SYCL)"
#    endif


#    if defined(DFTEFE_WITH_DEVICE_LANG_CUDA) || \
      defined(DFTEFE_WITH_DEVICE_LANG_HIP)
#      define SYNCTHREADS __syncthreads()
#    elif defined(DFTEFE_WITH_DEVICE_LANG_SYCL)
#      define SYNCTHREADS sycl::group_barrier(ind.get_group());
#    else
#      error \
        "No device backend defined (DFTEFE_WITH_DEVICE_LANG_CUDA or DFTEFE_WITH_DEVICE_LANG_HIP or DFTEFE_WITH_DEVICE_LANG_SYCL)"
#    endif

#if defined(DFTEFE_WITH_DEVICE_LANG_CUDA) || \
    defined(DFTEFE_WITH_DEVICE_LANG_HIP)

  #define DFTEFE_DEVICE        __device__
  #define DFTEFE_HOST_DEVICE   __host__ __device__
  #define DFTEFE_FORCEINLINE   __forceinline__

#elif defined(DFTEFE_WITH_DEVICE_LANG_SYCL)

  // SYCL: no host/device qualifiers
  #define DFTEFE_DEVICE
  #define DFTEFE_HOST_DEVICE
  #if defined(__clang__)
    #define DFTEFE_FORCEINLINE inline __attribute__((always_inline))
  #else
    #define DFTEFE_FORCEINLINE inline
  #endif

#else
  #error "DFTEFE_WITH_DEVICE is set but no backend (CUDA/HIP/SYCL) defined"
#endif

#else

  #define DFTEFE_HOST_DEVICE
  #define DFTEFE_FORCEINLINE inline

#endif   // DFTEFE_WITH_DEVICE

#define DFTEFE_HOST_DEVICE_FUNC DFTEFE_FORCEINLINE DFTEFE_HOST_DEVICE

#endif // dftefeDeviceKernelLauncherHelpers_h