#ifndef dftefeDeviceExceptions_cuh
#define dftefeDeviceExceptions_cuh


#define CUDA_API_CHECK(cmd)                         \
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

#endif // dftefeDeviceExceptions_cuh
