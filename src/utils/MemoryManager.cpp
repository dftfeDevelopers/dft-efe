#include "MemoryManager.h"

#include <cstring>

namespace dftefe
{
  template <typename NumType>
  NumType *
  MemoryManager<NumType, MemorySpace::HOST>::allocate(const size_type size)
  {
    NumType *tempPtr = new NumType[size];
    std::memset(tempPtr, 0, size * sizeof(NumType));
    return tempPtr;
  }

  template <typename NumType>
  void
  MemoryManager<NumType, MemorySpace::HOST>::deallocate(NumType *ptr)
  {
    if (ptr != nullptr)
      delete[] ptr;
  }

#ifdef DFTEFE_WITH_CUDA
  template <typename NumType>
  NumType *
  MemoryManager<NumType, MemorySpace::DEVICE_CUDA>::allocate(
    const size_type size)
  {
    NumType *tempPtr;
    cudaMalloc((void **)&tempPtr, size * sizeof(NumType));
    cudaMemset(tempPtr, 0, size * sizeof(NumType));

    return tempPtr;
  }

  template <typename NumType>
  void
  MemoryManager<NumType, MemorySpace::DEVICE_CUDA>::deallocate(NumType *ptr)
  {
    if (ptr != nullptr)
      cudaFree(ptr);
  }
#endif

  //
  // explicitly instantiating MemoryManager
  //
  template class MemoryManager<double, MemorySpace::HOST>;

#ifdef DFTEFE_WITH_CUDA
  template class MemoryManager<double, MemorySpace::DEVICE_CUDA>;
#endif

} // namespace dftefe
