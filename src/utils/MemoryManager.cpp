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
    if ( ptr != nullptr)
    	delete[] ptr;
  }

  template <typename NumType>
  NumType *
  MemoryManager<NumType, MemorySpace::DEVICE_NVIDIA>::allocate(
    const size_type size)
  {
    NumType *tempPtr;
    // cudaMalloc((void **)&tempPtr, size * sizeof(NumType));
    // cudaMemset(tempPtr, 0, size * sizeof(NumType));

    return tempPtr;
  }

  template <typename NumType>
  void
  MemoryManager<NumType, MemorySpace::DEVICE_NVIDIA>::deallocate(NumType *ptr)
  {
// FIXME check if the pointer is not null before de allocating 
    // cudaFree(ptr);
  }


  template class MemoryManager<double, MemorySpace::HOST>;

} // namespace dftefe
