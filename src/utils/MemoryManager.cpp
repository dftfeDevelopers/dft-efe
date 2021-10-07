//#include "MemoryManager.h"

#include <cstring>
#include <complex>

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

#ifdef DFTEFE_WITH_DEVICE_CUDA
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
  /*
  template class MemoryManager<double, MemorySpace::HOST>;
  template class MemoryManager<float, MemorySpace::HOST>;
  template class MemoryManager<std::complex<double>, MemorySpace::HOST>;
  template class MemoryManager<std::complex<float>, MemorySpace::HOST>;

#ifdef DFTEFE_WITH_DEVICE_CUDA
  template class MemoryManager<double, MemorySpace::DEVICE_CUDA>;
  template class MemoryManager<float, MemorySpace::DEVICE_CUDA>;
  template class MemoryManager<std::complex<double>, MemorySpace::DEVICE_CUDA>;
  template class MemoryManager<std::complex<float>, MemorySpace::DEVICE_CUDA>;
#endif
  */

} // namespace dftefe
