#include "DeviceAPICalls.h"
#include <cstring>


namespace dftefe
{
  namespace utils
  {
    template <typename NumType>
    void
    MemoryManager<NumType, MemorySpace::HOST>::allocate(size_type size,
                                                        NumType  *ptr)
    {
      ptr = new NumType[size];
    }

    template <typename NumType>
    void
    MemoryManager<NumType, MemorySpace::HOST>::deallocate(NumType *ptr)
    {
      delete[] ptr;
    }

    template <typename NumType>
    void
    MemoryManager<NumType, MemorySpace::HOST>::set(size_type size,
                                                   NumType  *ptr,
                                                   NumType   val)
    {
      std::memset(ptr, val, size * sizeof(NumType));
    }

    template <typename NumType>
    void
    MemoryManager<NumType, MemorySpace::DEVICE>::allocate(size_type size,
                                                          NumType  *ptr)
    {
      deviceMalloc((void **)&ptr, size * sizeof(NumType));
      deviceMemset(ptr, 0, size * sizeof(NumType));
    }

    template <typename NumType>
    void
    MemoryManager<NumType, MemorySpace::DEVICE>::deallocate(NumType *ptr)
    {
      deviceFree(ptr);
    }

    template <typename NumType>
    void
    MemoryManager<NumType, MemorySpace::DEVICE>::set(size_type size,
                                                     NumType  *ptr,
                                                     NumType   val)
    {
      deviceMemset(ptr, val, size * sizeof(NumType));
    }
  } // namespace utils

} // namespace dftefe
