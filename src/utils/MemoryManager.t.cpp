#include "DeviceAPICalls.h"
#include <cstring>
#include <algorithm>


namespace dftefe
{
  namespace utils
  {
    template <typename NumType>
    void
    MemoryManager<NumType, MemorySpace::HOST>::allocate(size_type size,
                                                        NumType **ptr)
    {
      *ptr = new NumType[size];
    }

    template <typename NumType>
    void
    MemoryManager<NumType, MemorySpace::HOST>::deallocate(NumType *ptr)
    {
      delete[] ptr;
    }

    // todo
    template <typename NumType>
    void
    MemoryManager<NumType, MemorySpace::HOST>::set(size_type size,
                                                   NumType  *ptr,
                                                   NumType   val)
    {
      for (int i = 0; i < size; ++i)
        {
          ptr[i] = val;
        }
    }

    template <typename NumType>
    void
    MemoryManager<NumType, MemorySpace::DEVICE>::allocate(size_type size,
                                                          NumType **ptr)
    {
      deviceMalloc((void **)ptr, size * sizeof(NumType));
      deviceMemset(*ptr, size * sizeof(NumType));
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
      // todo
      deviceMemset(ptr, size * sizeof(NumType));
    }
  } // namespace utils

} // namespace dftefe
