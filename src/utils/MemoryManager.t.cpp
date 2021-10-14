#include "DeviceAPICalls.h"
#include <cstring>


namespace dftefe
{
  namespace utils
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

    template <typename NumType>
    NumType *
    MemoryManager<NumType, MemorySpace::DEVICE>::allocate(const size_type size)
    {
      NumType *tempPtr;
      deviceMalloc((void **)&tempPtr, size * sizeof(NumType));
      deviceMemset(tempPtr, 0, size * sizeof(NumType));

      return tempPtr;
    }

    template <typename NumType>
    void
    MemoryManager<NumType, MemorySpace::DEVICE>::deallocate(NumType *ptr)
    {
      if (ptr != nullptr)
        {
          deviceFree(ptr);
        }
    }
  } // namespace utils

} // namespace dftefe
