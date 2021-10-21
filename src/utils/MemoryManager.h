#ifndef dftefeMemoryManager_h
#define dftefeMemoryManager_h

#include "TypeConfig.h"
#include "MemorySpaceType.h"

namespace dftefe
{
  namespace utils
  {
    //
    // MemoryManager
    //
    template <typename NumType, MemorySpace memorySpace>
    class MemoryManager
    {
    public:
      static void
      allocate(size_type size, NumType *ptr);

      static void
      deallocate(NumType *ptr);

      static void
      set(size_type size, NumType *ptr, NumType val);
    };

    template <typename NumType>
    class MemoryManager<NumType, MemorySpace::HOST>
    {
    public:
      static void
      allocate(size_type size, NumType *ptr);

      static void
      deallocate(NumType *ptr);

      static void
      set(size_type size, NumType *ptr, NumType val);
    };

    template <typename NumType>
    class MemoryManager<NumType, MemorySpace::DEVICE>
    {
    public:
      static void
      allocate(size_type size, NumType *ptr);

      static void
      deallocate(NumType *ptr);

      static void
      set(size_type size, NumType *ptr, NumType val);
    };
  } // namespace utils

} // namespace dftefe

#include "MemoryManager.t.cpp"

#endif
