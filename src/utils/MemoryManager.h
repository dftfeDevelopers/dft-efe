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
    template <typename ValueType, MemorySpace memorySpace>
    class MemoryManager
    {
    public:
      static void
      allocate(size_type size, ValueType **ptr);

      static void
      deallocate(ValueType *ptr);

      static void
      set(size_type size, ValueType *ptr, ValueType val);
    };

    template <typename ValueType>
    class MemoryManager<ValueType, MemorySpace::HOST>
    {
    public:
      static void
      allocate(size_type size, ValueType **ptr);

      static void
      deallocate(ValueType *ptr);

      static void
      set(size_type size, ValueType *ptr, ValueType val);
    };

#ifdef DFTEFE_WITH_DEVICE
    template <typename ValueType>
    class MemoryManager<ValueType, MemorySpace::HOST_PINNED>
    {
    public:
      static void
      allocate(size_type size, ValueType **ptr);

      static void
      deallocate(ValueType *ptr);

      static void
      set(size_type size, ValueType *ptr, ValueType val);
    };


    template <typename ValueType>
    class MemoryManager<ValueType, MemorySpace::DEVICE>
    {
    public:
      static void
      allocate(size_type size, ValueType **ptr);

      static void
      deallocate(ValueType *ptr);

      static void
      set(size_type size, ValueType *ptr, ValueType val);
    };
#endif // DFTEFE_WITH_DEVICE
  }    // namespace utils

} // namespace dftefe

#include "MemoryManager.t.cpp"

#endif
