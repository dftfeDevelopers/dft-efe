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

      static void
      setZero(size_type size, ValueType *ptr);
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

      static void
      setZero(size_type size, ValueType *ptr);
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

      static void
      setZero(size_type size, ValueType *ptr);
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

      static void
      setZero(size_type size, ValueType *ptr);
    };
#endif // DFTEFE_WITH_DEVICE
  }    // namespace utils

} // namespace dftefe

#include "MemoryManager.t.cpp"

#endif
