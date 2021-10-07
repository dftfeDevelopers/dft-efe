#ifndef dftefeMemoryManager_h
#define dftefeMemoryManager_h

#include "TypeConfig.h"

namespace dftefe {
    //
    // MemorySpace
    //
    enum class MemorySpace {
        HOST, //
        DEVICE //
    };

    //
    // MemoryManager
    //
    template<typename NumType, MemorySpace memorySpace>
    class MemoryManager {
    public:
        static NumType *
        allocate(const size_type size);

        static void
        deallocate(NumType *ptr);
    };

    template<typename NumType>
    class MemoryManager<NumType, MemorySpace::HOST> {
    public:
        static NumType *
        allocate(const size_type size);

        static void
        deallocate(NumType *ptr);
    };

//#ifdef DFTEFE_WITH_DEVICE_CUDA
    template<typename NumType>
    class MemoryManager<NumType, MemorySpace::DEVICE> {
    public:
        static NumType *
        allocate(const size_type size);

        static void
        deallocate(NumType *ptr);
    };
//#endif

} // namespace dftefe

#include "MemoryManager.t.cpp"

#endif
