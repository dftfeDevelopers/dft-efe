#ifndef dftefeVectorKernels_h
#define dftefeVectorKernels_h

#include "MemoryManager.h"

namespace dftefe
{
  namespace linearAlgebra
  {
    template <typename NumberType, dftefe::utils::MemorySpace memorySpace>
    class VectorKernels
    {
    public:
      /**
       * @tparam NumberType
       * @tparam memorySpace
       * @param u
       * @param v
       */
      static void
      add(size_type size, const NumberType *u, NumberType *v);
    };

    template <typename NumberType>
    class VectorKernels<NumberType, dftefe::utils::MemorySpace::HOST>
    {
    public:
      /**
       * @tparam NumberType
       * @param u
       * @param v
       */
      static void
      add(size_type size, const NumberType *u, NumberType *v);
    };

#ifdef DFTEFE_WITH_DEVICE
    template <typename NumberType>
    class VectorKernels<NumberType, dftefe::utils::MemorySpace::DEVICE>
    {
    public:
      /**
       * @tparam NumberType
       * @param u
       * @param v
       */
      static void
      add(size_type size, const NumberType *u, NumberType *v);
    };
#endif

  } // namespace linearAlgebra
} // namespace dftefe

#endif // dftefeVectorKernels_h
