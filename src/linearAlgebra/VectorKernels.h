#ifndef dftefeVectorKernels_h
#define dftefeVectorKernels_h

#include "MemoryManager.h"

namespace dftefe
{
  namespace linearAlgebra
  {
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    class VectorKernels
    {
    public:
      /**
       * @brief Function template for architecture adaptable compound addition to perform v += u element-wisely
       * @tparam ValueType the type of the number
       * @tparam memorySpace dftefe::utils::MemorySpace::HOST of dftefe::utils::MemorySpace::DEVICE
       * @param u
       * @param v
       */
      static void
      add(size_type size, const ValueType *u, ValueType *v);

      /**
       * @brief Function template for architecture adaptable compound subtraction to perform v -= u element-wisely
       * @tparam ValueType the type of the number
       * @tparam memorySpace dftefe::utils::MemorySpace::HOST of dftefe::utils::MemorySpace::DEVICE
       * @param u
       * @param v
       */
      static void
      sub(size_type size, const ValueType *u, ValueType *v);

      /**
       * @brief Template for performing \f$ w = au + bv \f$
       * @param[in] size size of the array
       * @param[in] a scalar
       * @param[in] u array
       * @param[in] b scalar
       * @param[in] v array
       * @param[out] w array of the result
       */
      static void
      add(size_type        size,
          ValueType        a,
          const ValueType *u,
          ValueType        b,
          const ValueType *v,
          ValueType *      w);
    };

    template <typename ValueType>
    class VectorKernels<ValueType, dftefe::utils::MemorySpace::HOST>
    {
    public:
      /**
       * @tparam ValueType
       * @param u
       * @param v
       */
      static void
      add(size_type size, const ValueType *u, ValueType *v);

      /**
       * @tparam ValueType
       * @param u
       * @param v
       */
      static void
      sub(size_type size, const ValueType *u, ValueType *v);

      /**
       * @brief Performing \f$ w = au + bv \f$ for the host
       * @param[in] size size of the array
       * @param[in] a scalar
       * @param[in] u array
       * @param[in] b scalar
       * @param[in] v array
       * @param[out] w array of the result
       */
      static void
      add(size_type        size,
          ValueType        a,
          const ValueType *u,
          ValueType        b,
          const ValueType *v,
          ValueType *      w);
    };

#ifdef DFTEFE_WITH_DEVICE
    template <typename ValueType>
    class VectorKernels<ValueType, dftefe::utils::MemorySpace::DEVICE>
    {
    public:
      /**
       * @tparam ValueType
       * @param u
       * @param v
       */
      static void
      add(size_type size, const ValueType *u, ValueType *v);

      /**
       * @tparam ValueType
       * @param u
       * @param v
       */
      static void
      sub(size_type size, const ValueType *u, ValueType *v);

      /**
       * @brief Performing \f$ w = au + bv \f$ for the device
       * @param[in] size size of the array
       * @param[in] a scalar
       * @param[in] u array
       * @param[in] b scalar
       * @param[in] v array
       * @param[out] w array of the result
       */
      static void
      add(size_type        size,
          ValueType        a,
          const ValueType *u,
          ValueType        b,
          const ValueType *v,
          ValueType *      w);
    };
#endif

  } // namespace linearAlgebra
} // namespace dftefe

#endif // dftefeVectorKernels_h
