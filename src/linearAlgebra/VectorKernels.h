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
       * @brief Function template for architecture adaptable compound addition to perform v += u element-wisely
       * @tparam NumberType the type of the number
       * @tparam memorySpace dftefe::utils::MemorySpace::HOST of dftefe::utils::MemorySpace::DEVICE
       * @param u
       * @param v
       */
      static void
      add(size_type size, const NumberType *u, NumberType *v);

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
      add(size_type         size,
          NumberType        a,
          const NumberType *u,
          NumberType        b,
          const NumberType *v,
          NumberType       *w);
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
      add(size_type         size,
          NumberType        a,
          const NumberType *u,
          NumberType        b,
          const NumberType *v,
          NumberType       *w);
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
      add(size_type         size,
          NumberType        a,
          const NumberType *u,
          NumberType        b,
          const NumberType *v,
          NumberType       *w);
    };
#endif

  } // namespace linearAlgebra
} // namespace dftefe

#endif // dftefeVectorKernels_h
