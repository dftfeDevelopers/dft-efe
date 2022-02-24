#ifndef dftefeMemoryOperations_h
#define dftefeMemoryOperations_h

#include <utils/MemoryManager.h>

namespace dftefe
{
  namespace linearAlgebra
  {
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    class MemoryOperations
    {
    public:
      /**
       * @brief Function template for architecture adaptable compound addition to perform v += u element-wise
       * @tparam ValueType the type of the number
       * @tparam memorySpace
       * @param[in] size size of array
       * @param[in] u array
       * @param[out] v array
       */
      static void
      add(size_type size, const ValueType *u, ValueType *v);

      /**
       * @brief Function template for architecture adaptable compound subtraction to perform v -= u element-wise
       * @tparam ValueType the type of the number
       * @tparam memorySpace
       * @param[in] size size of array
       * @param[in] u array
       * @param[out] v array
       */
      static void
      sub(size_type size, const ValueType *u, ValueType *v);


      /**
       * @brief compute \f$ l_2 \f$ norm of vector
       * @tparam ValueType the type of the number
       * @tparam memorySpace
       * @param[in] size size of array
       * @param[in] u array
       * @return \f$ l_2 \f$ norm of u as double type
       */
      static double
      l2Norm(size_type size, const ValueType *u);

      /**
       * @brief compute  \f$ l_{\inf} \f$ norm norm of vector
       * @tparam ValueType the type of the number
       * @tparam memorySpace
       * @param[in] size size of array
       * @param[in] u array
       * @return  \f$ l_{\inf} \f$ norm of u as double type
       */
      static double
      lInfNorm(size_type size, const ValueType *u);

      /**
       * @brief computes  the dot product between the norms of the vector
       * @tparam ValueType the type of the number
       * @tparam memorySpace
       * @param[in] size size of array
       * @param[in] v array
       * @param[in] u array
       * @return  u.v as double type
       */
      static double
      dotProduct(size_type size, const ValueType *v, const ValueType *u);


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

#ifdef DFTEFE_WITH_DEVICE
    template <typename ValueType>
    class MemoryOperations<ValueType, dftefe::utils::MemorySpace::DEVICE>
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

      static double
      l2Norm(size_type size, const ValueType *u);

      static double
      lInfNorm(size_type size, const ValueType *u);

      static double
      dotProduct(size_type size, const ValueType *v, const ValueType *u);

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

#endif // dftefeMemoryOperations_h
