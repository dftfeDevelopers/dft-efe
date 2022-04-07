#ifndef dftefeVectorKernels_h
#define dftefeVectorKernels_h

#include <utils/MemoryManager.h>
#include <vector>

namespace dftefe
{
  namespace linearAlgebra
  {
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    class VectorKernels
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
       * @brief compute \f$ l_2 \f$ norms of each vector of multi vector
       * @tparam ValueType the type of the number
       * @tparam memorySpace
       * @param[in] size size of vector
       * @param[in] size number of vectors
       * @param[in] u array
       * @return \f$ l_2 \f$ norm of u as double type
       */
      static std::vector<double>
      l2Norms(size_type size, size_type numVectors, const ValueType *u);

      /**
       * @brief compute  \f$ l_{\inf} \f$ norm of each vector of multi vector
       * @tparam ValueType the type of the number
       * @tparam memorySpace
       * @param[in] size size of vector
       * @param[in] size number of vectors
       * @param[in] u array
       * @return  \f$ l_{\inf} \f$ norm of u as double type
       */
      static std::vector<double>
      lInfNorms(size_type size, size_type numVectors, const ValueType *u);


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
    class VectorKernels<ValueType, dftefe::utils::MemorySpace::DEVICE>
    {
    public:
      static void
      add(size_type size, const ValueType *u, ValueType *v);

      static void
      sub(size_type size, const ValueType *u, ValueType *v);

      static double
      l2Norm(size_type size, const ValueType *u);

      static double
      lInfNorm(size_type size, const ValueType *u);

      static std::vector<double>
      l2Norms(size_type size, size_type numVectors, const ValueType *u);

      static std::vector<double>
      lInfNorms(size_type size, size_type numVectors, const ValueType *u);

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
