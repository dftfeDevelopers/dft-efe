#ifndef dftefeKernels_h
#define dftefeKernels_h

#include <utils/MemoryManager.h>
#include <linearAlgebra/BlasLapackTypedef.h>
#include <blas.hh>
#include <vector>

namespace dftefe
{
  namespace linearAlgebra
  {
    namespace blasLapack
    {
      /**
       * @brief namespace class for BlasLapack kernels not present in blaspp.
       */
      template <dftefe::utils::MemorySpace memorySpace,
                typename ValueType1,
                typename ValueType2 = ValueType1>
      class Kernels
      {
      public:
        /**
         * @brief Template for performing \f$ z = \alpha x$
         * @param[in] size size of the array
         * @param[in] \f$ alpha \f$ scalar
         * @param[in] x array
         * @param[out] z array
         */
        static void
        ascale(size_type                            size,
               ValueType1                           alpha,
               const ValueType2 *                   x,
               scalar_type<ValueType1, ValueType2> *z);

        /**
         * @brief Template for performing \f$ z_i = x_i * y_i$
         * @param[in] size size of the array
         * @param[in] x array
         * @param[in] y array
         * @param[out] z array
         */
        static void
        hadamardProduct(size_type                            size,
                        const ValueType1 *                   x,
                        const ValueType2 *                   y,
                        scalar_type<ValueType1, ValueType2> *z);

        /**
         * @brief Template for performing \f$ z = \alpha x + \beta y \f$
         * @param[in] size size of the array
         * @param[in] \f$ alpha \f$ scalar
         * @param[in] x array
         * @param[in] \f$ beta \f$ scalar
         * @param[in] y array
         * @param[out] z array
         */
        static void
        axpby(size_type                            size,
              scalar_type<ValueType1, ValueType2>  alpha,
              const ValueType1 *                   x,
              scalar_type<ValueType1, ValueType2>  beta,
              const ValueType2 *                   y,
              scalar_type<ValueType1, ValueType2> *z);

        /**
         * @brief Template for computing \f$ l_{\inf} \f$ norms of all the numVec vectors in a multi Vector
         * @param[in] vecSize size of each vector
         * @param[in] numVec number of vectors in the multi Vector
         * @param[in] multiVecData multi vector data in row major format i.e.
         * vector index is the fastest index
         *
         * @return \f$ l_{\inf} \f$  norms of all the vectors
         */
        static std::vector<double>
        amaxsMultiVector(size_type         vecSize,
                         size_type         numVec,
                         const ValueType1 *multiVecData);

        /**
         * @brief Template for computing \f$ l_2 \f$ norms of all the numVec vectors in a multi Vector
         * @param[in] vecSize size of each vector
         * @param[in] numVec number of vectors in the multi Vector
         * @param[in] multiVecData multi vector data in row major format i.e.
         * vector index is the fastest index
         *
         * @return \f$ l_2 \f$  norms of all the vectors
         */
        static std::vector<double>
        nrms2MultiVector(size_type               vecSize,
                         size_type               numVec,
                         const ValueType1 *      multiVecData,
                         BlasQueue<memorySpace> &BlasQueue);
      };

#ifdef DFTEFE_WITH_DEVICE
      template <typename ValueType1, typename ValueType2>
      class Kernels<dftefe::utils::MemorySpace::DEVICE, ValueType1, ValueType2>
      {
      public:
        static void
        ascale(size_type                            size,
               ValueType1                           alpha,
               const ValueType2 *                   x,
               scalar_type<ValueType1, ValueType2> *z);


        static void
        hadamardProduct(size_type                            size,
                        const ValueType1 *                   x,
                        const ValueType2 *                   y,
                        scalar_type<ValueType1, ValueType2> *z);

        static void
        axpby(size_type                            size,
              scalar_type<ValueType1, ValueType2>  alpha,
              const ValueType1 *                   x,
              scalar_type<ValueType1, ValueType2>  beta,
              const ValueType2 *                   y,
              scalar_type<ValueType1, ValueType2> *z);

        static std::vector<double>
        amaxsMultiVector(size_type         vecSize,
                         size_type         numVec,
                         const ValueType1 *multiVecData);


        static std::vector<double>
        nrms2MultiVector(
          size_type                                      vecSize,
          size_type                                      numVec,
          const ValueType1 *                             multiVecData,
          BlasQueue<dftefe::utils::MemorySpace::DEVICE> &BlasQueue);
      };

#endif

    } // namespace blasLapack
  }   // namespace linearAlgebra
} // namespace dftefe

#endif // dftefeKernels_h
