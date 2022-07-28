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

      template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
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
        ascale(size_type        size,
               ValueType        alpha,
               const ValueType *x,
               ValueType *      z);

        /**
         * @brief Template for performing \f$ z = 1 /x$, does not check if x[i] is zero
         * @param[in] size size of the array
         * @param[in] x array
         * @param[out] z array
         */
        static void
        reciprocalX(size_type        size,
                    const ValueType alpha,
                    const ValueType *x,
                    ValueType *      z);

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
        axpby(size_type        size,
              ValueType        alpha,
              const ValueType *x,
              ValueType        beta,
              const ValueType *y,
              ValueType *      z);


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
        amaxsMultiVector(size_type        vecSize,
                         size_type        numVec,
                         const ValueType *multiVecData);

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
                         const ValueType *       multiVecData,
                         BlasQueue<memorySpace> &BlasQueue);
      };


#ifdef DFTEFE_WITH_DEVICE
      template <typename ValueType>
      class Kernels<ValueType, dftefe::utils::MemorySpace::DEVICE>
      {
      public:
        static void
        ascale(size_type        size,
               ValueType        alpha,
               const ValueType *x,
               ValueType *      z);

        /*
        * @brief Template for performing \f$ z = 1 /x$, does not check if x[i] is zero
        * @param[in] size size of the array
        * @param[in] x array
        * @param[out] z array
        */
        static void
        reciprocalX(size_type        size,
                    const ValueType alpha,
                    const ValueType *x,
                    ValueType *      z);
        static void
        axpby(size_type        size,
              ValueType        alpha,
              const ValueType *x,
              ValueType        beta,
              const ValueType *y,
              ValueType *      z);


        static std::vector<double>
        amaxsMultiVector(size_type        vecSize,
                         size_type        numVec,
                         const ValueType *multiVecData);

        static std::vector<double>
        nrms2MultiVector(
          size_type                                      vecSize,
          size_type                                      numVec,
          const ValueType *                              multiVecData,
          BlasQueue<dftefe::utils::MemorySpace::DEVICE> &BlasQueue);
      };
#endif
    } // namespace blasLapack
  }   // namespace linearAlgebra
} // namespace dftefe

#endif // dftefeKernels_h
