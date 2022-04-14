#ifndef dftefeBlasLapackKernels_h
#define dftefeBlasLapackKernels_h

#include <utils/MemoryManager.h>
#include <vector>

namespace dftefe
{
  namespace linearAlgebra
  {
    /**
     * @brief namespace class for BlasLapack kernels not present in blaspp.
     */

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    class BlasLapackKernels
    {
    public:
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
    };


#ifdef DFTEFE_WITH_DEVICE
    template <typename ValueType>
    class BlasLapackKernels<ValueType, dftefe::utils::MemorySpace::DEVICE>
    {
    public:
      static void
      axpby(size_type        size,
            ValueType        alpha,
            const ValueType *x,
            ValueType        beta,
            const ValueType *y,
            ValueType *      z);
    };
#endif

  } // namespace linearAlgebra
} // namespace dftefe

#endif // dftefeBlasLapackKernels_h
