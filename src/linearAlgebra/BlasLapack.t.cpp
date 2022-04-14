#include <utils/Exceptions.h>
#include <utils/MemorySpaceType.h>
#include <linearAlgebra/BlasLapackKernels.h>
#include <type_traits>

namespace dftefe
{
  namespace linearAlgebra
  {
    namespace blasLapack
    {
      template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
      real_type<ValueType>
      asum(const size_type             n,
           ValueType const *           x,
           const size_type             incx,
           blasQueueType<memorySpace> &blasQueue)
      {
        //      auto memorySpaceDevice = dftefe::utils::MemorySpace::DEVICE;
        utils::throwException(
          memorySpace != dftefe::utils::MemorySpace::DEVICE,
          "blas::asum() is not implemented for dftefe::utils::MemorySpace::DEVICE .... ");
        real_type<ValueType> output;
        output = blas::asum(n, x, incx);
        return output;
      }

      template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
      real_type<ValueType>
      amax(const size_type             n,
           ValueType const *           x,
           const size_type             incx,
           blasQueueType<memorySpace> &blasQueue)
      {
        utils::throwException(
          memorySpace != dftefe::utils::MemorySpace::DEVICE,
          "amax() is not implemented for dftefe::utils::MemorySpace::DEVICE .... ");

        size_type outputIndex;
        outputIndex = blas::iamax(n, x, incx);
        return std::abs(*(x + outputIndex));
      }

      template <typename ValueType1,
                typename ValueType2,
                dftefe::utils::MemorySpace memorySpace>
      void
      axpy(const size_type                           n,
           const scalar_type<ValueType1, ValueType2> alpha,
           ValueType1 const *                        x,
           const size_type                           incx,
           ValueType2 *                              y,
           const size_type                           incy,
           blasQueueType<memorySpace> &              blasQueue)
      {
        utils::throwException(
          memorySpace != dftefe::utils::MemorySpace::DEVICE,
          "blas::axpy() is not implemented for dftefe::utils::MemorySpace::DEVICE .... ");
        blas::axpy(n, alpha, x, incx, y, incy);
      }

      template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
      void
      axpby(const size_type             n,
            const ValueType             alpha,
            const ValueType *           x,
            const ValueType             beta,
            const ValueType *           y,
            ValueType *                 z,
            blasQueueType<memorySpace> &blasQueue)
      {
        BlasLapackKernels<ValueType, memorySpace>::axpby(
          n, alpha, x, beta, y, z);
      }


      template <typename ValueType1,
                typename ValueType2,
                dftefe::utils::MemorySpace memorySpace>
      scalar_type<ValueType1, ValueType2>
      dot(const size_type             n,
          ValueType1 const *          x,
          const size_type             incx,
          ValueType2 const *          y,
          const size_type             incy,
          blasQueueType<memorySpace> &blasQueue)
      {
        utils::throwException(
          memorySpace != dftefe::utils::MemorySpace::DEVICE,
          "blas::dot() is not implemented for dftefe::utils::MemorySpace::DEVICE .... ");

        scalar_type<ValueType1, ValueType2> output;
        output = blas::dot(n, x, incx, y, incy);
        return output;
      }

      template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
      real_type<ValueType>
      nrm2(const size_type             n,
           ValueType const *           x,
           const size_type             incx,
           blasQueueType<memorySpace> &blasQueue)
      {
        utils::throwException(
          memorySpace != dftefe::utils::MemorySpace::DEVICE,
          "blas::nrm2() is not implemented for dftefe::utils::MemorySpace::DEVICE .... ");
        real_type<ValueType> output;
        output = blas::nrm2(n, x, incx);
        return output;
      }


      template <typename ValueType>
      void
      gemm(const Layout                                     layout,
           const Op                                         transA,
           const Op                                         transB,
           const size_type                                  m,
           const size_type                                  n,
           const size_type                                  k,
           const ValueType                                  alpha,
           ValueType const *                                dA,
           const size_type                                  ldda,
           ValueType const *                                dB,
           const size_type                                  lddb,
           const ValueType                                  beta,
           ValueType *                                      dC,
           const size_type                                  lddc,
           blasQueueType<dftefe::utils::MemorySpace::HOST> &blasQueue)
      {
        blas::gemm(layout,
                   transA,
                   transB,
                   m,
                   n,
                   k,
                   alpha,
                   dA,
                   ldda,
                   dB,
                   lddb,
                   beta,
                   dC,
                   lddc);
      }

      template <typename ValueType>
      void
      gemm(const Layout                                       layout,
           const Op                                           transA,
           const Op                                           transB,
           const size_type                                    m,
           const size_type                                    n,
           const size_type                                    k,
           const ValueType                                    alpha,
           ValueType const *                                  dA,
           const size_type                                    ldda,
           ValueType const *                                  dB,
           const size_type                                    lddb,
           const ValueType                                    beta,
           ValueType *                                        dC,
           const size_type                                    lddc,
           blasQueueType<dftefe::utils::MemorySpace::DEVICE> &blasQueue)
      {
        blas::gemm(layout,
                   transA,
                   transB,
                   m,
                   n,
                   k,
                   alpha,
                   dA,
                   ldda,
                   dB,
                   lddb,
                   beta,
                   dC,
                   lddc,
                   blasQueue);
      }
    } // namespace blasLapack
  }   // namespace linearAlgebra

} // namespace dftefe
