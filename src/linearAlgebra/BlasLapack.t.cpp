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
      asum(const size_type               n,
           ValueType const *             x,
           const size_type               incx,
           LinAlgOpContext<memorySpace> &context)
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
      amax(const size_type               n,
           ValueType const *             x,
           const size_type               incx,
           LinAlgOpContext<memorySpace> &context)
      {
        utils::throwException(
          memorySpace != dftefe::utils::MemorySpace::DEVICE,
          "amax() is not implemented for dftefe::utils::MemorySpace::DEVICE .... ");

        size_type outputIndex;
        outputIndex = blas::iamax(n, x, incx);
        return std::abs(*(x + outputIndex));
      }

      template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
      std::vector<double>
      amaxsMultiVector(const size_type               vecSize,
                       const size_type               numVec,
                       const ValueType *             multiVecData,
                       LinAlgOpContext<memorySpace> &context)
      {
        utils::throwException(
          memorySpace != dftefe::utils::MemorySpace::DEVICE,
          "amaxsMultiVector() is not implemented for dftefe::utils::MemorySpace::DEVICE .... ");


        return Kernels<ValueType, memorySpace>::amaxsMultiVector(vecSize,
                                                                 numVec,
                                                                 multiVecData);
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
           LinAlgOpContext<memorySpace> &            context)
      {
        utils::throwException(
          memorySpace != dftefe::utils::MemorySpace::DEVICE,
          "blas::axpy() is not implemented for dftefe::utils::MemorySpace::DEVICE .... ");
        blas::axpy(n, alpha, x, incx, y, incy);
      }


      template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
      void
      ascale(const size_type               n,
             const ValueType               alpha,
             const ValueType *             x,
             ValueType *                   z,
             LinAlgOpContext<memorySpace> &context)
      {
        Kernels<ValueType, memorySpace>::ascale(n, alpha, x, z);
      }

      template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
      void
      hadamardProduct(const size_type               n,
                      const ValueType *             x,
                      const ValueType *             y,
                      ValueType *                   z,
                      LinAlgOpContext<memorySpace> &context)
      {
        Kernels<ValueType, memorySpace>::hadamardProduct(n, x, y, z);
      }

      template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
      void
      axpby(const size_type               n,
            const ValueType               alpha,
            const ValueType *             x,
            const ValueType               beta,
            const ValueType *             y,
            ValueType *                   z,
            LinAlgOpContext<memorySpace> &context)
      {
        Kernels<ValueType, memorySpace>::axpby(n, alpha, x, beta, y, z);
      }


      template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
      void
      axpbyMultiVector(const size_type               vecSize,
                       const size_type               numVec,
                       const ValueType               alpha,
                       const ValueType *             x,
                       const ValueType               beta,
                       const ValueType *             y,
                       ValueType *                   z,
                       LinAlgOpContext<memorySpace> &context)
      {
        Kernels<ValueType, memorySpace>::axpbyMultiVector(
          vecSize, numVec, alpha, x, beta, y, z);
      }


      template <typename ValueType1,
                typename ValueType2,
                dftefe::utils::MemorySpace memorySpace>
      scalar_type<ValueType1, ValueType2>
      dot(const size_type               n,
          ValueType1 const *            x,
          const size_type               incx,
          ValueType2 const *            y,
          const size_type               incy,
          LinAlgOpContext<memorySpace> &context)
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
      nrm2(const size_type               n,
           ValueType const *             x,
           const size_type               incx,
           LinAlgOpContext<memorySpace> &context)
      {
        utils::throwException(
          memorySpace != dftefe::utils::MemorySpace::DEVICE,
          "blas::nrm2() is not implemented for dftefe::utils::MemorySpace::DEVICE .... ");
        real_type<ValueType> output;
        output = blas::nrm2(n, x, incx);
        return output;
      }


      template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
      std::vector<double>
      nrms2MultiVector(const size_type               vecSize,
                       const size_type               numVec,
                       const ValueType *             multiVecData,
                       LinAlgOpContext<memorySpace> &context)
      {
        return Kernels<ValueType, memorySpace>::nrms2MultiVector(
          vecSize, numVec, multiVecData, context.setBlasQueue());
      }


      template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
      void
      gemm(const Layout                  layout,
           const Op                      transA,
           const Op                      transB,
           const size_type               m,
           const size_type               n,
           const size_type               k,
           const ValueType               alpha,
           ValueType const *             dA,
           const size_type               ldda,
           ValueType const *             dB,
           const size_type               lddb,
           const ValueType               beta,
           ValueType *                   dC,
           const size_type               lddc,
           LinAlgOpContext<memorySpace> &context)
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
      gemm(const Layout                                         layout,
           const Op                                             transA,
           const Op                                             transB,
           const size_type                                      m,
           const size_type                                      n,
           const size_type                                      k,
           const ValueType                                      alpha,
           ValueType const *                                    dA,
           const size_type                                      ldda,
           ValueType const *                                    dB,
           const size_type                                      lddb,
           const ValueType                                      beta,
           ValueType *                                          dC,
           const size_type                                      lddc,
           LinAlgOpContext<dftefe::utils::MemorySpace::DEVICE> &context)
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
                   context.getBlasQueue());
      }



      template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
      void
      gemmStridedVarBatched(const Layout                  layout,
                            const size_type               numMats,
                            const Op *                    transA,
                            const Op *                    transB,
                            const size_type *             stridea,
                            const size_type *             strideb,
                            const size_type *             stridec,
                            const size_type *             m,
                            const size_type *             n,
                            const size_type *             k,
                            const ValueType               alpha,
                            const ValueType *             dA,
                            const size_type *             ldda,
                            const ValueType *             dB,
                            const size_type *             lddb,
                            const ValueType               beta,
                            ValueType *                   dC,
                            const size_type *             lddc,
                            LinAlgOpContext<memorySpace> &context)
      {
        size_type cumulativeA = 0;
        size_type cumulativeB = 0;
        size_type cumulativeC = 0;
        for (size_type ibatch = 0; ibatch < numMats; ++ibatch)
          {
            blas::gemm(layout,
                       *(transA + ibatch),
                       *(transB + ibatch),
                       *(m + ibatch),
                       *(n + ibatch),
                       *(k + ibatch),
                       alpha,
                       dA + cumulativeA,
                       *(ldda + ibatch),
                       dB + cumulativeB,
                       *(lddb + ibatch),
                       beta,
                       dC + cumulativeC,
                       *(lddc + ibatch));

            cumulativeA += *(stridea + ibatch);
            cumulativeB += *(strideb + ibatch);
            cumulativeC += *(stridec + ibatch);
          }
      }

      template <typename ValueType>
      void
      gemmStridedVarBatched(
        const Layout                                         layout,
        const size_type                                      numMats,
        const Op *                                           transA,
        const Op *                                           transB,
        const size_type *                                    stridea,
        const size_type *                                    strideb,
        const size_type *                                    stridec,
        const size_type *                                    m,
        const size_type *                                    n,
        const size_type *                                    k,
        const ValueType                                      alpha,
        const ValueType *                                    dA,
        const size_type *                                    ldda,
        const ValueType *                                    dB,
        const size_type *                                    lddb,
        const ValueType                                      beta,
        ValueType *                                          dC,
        const size_type *                                    lddc,
        LinAlgOpContext<dftefe::utils::MemorySpace::DEVICE> &context)
      {
        size_type cumulativeA = 0;
        size_type cumulativeB = 0;
        size_type cumulativeC = 0;
        for (size_type ibatch = 0; ibatch < numMats; ++ibatch)
          {
            blas::gemm(layout,
                       *(transA + ibatch),
                       *(transB + ibatch),
                       *(m + ibatch),
                       *(n + ibatch),
                       *(k + ibatch),
                       alpha,
                       dA + cumulativeA,
                       *(ldda + ibatch),
                       dB + cumulativeB,
                       *(lddb + ibatch),
                       beta,
                       dC + cumulativeC,
                       *(lddc + ibatch),
                       context.getBlasQueue());


            cumulativeA += *(stridea + ibatch);
            cumulativeB += *(strideb + ibatch);
            cumulativeC += *(stridec + ibatch);
          }
      }


    } // namespace blasLapack
  }   // namespace linearAlgebra

} // namespace dftefe
