#include <utils/Exceptions.h>
#include <utils/MemorySpaceType.h>
#include <linearAlgebra/BlasLapackKernels.h>
#include <type_traits>
#include <utils/MemoryStorage.h>

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


        return KernelsOneValueType<ValueType, memorySpace>::amaxsMultiVector(
          vecSize, numVec, multiVecData);
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

      template <typename ValueType1,
                typename ValueType2,
                dftefe::utils::MemorySpace memorySpace>
      void
      reciprocalX(size_type                            n,
                  const ValueType1                     alpha,
                  ValueType2 const *                   x,
                  scalar_type<ValueType1, ValueType2> *y,
                  LinAlgOpContext<memorySpace> &       context)
      {
        KernelsTwoValueTypes<ValueType1, ValueType2, memorySpace>::reciprocalX(
          n, alpha, x, y);
      }

      template <typename ValueType1,
                typename ValueType2,
                dftefe::utils::MemorySpace memorySpace>
      void
      ascale(const size_type                      n,
             const ValueType1                     alpha,
             const ValueType2 *                   x,
             scalar_type<ValueType1, ValueType2> *z,
             LinAlgOpContext<memorySpace> &       context)
      {
        KernelsTwoValueTypes<ValueType1, ValueType2, memorySpace>::ascale(n,
                                                                          alpha,
                                                                          x,
                                                                          z);
      }

      template <typename ValueType1,
                typename ValueType2,
                dftefe::utils::MemorySpace memorySpace>
      void
      hadamardProduct(const size_type                      n,
                      const ValueType1 *                   x,
                      const ValueType2 *                   y,
                      scalar_type<ValueType1, ValueType2> *z,
                      LinAlgOpContext<memorySpace> &       context)
      {
        KernelsTwoValueTypes<ValueType1, ValueType2, memorySpace>::
          hadamardProduct(n, x, y, z);
      }

      // template <typename ValueType1,
      //           typename ValueType2,
      //           dftefe::utils::MemorySpace memorySpace>
      // void
      //   blockedHadamardProduct(const size_type                      vecSize,
      //                   const size_type                      numComponents,
      //                   const ValueType1 *                   blockedInput,
      //                   const ValueType2 * singleVectorInput,
      //                   scalar_type<ValueType1, ValueType2> *blockedOutput,
      //                   LinAlgOpContext<memorySpace> &       context)
      // {
      //   KernelsTwoValueTypes<ValueType1, ValueType2, memorySpace>::
      //   blockedHadamardProduct(vecSize,
      //                   numComponents,
      //                   blockedInput,
      //                   singleVectorInput,
      //                   blockedOutput);
      // }

      template <typename ValueType1,
                typename ValueType2,
                typename dftefe::utils::MemorySpace memorySpace>
      void
      hadamardProduct(size_type                            n,
                      const ValueType1 *                   x,
                      const ValueType2 *                   y,
                      const ScalarOp &                     opx,
                      const ScalarOp &                     opy,
                      scalar_type<ValueType1, ValueType2> *z,
                      LinAlgOpContext<memorySpace> &       context)
      {
        KernelsTwoValueTypes<ValueType1, ValueType2, memorySpace>::
          hadamardProduct(n, x, y, opx, opy, z);
      }


      template <typename ValueType1,
                typename ValueType2,
                typename dftefe::utils::MemorySpace memorySpace>
      void
      khatriRaoProduct(const Layout                         layout,
                       const size_type                      sizeI,
                       const size_type                      sizeJ,
                       const size_type                      sizeK,
                       const ValueType1 *                   A,
                       const ValueType2 *                   B,
                       scalar_type<ValueType1, ValueType2> *Z,
                       LinAlgOpContext<memorySpace> &       context)
      {
        KernelsTwoValueTypes<ValueType1, ValueType2, memorySpace>::
          khatriRaoProduct(layout, sizeI, sizeJ, sizeK, A, B, Z);
      }

      template <typename ValueType1,
                typename ValueType2,
                typename dftefe::utils::MemorySpace memorySpace>
      void
      transposedKhatriRaoProduct(const Layout                         layout,
                                 const size_type                      sizeI,
                                 const size_type                      sizeJ,
                                 const size_type                      sizeK,
                                 const ValueType1 *                   A,
                                 const ValueType2 *                   B,
                                 scalar_type<ValueType1, ValueType2> *Z,
                                 LinAlgOpContext<memorySpace> &       context)
      {
        KernelsTwoValueTypes<ValueType1, ValueType2, memorySpace>::
          transposedKhatriRaoProduct(layout, sizeI, sizeJ, sizeK, A, B, Z);
      }


      template <typename ValueType1,
                typename ValueType2,
                dftefe::utils::MemorySpace memorySpace>
      void
      axpby(const size_type                           n,
            const scalar_type<ValueType1, ValueType2> alpha,
            const ValueType1 *                        x,
            const scalar_type<ValueType1, ValueType2> beta,
            const ValueType2 *                        y,
            scalar_type<ValueType1, ValueType2> *     z,
            LinAlgOpContext<memorySpace> &            context)
      {
        KernelsTwoValueTypes<ValueType1, ValueType2, memorySpace>::axpby(
          n, alpha, x, beta, y, z);
      }

      template <typename ValueType1,
                typename ValueType2,
                typename dftefe::utils::MemorySpace memorySpace>
      void
      axpbyBlocked(const size_type                            n,
                   const size_type                            blockSize,
                   const scalar_type<ValueType1, ValueType2> *alpha,
                   const ValueType1 *                         x,
                   const scalar_type<ValueType1, ValueType2> *beta,
                   const ValueType2 *                         y,
                   scalar_type<ValueType1, ValueType2> *      z,
                   LinAlgOpContext<memorySpace> &             context)
      {
        KernelsTwoValueTypes<ValueType1, ValueType2, memorySpace>::axpbyBlocked(
          n, blockSize, alpha, x, beta, y, z);
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

      template <typename ValueType1,
                typename ValueType2,
                dftefe::utils::MemorySpace memorySpace>
      void
      dotMultiVector(const size_type                      vecSize,
                     const size_type                      numVec,
                     const ValueType1 *                   multiVecDataX,
                     const ValueType2 *                   multiVecDataY,
                     const ScalarOp &                     opX,
                     const ScalarOp &                     opY,
                     scalar_type<ValueType1, ValueType2> *multiVecDotProduct,
                     LinAlgOpContext<memorySpace> &       context)
      {
        KernelsTwoValueTypes<ValueType1, ValueType2, memorySpace>::
          dotMultiVector(vecSize,
                         numVec,
                         multiVecDataX,
                         multiVecDataY,
                         opX,
                         opY,
                         multiVecDotProduct,
                         context);
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
        return KernelsOneValueType<ValueType, memorySpace>::nrms2MultiVector(
          vecSize, numVec, multiVecData, context);
      }


      template <typename ValueType1,
                typename ValueType2,
                dftefe::utils::MemorySpace memorySpace>
      void
      gemm(const Layout                              layout,
           const Op                                  transA,
           const Op                                  transB,
           const size_type                           m,
           const size_type                           n,
           const size_type                           k,
           const scalar_type<ValueType1, ValueType2> alpha,
           ValueType1 const *                        dA,
           const size_type                           ldda,
           ValueType2 const *                        dB,
           const size_type                           lddb,
           const scalar_type<ValueType1, ValueType2> beta,
           scalar_type<ValueType1, ValueType2> *     dC,
           const size_type                           lddc,
           LinAlgOpContext<memorySpace> &            context)
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

      template <typename ValueType1, typename ValueType2>
      void
      gemm(const Layout                                         layout,
           const Op                                             transA,
           const Op                                             transB,
           const size_type                                      m,
           const size_type                                      n,
           const size_type                                      k,
           const scalar_type<ValueType1, ValueType2>            alpha,
           ValueType1 const *                                   dA,
           const size_type                                      ldda,
           ValueType2 const *                                   dB,
           const size_type                                      lddb,
           const scalar_type<ValueType1, ValueType2>            beta,
           scalar_type<ValueType1, ValueType2> *                dC,
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



      template <typename ValueType1,
                typename ValueType2,
                dftefe::utils::MemorySpace memorySpace>
      void
      gemmStridedVarBatched(const Layout                              layout,
                            const size_type                           numMats,
                            const Op *                                transA,
                            const Op *                                transB,
                            const size_type *                         stridea,
                            const size_type *                         strideb,
                            const size_type *                         stridec,
                            const size_type *                         m,
                            const size_type *                         n,
                            const size_type *                         k,
                            const scalar_type<ValueType1, ValueType2> alpha,
                            const ValueType1 *                        dA,
                            const size_type *                         ldda,
                            const ValueType2 *                        dB,
                            const size_type *                         lddb,
                            const scalar_type<ValueType1, ValueType2> beta,
                            scalar_type<ValueType1, ValueType2> *     dC,
                            const size_type *                         lddc,
                            LinAlgOpContext<memorySpace> &            context)
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

      template <typename ValueType1, typename ValueType2>
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
        const scalar_type<ValueType1, ValueType2>            alpha,
        const ValueType1 *                                   dA,
        const size_type *                                    ldda,
        const ValueType2 *                                   dB,
        const size_type *                                    lddb,
        const scalar_type<ValueType1, ValueType2>            beta,
        scalar_type<ValueType1, ValueType2> *                dC,
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

      // ------------ lapack calls with memSpace template specialization-------
      template <typename ValueType,
                typename dftefe::utils::MemorySpace memorySpace>
      LapackError
      inverse(size_type n, ValueType *A, LinAlgOpContext<memorySpace> &context)
      {
        LapackError                                         returnVal;
        global_size_type                                    error1, error2;
        utils::MemoryStorage<global_size_type, memorySpace> ipiv(n);

        error1 = lapack::getrf(n, n, A, n, ipiv.data());

        error2 = lapack::getri(n, A, n, ipiv.data());

        if (error1 != 0 || error2 != 0)
          {
            returnVal = LapackErrorMsg::isSuccessAndMsg(
              LapackErrorCode::FAILED_DENSE_MATRIX_INVERSE);
            returnVal.msg +=
              std::to_string(error1) + ", " + std::to_string(error2) + " .";
          }
        else
          returnVal = LapackErrorMsg::isSuccessAndMsg(LapackErrorCode::SUCCESS);

        return returnVal;
      }

      template <typename ValueType,
                typename dftefe::utils::MemorySpace memorySpace>
      LapackError
      trtri(Uplo                          uplo,
            Diag                          diag,
            size_type                     n,
            ValueType *                   A,
            size_type                     lda,
            LinAlgOpContext<memorySpace> &context)
      {
        LapackError      returnVal;
        global_size_type error;

        error = lapack::trtri(uplo, diag, n, A, lda);

        if (error != 0)
          {
            returnVal = LapackErrorMsg::isSuccessAndMsg(
              LapackErrorCode::FAILED_TRIA_MATRIX_INVERSE);
            returnVal.msg += std::to_string(error) + " .";
          }
        else
          returnVal = LapackErrorMsg::isSuccessAndMsg(LapackErrorCode::SUCCESS);

        return returnVal;
      }

      template <typename ValueType,
                typename dftefe::utils::MemorySpace memorySpace>
      LapackError
      potrf(Uplo                          uplo,
            size_type                     n,
            ValueType *                   A,
            size_type                     lda,
            LinAlgOpContext<memorySpace> &context)
      {
        LapackError      returnVal;
        global_size_type error;

        error = lapack::potrf(uplo, n, A, lda);

        if (error != 0)
          {
            returnVal = LapackErrorMsg::isSuccessAndMsg(
              LapackErrorCode::FAILED_CHOLESKY_FACTORIZATION);
            returnVal.msg += std::to_string(error) + " .";
          }
        else
          returnVal = LapackErrorMsg::isSuccessAndMsg(LapackErrorCode::SUCCESS);

        return returnVal;
      }

      template <typename ValueType,
                typename dftefe::utils::MemorySpace memorySpace>
      LapackError
      heevd(Job                           jobz,
            Uplo                          uplo,
            size_type                     n,
            ValueType *                   A,
            size_type                     lda,
            real_type<ValueType> *        W,
            LinAlgOpContext<memorySpace> &context)
      {
        LapackError      returnVal;
        global_size_type error;

        error = lapack::heevd(jobz, uplo, n, A, lda, W);

        if (error != 0)
          {
            returnVal = LapackErrorMsg::isSuccessAndMsg(
              LapackErrorCode::FAILED_STANDARD_EIGENPROBLEM);
            returnVal.msg += std::to_string(error) + " .";
          }
        else
          returnVal = LapackErrorMsg::isSuccessAndMsg(LapackErrorCode::SUCCESS);

        return returnVal;
      }

      template <typename ValueType,
                typename dftefe::utils::MemorySpace memorySpace>
      LapackError
      hegv(size_type                     itype,
           Job                           jobz,
           Uplo                          uplo,
           size_type                     n,
           ValueType *                   A,
           size_type                     lda,
           ValueType *                   B,
           size_type                     ldb,
           real_type<ValueType> *        W,
           LinAlgOpContext<memorySpace> &context)
      {
        LapackError      returnVal;
        global_size_type error;

        error = lapack::hegv(itype, jobz, uplo, n, A, lda, B, ldb, W);

        if (error != 0)
          {
            returnVal = LapackErrorMsg::isSuccessAndMsg(
              LapackErrorCode::FAILED_GENERALIZED_EIGENPROBLEM);
            returnVal.msg += std::to_string(error) + " .";
          }
        else
          returnVal = LapackErrorMsg::isSuccessAndMsg(LapackErrorCode::SUCCESS);

        return returnVal;
      }

      // ------------ lapack calls with device template specialization-------
      template <typename ValueType>
      LapackError
      inverse(size_type                                            n,
              ValueType *                                          A,
              LinAlgOpContext<dftefe::utils::MemorySpace::DEVICE> &context)
      {
        LapackError      returnVal;
        global_size_type error1, error2;

        utils::MemoryStorage<global_size_type, dftefe::utils::MemorySpace::HOST>
          ipiv(n);
        utils::MemoryStorage<ValueType, dftefe::utils::MemorySpace::HOST> Ahost(
          n * n);

        utils::MemoryTransfer<utils::MemorySpace::HOST,
                              utils::MemorySpace::DEVICE>::copy(n * n,
                                                                Ahost,
                                                                A);

        error1 = lapack::getrf(n, n, Ahost, n, ipiv.data());

        error2 = lapack::getri(n, Ahost, n, ipiv.data());

        utils::MemoryTransfer<utils::MemorySpace::DEVICE,
                              utils::MemorySpace::HOST>::copy(n * n, A, Ahost);

        if (error1 != 0 || error2 != 0)
          {
            returnVal = LapackErrorMsg::isSuccessAndMsg(
              LapackErrorCode::FAILED_DENSE_MATRIX_INVERSE);
            returnVal.msg +=
              std::to_string(error1) + ", " + std::to_string(error2) + " .";
          }
        else
          returnVal = LapackErrorMsg::isSuccessAndMsg(LapackErrorCode::SUCCESS);

        return returnVal;
      }

      template <typename ValueType>
      LapackError
      trtri(Uplo                                                 uplo,
            Diag                                                 diag,
            size_type                                            n,
            ValueType *                                          A,
            size_type                                            lda,
            LinAlgOpContext<dftefe::utils::MemorySpace::DEVICE> &context)
      {
        LapackError      returnVal;
        global_size_type error;

        utils::MemoryStorage<ValueType, dftefe::utils::MemorySpace::HOST> Ahost(
          n * n);
        utils::MemoryTransfer<utils::MemorySpace::HOST,
                              utils::MemorySpace::DEVICE>::copy(n * n,
                                                                Ahost,
                                                                A);

        error = lapack::trtri(uplo, diag, n, Ahost, lda);

        utils::MemoryTransfer<utils::MemorySpace::DEVICE,
                              utils::MemorySpace::HOST>::copy(n * n, A, Ahost);

        if (error != 0)
          {
            returnVal = LapackErrorMsg::isSuccessAndMsg(
              LapackErrorCode::FAILED_TRIA_MATRIX_INVERSE);
            returnVal.msg += std::to_string(error) + " .";
          }
        else
          returnVal = LapackErrorMsg::isSuccessAndMsg(LapackErrorCode::SUCCESS);

        return returnVal;
      }

      template <typename ValueType>
      LapackError
      potrf(Uplo                                                 uplo,
            size_type                                            n,
            ValueType *                                          A,
            size_type                                            lda,
            LinAlgOpContext<dftefe::utils::MemorySpace::DEVICE> &context)
      {
        LapackError      returnVal;
        global_size_type error;

        utils::MemoryStorage<ValueType, dftefe::utils::MemorySpace::HOST> Ahost(
          n * n);
        utils::MemoryTransfer<utils::MemorySpace::HOST,
                              utils::MemorySpace::DEVICE>::copy(n * n,
                                                                Ahost,
                                                                A);

        error = lapack::potrf(uplo, n, Ahost, lda);

        utils::MemoryTransfer<utils::MemorySpace::DEVICE,
                              utils::MemorySpace::HOST>::copy(n * n, A, Ahost);

        if (error != 0)
          {
            returnVal = LapackErrorMsg::isSuccessAndMsg(
              LapackErrorCode::FAILED_CHOLESKY_FACTORIZATION);
            returnVal.msg += std::to_string(error) + " .";
          }
        else
          returnVal = LapackErrorMsg::isSuccessAndMsg(LapackErrorCode::SUCCESS);

        return returnVal;
      }

      template <typename ValueType>
      LapackError
      heevd(Job                                                  jobz,
            Uplo                                                 uplo,
            size_type                                            n,
            ValueType *                                          A,
            size_type                                            lda,
            real_type<ValueType> *                               W,
            LinAlgOpContext<dftefe::utils::MemorySpace::DEVICE> &context)
      {
        LapackError      returnVal;
        global_size_type error;

        utils::MemoryStorage<ValueType, dftefe::utils::MemorySpace::HOST> Ahost(
          n * n);
        utils::MemoryTransfer<utils::MemorySpace::HOST,
                              utils::MemorySpace::DEVICE>::copy(n * n,
                                                                Ahost,
                                                                A);
        utils::MemoryStorage<ValueType, dftefe::utils::MemorySpace::HOST> Whost(
          n);
        utils::MemoryTransfer<utils::MemorySpace::HOST,
                              utils::MemorySpace::DEVICE>::copy(n, Whost, W);

        error = lapack::heevd(jobz, uplo, n, Ahost, lda, Whost);

        utils::MemoryTransfer<utils::MemorySpace::DEVICE,
                              utils::MemorySpace::HOST>::copy(n * n, A, Ahost);
        utils::MemoryTransfer<utils::MemorySpace::DEVICE,
                              utils::MemorySpace::HOST>::copy(n, W, Whost);

        if (error != 0)
          {
            returnVal = LapackErrorMsg::isSuccessAndMsg(
              LapackErrorCode::FAILED_STANDARD_EIGENPROBLEM);
            returnVal.msg += std::to_string(error) + " .";
          }
        else
          returnVal = LapackErrorMsg::isSuccessAndMsg(LapackErrorCode::SUCCESS);

        return returnVal;
      }

      template <typename ValueType>
      LapackError
      hegv(size_type                                            itype,
           Job                                                  jobz,
           Uplo                                                 uplo,
           size_type                                            n,
           ValueType *                                          A,
           size_type                                            lda,
           ValueType *                                          B,
           size_type                                            ldb,
           real_type<ValueType> *                               W,
           LinAlgOpContext<dftefe::utils::MemorySpace::DEVICE> &context)
      {
        LapackError      returnVal;
        global_size_type error;

        utils::MemoryStorage<ValueType, dftefe::utils::MemorySpace::HOST> Ahost(
          n * n);
        utils::MemoryTransfer<utils::MemorySpace::HOST,
                              utils::MemorySpace::DEVICE>::copy(n * n,
                                                                Ahost,
                                                                A);
        utils::MemoryStorage<ValueType, dftefe::utils::MemorySpace::HOST> Bhost(
          n * n);
        utils::MemoryTransfer<utils::MemorySpace::HOST,
                              utils::MemorySpace::DEVICE>::copy(n * n,
                                                                Bhost,
                                                                B);
        utils::MemoryStorage<ValueType, dftefe::utils::MemorySpace::HOST> Whost(
          n);
        utils::MemoryTransfer<utils::MemorySpace::HOST,
                              utils::MemorySpace::DEVICE>::copy(n, Whost, W);

        error =
          lapack::hegv(itype, jobz, uplo, n, Ahost, lda, Bhost, ldb, Whost);

        utils::MemoryTransfer<utils::MemorySpace::DEVICE,
                              utils::MemorySpace::HOST>::copy(n * n, A, Ahost);
        utils::MemoryTransfer<utils::MemorySpace::DEVICE,
                              utils::MemorySpace::HOST>::copy(n * n, B, Bhost);
        utils::MemoryTransfer<utils::MemorySpace::DEVICE,
                              utils::MemorySpace::HOST>::copy(n, W, Whost);

        if (error != 0)
          {
            returnVal = LapackErrorMsg::isSuccessAndMsg(
              LapackErrorCode::FAILED_GENERALIZED_EIGENPROBLEM);
            returnVal.msg += std::to_string(error) + " .";
          }
        else
          returnVal = LapackErrorMsg::isSuccessAndMsg(LapackErrorCode::SUCCESS);

        return returnVal;
      }

    } // namespace blasLapack
  }   // namespace linearAlgebra

} // namespace dftefe
