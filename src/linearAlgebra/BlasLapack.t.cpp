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
        real_type<ValueType> output;
        output = xasum<ValueType , memorySpace>(n, x, incx , context);
        return output;
      }

      template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
      real_type<ValueType>
      amax(const size_type               n,
           ValueType const *             x,
           const size_type               incx,
           LinAlgOpContext<memorySpace> &context)
      {
        size_type outputIndex;
        outputIndex = xiamax<ValueType , memorySpace>(n, x, incx , context);
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
        DFTEFE_AssertWithMsg(
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
                dftefe::utils::MemorySpace memorySpace>
      void
      scaleStridedVarBatched(const size_type                      numMats,
                             const Layout                         layout,
                             const ScalarOp &                     scalarOpA,
                             const ScalarOp &                     scalarOpB,
                             const size_type *                    stridea,
                             const size_type *                    strideb,
                             const size_type *                    stridec,
                             const size_type *                    m,
                             const size_type *                    n,
                             const size_type *                    k,
                             const ValueType1 *                   dA,
                             const ValueType2 *                   dB,
                             scalar_type<ValueType1, ValueType2> *dC,
                             LinAlgOpContext<memorySpace> &       context)
      {
        KernelsTwoValueTypes<ValueType1, ValueType2, memorySpace>::
          scaleStridedVarBatched(numMats,
                                 layout,
                                 scalarOpA,
                                 scalarOpB,
                                 stridea,
                                 strideb,
                                 stridec,
                                 m,
                                 n,
                                 k,
                                 dA,
                                 dB,
                                 dC,
                                 context);
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
                   const scalar_type<ValueType1, ValueType2>  alpha1,
                   const scalar_type<ValueType1, ValueType2> *alpha,
                   const ValueType1 *                         x,
                   const scalar_type<ValueType1, ValueType2>  beta1,
                   const scalar_type<ValueType1, ValueType2> *beta,
                   const ValueType2 *                         y,
                   scalar_type<ValueType1, ValueType2> *      z,
                   LinAlgOpContext<memorySpace> &             context)
      {
        KernelsTwoValueTypes<ValueType1, ValueType2, memorySpace>::axpbyBlocked(
          n, blockSize, alpha1, alpha, x, beta1, beta, y, z);
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
        scalar_type<ValueType1, ValueType2> output;
        output = xdot<ValueType1, ValueType2, memorySpace>(n, x, incx, y, incy, context);
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
        real_type<ValueType> output;
        output = xnrm2<ValueType, memorySpace>(n, x, incx , context);
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
      gemm(const char                                &transA,
           const char                                &transB,
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
        xgemm<ValueType1,
                ValueType2,
                memorySpace>(transA,
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
                   context);
      }

      template <typename ValueType1, typename ValueType2>
      void
      gemm(const char                                           &transA,
           const char                                           &transB,
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
        utils::throwException(
          false,
          "blasLapack::gemm() is not implemented for dftefe::utils::MemorySpace::DEVICE .... ");
      }



      template <typename ValueType1,
                typename ValueType2,
                dftefe::utils::MemorySpace memorySpace>
      void
      gemmStridedVarBatched(const size_type                           numMats,
                            const char   *                                transA,
                            const char   *                                transB,
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
            if (*(m + ibatch) > 0 && *(n + ibatch) > 0 && *(k + ibatch) > 0)
              xgemm<ValueType1,
                    ValueType2,
                    memorySpace>(
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
                        context);

            cumulativeA += *(stridea + ibatch);
            cumulativeB += *(strideb + ibatch);
            cumulativeC += *(stridec + ibatch);
          }
      }

      template <typename ValueType1, typename ValueType2>
      void
      gemmStridedVarBatched(const size_type                  numMats,
        const char   *                                         transA,
        const char   *                                         transB,
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
        utils::throwException(
          false,
          "blasLapack::gemmStridedVarBatched() is not implemented for dftefe::utils::MemorySpace::DEVICE .... ");
      }

      // ------------ lapack calls -------
      template <typename ValueType,
                typename dftefe::utils::MemorySpace memorySpace>
      LapackError
      inverse(size_type n, ValueType *A, LinAlgOpContext<memorySpace> &context)
      {
        LapackError                                  returnVal;
        global_size_type                             error1, error2;
        utils::MemoryStorage<LapackInt, memorySpace> ipiv(n);

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
      steqr(Job                           jobz,
            size_type                     n,
            real_type<ValueType> *        D,
            real_type<ValueType> *        E,
            ValueType *                   Z,
            size_type                     ldz,
            LinAlgOpContext<memorySpace> &context)
      {
        LapackError      returnVal;
        global_size_type error;

        error = lapack::steqr(jobz, n, D, E, Z, ldz);

        if (error != 0)
          {
            returnVal = LapackErrorMsg::isSuccessAndMsg(
              LapackErrorCode::FAILED_REAL_TRIDIAGONAL_EIGENPROBLEM);
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

      template <typename ValueType,
                typename dftefe::utils::MemorySpace memorySpace>
      LapackError
      gesv(size_type                     n,
           size_type                     nrhs,
           ValueType *                   A,
           size_type                     lda,
           LapackInt *                   ipiv,
           ValueType *                   B,
           size_type                     ldb,
           LinAlgOpContext<memorySpace> &context)
      {
        LapackError      returnVal;
        global_size_type error;

        error = lapack::gesv(n, nrhs, A, lda, ipiv, B, ldb);

        if (error != 0)
          {
            returnVal = LapackErrorMsg::isSuccessAndMsg(
              LapackErrorCode::FAILED_LINEAR_SYSTEM_SOLVE);
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
        utils::throwException(
          false,
          "inverse() is not implemented for dftefe::utils::MemorySpace::DEVICE .... ");
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
        utils::throwException(
          false,
          "trtri() is not implemented for dftefe::utils::MemorySpace::DEVICE .... ");
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
        utils::throwException(
          false,
          "potrf() is not implemented for dftefe::utils::MemorySpace::DEVICE .... ");
        return returnVal;
      }

      template <typename ValueType>
      LapackError
      steqr(Job                                                  jobz,
            size_type                                            n,
            real_type<ValueType> *                               D,
            real_type<ValueType> *                               E,
            ValueType *                                          Z,
            size_type                                            ldz,
            LinAlgOpContext<dftefe::utils::MemorySpace::DEVICE> &context)
      {
        LapackError      returnVal;
        utils::throwException(
          false,
          "steqr() is not implemented for dftefe::utils::MemorySpace::DEVICE .... ");
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
        utils::throwException(
          false,
          "heevd() is not implemented for dftefe::utils::MemorySpace::DEVICE .... ");
        return returnVal;
      }

      template <typename ValueType>
      LapackError
      heevr(Job                                                  jobz,
            Range                                                range,
            Uplo                                                 uplo,
            size_type                                            n,
            ValueType *                                          A,
            size_type                                            lda,
            real_type<ValueType>                                 vl,
            real_type<ValueType>                                 vu,
            size_type                                            il,
            size_type                                            iu,
            real_type<ValueType>                                 abstol,
            size_type                                            nfound,
            real_type<ValueType> *                               W,
            ValueType *                                          Z,
            size_type                                            ldz,
            LinAlgOpContext<dftefe::utils::MemorySpace::DEVICE> &context)
      {
        LapackError      returnVal;
        utils::throwException(
          false,
          "heevr() is not implemented for dftefe::utils::MemorySpace::DEVICE .... ");
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
        utils::throwException(
          false,
          "hegv() is not implemented for dftefe::utils::MemorySpace::DEVICE .... ");
        return returnVal;
      }

      template <typename ValueType>
      LapackError
      gesv(size_type                                            n,
           size_type                                            nrhs,
           ValueType *                                          A,
           size_type                                            lda,
           LapackInt *                                          ipiv,
           ValueType *                                          B,
           size_type                                            ldb,
           LinAlgOpContext<dftefe::utils::MemorySpace::DEVICE> &context)
      {
        LapackError      returnVal;
        utils::throwException(
          false,
          "gesv() is not implemented for dftefe::utils::MemorySpace::DEVICE .... ");
        return returnVal;
      }

    } // namespace blasLapack
  }   // namespace linearAlgebra

} // namespace dftefe
