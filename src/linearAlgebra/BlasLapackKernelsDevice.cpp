#ifdef DFTEFE_WITH_DEVICE
#  include <utils/DeviceUtils.h>
#  include <utils/DeviceTypeConfig.h>
#  include <utils/DeviceKernelLauncherHelpers.h>
#  include <utils/DeviceAPICalls.h>
#  include <utils/DeviceDataTypeOverloads.h>
#  include <utils/DeviceTypeConfigHalfPrec.h>
#  include <linearAlgebra/BlasLapackKernels.h>
#  include <linearAlgebra/BlasLapack.h>
#  include <complex>
#  include <algorithm>
namespace dftefe
{
  namespace linearAlgebra
  {
    namespace blasLapack
    {
      namespace
      {
        template <typename ValueType1, typename ValueType2, typename ValueType3>
        DFTEFE_CREATE_KERNEL(
            void,
            ascaleDeviceKernel,
            {
              for (size_type i = globalThreadId; i < size;
                  i += nThreadsPerBlock * nThreadBlock)
                {
                  z[i] = utils::mult(alpha, x[i]);
                }
            },
            const size_type   size,
            const ValueType1  alpha,
            const ValueType2 *x,
            ValueType3 *      z);

        template <typename ValueType1, typename ValueType2, typename ValueType3>
        DFTEFE_CREATE_KERNEL(
            void,
            reciprocalXDeviceKernel,
            {
              for (size_type i = globalThreadId; i < size;
                  i += nThreadsPerBlock * nThreadBlock)
                {
                  z[i] = utils::div(alpha, x[i]);
                }
            },
            const size_type   size,
            const ValueType1  alpha,
            const ValueType2 *x,
            ValueType3 *      z);

        template <typename ValueType1, typename ValueType2, typename ValueType3>
        DFTEFE_CREATE_KERNEL(
            void,
            hadamardProductDeviceKernel,
            {
              for (size_type i = globalThreadId; i < size;
                  i += nThreadsPerBlock * nThreadBlock)
                {
                  z[i] = utils::mult(x[i], y[i]);
                }
            },
            const size_type   size,
            const ValueType1 *x,
            const ValueType2 *y,
            ValueType3 *      z);

        template <typename ValueType1, typename ValueType2, typename ValueType3>
        DFTEFE_CREATE_KERNEL(
            void,
            hadamardProductDeviceKernelConj,
            {
              for (size_type i = globalThreadId; i < size;
                  i += nThreadsPerBlock * nThreadBlock)
                {
                  z[i] = utils::mult(utils::conj(x[i]), y[i]);
                }
            },
            const size_type   size,
            const ValueType1 *x,
            const ValueType2 *y,
            ValueType3 *      z);

        template <typename ValueType1, typename ValueType2, typename ValueType3>
        DFTEFE_CREATE_KERNEL(
            void,
            hadamardProductDeviceKernelConjConj,
            {
              for (size_type i = globalThreadId; i < size;
                  i += nThreadsPerBlock * nThreadBlock)
                {
                  z[i] = utils::mult(utils::conj(x[i]),
                                         utils::conj(y[i]));
                }
            },
            const size_type   size,
            const ValueType1 *x,
            const ValueType2 *y,
            ValueType3 *      z);

        template <typename ValueType1, typename ValueType2, typename ValueType3>
        DFTEFE_CREATE_KERNEL(
            void,
            khatriRaoProductColMajorDeviceKernel,
            {
              const size_type totalSize = sizeJ * sizeI * sizeK;
              for (size_type kij = globalThreadId; kij < totalSize;
                  kij += nThreadsPerBlock * nThreadBlock)
                {
                  const size_type k     = kij / (sizeI * sizeJ);
                  const size_type ijRem = kij - k * sizeI * sizeJ;
                  const size_type i     = ijRem / sizeJ;
                  const size_type j     = ijRem - i * sizeJ;
                  Z[kij] = utils::mult(A[k * sizeI + i], B[k * sizeJ + j]);
                }
            },
            const size_type   sizeI,
            const size_type   sizeJ,
            const size_type   sizeK,
            const ValueType1 *A,
            const ValueType2 *B,
            ValueType3 *      Z);

        template <typename ValueType1, typename ValueType2, typename ValueType3>
        DFTEFE_CREATE_KERNEL(
            void,
            khatriRaoProductRowMajorDeviceKernel,
            {
              const size_type totalSize = sizeJ * sizeI * sizeK;
              for (size_type jik = globalThreadId; jik < totalSize;
                  jik += nThreadsPerBlock * nThreadBlock)
                {
                  const size_type j     = jik / (sizeK * sizeI);
                  const size_type ikRem = jik - j * sizeK * sizeI;
                  const size_type i     = ikRem / sizeK;
                  const size_type k     = ikRem - i * sizeK;
                  Z[jik] = utils::mult(A[i * sizeK + k], B[j * sizeK + k]);
                }
            },
            const size_type   sizeI,
            const size_type   sizeJ,
            const size_type   sizeK,
            const ValueType1 *A,
            const ValueType2 *B,
            ValueType3 *      Z);

        template <typename ValueType1, typename ValueType2, typename ValueType3>
        DFTEFE_CREATE_KERNEL(
            void,
            axpbyDeviceKernel,
            {
              for (size_type i = globalThreadId; i < size;
                  i += nThreadsPerBlock * nThreadBlock)
                {
                  z[i] = utils::add(utils::mult(alpha, x[i]),
                                            utils::mult(beta, y[i]));
                }
            },
            const size_type   size,
            const ValueType3  alpha,
            const ValueType1 *x,
            const ValueType3  beta,
            const ValueType2 *y,
            ValueType3 *      z);

        template <typename ValueType>
        DFTEFE_CREATE_KERNEL(
            void,
            absSquareEntriesDeviceKernel,
            {
              for (size_type i = globalThreadId; i < size;
                  i += nThreadsPerBlock * nThreadBlock)
                {
                  const double temp = utils::abs(x[i]);
                  y[i]              = temp * temp;
                }
            },
            const size_type  size,
            const ValueType *x,
            double *         y);

      } // namespace


      template <typename ValueType1, typename ValueType2>
      void
      KernelsTwoValueTypes<ValueType1,
                           ValueType2,
                           utils::MemorySpace::DEVICE>::
        ascale(const size_type                      size,
               const ValueType1                     alpha,
               const ValueType2 *                   x,
               scalar_type<ValueType1, ValueType2> *z,
               LinAlgOpContext<utils::MemorySpace::DEVICE> &context)
      {
        DFTEFE_LAUNCH_KERNEL(
          ascaleDeviceKernel,
          size / utils::DEVICE_BLOCK_SIZE + 1,
          utils::DEVICE_BLOCK_SIZE,
          context.getBlasStream(),
          size,
          utils::makeDataTypeDeviceCompatible(alpha),
          utils::makeDataTypeDeviceCompatible(x),
          utils::makeDataTypeDeviceCompatible(z));
      }

      template <typename ValueType1, typename ValueType2>
      void
      KernelsTwoValueTypes<ValueType1,
                           ValueType2,
                           utils::MemorySpace::DEVICE>::
        reciprocalX(const size_type                      size,
                    const ValueType1                     alpha,
                    const ValueType2 *                   x,
                    scalar_type<ValueType1, ValueType2> *z,
                    LinAlgOpContext<utils::MemorySpace::DEVICE> &context)
      {
        DFTEFE_LAUNCH_KERNEL(
          reciprocalXDeviceKernel,
          size / utils::DEVICE_BLOCK_SIZE + 1,
          utils::DEVICE_BLOCK_SIZE,
          context.getBlasStream(),
          size,
          utils::makeDataTypeDeviceCompatible(alpha),
          utils::makeDataTypeDeviceCompatible(x),
          utils::makeDataTypeDeviceCompatible(z));
      }

      template <typename ValueType1, typename ValueType2>
      void
      KernelsTwoValueTypes<ValueType1,
                           ValueType2,
                           utils::MemorySpace::DEVICE>::
        hadamardProduct(const size_type                      size,
                        const ValueType1 *                   x,
                        const ValueType2 *                   y,
                        scalar_type<ValueType1, ValueType2> *z,
                        LinAlgOpContext<utils::MemorySpace::DEVICE> &context)
      {
        DFTEFE_LAUNCH_KERNEL(
          hadamardProductDeviceKernel,
          size / utils::DEVICE_BLOCK_SIZE + 1,
          utils::DEVICE_BLOCK_SIZE,
          context.getBlasStream(),
          size,
          utils::makeDataTypeDeviceCompatible(x),
          utils::makeDataTypeDeviceCompatible(y),
          utils::makeDataTypeDeviceCompatible(z));
      }

      template <typename ValueType1, typename ValueType2>
      void
      KernelsTwoValueTypes<ValueType1,
                           ValueType2,
                           utils::MemorySpace::DEVICE>::
        hadamardProduct(const size_type                      size,
                        const ValueType1 *                   x,
                        const ValueType2 *                   y,
                        const ScalarOp &                     opx,
                        const ScalarOp &                     opy,
                        scalar_type<ValueType1, ValueType2> *z,
                        LinAlgOpContext<utils::MemorySpace::DEVICE> &context)
      {
        if (opx == ScalarOp::Identity && opy == ScalarOp::Identity)
          {
            DFTEFE_LAUNCH_KERNEL(
              hadamardProductDeviceKernel,
              size / utils::DEVICE_BLOCK_SIZE + 1,
              utils::DEVICE_BLOCK_SIZE,
              context.getBlasStream(),
              size,
              utils::makeDataTypeDeviceCompatible(x),
              utils::makeDataTypeDeviceCompatible(y),
              utils::makeDataTypeDeviceCompatible(z));
          }

        else if (opx == ScalarOp::Identity && opy == ScalarOp::Conj)
          {
            //
            // @note hadamardProductDeviceKernelConj takes the conjgate of
            // the first entry. In order to take the conjugate of second entry,
            // we flip x and y
            //
            DFTEFE_LAUNCH_KERNEL(
              hadamardProductDeviceKernelConj,
              size / utils::DEVICE_BLOCK_SIZE + 1,
              utils::DEVICE_BLOCK_SIZE,
              context.getBlasStream(),
              size,
              utils::makeDataTypeDeviceCompatible(y),
              utils::makeDataTypeDeviceCompatible(x),
              utils::makeDataTypeDeviceCompatible(z));
          }

        else if (opx == ScalarOp::Conj && opy == ScalarOp::Identity)
          {
            DFTEFE_LAUNCH_KERNEL(
              hadamardProductDeviceKernelConj,
              size / utils::DEVICE_BLOCK_SIZE + 1,
              utils::DEVICE_BLOCK_SIZE,
              context.getBlasStream(),
              size,
              utils::makeDataTypeDeviceCompatible(x),
              utils::makeDataTypeDeviceCompatible(y),
              utils::makeDataTypeDeviceCompatible(z));
          }

        else
          {
            DFTEFE_LAUNCH_KERNEL(
              hadamardProductDeviceKernelConjConj,
              size / utils::DEVICE_BLOCK_SIZE + 1,
              utils::DEVICE_BLOCK_SIZE,
              context.getBlasStream(),
              size,
              utils::makeDataTypeDeviceCompatible(x),
              utils::makeDataTypeDeviceCompatible(y),
              utils::makeDataTypeDeviceCompatible(z));
          }
      }

      template <typename ValueType1,
                typename ValueType2>
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
                               LinAlgOpContext<utils::MemorySpace::DEVICE> &       context)
      {
        utils::throwException(
          false,
          "scaleStridedVarBatched() is not implemented for utils::MemorySpace::DEVICE .... ");
      }

      template <typename ValueType1,
                typename ValueType2>
      void
        axpbyBlocked(const size_type                            size,
                     const size_type                            blockSize,
                     const scalar_type<ValueType1, ValueType2>  alpha1,
                     const scalar_type<ValueType1, ValueType2> *alpha,
                     const ValueType1 *                         x,
                     const scalar_type<ValueType1, ValueType2>  beta1,
                     const scalar_type<ValueType1, ValueType2> *beta,
                     const ValueType2 *                         y,
                     scalar_type<ValueType1, ValueType2> *      z,
                     LinAlgOpContext<dftefe::utils::MemorySpace::DEVICE> &       context)
      {
        utils::throwException(
          false,
          "axpbyBlocked() is not implemented for utils::MemorySpace::DEVICE .... ");
      }

      template <typename ValueType1,
                typename ValueType2>
      void
      ascale(size_type                            size,
              ValueType1                           alpha,
              const ValueType2 *                   x,
              const ScalarOp &                     opalpha,
              const ScalarOp &                     opx,
              scalar_type<ValueType1, ValueType2> *z,
              LinAlgOpContext<dftefe::utils::MemorySpace::DEVICE> &       context)
      {
        utils::throwException(
          false,
          "ascale() is not implemented for utils::MemorySpace::DEVICE .... ");
      }        

      template <typename ValueType1, typename ValueType2>
      void
      KernelsTwoValueTypes<ValueType1,
                           ValueType2,
                           utils::MemorySpace::DEVICE>::
        khatriRaoProduct(const Layout                         layout,
                         const size_type                      sizeI,
                         const size_type                      sizeJ,
                         const size_type                      sizeK,
                         const ValueType1 *                   A,
                         const ValueType2 *                   B,
                         scalar_type<ValueType1, ValueType2> *Z,
                         LinAlgOpContext<utils::MemorySpace::DEVICE> &context)
      {
        if (layout == Layout::ColMajor)
          {
            DFTEFE_LAUNCH_KERNEL(
              khatriRaoProductColMajorDeviceKernel,
              (sizeI * sizeJ * sizeK) / utils::DEVICE_BLOCK_SIZE + 1,
              utils::DEVICE_BLOCK_SIZE,
              context.getBlasStream(),
              sizeI,
              sizeJ,
              sizeK,
              utils::makeDataTypeDeviceCompatible(A),
              utils::makeDataTypeDeviceCompatible(B),
              utils::makeDataTypeDeviceCompatible(Z));
          }
        else if (layout == Layout::RowMajor)
          {
            DFTEFE_LAUNCH_KERNEL(
              khatriRaoProductRowMajorDeviceKernel,
              (sizeI * sizeJ * sizeK) / utils::DEVICE_BLOCK_SIZE + 1,
              utils::DEVICE_BLOCK_SIZE,
              context.getBlasStream(),
              sizeI,
              sizeJ,
              sizeK,
              utils::makeDataTypeDeviceCompatible(A),
              utils::makeDataTypeDeviceCompatible(B),
              utils::makeDataTypeDeviceCompatible(Z));
          }
      }

      template <typename ValueType1, typename ValueType2>
      void
      KernelsTwoValueTypes<ValueType1,
                           ValueType2,
                           utils::MemorySpace::DEVICE>::
        axpby(const size_type                           size,
              const scalar_type<ValueType1, ValueType2> alpha,
              const ValueType1 *                        x,
              const scalar_type<ValueType1, ValueType2> beta,
              const ValueType2 *                        y,
              scalar_type<ValueType1, ValueType2> *     z,
              LinAlgOpContext<utils::MemorySpace::DEVICE> &context)
      {
        DFTEFE_LAUNCH_KERNEL(
          axpbyDeviceKernel,
          size / utils::DEVICE_BLOCK_SIZE + 1,
          utils::DEVICE_BLOCK_SIZE,
          context.getBlasStream(),
          size,
          utils::makeDataTypeDeviceCompatible(alpha),
          utils::makeDataTypeDeviceCompatible(x),
          utils::makeDataTypeDeviceCompatible(beta),
          utils::makeDataTypeDeviceCompatible(y),
          utils::makeDataTypeDeviceCompatible(z));
      }


      template <typename ValueType1, typename ValueType2>
      void
      KernelsTwoValueTypes<ValueType1,
                           ValueType2,
                           utils::MemorySpace::DEVICE>::
        dotMultiVector(
          const size_type                      vecSize,
          const size_type                      numVec,
          const ValueType1 *                   multiVecDataX,
          const ValueType2 *                   multiVecDataY,
          const ScalarOp &                     opX,
          const ScalarOp &                     opY,
          scalar_type<ValueType1, ValueType2> *multiVecDotProduct,
          LinAlgOpContext<utils::MemorySpace::DEVICE> &context)
      {
        utils::MemoryStorage<scalar_type<ValueType1, ValueType2>,
                                     utils::MemorySpace::DEVICE>
          onesVecDevice(vecSize, 1.0);
        utils::MemoryStorage<scalar_type<ValueType1, ValueType2>,
                                     utils::MemorySpace::DEVICE>
          hadamardProductDevice(vecSize * numVec, 0.0);

        hadamardProduct(vecSize * numVec,
                        multiVecDataX,
                        multiVecDataY,
                        opX,
                        opY,
                        hadamardProductDevice.data(),
                        context);

        gemm<scalar_type<ValueType1, ValueType2>,
             scalar_type<ValueType1, ValueType2>,
             utils::MemorySpace::DEVICE>('N',
                                                 'T',
                                                 1,
                                                 numVec,
                                                 vecSize,
                                                 1.0,
                                                 onesVecDevice.data(),
                                                 1,
                                                 hadamardProductDevice.data(),
                                                 numVec,
                                                 1.0,
                                                 multiVecDotProduct,
                                                 1,
                                                 context);
      }


      template <typename ValueType>
      std::vector<double>
      KernelsOneValueType<ValueType, utils::MemorySpace::DEVICE>::
        amaxsMultiVector(const size_type                     vecSize,
                       const size_type                     numVec,
                       ValueType const *             multiVecData,
                       LinAlgOpContext<utils::MemorySpace::DEVICE> &context)
      {
        std::vector<double> amaxs(numVec, 0);

        utils::throwException(
          false,
          "amaxsMultiVector() is not implemented for utils::MemorySpace::DEVICE .... ");
        return amaxs;
      }


      template <typename ValueType>
      std::vector<double>
      KernelsOneValueType<ValueType, utils::MemorySpace::DEVICE>::
        nrms2MultiVector(
          size_type                                            vecSize,
          size_type                                            numVec,
          ValueType const *                                    multiVecData,
          LinAlgOpContext<utils::MemorySpace::DEVICE> &context)
      {
        std::vector<double> nrms2(numVec, 0);

        utils::MemoryStorage<double, utils::MemorySpace::DEVICE>
          nrmsSqVecDevice(numVec, 0.0);
        utils::MemoryStorage<double, utils::MemorySpace::DEVICE>
          onesVecDevice(vecSize, 1.0);
        utils::MemoryStorage<double, utils::MemorySpace::DEVICE>
          squaredEntriesDevice(vecSize * numVec, 0.0);

        DFTEFE_LAUNCH_KERNEL(
          absSquareEntriesDeviceKernel,
          (vecSize * numVec) / utils::DEVICE_BLOCK_SIZE + 1,
          utils::DEVICE_BLOCK_SIZE,
          context.getBlasStream(),
          vecSize * numVec,
          utils::makeDataTypeDeviceCompatible(multiVecData),
          utils::makeDataTypeDeviceCompatible(
            squaredEntriesDevice.begin()));


        gemm<double, double, utils::MemorySpace::DEVICE>(
          'N',
          'T',
          1,
          numVec,
          vecSize,
          1.0,
          onesVecDevice.data(),
          1,
          squaredEntriesDevice.data(),
          numVec,
          1.0,
          nrmsSqVecDevice.data(),
          1,
          context);


        nrmsSqVecDevice.copyTo<utils::MemorySpace::DEVICE>(&nrms2[0]);

        for (size_type i = 0; i < numVec; i++)
          nrms2[i] = std::sqrt(nrms2[i]);

        return nrms2;
      }

#  define EXPLICITLY_INSTANTIATE_2T(T1, T2, M) \
    template class KernelsTwoValueTypes<T1, T2, M>;

#  define EXPLICITLY_INSTANTIATE_1T(T, M) \
    template class KernelsOneValueType<T, M>;


      EXPLICITLY_INSTANTIATE_1T(float, utils::MemorySpace::DEVICE);
      EXPLICITLY_INSTANTIATE_1T(double, utils::MemorySpace::DEVICE);
      EXPLICITLY_INSTANTIATE_1T(std::complex<float>,
                                utils::MemorySpace::DEVICE);
      EXPLICITLY_INSTANTIATE_1T(std::complex<double>,
                                utils::MemorySpace::DEVICE);


      EXPLICITLY_INSTANTIATE_2T(float,
                                float,
                                utils::MemorySpace::DEVICE);
      EXPLICITLY_INSTANTIATE_2T(float,
                                double,
                                utils::MemorySpace::DEVICE);
      EXPLICITLY_INSTANTIATE_2T(float,
                                std::complex<float>,
                                utils::MemorySpace::DEVICE);
      EXPLICITLY_INSTANTIATE_2T(float,
                                std::complex<double>,
                                utils::MemorySpace::DEVICE);
      EXPLICITLY_INSTANTIATE_2T(double,
                                float,
                                utils::MemorySpace::DEVICE);
      EXPLICITLY_INSTANTIATE_2T(double,
                                double,
                                utils::MemorySpace::DEVICE);
      EXPLICITLY_INSTANTIATE_2T(double,
                                std::complex<float>,
                                utils::MemorySpace::DEVICE);
      EXPLICITLY_INSTANTIATE_2T(double,
                                std::complex<double>,
                                utils::MemorySpace::DEVICE);
      EXPLICITLY_INSTANTIATE_2T(std::complex<float>,
                                float,
                                utils::MemorySpace::DEVICE);
      EXPLICITLY_INSTANTIATE_2T(std::complex<float>,
                                double,
                                utils::MemorySpace::DEVICE);
      EXPLICITLY_INSTANTIATE_2T(std::complex<float>,
                                std::complex<float>,
                                utils::MemorySpace::DEVICE);
      EXPLICITLY_INSTANTIATE_2T(std::complex<float>,
                                std::complex<double>,
                                utils::MemorySpace::DEVICE);
      EXPLICITLY_INSTANTIATE_2T(std::complex<double>,
                                float,
                                utils::MemorySpace::DEVICE);
      EXPLICITLY_INSTANTIATE_2T(std::complex<double>,
                                double,
                                utils::MemorySpace::DEVICE);
      EXPLICITLY_INSTANTIATE_2T(std::complex<double>,
                                std::complex<float>,
                                utils::MemorySpace::DEVICE);
      EXPLICITLY_INSTANTIATE_2T(std::complex<double>,
                                std::complex<double>,
                                utils::MemorySpace::DEVICE);
    } // namespace blasLapack
  }   // namespace linearAlgebra
} // namespace dftefe
#endif
