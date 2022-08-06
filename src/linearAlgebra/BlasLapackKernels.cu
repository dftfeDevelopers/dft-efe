#ifdef DFTEFE_WITH_DEVICE_CUDA
#  include <utils/DeviceKernelLauncher.h>
#  include <utils/DeviceDataTypeOverloads.cuh>
#  include <utils/DataTypeOverloads.h>
#  include <utils/MemoryTransfer.h>
#  include <utils/Exceptions.h>
#  include <linearAlgebra/BlasLapackKernels.h>
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
        __global__ void
        ascaleDeviceKernel(const size_type   size,
                           const ValueType1  alpha,
                           const ValueType2 *x,
                           ValueType3 *      z)
        {
          const size_type globalThreadId =
            blockIdx.x * blockDim.x + threadIdx.x;
          for (size_type i = globalThreadId; i < size;
               i += blockDim.x * gridDim.x)
            {
              z[i] = dftefe::utils::mult(alpha, x[i]);
            }
        }

        template <typename ValueType1, typename ValueType2, typename ValueType3>
        __global__ void
        hadamardProductDeviceKernel(const size_type   size,
                                    const ValueType1 *x,
                                    const ValueType2 *y,
                                    ValueType3 *      z)
        {
          const size_type globalThreadId =
            blockIdx.x * blockDim.x + threadIdx.x;
          for (size_type i = globalThreadId; i < size;
               i += blockDim.x * gridDim.x)
            {
              z[i] = dftefe::utils::mult(x[i], y[i]);
            }
        }

        template <typename ValueType1, typename ValueType2, typename ValueType3>
        __global__ void
        axpbyDeviceKernel(const size_type   size,
                          const ValueType3  alpha,
                          const ValueType1 *x,
                          const ValueType3  beta,
                          const ValueType2 *y,
                          ValueType3 *      z)
        {
          const size_type globalThreadId =
            blockIdx.x * blockDim.x + threadIdx.x;
          for (size_type i = globalThreadId; i < size;
               i += blockDim.x * gridDim.x)
            {
              z[i] = dftefe::utils::add(dftefe::utils::mult(alpha, x[i]),
                                        dftefe::utils::mult(beta, y[i]));
            }
        }


        template <typename ValueType>
        __global__ void
        absSquareEntriesDeviceKernel(const size_type  size,
                                     const ValueType *x,
                                     double *         y)
        {
          const size_type globalThreadId =
            blockIdx.x * blockDim.x + threadIdx.x;
          for (size_type i = globalThreadId; i < size;
               i += blockDim.x * gridDim.x)
            {
              const double temp = dftefe::utils::abs(x[i]);
              y[i]              = temp * temp;
            }
        }

      } // namespace


      template <typename ValueType1, typename ValueType2>
      void
      Kernels<dftefe::utils::MemorySpace::DEVICE, ValueType1, ValueType2>::
        ascale(const size_type                      size,
               const ValueType1                     alpha,
               const ValueType2 *                   x,
               scalar_type<ValueType1, ValueType2> *z)
      {
        ascaleDeviceKernel<<<size / dftefe::utils::BLOCK_SIZE + 1,
                             dftefe::utils::BLOCK_SIZE>>>(
          size,
          dftefe::utils::makeDataTypeDeviceCompatible(alpha),
          dftefe::utils::makeDataTypeDeviceCompatible(x),
          dftefe::utils::makeDataTypeDeviceCompatible(z));
      }

      template <typename ValueType1, typename ValueType2>
      void
      Kernels<dftefe::utils::MemorySpace::DEVICE, ValueType1, ValueType2>::
        hadamardProduct(const size_type                      size,
                        const ValueType1 *                   x,
                        const ValueType2 *                   y,
                        scalar_type<ValueType1, ValueType2> *z)
      {
        hadamardProductDeviceKernel<<<size / dftefe::utils::BLOCK_SIZE + 1,
                                      dftefe::utils::BLOCK_SIZE>>>(
          size,
          dftefe::utils::makeDataTypeDeviceCompatible(x),
          dftefe::utils::makeDataTypeDeviceCompatible(y),
          dftefe::utils::makeDataTypeDeviceCompatible(z));
      }

      template <typename ValueType1, typename ValueType2>
      void
      Kernels<dftefe::utils::MemorySpace::DEVICE, ValueType1, ValueType2>::
        axpby(const size_type                           size,
              const scalar_type<ValueType1, ValueType2> alpha,
              const ValueType1 *                        x,
              const scalar_type<ValueType1, ValueType2> beta,
              const ValueType2 *                        y,
              scalar_type<ValueType1, ValueType2> *     z)
      {
        axpbyDeviceKernel<<<size / dftefe::utils::BLOCK_SIZE + 1,
                            dftefe::utils::BLOCK_SIZE>>>(
          size,
          dftefe::utils::makeDataTypeDeviceCompatible(alpha),
          dftefe::utils::makeDataTypeDeviceCompatible(x),
          dftefe::utils::makeDataTypeDeviceCompatible(beta),
          dftefe::utils::makeDataTypeDeviceCompatible(y),
          dftefe::utils::makeDataTypeDeviceCompatible(z));
      }


      template <typename ValueType1, typename ValueType2>
      std::vector<double>
      Kernels<dftefe::utils::MemorySpace::DEVICE, ValueType1, ValueType2>::
        amaxsMultiVector(size_type         vecSize,
                         size_type         numVec,
                         ValueType1 const *multiVecData)
      {
        std::vector<double> amaxs(numVec, 0);

        utils::throwException(
          false,
          "amaxsMultiVector() is not implemented for dftefe::utils::MemorySpace::DEVICE .... ");
        return amaxs;
      }



      template <typename ValueType1, typename ValueType2>
      std::vector<double>
      Kernels<dftefe::utils::MemorySpace::DEVICE, ValueType1, ValueType2>::
        nrms2MultiVector(
          size_type                                      vecSize,
          size_type                                      numVec,
          ValueType1 const *                             multiVecData,
          BlasQueue<dftefe::utils::MemorySpace::DEVICE> &BlasQueue)
      {
        std::vector<double> nrms2(numVec, 0);

        dftefe::utils::MemoryStorage<double, dftefe::utils::MemorySpace::DEVICE>
          nrmsSqVecDevice(numVec, 0.0);
        dftefe::utils::MemoryStorage<double, dftefe::utils::MemorySpace::DEVICE>
          onesVecDevice(vecSize, 1.0);
        dftefe::utils::MemoryStorage<double, dftefe::utils::MemorySpace::DEVICE>
          squaredEntriesDevice(vecSize * numVec, 0.0);

        absSquareEntriesDeviceKernel<<<
          (vecSize * numVec) / dftefe::utils::BLOCK_SIZE + 1,
          dftefe::utils::BLOCK_SIZE>>>(
          vecSize * numVec,
          dftefe::utils::makeDataTypeDeviceCompatible(multiVecData),
          dftefe::utils::makeDataTypeDeviceCompatible(
            squaredEntriesDevice.begin()));

        blas::gemm(Layout::ColMajor,
                   Op::NoTrans,
                   Op::Trans,
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
                   BlasQueue);


        nrmsSqVecDevice.copyTo<dftefe::utils::MemorySpace::DEVICE>(&nrms2[0]);

        for (size_type i = 0; i < numVec; i++)
          nrms2[i] = std::sqrt(nrms2[i]);

        return nrms2;
      }

#  define EXPLICITLY_INSTANTIATE(T1, T2, M) template class Kernels<M, T1, T2>;



      EXPLICITLY_INSTANTIATE(float, float, dftefe::utils::MemorySpace::DEVICE);
      EXPLICITLY_INSTANTIATE(float, double, dftefe::utils::MemorySpace::DEVICE);
      EXPLICITLY_INSTANTIATE(float,
                             std::complex<float>,
                             dftefe::utils::MemorySpace::DEVICE);
      EXPLICITLY_INSTANTIATE(float,
                             std::complex<double>,
                             dftefe::utils::MemorySpace::DEVICE);
      EXPLICITLY_INSTANTIATE(double, float, dftefe::utils::MemorySpace::DEVICE);
      EXPLICITLY_INSTANTIATE(double,
                             double,
                             dftefe::utils::MemorySpace::DEVICE);
      EXPLICITLY_INSTANTIATE(double,
                             std::complex<float>,
                             dftefe::utils::MemorySpace::DEVICE);
      EXPLICITLY_INSTANTIATE(double,
                             std::complex<double>,
                             dftefe::utils::MemorySpace::DEVICE);
      EXPLICITLY_INSTANTIATE(std::complex<float>,
                             float,
                             dftefe::utils::MemorySpace::DEVICE);
      EXPLICITLY_INSTANTIATE(std::complex<float>,
                             double,
                             dftefe::utils::MemorySpace::DEVICE);
      EXPLICITLY_INSTANTIATE(std::complex<float>,
                             std::complex<float>,
                             dftefe::utils::MemorySpace::DEVICE);
      EXPLICITLY_INSTANTIATE(std::complex<float>,
                             std::complex<double>,
                             dftefe::utils::MemorySpace::DEVICE);
      EXPLICITLY_INSTANTIATE(std::complex<double>,
                             float,
                             dftefe::utils::MemorySpace::DEVICE);
      EXPLICITLY_INSTANTIATE(std::complex<double>,
                             double,
                             dftefe::utils::MemorySpace::DEVICE);
      EXPLICITLY_INSTANTIATE(std::complex<double>,
                             std::complex<float>,
                             dftefe::utils::MemorySpace::DEVICE);
      EXPLICITLY_INSTANTIATE(std::complex<double>,
                             std::complex<double>,
                             dftefe::utils::MemorySpace::DEVICE);
    } // namespace blasLapack
  }   // namespace linearAlgebra
} // namespace dftefe
#endif
