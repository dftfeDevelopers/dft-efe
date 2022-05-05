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
        template <typename ValueType>
        __global__ void
        axpbyDeviceKernel(const size_type  size,
                          const ValueType  alpha,
                          const ValueType *x,
                          const ValueType  beta,
                          const ValueType *y,
                          ValueType *      z)
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



      template <typename ValueType>
      void
      Kernels<ValueType, dftefe::utils::MemorySpace::DEVICE>::axpby(
        const size_type  size,
        const ValueType  alpha,
        const ValueType *x,
        const ValueType  beta,
        const ValueType *y,
        ValueType *      z)
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


      template <typename ValueType>
      std::vector<double>
      Kernels<ValueType, dftefe::utils::MemorySpace::DEVICE>::amaxsMultiVector(
        size_type        vecSize,
        size_type        numVec,
        ValueType const *multiVecData)
      {
        std::vector<double> amaxs(numVec, 0);

        utils::throwException(
          false,
          "amaxsMultiVector() is not implemented for dftefe::utils::MemorySpace::DEVICE .... ");
        return amaxs;
      }



      template <typename ValueType>
      std::vector<double>
      Kernels<ValueType, dftefe::utils::MemorySpace::DEVICE>::nrms2MultiVector(
        size_type                                          vecSize,
        size_type                                          numVec,
        ValueType const *                                  multiVecData,
        BlasQueueType<dftefe::utils::MemorySpace::DEVICE> &BlasQueue)
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


        nrmsSqVecDevice.copyTo<dftefe::utils::MemorySpace::HOST>(&nrms2[0]);

        for (size_type i = 0; i < numVec; i++)
          nrms2[i] = std::sqrt(nrms2[i]);

        return nrms2;
      }

      template class Kernels<double, dftefe::utils::MemorySpace::DEVICE>;
      template class Kernels<float, dftefe::utils::MemorySpace::DEVICE>;
      template class Kernels<std::complex<double>,
                             dftefe::utils::MemorySpace::DEVICE>;
      template class Kernels<std::complex<float>,
                             dftefe::utils::MemorySpace::DEVICE>;
    } // namespace blasLapack
  }   // namespace linearAlgebra
} // namespace dftefe
#endif
