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
          const unsigned int globalThreadId =
            blockIdx.x * blockDim.x + threadIdx.x;
          for (unsigned int i = globalThreadId; i < size;
               i += blockDim.x * gridDim.x)
            {
              z[i] = dftefe::utils::add(dftefe::utils::mult(alpha, x[i]),
                                        dftefe::utils::mult(beta, y[i]));
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
        size_type                                        vecSize,
        size_type                                        numVec,
        ValueType const *                                multiVecData,
        blasQueueType<dftefe::utils::MemorySpace::HOST> &blasQueue)
      {
        std::vector<double> nrms2(numVec, 0);
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
