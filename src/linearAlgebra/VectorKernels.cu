#ifdef DFTEFE_WITH_DEVICE_CUDA
#  include <utils/DeviceKernelLauncher.h>
#  include <utils/DeviceDataTypeOverloads.cuh>
#  include <utils/DataTypeOverloads.h>
#  include <utils/MemoryTransfer.h>
#  include <linearAlgebra/VectorKernels.h>
#  include <linearAlgebra/DeviceBlasLapackTemplates.h>
#  include <linearAlgebra/DeviceLAContextsSingleton.h>
#  include <complex>
#  include <algorithm>
namespace dftefe
{
  namespace linearAlgebra
  {
    namespace
    {
      template <typename ValueType>
      __global__ void
      addCUDAKernel(const size_type size, const ValueType *u, ValueType *v)
      {
        const unsigned int globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        for (unsigned int i = globalThreadId; i < size;
             i += blockDim.x * gridDim.x)
          {
            v[i] = dftefe::utils::add(v[i], u[i]);
          }
      }

      template <typename ValueType>
      __global__ void
      subCUDAKernel(const size_type size, const ValueType *u, ValueType *v)
      {
        const unsigned int globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        for (unsigned int i = globalThreadId; i < size;
             i += blockDim.x * gridDim.x)
          {
            v[i] = dftefe::utils::sub(v[i], u[i]);
          }
      }

      template <typename ValueType>
      __global__ void
      addCUDAKernel(const size_type  size,
                    const ValueType  a,
                    const ValueType *u,
                    const ValueType  b,
                    const ValueType *v,
                    ValueType *      w)
      {
        const unsigned int globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        for (unsigned int i = globalThreadId; i < size;
             i += blockDim.x * gridDim.x)
          {
            w[i] = dftefe::utils::add(dftefe::utils::mult(a, u[i]),
                                      dftefe::utils::mult(b, v[i]));
          }
      }

    } // namespace

    template <typename ValueType>
    void
    VectorKernels<ValueType, dftefe::utils::MemorySpace::DEVICE>::add(
      const size_type  size,
      const ValueType *u,
      ValueType *      v)
    {
      addCUDAKernel<<<size / dftefe::utils::BLOCK_SIZE + 1,
                      dftefe::utils::BLOCK_SIZE>>>(
        size,
        dftefe::utils::makeDataTypeDeviceCompatible(u),
        dftefe::utils::makeDataTypeDeviceCompatible(v));
    }

    template <typename ValueType>
    void
    VectorKernels<ValueType, dftefe::utils::MemorySpace::DEVICE>::sub(
      const size_type  size,
      const ValueType *u,
      ValueType *      v)
    {
      subCUDAKernel<<<size / dftefe::utils::BLOCK_SIZE + 1,
                      dftefe::utils::BLOCK_SIZE>>>(
        size,
        dftefe::utils::makeDataTypeDeviceCompatible(u),
        dftefe::utils::makeDataTypeDeviceCompatible(v));
    }

    template <typename ValueType>
    double
    VectorKernels<ValueType, dftefe::utils::MemorySpace::DEVICE>::l2Norm(
      const size_type  size,
      const ValueType *u)
    {
      double l2norm = 0;
      dftefe::linearAlgebra::DeviceBlasLapack<ValueType>::nrm2(
        dftefe::linearAlgebra::DeviceLAContextsSingleton::getInstance()
          ->getDeviceBlasHandle(),
        size,
        u,
        1,
        &l2norm);
      return l2norm;
    }


    template <typename ValueType>
    double
    VectorKernels<ValueType, dftefe::utils::MemorySpace::DEVICE>::lInfNorm(
      const size_type  size,
      const ValueType *u)
    {
      int maxIndex = 0;
      dftefe::linearAlgebra::DeviceBlasLapack<ValueType>::iamax(
        dftefe::linearAlgebra::DeviceLAContextsSingleton::getInstance()
          ->getDeviceBlasHandle(),
        size,
        u,
        1,
        &maxIndex);


      ValueType temp = 0.0;
      utils::MemoryTransfer<
        dftefe::utils::MemorySpace::HOST,
        dftefe::utils::MemorySpace::DEVICE>::copy(1, &temp, u + maxIndex - 1);

      return dftefe::utils::abs_(temp);
    }


    template <typename ValueType>
    std::vector<double>
    VectorKernels<ValueType, dftefe::utils::MemorySpace::DEVICE>::l2Norms(
      const size_type  size,
      const size_type  numVectors,
      const ValueType *u)
    {
      std::vector<double> l2norms;
      return l2norms;
    }


    template <typename ValueType>
    std::vector<double>
    VectorKernels<ValueType, dftefe::utils::MemorySpace::DEVICE>::lInfNorms(
      const size_type  size,
      const size_type  numVectors,
      const ValueType *u)
    {
      std::vector<double> linfnorms;
      return linfnorms;
    }


    template <typename ValueType>
    void
    VectorKernels<ValueType, dftefe::utils::MemorySpace::DEVICE>::add(
      const size_type  size,
      const ValueType  a,
      const ValueType *u,
      const ValueType  b,
      const ValueType *v,
      ValueType *      w)
    {
      addCUDAKernel<<<size / dftefe::utils::BLOCK_SIZE + 1,
                      dftefe::utils::BLOCK_SIZE>>>(
        size,
        dftefe::utils::makeDataTypeDeviceCompatible(a),
        dftefe::utils::makeDataTypeDeviceCompatible(u),
        dftefe::utils::makeDataTypeDeviceCompatible(b),
        dftefe::utils::makeDataTypeDeviceCompatible(v),
        dftefe::utils::makeDataTypeDeviceCompatible(w));
    }

    template class VectorKernels<size_type, dftefe::utils::MemorySpace::DEVICE>;
    template class VectorKernels<int, dftefe::utils::MemorySpace::DEVICE>;
    template class VectorKernels<double, dftefe::utils::MemorySpace::DEVICE>;
    template class VectorKernels<float, dftefe::utils::MemorySpace::DEVICE>;
    template class VectorKernels<std::complex<double>,
                                 dftefe::utils::MemorySpace::DEVICE>;
    template class VectorKernels<std::complex<float>,
                                 dftefe::utils::MemorySpace::DEVICE>;

  } // namespace linearAlgebra
} // namespace dftefe
#endif
