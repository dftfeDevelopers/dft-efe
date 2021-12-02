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
      addCUDAKernel(size_type size, const ValueType *u, ValueType *v)
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
      subCUDAKernel(size_type size, const ValueType *u, ValueType *v)
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
      addCUDAKernel(size_type        size,
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
    void
    VectorKernels<ValueType, dftefe::utils::MemorySpace::DEVICE>::add(
      size_type        size,
      ValueType        a,
      const ValueType *u,
      ValueType        b,
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

    template <typename ValueType>
    void
    VectorKernels<ValueType, dftefe::utils::MemorySpace::HOST_PINNED>::add(
      const size_type  size,
      const ValueType *u,
      ValueType *      v)
    {
      for (size_type i = 0; i < size; ++i)
        {
          v[i] += u[i];
        }
    }

    template <typename ValueType>
    void
    VectorKernels<ValueType, dftefe::utils::MemorySpace::HOST_PINNED>::sub(
      const size_type  size,
      const ValueType *u,
      ValueType *      v)
    {
      for (size_type i = 0; i < size; ++i)
        {
          v[i] -= u[i];
        }
    }

    template <typename ValueType>
    double
    VectorKernels<ValueType, dftefe::utils::MemorySpace::HOST_PINNED>::l2Norm(
      const size_type  size,
      const ValueType *u)
    {
      double temp = 0.0;
      for (size_type i = 0; i < size; ++i)
        {
          temp += dftefe::utils::absSq(u[i]);
        }
      return std::sqrt(temp);
    }


    template <typename ValueType>
    double
    VectorKernels<ValueType, dftefe::utils::MemorySpace::HOST_PINNED>::lInfNorm(
      const size_type  size,
      const ValueType *u)
    {
      return dftefe::utils::abs_(
        *std::max_element(u, u + size, dftefe::utils::absCompare<ValueType>));
    }


    template <typename ValueType>
    void
    VectorKernels<ValueType, dftefe::utils::MemorySpace::HOST_PINNED>::add(
      size_type        size,
      ValueType        a,
      const ValueType *u,
      ValueType        b,
      const ValueType *v,
      ValueType *      w)
    {
      for (int i = 0; i < size; ++i)
        {
          w[i] = a * u[i] + b * v[i];
        }
    }

    template class VectorKernels<size_type, dftefe::utils::MemorySpace::DEVICE>;
    template class VectorKernels<int, dftefe::utils::MemorySpace::DEVICE>;
    template class VectorKernels<double, dftefe::utils::MemorySpace::DEVICE>;
    template class VectorKernels<float, dftefe::utils::MemorySpace::DEVICE>;
    template class VectorKernels<std::complex<double>,
                                 dftefe::utils::MemorySpace::DEVICE>;
    template class VectorKernels<std::complex<float>,
                                 dftefe::utils::MemorySpace::DEVICE>;

    template class VectorKernels<size_type,
                                 dftefe::utils::MemorySpace::HOST_PINNED>;
    template class VectorKernels<int, dftefe::utils::MemorySpace::HOST_PINNED>;
    template class VectorKernels<double,
                                 dftefe::utils::MemorySpace::HOST_PINNED>;
    template class VectorKernels<float,
                                 dftefe::utils::MemorySpace::HOST_PINNED>;
    template class VectorKernels<std::complex<double>,
                                 dftefe::utils::MemorySpace::HOST_PINNED>;
    template class VectorKernels<std::complex<float>,
                                 dftefe::utils::MemorySpace::HOST_PINNED>;
  } // namespace linearAlgebra
} // namespace dftefe
#endif
