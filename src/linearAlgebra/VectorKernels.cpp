#include <linearAlgebra/VectorKernels.h>
#include <utils/DataTypeOverloads.h>
#include <complex>
#include <algorithm>

namespace dftefe
{
  namespace linearAlgebra
  {
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    VectorKernels<ValueType, memorySpace>::add(const size_type  size,
                                               const ValueType *u,
                                               ValueType *      v)
    {
      for (size_type i = 0; i < size; ++i)
        {
          v[i] += u[i];
        }
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    VectorKernels<ValueType, memorySpace>::sub(const size_type  size,
                                               const ValueType *u,
                                               ValueType *      v)
    {
      for (size_type i = 0; i < size; ++i)
        {
          v[i] -= u[i];
        }
    }


    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    double
    VectorKernels<ValueType, memorySpace>::l2Norm(const size_type  size,
                                                  const ValueType *u)
    {
      double temp = 0.0;
      for (size_type i = 0; i < size; ++i)
        {
          temp += dftefe::utils::absSq(u[i]);
        }
      return std::sqrt(temp);
    }


    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    double
    VectorKernels<ValueType, memorySpace>::lInfNorm(const size_type  size,
                                                    const ValueType *u)
    {
      return dftefe::utils::abs_(
        *std::max_element(u, u + size, dftefe::utils::absCompare<ValueType>));
    }


    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    std::vector<double>
    VectorKernels<ValueType, memorySpace>::l2Norms(const size_type  size,
                                                   const size_type  numVectors,
                                                   const ValueType *u)
    {
      std::vector<double> l2norms(numVectors, 0.0);
      return l2norms;
    }


    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    std::vector<double>
    VectorKernels<ValueType, memorySpace>::lInfNorms(const size_type size,
                                                     const size_type numVectors,
                                                     const ValueType *u)
    {
      std::vector<double> linfnorms(numVectors, 0.0);
      return linfnorms;
    }


    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    VectorKernels<ValueType, memorySpace>::add(const size_type  size,
                                               const ValueType  a,
                                               const ValueType *u,
                                               const ValueType  b,
                                               const ValueType *v,
                                               ValueType *      w)
    {
      for (int i = 0; i < size; ++i)
        {
          w[i] = a * u[i] + b * v[i];
        }
    }

    template class VectorKernels<size_type, dftefe::utils::MemorySpace::HOST>;
    template class VectorKernels<int, dftefe::utils::MemorySpace::HOST>;
    template class VectorKernels<double, dftefe::utils::MemorySpace::HOST>;
    template class VectorKernels<float, dftefe::utils::MemorySpace::HOST>;
    template class VectorKernels<std::complex<double>,
                                 dftefe::utils::MemorySpace::HOST>;
    template class VectorKernels<std::complex<float>,
                                 dftefe::utils::MemorySpace::HOST>;

#ifdef DFTEFE_WITH_DEVICE
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
#endif


  } // namespace linearAlgebra
} // namespace dftefe
