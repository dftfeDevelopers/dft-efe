#include <linearAlgebra/VectorKernels.h>
#include <utils/DataTypeOverloads.h>
#include <complex>
#include <algorithm>

namespace dftefe
{
  namespace linearAlgebra
  {
    template <typename ValueType>
    void
    VectorKernels<ValueType, dftefe::utils::MemorySpace::HOST>::add(
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
    VectorKernels<ValueType, dftefe::utils::MemorySpace::HOST>::sub(
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
    VectorKernels<ValueType, dftefe::utils::MemorySpace::HOST>::l2Norm(
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
    VectorKernels<ValueType, dftefe::utils::MemorySpace::HOST>::lInfNorm(
      const size_type  size,
      const ValueType *u)
    {
      return dftefe::utils::abs_(
        *std::max_element(u, u + size, dftefe::utils::absCompare<ValueType>));
    }


    template <typename ValueType>
    void
    VectorKernels<ValueType, dftefe::utils::MemorySpace::HOST>::add(
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

    template class VectorKernels<size_type, dftefe::utils::MemorySpace::HOST>;
    template class VectorKernels<int, dftefe::utils::MemorySpace::HOST>;
    template class VectorKernels<double, dftefe::utils::MemorySpace::HOST>;
    template class VectorKernels<float, dftefe::utils::MemorySpace::HOST>;
    template class VectorKernels<std::complex<double>,
                                 dftefe::utils::MemorySpace::HOST>;
    template class VectorKernels<std::complex<float>,
                                 dftefe::utils::MemorySpace::HOST>;
  } // namespace linearAlgebra
} // namespace dftefe
