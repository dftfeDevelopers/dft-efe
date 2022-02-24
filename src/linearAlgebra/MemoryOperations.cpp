#include <linearAlgebra/MemoryOperations.h>
#include "BlasWrappers.h"
#include <utils/DataTypeOverloads.h>
#include <complex>
#include <algorithm>

namespace dftefe
{
  namespace linearAlgebra
  {
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    MemoryOperations<ValueType, memorySpace >::add(
      const size_type  size,
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
    MemoryOperations<ValueType, memorySpace>::sub(
      const size_type  size,
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
    MemoryOperations<ValueType, memorySpace>::l2Norm(
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


    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    double
    MemoryOperations<ValueType, memorySpace>::lInfNorm(
      const size_type  size,
      const ValueType *u)
    {
      return dftefe::utils::abs_(
        *std::max_element(u, u + size, dftefe::utils::absCompare<ValueType>));
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    double
    MemoryOperations<ValueType, memorySpace>::dotProduct
      (size_type size, const ValueType *v , const ValueType *u)
    {
      return blasWrapper::dot<ValueType,ValueType,memorySpace > (size, v, 1, u, 1);
    }



    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    MemoryOperations<ValueType, memorySpace>::add(
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

    template class MemoryOperations<size_type, dftefe::utils::MemorySpace::HOST>;
    template class MemoryOperations<int, dftefe::utils::MemorySpace::HOST>;
    template class MemoryOperations<double, dftefe::utils::MemorySpace::HOST>;
    template class MemoryOperations<float, dftefe::utils::MemorySpace::HOST>;
    template class MemoryOperations<std::complex<double>,
                                 dftefe::utils::MemorySpace::HOST>;
    template class MemoryOperations<std::complex<float>,
                                 dftefe::utils::MemorySpace::HOST>;


#ifdef DFTEFE_WITH_DEVICE
    template class MemoryOperations<size_type,
                                 dftefe::utils::MemorySpace::HOST_PINNED>;
    template class MemoryOperations<int, dftefe::utils::MemorySpace::HOST_PINNED>;
    template class MemoryOperations<double,
                                 dftefe::utils::MemorySpace::HOST_PINNED>;
    template class MemoryOperations<float,
                                 dftefe::utils::MemorySpace::HOST_PINNED>;
    template class MemoryOperations<std::complex<double>,
                                 dftefe::utils::MemorySpace::HOST_PINNED>;
    template class MemoryOperations<std::complex<float>,
                                 dftefe::utils::MemorySpace::HOST_PINNED>;

#endif
  } // namespace linearAlgebra
} // namespace dftefe
