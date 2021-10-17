#include "VectorKernels.h"
#include <complex>

namespace dftefe
{
  namespace linearAlgebra
  {
    template <typename NumberType>
    void
    VectorKernels<NumberType, dftefe::utils::MemorySpace::HOST>::add(
      const size_type   size,
      const NumberType *u,
      NumberType *      v)
    {
      for (size_type i = 0; i < size; ++i)
        {
          v[i] += u[i];
        }
    }

    template class VectorKernels<double, dftefe::utils::MemorySpace::HOST>;
    template class VectorKernels<float, dftefe::utils::MemorySpace::HOST>;
    template class VectorKernels<std::complex<double>,
                                 dftefe::utils::MemorySpace::HOST>;
    template class VectorKernels<std::complex<float>,
                                 dftefe::utils::MemorySpace::HOST>;
  } // namespace linearAlgebra
} // namespace dftefe
