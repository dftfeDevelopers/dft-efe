#include <linearAlgebra/BlasLapackKernels.h>
#include <utils/DataTypeOverloads.h>
#include <complex>
#include <algorithm>

namespace dftefe
{
  namespace linearAlgebra
  {
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    BlasLapackKernels<ValueType, memorySpace>::axpby(const size_type  size,
                                                     const ValueType  alpha,
                                                     const ValueType *x,
                                                     const ValueType  beta,
                                                     const ValueType *y,
                                                     ValueType *      z)
    {
      for (size_type i = 0; i < size; ++i)
        {
          z[i] = alpha * x[i] + beta * y[i];
        }
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    std::vector<double>
    BlasLapackKernels<ValueType, memorySpace>::nrms2MultiVector(
      const size_type  vecSize,
      const size_type  numVec,
      const ValueType *multiVecData)
    {
      std::vector<double> nrms2(numVec, 0);
      return nrms2;
    }


    template class BlasLapackKernels<size_type,
                                     dftefe::utils::MemorySpace::HOST>;
    template class BlasLapackKernels<int, dftefe::utils::MemorySpace::HOST>;
    template class BlasLapackKernels<double, dftefe::utils::MemorySpace::HOST>;
    template class BlasLapackKernels<float, dftefe::utils::MemorySpace::HOST>;
    template class BlasLapackKernels<std::complex<double>,
                                     dftefe::utils::MemorySpace::HOST>;
    template class BlasLapackKernels<std::complex<float>,
                                     dftefe::utils::MemorySpace::HOST>;

#ifdef DFTEFE_WITH_DEVICE
    template class BlasLapackKernels<size_type,
                                     dftefe::utils::MemorySpace::HOST_PINNED>;
    template class BlasLapackKernels<int,
                                     dftefe::utils::MemorySpace::HOST_PINNED>;
    template class BlasLapackKernels<double,
                                     dftefe::utils::MemorySpace::HOST_PINNED>;
    template class BlasLapackKernels<float,
                                     dftefe::utils::MemorySpace::HOST_PINNED>;
    template class BlasLapackKernels<std::complex<double>,
                                     dftefe::utils::MemorySpace::HOST_PINNED>;
    template class BlasLapackKernels<std::complex<float>,
                                     dftefe::utils::MemorySpace::HOST_PINNED>;
#endif
  } // namespace linearAlgebra
} // namespace dftefe
