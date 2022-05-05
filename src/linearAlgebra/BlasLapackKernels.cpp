#include <linearAlgebra/BlasLapackKernels.h>
#include <utils/DataTypeOverloads.h>
#include <complex>
#include <algorithm>

namespace dftefe
{
  namespace linearAlgebra
  {
    namespace blasLapack
    {
      template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
      void
      Kernels<ValueType, memorySpace>::axpby(const size_type  size,
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
      Kernels<ValueType, memorySpace>::amaxsMultiVector(
        const size_type  vecSize,
        const size_type  numVec,
        const ValueType *multiVecData)
      {
        std::vector<double> amaxs(numVec, 0);

        std::vector<ValueType> tempVec(vecSize, 0);
        for (size_type i = 0; i < numVec; ++i)
          {
            for (size_type j = 0; j < vecSize; ++j)
              tempVec[j] = multiVecData[j * numVec + i];

            amaxs[i] = dftefe::utils::abs_(
              *std::max_element(tempVec.begin(),
                                tempVec.end(),
                                dftefe::utils::absCompare<ValueType>));
          }


        return amaxs;
      }



      template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
      std::vector<double>
      Kernels<ValueType, memorySpace>::nrms2MultiVector(
        const size_type         vecSize,
        const size_type         numVec,
        const ValueType *       multiVecData,
        BlasQueue<memorySpace> &BlasQueue)
      {
        std::vector<double> nrms2(numVec, 0);

        for (size_type i = 0; i < vecSize; ++i)
          for (size_type j = 0; j < numVec; ++j)
            nrms2[j] += utils::absSq(multiVecData[i * numVec + j]);

        for (size_type i = 0; i < numVec; ++i)
          nrms2[i] = std::sqrt(nrms2[i]);

        return nrms2;
      }


      template class Kernels<double, dftefe::utils::MemorySpace::HOST>;
      template class Kernels<float, dftefe::utils::MemorySpace::HOST>;
      template class Kernels<std::complex<double>,
                             dftefe::utils::MemorySpace::HOST>;
      template class Kernels<std::complex<float>,
                             dftefe::utils::MemorySpace::HOST>;

#ifdef DFTEFE_WITH_DEVICE
      template class Kernels<double, dftefe::utils::MemorySpace::HOST_PINNED>;
      template class Kernels<float, dftefe::utils::MemorySpace::HOST_PINNED>;
      template class Kernels<std::complex<double>,
                             dftefe::utils::MemorySpace::HOST_PINNED>;
      template class Kernels<std::complex<float>,
                             dftefe::utils::MemorySpace::HOST_PINNED>;
#endif
    } // namespace blasLapack
  }   // namespace linearAlgebra
} // namespace dftefe
