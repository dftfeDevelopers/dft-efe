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
      template <dftefe::utils::MemorySpace memorySpace,
                typename ValueType1,
                typename ValueType2>
      void
      Kernels<memorySpace, ValueType1, ValueType2>::ascale(
        const size_type                      size,
        const ValueType1                     alpha,
        const ValueType2 *                   x,
        scalar_type<ValueType1, ValueType2> *z)
      {
        for (size_type i = 0; i < size; ++i)
          {
            z[i] = ((scalar_type<ValueType1, ValueType2>)alpha) *
                   ((scalar_type<ValueType1, ValueType2>)x[i]);
          }
      }


      template <dftefe::utils::MemorySpace memorySpace,
                typename ValueType1,
                typename ValueType2>
      void
      Kernels<memorySpace, ValueType1, ValueType2>::hadamardProduct(
        const size_type                      size,
        const ValueType1 *                   x,
        const ValueType2 *                   y,
        scalar_type<ValueType1, ValueType2> *z)
      {
        for (size_type i = 0; i < size; ++i)
          {
            z[i] = ((scalar_type<ValueType1, ValueType2>)x[i]) *
                   ((scalar_type<ValueType1, ValueType2>)y[i]);
          }
      }

      template <dftefe::utils::MemorySpace memorySpace,
                typename ValueType1,
                typename ValueType2>
      void
      Kernels<memorySpace, ValueType1, ValueType2>::axpby(
        const size_type                           size,
        const scalar_type<ValueType1, ValueType2> alpha,
        const ValueType1 *                        x,
        const scalar_type<ValueType1, ValueType2> beta,
        const ValueType2 *                        y,
        scalar_type<ValueType1, ValueType2> *     z)
      {
        for (size_type i = 0; i < size; ++i)
          {
            z[i] = ((scalar_type<ValueType1, ValueType2>)alpha) *
                     ((scalar_type<ValueType1, ValueType2>)x[i]) +
                   ((scalar_type<ValueType1, ValueType2>)beta) *
                     ((scalar_type<ValueType1, ValueType2>)y[i]);
          }
      }


      template <dftefe::utils::MemorySpace memorySpace,
                typename ValueType1,
                typename ValueType2>
      std::vector<double>
      Kernels<memorySpace, ValueType1, ValueType2>::amaxsMultiVector(
        const size_type   vecSize,
        const size_type   numVec,
        const ValueType1 *multiVecData)
      {
        std::vector<double> amaxs(numVec, 0);

        std::vector<ValueType1> tempVec(vecSize, 0);
        for (size_type i = 0; i < numVec; ++i)
          {
            for (size_type j = 0; j < vecSize; ++j)
              tempVec[j] = multiVecData[j * numVec + i];

            amaxs[i] = dftefe::utils::abs_(
              *std::max_element(tempVec.begin(),
                                tempVec.end(),
                                dftefe::utils::absCompare<ValueType1>));
          }


        return amaxs;
      }



      template <dftefe::utils::MemorySpace memorySpace,
                typename ValueType1,
                typename ValueType2>
      std::vector<double>
      Kernels<memorySpace, ValueType1, ValueType2>::nrms2MultiVector(
        const size_type         vecSize,
        const size_type         numVec,
        const ValueType1 *      multiVecData,
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

#define EXPLICITLY_INSTANTIATE(T1, T2, M) template class Kernels<M, T1, T2>;


      EXPLICITLY_INSTANTIATE(float, float, dftefe::utils::MemorySpace::HOST);
      EXPLICITLY_INSTANTIATE(float, double, dftefe::utils::MemorySpace::HOST);
      EXPLICITLY_INSTANTIATE(float,
                             std::complex<float>,
                             dftefe::utils::MemorySpace::HOST);
      EXPLICITLY_INSTANTIATE(float,
                             std::complex<double>,
                             dftefe::utils::MemorySpace::HOST);
      EXPLICITLY_INSTANTIATE(double, float, dftefe::utils::MemorySpace::HOST);
      EXPLICITLY_INSTANTIATE(double, double, dftefe::utils::MemorySpace::HOST);
      EXPLICITLY_INSTANTIATE(double,
                             std::complex<float>,
                             dftefe::utils::MemorySpace::HOST);
      EXPLICITLY_INSTANTIATE(double,
                             std::complex<double>,
                             dftefe::utils::MemorySpace::HOST);
      EXPLICITLY_INSTANTIATE(std::complex<float>,
                             float,
                             dftefe::utils::MemorySpace::HOST);
      EXPLICITLY_INSTANTIATE(std::complex<float>,
                             double,
                             dftefe::utils::MemorySpace::HOST);
      EXPLICITLY_INSTANTIATE(std::complex<float>,
                             std::complex<float>,
                             dftefe::utils::MemorySpace::HOST);
      EXPLICITLY_INSTANTIATE(std::complex<float>,
                             std::complex<double>,
                             dftefe::utils::MemorySpace::HOST);
      EXPLICITLY_INSTANTIATE(std::complex<double>,
                             float,
                             dftefe::utils::MemorySpace::HOST);
      EXPLICITLY_INSTANTIATE(std::complex<double>,
                             double,
                             dftefe::utils::MemorySpace::HOST);
      EXPLICITLY_INSTANTIATE(std::complex<double>,
                             std::complex<float>,
                             dftefe::utils::MemorySpace::HOST);
      EXPLICITLY_INSTANTIATE(std::complex<double>,
                             std::complex<double>,
                             dftefe::utils::MemorySpace::HOST);

#ifdef DFTEFE_WITH_DEVICE
      EXPLICITLY_INSTANTIATE(float,
                             float,
                             dftefe::utils::MemorySpace::HOST_PINNED);
      EXPLICITLY_INSTANTIATE(float,
                             double,
                             dftefe::utils::MemorySpace::HOST_PINNED);
      EXPLICITLY_INSTANTIATE(float,
                             std::complex<float>,
                             dftefe::utils::MemorySpace::HOST_PINNED);
      EXPLICITLY_INSTANTIATE(float,
                             std::complex<double>,
                             dftefe::utils::MemorySpace::HOST_PINNED);
      EXPLICITLY_INSTANTIATE(double,
                             float,
                             dftefe::utils::MemorySpace::HOST_PINNED);
      EXPLICITLY_INSTANTIATE(double,
                             double,
                             dftefe::utils::MemorySpace::HOST_PINNED);
      EXPLICITLY_INSTANTIATE(double,
                             std::complex<float>,
                             dftefe::utils::MemorySpace::HOST_PINNED);
      EXPLICITLY_INSTANTIATE(double,
                             std::complex<double>,
                             dftefe::utils::MemorySpace::HOST_PINNED);
      EXPLICITLY_INSTANTIATE(std::complex<float>,
                             float,
                             dftefe::utils::MemorySpace::HOST_PINNED);
      EXPLICITLY_INSTANTIATE(std::complex<float>,
                             double,
                             dftefe::utils::MemorySpace::HOST_PINNED);
      EXPLICITLY_INSTANTIATE(std::complex<float>,
                             std::complex<float>,
                             dftefe::utils::MemorySpace::HOST_PINNED);
      EXPLICITLY_INSTANTIATE(std::complex<float>,
                             std::complex<double>,
                             dftefe::utils::MemorySpace::HOST_PINNED);
      EXPLICITLY_INSTANTIATE(std::complex<double>,
                             float,
                             dftefe::utils::MemorySpace::HOST_PINNED);
      EXPLICITLY_INSTANTIATE(std::complex<double>,
                             double,
                             dftefe::utils::MemorySpace::HOST_PINNED);
      EXPLICITLY_INSTANTIATE(std::complex<double>,
                             std::complex<float>,
                             dftefe::utils::MemorySpace::HOST_PINNED);
      EXPLICITLY_INSTANTIATE(std::complex<double>,
                             std::complex<double>,
                             dftefe::utils::MemorySpace::HOST_PINNED);
#endif
    } // namespace blasLapack
  }   // namespace linearAlgebra
} // namespace dftefe
