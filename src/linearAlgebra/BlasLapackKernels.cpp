#include <linearAlgebra/BlasLapackKernels.h>
#include <linearAlgebra/BlasLapack.h>
#include <utils/DataTypeOverloads.h>
#include <complex>
#include <algorithm>

namespace dftefe
{
  namespace linearAlgebra
  {
    namespace blasLapack
    {
      template <typename ValueType1,
                typename ValueType2,
                dftefe::utils::MemorySpace memorySpace>
      void
      KernelsTwoValueTypes<ValueType1, ValueType2, memorySpace>::ascale(
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


      template <typename ValueType1,
                typename ValueType2,
                dftefe::utils::MemorySpace memorySpace>
      void
      KernelsTwoValueTypes<ValueType1, ValueType2, memorySpace>::reciprocalX(
        const size_type                      size,
        const ValueType1                     alpha,
        const ValueType2 *                   x,
        scalar_type<ValueType1, ValueType2> *z)
      {
        for (size_type i = 0; i < size; ++i)
          {
            z[i] = ((scalar_type<ValueType1, ValueType2>)alpha) /
                   ((scalar_type<ValueType1, ValueType2>)x[i]);
          }
      }

      template <typename ValueType1,
                typename ValueType2,
                dftefe::utils::MemorySpace memorySpace>
      void
      KernelsTwoValueTypes<ValueType1, ValueType2, memorySpace>::
        hadamardProduct(const size_type                      size,
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


      template <typename ValueType1,
                typename ValueType2,
                dftefe::utils::MemorySpace memorySpace>
      void
      KernelsTwoValueTypes<ValueType1, ValueType2, memorySpace>::
        khatriRaoProduct(const size_type                      sizeI,
                         const size_type                      sizeJ,
                         const size_type                      sizeK,
                         const ValueType1 *                   A,
                         const ValueType2 *                   B,
                         scalar_type<ValueType1, ValueType2> *Z)
      {
        for (size_type k = 0; k < sizeK; ++k)
          for (size_type i = 0; i < sizeI; ++i)
            for (size_type j = 0; j < sizeJ; ++j)
              Z[k * sizeI * sizeJ + i * sizeJ + j] =
                ((scalar_type<ValueType1, ValueType2>)A[k * sizeI + i]) *
                ((scalar_type<ValueType1, ValueType2>)B[k * sizeJ + j]);
      }

      template <typename ValueType1,
                typename ValueType2,
                dftefe::utils::MemorySpace memorySpace>
      void
      KernelsTwoValueTypes<ValueType1, ValueType2, memorySpace>::axpby(
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

      template <typename ValueType1,
                typename ValueType2,
                dftefe::utils::MemorySpace memorySpace>
      void
      KernelsTwoValueTypes<ValueType1, ValueType2, memorySpace>::dotMultiVector(
        const size_type                      vecSize,
        const size_type                      numVec,
        const ValueType1 *                   multiVecDataX,
        const ValueType2 *                   multiVecDataY,
        scalar_type<ValueType1, ValueType2> *multiVecDotProduct,
        LinAlgOpContext<memorySpace> &       context)
      {
        std::fill(multiVecDotProduct, multiVecDotProduct + numVec, 0);
        for (size_type i = 0; i < vecSize; ++i)
          for (size_type j = 0; j < numVec; ++j)
            multiVecDotProduct[j] += ((scalar_type<ValueType1, ValueType2>)
                                        multiVecDataX[i * numVec + j]) *
                                     ((scalar_type<ValueType1, ValueType2>)
                                        multiVecDataY[i * numVec + j]);
      }

      template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
      std::vector<double>
      KernelsOneValueType<ValueType, memorySpace>::amaxsMultiVector(
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
      KernelsOneValueType<ValueType, memorySpace>::nrms2MultiVector(
        const size_type               vecSize,
        const size_type               numVec,
        const ValueType *             multiVecData,
        LinAlgOpContext<memorySpace> &context)
      {
        std::vector<double> nrms2(numVec, 0);

        for (size_type i = 0; i < vecSize; ++i)
          for (size_type j = 0; j < numVec; ++j)
            nrms2[j] += utils::absSq(multiVecData[i * numVec + j]);

        for (size_type i = 0; i < numVec; ++i)
          nrms2[i] = std::sqrt(nrms2[i]);

        return nrms2;
      }

#define EXPLICITLY_INSTANTIATE_2T(T1, T2, M) \
  template class KernelsTwoValueTypes<T1, T2, M>;

#define EXPLICITLY_INSTANTIATE_1T(T, M) \
  template class KernelsOneValueType<T, M>;


      EXPLICITLY_INSTANTIATE_1T(float, dftefe::utils::MemorySpace::HOST);
      EXPLICITLY_INSTANTIATE_1T(double, dftefe::utils::MemorySpace::HOST);
      EXPLICITLY_INSTANTIATE_1T(std::complex<float>,
                                dftefe::utils::MemorySpace::HOST);
      EXPLICITLY_INSTANTIATE_1T(std::complex<double>,
                                dftefe::utils::MemorySpace::HOST);


      EXPLICITLY_INSTANTIATE_2T(float, float, dftefe::utils::MemorySpace::HOST);
      EXPLICITLY_INSTANTIATE_2T(float,
                                double,
                                dftefe::utils::MemorySpace::HOST);
      EXPLICITLY_INSTANTIATE_2T(float,
                                std::complex<float>,
                                dftefe::utils::MemorySpace::HOST);
      EXPLICITLY_INSTANTIATE_2T(float,
                                std::complex<double>,
                                dftefe::utils::MemorySpace::HOST);
      EXPLICITLY_INSTANTIATE_2T(double,
                                float,
                                dftefe::utils::MemorySpace::HOST);
      EXPLICITLY_INSTANTIATE_2T(double,
                                double,
                                dftefe::utils::MemorySpace::HOST);
      EXPLICITLY_INSTANTIATE_2T(double,
                                std::complex<float>,
                                dftefe::utils::MemorySpace::HOST);
      EXPLICITLY_INSTANTIATE_2T(double,
                                std::complex<double>,
                                dftefe::utils::MemorySpace::HOST);
      EXPLICITLY_INSTANTIATE_2T(std::complex<float>,
                                float,
                                dftefe::utils::MemorySpace::HOST);
      EXPLICITLY_INSTANTIATE_2T(std::complex<float>,
                                double,
                                dftefe::utils::MemorySpace::HOST);
      EXPLICITLY_INSTANTIATE_2T(std::complex<float>,
                                std::complex<float>,
                                dftefe::utils::MemorySpace::HOST);
      EXPLICITLY_INSTANTIATE_2T(std::complex<float>,
                                std::complex<double>,
                                dftefe::utils::MemorySpace::HOST);
      EXPLICITLY_INSTANTIATE_2T(std::complex<double>,
                                float,
                                dftefe::utils::MemorySpace::HOST);
      EXPLICITLY_INSTANTIATE_2T(std::complex<double>,
                                double,
                                dftefe::utils::MemorySpace::HOST);
      EXPLICITLY_INSTANTIATE_2T(std::complex<double>,
                                std::complex<float>,
                                dftefe::utils::MemorySpace::HOST);
      EXPLICITLY_INSTANTIATE_2T(std::complex<double>,
                                std::complex<double>,
                                dftefe::utils::MemorySpace::HOST);

#ifdef DFTEFE_WITH_DEVICE
      EXPLICITLY_INSTANTIATE_1T(float, dftefe::utils::MemorySpace::HOST_PINNED);
      EXPLICITLY_INSTANTIATE_1T(double,
                                dftefe::utils::MemorySpace::HOST_PINNED);
      EXPLICITLY_INSTANTIATE_1T(std::complex<float>,
                                dftefe::utils::MemorySpace::HOST_PINNED);
      EXPLICITLY_INSTANTIATE_1T(std::complex<double>,
                                dftefe::utils::MemorySpace::HOST_PINNED);

      EXPLICITLY_INSTANTIATE_2T(float,
                                float,
                                dftefe::utils::MemorySpace::HOST_PINNED);
      EXPLICITLY_INSTANTIATE_2T(float,
                                double,
                                dftefe::utils::MemorySpace::HOST_PINNED);
      EXPLICITLY_INSTANTIATE_2T(float,
                                std::complex<float>,
                                dftefe::utils::MemorySpace::HOST_PINNED);
      EXPLICITLY_INSTANTIATE_2T(float,
                                std::complex<double>,
                                dftefe::utils::MemorySpace::HOST_PINNED);
      EXPLICITLY_INSTANTIATE_2T(double,
                                float,
                                dftefe::utils::MemorySpace::HOST_PINNED);
      EXPLICITLY_INSTANTIATE_2T(double,
                                double,
                                dftefe::utils::MemorySpace::HOST_PINNED);
      EXPLICITLY_INSTANTIATE_2T(double,
                                std::complex<float>,
                                dftefe::utils::MemorySpace::HOST_PINNED);
      EXPLICITLY_INSTANTIATE_2T(double,
                                std::complex<double>,
                                dftefe::utils::MemorySpace::HOST_PINNED);
      EXPLICITLY_INSTANTIATE_2T(std::complex<float>,
                                float,
                                dftefe::utils::MemorySpace::HOST_PINNED);
      EXPLICITLY_INSTANTIATE_2T(std::complex<float>,
                                double,
                                dftefe::utils::MemorySpace::HOST_PINNED);
      EXPLICITLY_INSTANTIATE_2T(std::complex<float>,
                                std::complex<float>,
                                dftefe::utils::MemorySpace::HOST_PINNED);
      EXPLICITLY_INSTANTIATE_2T(std::complex<float>,
                                std::complex<double>,
                                dftefe::utils::MemorySpace::HOST_PINNED);
      EXPLICITLY_INSTANTIATE_2T(std::complex<double>,
                                float,
                                dftefe::utils::MemorySpace::HOST_PINNED);
      EXPLICITLY_INSTANTIATE_2T(std::complex<double>,
                                double,
                                dftefe::utils::MemorySpace::HOST_PINNED);
      EXPLICITLY_INSTANTIATE_2T(std::complex<double>,
                                std::complex<float>,
                                dftefe::utils::MemorySpace::HOST_PINNED);
      EXPLICITLY_INSTANTIATE_2T(std::complex<double>,
                                std::complex<double>,
                                dftefe::utils::MemorySpace::HOST_PINNED);
#endif
    } // namespace blasLapack
  }   // namespace linearAlgebra
} // namespace dftefe
