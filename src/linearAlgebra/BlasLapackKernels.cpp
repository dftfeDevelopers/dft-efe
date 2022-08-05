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
      namespace kernels
      {
        template <typename ValueType1,
                  typename ValueType2,
                  dftefe::utils::MemorySpace memorySpace>
        void
        ascale(const size_type                      size,
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
        axpby(const size_type                           size,
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


        template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
        std::vector<double>
        amaxsMultiVector(const size_type  vecSize,
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
        nrms2MultiVector(const size_type         vecSize,
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

#define EXPLICITLY_INSTANTIATE_2T(T1, T2, M)                          \
  template void hadamardProduct<T1, T2, M>(const size_type      size, \
                                           const T1 *           x,    \
                                           const T2 *           y,    \
                                           scalar_type<T1, T2> *z);   \
                                                                      \
  template void axpby<T1, T2, M>(const size_type           size,      \
                                 const scalar_type<T1, T2> alpha,     \
                                 const T1 *                x,         \
                                 const scalar_type<T1, T2> beta,      \
                                 const T2 *                y,         \
                                 scalar_type<T1, T2> *     z);             \
  template void ascale<T1, T2, M>(const size_type      size,          \
                                  const T1             alpha,         \
                                  const T2 *           x,             \
                                  scalar_type<T1, T2> *z);



#define EXPLICITLY_INSTANTIATE_1T(T, M)                                        \
  template std::vector<double> amaxsMultiVector<T, M>(const size_type vecSize, \
                                                      const size_type numVec,  \
                                                      const T *multiVecData);  \
  template std::vector<double> nrms2MultiVector<T, M>(                         \
    const size_type vecSize,                                                   \
    const size_type numVec,                                                    \
    const T *       multiVecData,                                              \
    BlasQueue<M> &  BlasQueue);


        EXPLICITLY_INSTANTIATE_1T(float, dftefe::utils::MemorySpace::HOST);
        EXPLICITLY_INSTANTIATE_1T(double, dftefe::utils::MemorySpace::HOST);
        EXPLICITLY_INSTANTIATE_1T(std::complex<float>,
                                  dftefe::utils::MemorySpace::HOST);
        EXPLICITLY_INSTANTIATE_1T(std::complex<double>,
                                  dftefe::utils::MemorySpace::HOST);


        EXPLICITLY_INSTANTIATE_2T(float,
                                  float,
                                  dftefe::utils::MemorySpace::HOST);
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
        EXPLICITLY_INSTANTIATE_1T(float,
                                  dftefe::utils::MemorySpace::HOST_PINNED);
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
      } // namespace kernels
    }   // namespace blasLapack
  }     // namespace linearAlgebra
} // namespace dftefe
