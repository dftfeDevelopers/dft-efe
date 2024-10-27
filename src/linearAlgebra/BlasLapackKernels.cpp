#include <linearAlgebra/BlasLapackKernels.h>
#include <linearAlgebra/BlasLapack.h>
#include <utils/DataTypeOverloads.h>
#include <complex>
#include <algorithm>
#include <iostream>

namespace dftefe
{
  namespace linearAlgebra
  {
    namespace blasLapack
    {
      namespace blasLapackKernelsInternal
      {
        template <typename T>
        inline T
        conjugate(const T &x)
        {
          return std::conj(x);
        }

        template <>
        inline double
        conjugate(const double &x)
        {
          return x;
        }

        template <>
        inline float
        conjugate(const float &x)
        {
          return x;
        }

        template <typename T1, typename T2, ScalarOp op1, ScalarOp op2>
        class ScalarProduct
        {
        public:
          inline static scalar_type<T1, T2>
          prod(T1 t1, T2 t2)
          {
            return ((scalar_type<T1, T2>)t1) * ((scalar_type<T1, T2>)t2);
          }
        };

        template <typename T1, typename T2>
        class ScalarProduct<T1, T2, ScalarOp::Identity, ScalarOp::Conj>
        {
        public:
          inline static scalar_type<T1, T2>
          prod(T1 t1, T2 t2)
          {
            return ((scalar_type<T1, T2>)t1) *
                   ((scalar_type<T1, T2>)conjugate(t2));
          }
        };

        template <typename T1, typename T2>
        class ScalarProduct<T1, T2, ScalarOp::Conj, ScalarOp::Identity>
        {
        public:
          inline static scalar_type<T1, T2>
          prod(T1 t1, T2 t2)
          {
            return ((scalar_type<T1, T2>)conjugate(t1)) *
                   ((scalar_type<T1, T2>)t2);
          }
        };

        template <typename T1, typename T2>
        class ScalarProduct<T1, T2, ScalarOp::Conj, ScalarOp::Conj>
        {
        public:
          inline static scalar_type<T1, T2>
          prod(T1 t1, T2 t2)
          {
            return ((scalar_type<T1, T2>)conjugate(t1)) *
                   ((scalar_type<T1, T2>)conjugate(t2));
          }
        };


      } // namespace blasLapackKernelsInternal

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

      // template <typename ValueType1,
      //           typename ValueType2,
      //           dftefe::utils::MemorySpace memorySpace>
      // void
      // KernelsTwoValueTypes<ValueType1, ValueType2, memorySpace>::
      //   blockedHadamardProduct(const size_type                      vecSize,
      //                   const size_type                      numComponents,
      //                   const ValueType1 *                   blockedInput,
      //                   const ValueType2 * singleVectorInput,
      //                   scalar_type<ValueType1, ValueType2> *blockedOutput)
      // {
      //   for (size_type i = 0; i < vecSize; ++i)
      //     {
      //       for (size_type j = 0; j < numComponents; ++j)
      //       {
      //         blockedOutput[i * numComponents+j] =
      //           ((scalar_type<ValueType1, ValueType2>)blockedInput[i *
      //           numComponents+j]) *
      //           ((scalar_type<ValueType1, ValueType2>)singleVectorInput[i]);
      //       }
      //     }
      // }

      template <typename ValueType1,
                typename ValueType2,
                dftefe::utils::MemorySpace memorySpace>
      void
      KernelsTwoValueTypes<ValueType1, ValueType2, memorySpace>::
        hadamardProduct(const size_type                      size,
                        const ValueType1 *                   x,
                        const ValueType2 *                   y,
                        const ScalarOp &                     opx,
                        const ScalarOp &                     opy,
                        scalar_type<ValueType1, ValueType2> *z)
      {
        if (opx == ScalarOp::Identity && opy == ScalarOp::Identity)
          {
            blasLapackKernelsInternal::ScalarProduct<ValueType1,
                                                     ValueType2,
                                                     ScalarOp::Identity,
                                                     ScalarOp::Identity>
              sp;
            for (size_type i = 0; i < size; ++i)
              z[i] = sp.prod(x[i], y[i]);
          }

        else if (opx == ScalarOp::Identity && opy == ScalarOp::Conj)
          {
            blasLapackKernelsInternal::ScalarProduct<ValueType1,
                                                     ValueType2,
                                                     ScalarOp::Identity,
                                                     ScalarOp::Conj>
              sp;
            for (size_type i = 0; i < size; ++i)
              z[i] = sp.prod(x[i], y[i]);
          }

        else if (opx == ScalarOp::Conj && opy == ScalarOp::Identity)
          {
            blasLapackKernelsInternal::ScalarProduct<ValueType1,
                                                     ValueType2,
                                                     ScalarOp::Conj,
                                                     ScalarOp::Identity>
              sp;
            for (size_type i = 0; i < size; ++i)
              z[i] = sp.prod(x[i], y[i]);
          }

        // opx == ScalarOp::Conj && opy == ScalarOp::Conj
        else
          {
            blasLapackKernelsInternal::ScalarProduct<ValueType1,
                                                     ValueType2,
                                                     ScalarOp::Conj,
                                                     ScalarOp::Conj>
              sp;
            for (size_type i = 0; i < size; ++i)
              z[i] = sp.prod(x[i], y[i]);
          }
      }

      template <typename ValueType1,
                typename ValueType2,
                dftefe::utils::MemorySpace memorySpace>
      void
      KernelsTwoValueTypes<ValueType1, ValueType2, memorySpace>::
        scaleStridedVarBatched(const size_type                      numMats,
                               const ScalarOp *                     scalarOpA,
                               const ScalarOp *                     scalarOpB,
                               const size_type *                    stridea,
                               const size_type *                    strideb,
                               const size_type *                    stridec,
                               const size_type *                    m,
                               const size_type *                    n,
                               const size_type *                    k,
                               const ValueType1 *                   dA,
                               const ValueType2 *                   dB,
                               scalar_type<ValueType1, ValueType2> *dC,
                               LinAlgOpContext<memorySpace> &       context)
      {
        size_type cumulativeA = 0, cumulativeB = 0, cumulativeC = 0;
        for (size_type ibatch = 0; ibatch < numMats; ++ibatch)
          {
            for (size_type icolA = 0; icolA < *(m + ibatch); ++icolA)
              {
                for (size_type icolB = 0; icolB < *(n + ibatch); ++icolB)
                  {
                    size_type numrows = *(k + ibatch);
                    hadamardProduct(numrows,
                                    (dA + cumulativeA + icolA * numrows),
                                    (dB + cumulativeB + icolB * numrows),
                                    *(scalarOpA + ibatch),
                                    *(scalarOpB + ibatch),
                                    (dC + cumulativeC +
                                     icolA * *(n + ibatch) * numrows +
                                     icolB * numrows));
                  }
              }
            cumulativeA += *(stridea + ibatch);
            cumulativeB += *(strideb + ibatch);
            cumulativeC += *(stridec + ibatch);
          }
      }

      template <typename ValueType1,
                typename ValueType2,
                dftefe::utils::MemorySpace memorySpace>
      void
      KernelsTwoValueTypes<ValueType1, ValueType2, memorySpace>::
        khatriRaoProduct(const Layout                         layout,
                         const size_type                      sizeI,
                         const size_type                      sizeJ,
                         const size_type                      sizeK,
                         const ValueType1 *                   A,
                         const ValueType2 *                   B,
                         scalar_type<ValueType1, ValueType2> *Z)
      {
        if (layout == Layout::ColMajor)
          {
            for (size_type k = 0; k < sizeK; ++k)
              for (size_type i = 0; i < sizeI; ++i)
                for (size_type j = 0; j < sizeJ; ++j)
                  Z[k * sizeI * sizeJ + i * sizeJ + j] =
                    ((scalar_type<ValueType1, ValueType2>)A[k * sizeI + i]) *
                    ((scalar_type<ValueType1, ValueType2>)B[k * sizeJ + j]);
          }
        else if (layout == Layout::RowMajor)
          {
            for (size_type j = 0; j < sizeJ; ++j)
              for (size_type i = 0; i < sizeI; ++i)
                for (size_type k = 0; k < sizeK; ++k)
                  Z[j * sizeI * sizeK + i * sizeK + k] =
                    ((scalar_type<ValueType1, ValueType2>)A[i * sizeK + k]) *
                    ((scalar_type<ValueType1, ValueType2>)B[j * sizeK + k]);
          }
      }

      template <typename ValueType1,
                typename ValueType2,
                dftefe::utils::MemorySpace memorySpace>
      void
      KernelsTwoValueTypes<ValueType1, ValueType2, memorySpace>::
        khatriRaoProductStridedVarBatched(
          const Layout                         layout,
          const size_type                      numMats,
          const size_type *                    stridea,
          const size_type *                    strideb,
          const size_type *                    stridec,
          const size_type *                    m,
          const size_type *                    n,
          const size_type *                    k,
          const ValueType1 *                   dA,
          const ValueType2 *                   dB,
          scalar_type<ValueType1, ValueType2> *dC,
          LinAlgOpContext<memorySpace> &       context)
      {
        size_type cumulativeA = 0, cumulativeB = 0, cumulativeC = 0;
        for (size_type ibatch = 0; ibatch < numMats; ++ibatch)
          {
            khatriRaoProduct(layout,
                             *(m + ibatch),
                             *(n + ibatch),
                             *(k + ibatch),
                             (dA + cumulativeA),
                             (dB + cumulativeB),
                             (dC + cumulativeC));
            cumulativeA += *(stridea + ibatch);
            cumulativeB += *(strideb + ibatch);
            cumulativeC += *(stridec + ibatch);
          }
      }

      template <typename ValueType1,
                typename ValueType2,
                dftefe::utils::MemorySpace memorySpace>
      void
      KernelsTwoValueTypes<ValueType1, ValueType2, memorySpace>::
        transposedKhatriRaoProduct(const Layout                         layout,
                                   const size_type                      sizeI,
                                   const size_type                      sizeJ,
                                   const size_type                      sizeK,
                                   const ValueType1 *                   A,
                                   const ValueType2 *                   B,
                                   scalar_type<ValueType1, ValueType2> *Z)
      {
        if (layout == Layout::ColMajor)
          {
            for (size_type i = 0; i < sizeI; ++i)
              for (size_type j = 0; j < sizeJ; ++j)
                for (size_type k = 0; k < sizeK; ++k)
                  Z[i * sizeJ * sizeK + j * sizeK + k] =
                    ((scalar_type<ValueType1, ValueType2>)A[i * sizeK + k]) *
                    ((scalar_type<ValueType1, ValueType2>)B[j * sizeK + k]);
          }
        else if (layout == Layout::RowMajor)
          {
            for (size_type k = 0; k < sizeK; ++k)
              for (size_type i = 0; i < sizeI; ++i)
                for (size_type j = 0; j < sizeJ; ++j)
                  Z[k * sizeI * sizeJ + j * sizeJ + i] =
                    ((scalar_type<ValueType1, ValueType2>)A[k * sizeI + i]) *
                    ((scalar_type<ValueType1, ValueType2>)B[k * sizeJ + j]);
          }
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
      KernelsTwoValueTypes<ValueType1, ValueType2, memorySpace>::axpbyBlocked(
        const size_type                            size,
        const size_type                            blockSize,
        const scalar_type<ValueType1, ValueType2> *alpha,
        const ValueType1 *                         x,
        const scalar_type<ValueType1, ValueType2> *beta,
        const ValueType2 *                         y,
        scalar_type<ValueType1, ValueType2> *      z)
      {
        for (size_type i = 0; i < size; ++i)
          {
            for (size_type j = 0; j < blockSize; ++j)
              {
                z[i * blockSize + j] =
                  ((scalar_type<ValueType1, ValueType2>)alpha[j]) *
                    ((scalar_type<ValueType1, ValueType2>)
                       x[i * blockSize + j]) +
                  ((scalar_type<ValueType1, ValueType2>)beta[j]) *
                    ((scalar_type<ValueType1, ValueType2>)y[i * blockSize + j]);
              }
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
        const ScalarOp &                     opX,
        const ScalarOp &                     opY,
        scalar_type<ValueType1, ValueType2> *multiVecDotProduct,
        LinAlgOpContext<memorySpace> &       context)
      {
        std::fill(multiVecDotProduct, multiVecDotProduct + numVec, 0);
        if (opX == ScalarOp::Identity && opY == ScalarOp::Identity)
          {
            blasLapackKernelsInternal::ScalarProduct<ValueType1,
                                                     ValueType2,
                                                     ScalarOp::Identity,
                                                     ScalarOp::Identity>
              sp;
            for (size_type i = 0; i < vecSize; ++i)
              {
                for (size_type j = 0; j < numVec; ++j)
                  {
                    multiVecDotProduct[j] +=
                      sp.prod(multiVecDataX[i * numVec + j],
                              multiVecDataY[i * numVec + j]);
                  }
              }
          }

        else if (opX == ScalarOp::Identity && opY == ScalarOp::Conj)
          {
            blasLapackKernelsInternal::ScalarProduct<ValueType1,
                                                     ValueType2,
                                                     ScalarOp::Identity,
                                                     ScalarOp::Conj>
              sp;
            for (size_type i = 0; i < vecSize; ++i)
              {
                for (size_type j = 0; j < numVec; ++j)
                  {
                    multiVecDotProduct[j] +=
                      sp.prod(multiVecDataX[i * numVec + j],
                              multiVecDataY[i * numVec + j]);
                  }
              }
          }

        else if (opX == ScalarOp::Conj && opY == ScalarOp::Identity)
          {
            blasLapackKernelsInternal::ScalarProduct<ValueType1,
                                                     ValueType2,
                                                     ScalarOp::Conj,
                                                     ScalarOp::Identity>
              sp;
            for (size_type i = 0; i < vecSize; ++i)
              {
                for (size_type j = 0; j < numVec; ++j)
                  {
                    multiVecDotProduct[j] +=
                      sp.prod(multiVecDataX[i * numVec + j],
                              multiVecDataY[i * numVec + j]);
                  }
              }
          }

        // (opX == ScalarOp::Conj && opY == ScalarOp::Conj)
        else
          {
            blasLapackKernelsInternal::ScalarProduct<ValueType1,
                                                     ValueType2,
                                                     ScalarOp::Conj,
                                                     ScalarOp::Conj>
              sp;
            for (size_type i = 0; i < vecSize; ++i)
              {
                for (size_type j = 0; j < numVec; ++j)
                  {
                    multiVecDotProduct[j] +=
                      sp.prod(multiVecDataX[i * numVec + j],
                              multiVecDataY[i * numVec + j]);
                  }
              }
          }
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
