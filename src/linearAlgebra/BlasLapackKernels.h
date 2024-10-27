#ifndef dftefeKernels_h
#define dftefeKernels_h

#include <utils/MemoryManager.h>
#include <linearAlgebra/BlasLapackTypedef.h>
#include <linearAlgebra/LinAlgOpContext.h>
#include <blas.hh>
#include <vector>

namespace dftefe
{
  namespace linearAlgebra
  {
    namespace blasLapack
    {
      /**
       * @brief namespace class for BlasLapack kernels not present in blaspp.
       */
      template <typename ValueType1,
                typename ValueType2,
                dftefe::utils::MemorySpace memorySpace>
      class KernelsTwoValueTypes
      {
      public:
        /**
         * @brief Template for performing \f$ z = \alpha x$
         * @param[in] size size of the array
         * @param[in] \f$ alpha \f$ scalar
         * @param[in] x array
         * @param[out] z array
         */
        static void
        ascale(size_type                            size,
               ValueType1                           alpha,
               const ValueType2 *                   x,
               scalar_type<ValueType1, ValueType2> *z);

        /**
         * @brief Template for performing \f$ z = 1 /x$, does not check if x[i] is zero
         * @param[in] size size of the array
         * @param[in] x array
         * @param[out] z array
         */
        static void
        reciprocalX(size_type                            size,
                    const ValueType1                     alpha,
                    const ValueType2 *                   x,
                    scalar_type<ValueType1, ValueType2> *z);
        /*
         * @brief Template for performing \f$ z_i = x_i * y_i$
         * @param[in] size size of the array
         * @param[in] x array
         * @param[in] y array
         * @param[out] z array
         */
        static void
        hadamardProduct(size_type                            size,
                        const ValueType1 *                   x,
                        const ValueType2 *                   y,
                        scalar_type<ValueType1, ValueType2> *z);

        // /*
        //  * @brief Template for performing \f$ blockedOutput_ij = blockedInput_ij * singleVectorInput_i$
        //  * @param[in] size size of the blocked Input array
        //  * @param[in] numComponets no of componets
        //  * @param[in] blockedInput blocked array
        //  * @param[in] singleVectorInput array
        //  * @param[out] blockedOutput blocked array
        //  */
        // static void
        // blockedHadamardProduct(const size_type                      vecSize,
        //                 const size_type                      numComponents,
        //                 const ValueType1 *                   blockedInput,
        //                 const ValueType2 * singleVectorInput,
        //                 scalar_type<ValueType1, ValueType2> *blockedOutput);

        static void
        hadamardProduct(size_type                            size,
                        const ValueType1 *                   x,
                        const ValueType2 *                   y,
                        const ScalarOp &                     opx,
                        const ScalarOp &                     opy,
                        scalar_type<ValueType1, ValueType2> *z);

        /**
         * @brief Template for performing hadamard product of two columns
         * of batches of matrix A and B having num col A = m,
         * num common rows = k, num col B = n going through all the rows
         * of A and B with column B having the faster index than columnA.
         * This operation can be thought as the strided form of face-splitting
         * product between two matrices of variable strides but with common
         * rows in each stride. Also it is assumed that the matrices A and B
         * are column major. So for scalarop it represents either identity or
         * complex conjugate operation on a scalar. Size of C on output
         * will be (m*k) cols and n rows with strides.
         * @param[in] numMats number of batches
         * @param[in] scalarOpA scalar op of A
         * @param[in] scalarOpB scalar op of B
         * @param[in] stridea stride of matrix A
         * @param[in] stridea stride of matrix B
         * @param[in] stridec stride of matrix C
         * @param[in] m column of matrix A
         * @param[in] n column of matrix B
         * @param[in] k row of matrix B and A
         * @param[in] dA matrix A
         * @param[in] dB matrix B
         * @param[out] dC matrix C
         * @param[in] context memorySpace context
         */
        static void
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
                               LinAlgOpContext<memorySpace> &       context);

        /**
         * @brief Template for performing
         * In column major storage format:
         * \f$ {\bf Z}={\bf A} \odot {\bf B} = a_1 \otimes b_1
         * \quad a_2 \otimes b_2 \cdots \a_K \otimes b_K \f$, where \f${\bf
         * A}\f$ is  \f$I \times K\f$ matrix, \f${\bf B}\f$ is \f$J \times K\f$,
         * and \f$
         * {\bf Z} \f$ is \f$ (IJ)\times K \f$ matrix. \f$ a_1 \cdots \a_K \f$
         * are the columns of \f${\bf A}\f$
         * In row major storage format:
         * \f$ {\bf Z}^T={\bf A}^T \odot {\bf B}^T = a_1 \otimes b_1
         * \quad a_2 \otimes b_2 \cdots \a_K \otimes b_K \f$, where \f${\bf
         * A}\f$ is  \f$K \times I\f$ matrix, \f${\bf B}\f$ is \f$K \times J\f$,
         * and \f$
         * {\bf Z} \f$ is \f$ K\times (IJ) \f$ matrix. \f$ a_1 \cdots \a_K \f$
         * are the rows of \f${\bf A}\f$
         * @param[in] layout Layout::ColMajor or Layout::RowMajor
         * @param[in] size size I
         * @param[in] size size J
         * @param[in] size size K
         * @param[in] X array
         * @param[in] Y array
         * @param[out] Z array
         */
        static void
        khatriRaoProduct(const Layout                         layout,
                         const size_type                      sizeI,
                         const size_type                      sizeJ,
                         const size_type                      sizeK,
                         const ValueType1 *                   A,
                         const ValueType2 *                   B,
                         scalar_type<ValueType1, ValueType2> *Z);

        /**
         * @brief Template for performing khatriRao but with variable stride
         * In column major storage format:
         * \f$ {\bf Z}={\bf A} \odot {\bf B} = a_1 \otimes b_1
         * \quad a_2 \otimes b_2 \cdots \a_K \otimes b_K \f$, where \f${\bf
         * A}\f$ is  \f$I \times K\f$ matrix, \f${\bf B}\f$ is \f$J \times K\f$,
         * and \f$
         * {\bf Z} \f$ is \f$ (IJ)\times K \f$ matrix. \f$ a_1 \cdots \a_K \f$
         * are the columns of \f${\bf A}\f$
         * In row major storage format:
         * \f$ {\bf Z}^T={\bf A}^T \odot {\bf B}^T = a_1 \otimes b_1
         * \quad a_2 \otimes b_2 \cdots \a_K \otimes b_K \f$, where \f${\bf
         * A}\f$ is  \f$K \times I\f$ matrix, \f${\bf B}\f$ is \f$K \times J\f$,
         * and \f$
         * {\bf Z} \f$ is \f$ K\times (IJ) \f$ matrix. \f$ a_1 \cdots \a_K \f$
         * are the rows of \f${\bf A}\f$
         * @param[in] layout Layout::ColMajor or Layout::RowMajor
         * @param[in] size size I
         * @param[in] size size J
         * @param[in] size size K
         * @param[in] X array
         * @param[in] Y array
         * @param[out] Z array
         */
        static void
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
          LinAlgOpContext<memorySpace> &       context);

        /**
         * @brief Template for performing
         * In column major storage format:
         * \f$ {\bf Z}={\bf A} \odot {\bf B} = a_1 \otimes b_1
         * \quad a_2 \otimes b_2 \cdots \a_K \otimes b_K \f$, where \f${\bf
         * A}\f$ is  \f$K \times I\f$ matrix, \f${\bf B}\f$ is \f$K \times J\f$,
         * and \f$
         * {\bf Z} \f$ is \f$ K\times (IJ) \f$ matrix. \f$ a_1 \cdots \a_K \f$
         * are the rows of \f${\bf A}\f$
         * In row major storage format:
         * \f$ {\bf Z}^T={\bf A}^T \odot {\bf B}^T = a_1 \otimes b_1
         * \quad a_2 \otimes b_2 \cdots \a_K \otimes b_K \f$, where \f${\bf
         * A}\f$ is  \f$I \times K\f$ matrix, \f${\bf B}\f$ is \f$J \times K\f$,
         * and \f$
         * {\bf Z} \f$ is \f$ (IJ)\times K \f$ matrix. \f$ a_1 \cdots \a_K \f$
         * are the columns of \f${\bf A}\f$
         * @param[in] layout Layout::ColMajor or Layout::RowMajor
         * @param[in] size size I
         * @param[in] size size J
         * @param[in] size size K
         * @param[in] X array
         * @param[in] Y array
         * @param[out] Z array
         */
        static void
        transposedKhatriRaoProduct(const Layout                         layout,
                                   const size_type                      sizeI,
                                   const size_type                      sizeJ,
                                   const size_type                      sizeK,
                                   const ValueType1 *                   A,
                                   const ValueType2 *                   B,
                                   scalar_type<ValueType1, ValueType2> *Z);

        /**
         * @brief Template for performing \f$ z = \alpha x + \beta y \f$
         * @param[in] size size of the array
         * @param[in] \f$ alpha \f$ scalar
         * @param[in] x array
         * @param[in] \f$ beta \f$ scalar
         * @param[in] y array
         * @param[out] z array
         */
        static void
        axpby(size_type                            size,
              scalar_type<ValueType1, ValueType2>  alpha,
              const ValueType1 *                   x,
              scalar_type<ValueType1, ValueType2>  beta,
              const ValueType2 *                   y,
              scalar_type<ValueType1, ValueType2> *z);

        /**
         * @brief Template for performing \f$ z = \alpha x + \beta y \f$
         * @param[in] size size of the array
         * @param[in] \f$ alpha \f$ vector
         * @param[in] x array
         * @param[in] \f$ beta \f$ vector
         * @param[in] y array
         * @param[out] z array
         */
        static void
        axpbyBlocked(const size_type                            size,
                     const size_type                            blockSize,
                     const scalar_type<ValueType1, ValueType2> *alpha,
                     const ValueType1 *                         x,
                     const scalar_type<ValueType1, ValueType2> *beta,
                     const ValueType2 *                         y,
                     scalar_type<ValueType1, ValueType2> *      z);

        /**
         * @brief Template for computing dot products numVec vectors in a multi Vector
         * @param[in] vecSize size of each vector
         * @param[in] numVec number of vectors in the multi Vector
         * @param[in] multiVecDataX multi vector data in row major format i.e.
         * vector index is the fastest index
         * @param[in] multiVecDataY multi vector data in row major format i.e.
         * vector index is the fastest index
         * @param[in] opX blasLapack::ScalarOp defining the operation on each
         * entry of multiVecDataX. The available options are
         * (a) blasLapack::ScalarOp::Identity (identity operation on a scalar),
         * and (b) blasLapack::ScalarOp::Conj (complex conjugate on a scalar)
         * @param[in] opY blasLapack::ScalarOp defining the operation on each
         * entry of multiVecDataY.
         * @param[out] multiVecDotProduct multi vector dot product of size
         * numVec
         *
         */
        static void
        dotMultiVector(size_type                            vecSize,
                       size_type                            numVec,
                       const ValueType1 *                   multiVecDataX,
                       const ValueType2 *                   multiVecDataY,
                       const ScalarOp &                     opX,
                       const ScalarOp &                     opY,
                       scalar_type<ValueType1, ValueType2> *multiVecDotProduct,
                       LinAlgOpContext<memorySpace> &       context);
      };

      template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
      class KernelsOneValueType
      {
      public:
        /**
         * @brief Template for computing \f$ l_{\inf} \f$ norms of all the numVec vectors in a multi Vector
         * @param[in] vecSize size of each vector
         * @param[in] numVec number of vectors in the multi Vector
         * @param[in] multiVecData multi vector data in row major format i.e.
         * vector index is the fastest index
         *
         * @return \f$ l_{\inf} \f$  norms of all the vectors
         */
        static std::vector<double>
        amaxsMultiVector(size_type        vecSize,
                         size_type        numVec,
                         const ValueType *multiVecData);

        /**
         * @brief Template for computing \f$ l_2 \f$ norms of all the numVec vectors in a multi Vector
         * @param[in] vecSize size of each vector
         * @param[in] numVec number of vectors in the multi Vector
         * @param[in] multiVecData multi vector data in row major format i.e.
         * vector index is the fastest index
         *
         * @return \f$ l_2 \f$  norms of all the vectors
         */
        static std::vector<double>
        nrms2MultiVector(size_type                     vecSize,
                         size_type                     numVec,
                         const ValueType *             multiVecData,
                         LinAlgOpContext<memorySpace> &context);
      };

#ifdef DFTEFE_WITH_DEVICE
      template <typename ValueType1, typename ValueType2>
      class KernelsTwoValueTypes<ValueType1,
                                 ValueType2,
                                 dftefe::utils::MemorySpace::DEVICE>
      {
      public:
        static void
        ascale(size_type                            size,
               ValueType1                           alpha,
               const ValueType2 *                   x,
               scalar_type<ValueType1, ValueType2> *z);

        /*
         * @brief Template for performing \f$ z = 1 /x$, does not check if x[i] is zero
         * @param[in] size size of the array
         * @param[in] x array
         * @param[out] z array
         */
        static void
        reciprocalX(size_type                            size,
                    const ValueType1                     alpha,
                    const ValueType2 *                   x,
                    scalar_type<ValueType1, ValueType2> *z);

        static void
        hadamardProduct(size_type                            size,
                        const ValueType1 *                   x,
                        const ValueType2 *                   y,
                        scalar_type<ValueType1, ValueType2> *z);

        static void
        hadamardProduct(size_type                            size,
                        const ValueType1 *                   x,
                        const ValueType2 *                   y,
                        const ScalarOp &                     opx,
                        const ScalarOp &                     opy,
                        scalar_type<ValueType1, ValueType2> *z);


        static void
        khatriRaoProduct(const Layout                         layout,
                         const size_type                      sizeI,
                         const size_type                      sizeJ,
                         const size_type                      sizeK,
                         const ValueType1 *                   A,
                         const ValueType2 *                   B,
                         scalar_type<ValueType1, ValueType2> *Z);

        static void
        axpby(size_type                            size,
              scalar_type<ValueType1, ValueType2>  alpha,
              const ValueType1 *                   x,
              scalar_type<ValueType1, ValueType2>  beta,
              const ValueType2 *                   y,
              scalar_type<ValueType1, ValueType2> *z);

        static void
        dotMultiVector(
          size_type                            vecSize,
          size_type                            numVec,
          const ValueType1 *                   multiVecDataX,
          const ValueType2 *                   multiVecDataY,
          const ScalarOp &                     opX,
          const ScalarOp &                     opY,
          scalar_type<ValueType1, ValueType2> *multiVecDotProduct,
          LinAlgOpContext<dftefe::utils::MemorySpace::DEVICE> &context);
      };

      template <typename ValueType>
      class KernelsOneValueType<ValueType, dftefe::utils::MemorySpace::DEVICE>
      {
      public:
        static std::vector<double>
        amaxsMultiVector(size_type        vecSize,
                         size_type        numVec,
                         const ValueType *multiVecData);


        static std::vector<double>
        nrms2MultiVector(
          size_type                                            vecSize,
          size_type                                            numVec,
          const ValueType *                                    multiVecData,
          LinAlgOpContext<dftefe::utils::MemorySpace::DEVICE> &context);
      };

#endif

    } // namespace blasLapack
  }   // namespace linearAlgebra
} // namespace dftefe

#endif // dftefeKernels_h
