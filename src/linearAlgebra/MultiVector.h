/******************************************************************************
 * Copyright (c) 2021.                                                        *
 * The Regents of the University of Michigan and DFT-EFE developers.          *
 *                                                                            *
 * This file is part of the DFT-EFE code.                                     *
 *                                                                            *
 * DFT-EFE is free software: you can redistribute it and/or modify            *
 *   it under the terms of the Lesser GNU General Public License as           *
 *   published by the Free Software Foundation, either version 3 of           *
 *   the License, or (at your option) any later version.                      *
 *                                                                            *
 * DFT-EFE is distributed in the hope that it will be useful, but             *
 *   WITHOUT ANY WARRANTY; without even the implied warranty                  *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                     *
 *   See the Lesser GNU General Public License for more details.              *
 *                                                                            *
 * You should have received a copy of the GNU Lesser General Public           *
 *   License at the top level of DFT-EFE distribution.  If not, see           *
 *   <https://www.gnu.org/licenses/>.                                         *
 ******************************************************************************/

/*
 * @author Sambit Das, Bikash Kanungo
 */


#ifndef dftefeMultiVector_h
#define dftefeMultiVector_h

#include <utils/TypeConfig.h>
#include <utils/Defaults.h>
#include <utils/MemoryStorage.h>
#include <utils/MPITypes.h>
#include <utils/MPIPatternP2P.h>
#include <utils/MPICommunicatorP2P.h>
#include <linearAlgebra/VectorAttributes.h>
#include <linearAlgebra/LinAlgOpContext.h>
#include <linearAlgebra/BlasLapackTypedef.h>
#include <memory>
namespace dftefe
{
  namespace linearAlgebra
  {
    /**
     * @brief An class template to encapsulate a MultiVector.
     * A MultiVector is a collection of \f$N\f$ vectors belonging to the same
     * finite-dimensional vector space, where usual notion of vector size
     * denotes the dimension of the vector space. Note that this in the
     * mathematical sense and not in the sense of an multi-dimensional array.The
     * MultiVector is stored contiguously with the vector index being the
     * fastest index, or in other words a matrix of size \f$M \times N\f$ in row
     * major format with \f$M \f$ denoting the dimension of the vector space
     * (size of individual vector).
     *
     * This class handles both serial and distributed MultiVector
     * in a unfied way. There are different constructors provided for the
     * serial and distributed case.
     *
     * The serial MultiVector, as the name suggests, resides entirely in a
     * processor.
     *
     * The distributed MultiVector, on the other hand, is distributed across a
     * set of processors. The storage of each of the \f$N\f$ vectors in the
     * distributed MultiVector in a processor follows along similar lines to
     * a distributed Vector object and comprises of two parts:
     *   1. <b>locally owned part</b>: A part of the distribute MultiVector,
     * defined through a contiguous range of indices \f$[a,b)\f$ (\f$a\f$ is
     * included, but \f$b\f$ is not), for which the current processor is the
     * sole owner. The size of the locally owned part (i.e., \f$b-a\f$) is
     * termed as \e locallyOwnedSize. Note that the range of indices that
     * comprises the locally owned part (i.e., \f$[a,b)\f$) is same for all the
     * \f$N\f$ vectors in the MultiVector
     *   2. <b>ghost part</b>: Part of the MultiVector, defined
     *      through a set of ghost indices, that are owned by other processors.
     * The size of ghost indices for each vector is termed as \e ghostSize. Note
     * that the set of indices that define the ghost indices are same for all
     * the \f$N\f$ vectors in the MultiVector
     *
     * The global size of each vector in the distributed MultiVector
     * (i.e., the number of unique indices across all the processors) is simply
     * termed as \e size. Additionally, we define \e localSize = \e
     * locallyOwnedSize + \e ghostSize.
     *
     * We handle the serial MultiVector as a special case of the distributed
     * MultiVector, wherein \e size = \e locallyOwnedSize and \e ghostSize = 0.
     *
     * @note While typically one would link to an MPI library while compiling this class,
     * care is taken to seamlessly allow usage of this class even while not
     * linking to an MPI library. To do so, we have our own MPI wrappers that
     * redirect to the MPI library's function calls and definitions while
     * linking to an MPI library. While not linking to an MPI library, the MPI
     * wrappers provide equivalent functions and definitions that mimic the MPI
     * functions and definitions, albeit for a single processor. This allows the
     * user of this class to seamlessly switch between linking and de-linking to
     * an MPI library without any change in the code and with the expected
     * behavior.
     *
     * @note Note that the case of not linking to an MPI library and the case of
     * creating a serial mult-Vector are two independent things. One can still
     * create a serial MultiVector while linking to an MPI library and
     * running the code across multipe processors. That is, one can create a
     * serial MultiVector in one or more than one of the set of processors used
     * when running in parallel. Internally, we handle this by using
     * MPI_COMM_SELF as our MPI_Comm for the serial MultiVector (i.e., the
     * processor does self communication). However, while not linking to an MPI
     * library (which by definition means running on a single processor), there
     * is no notion of communication (neither with self nor with other
     * processors). In such case, both serial and distributed mult-Vector mean
     * the same thing and the MPI wrappers ensure the expected behavior (i.e.,
     * the behavior of a MultiVector while using just one processor)
     *
     * @tparam template parameter ValueType defines underlying datatype being stored
     *  in the MultiVector (i.e., int, double, complex<double>, etc.)
     * @tparam template parameter memorySpace defines the MemorySpace (i.e., HOST or
     * DEVICE) in which the MultiVector must reside.
     *
     * @note Broadly, there are two ways of constructing a distributed MultiVector.
     *  1. [<b>Prefered and efficient approach</b>] The first approach takes a
     *     pointer to an MPIPatternP2P as an input argument (along with other
     * arguments). The MPIPatternP2P, in turn, contains all the information
     *     regarding the locally owned and ghost part of the MultiVector as well
     * as the interaction map between processors. This is the most efficient way
     * of constructing a distributed MultiVector as it allows for reusing of an
     *     already constructed MPIPatternP2P.
     *  2. [<b> Expensive approach</b>] The second approach takes in the
     *     locally owned, ghost indices or the total number of indices
     *     across all the processors and internally creates an
     *     MPIPatternP2P object. Given that the creation of an MPIPatternP2P is
     *     expensive, this route of constructing a distributed MultiVector
     *     <b>should</b> be avoided.
     */
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    class MultiVector
    {
    public:
      //
      // typedefs
      //
      using Storage    = dftefe::utils::MemoryStorage<ValueType, memorySpace>;
      using value_type = typename Storage::value_type;
      using pointer    = typename Storage::pointer;
      using reference  = typename Storage::reference;
      using const_reference = typename Storage::const_reference;
      using iterator        = typename Storage::iterator;
      using const_iterator  = typename Storage::const_iterator;

    public:
      /**
       * @brief Default Constructor
       */
      MultiVector() = default;

      /**
       * @brief Default Destructor
       */
      ~MultiVector() = default;

      /**
       * @brief Constructor for \b serial MultiVector with vector size, number of vectors and initial value arguments
       * @param[in] size size of each vector in the MultiVector
       * @param[in] numVectors number of vectors in the MultiVector
       * @param[in] initVal initial value of elements of the MultiVector
       * @param[in] linAlgOpContext shared pointer to LinAlgOpContext object
       *
       */
      MultiVector(const size_type                               size,
                  const size_type                               numVectors,
                  std::shared_ptr<LinAlgOpContext<memorySpace>> linAlgOpContext,
                  const ValueType initVal = utils::Types<ValueType>::zero);

      /**
       * @brief Constructor for a \serial MultiVector with a predefined
       * MultiVector::Storage (i.e., utils::MemoryStorage).
       * This constructor transfers the ownership of the input Storage to the
       * MultiVector. This is useful when one does not want to allocate new
       * memory and instead use memory allocated in the MultiVector::Storage
       * (i.e., utils::MemoryStorage).
       * The \e locallyOwnedSize, \e ghostSize, etc., are automatically set
       * using the size of the input Storage object.
       *
       * @param[in] storage unique_ptr to MultiVector::Storage whose ownership
       * is to be transfered to the MultiVector
       * @param[in] numVectors number of vectors in the MultiVector
       * @param[in] linAlgOpContext shared pointer to LinAlgOpContext object
       * @note This Constructor transfers the ownership from the input
       * unique_ptr \p storage to the internal data member of the MultiVector.
       * Thus, after the function call \p storage will point to NULL and any
       * access through \p storage will lead to <b>undefined behavior</b>.
       *
       */
      MultiVector(
        std::unique_ptr<typename MultiVector<ValueType, memorySpace>::Storage>
                                                      storage,
        size_type                                     numVectors,
        std::shared_ptr<LinAlgOpContext<memorySpace>> linAlgOpContext);

      /**
       * @brief Constructor for a \b distributed MultiVector based on an input MPIPatternP2P.
       *
       * @param[in] mpiPatternP2P A shared_ptr to const MPIPatternP2P
       * based on which the distributed MultiVector will be created.
       * @param[in] linAlgOpContext shared pointer to LinAlgOpContext object
       * @param[in] numVectors number of vectors in the MultiVector
       * @param[in] initVal value with which the MultiVector shoud be
       * initialized
       */
      MultiVector(std::shared_ptr<const utils::mpi::MPIPatternP2P<memorySpace>>
                                                                mpiPatternP2P,
                  std::shared_ptr<LinAlgOpContext<memorySpace>> linAlgOpContext,
                  const size_type                               numVectors,
                  const ValueType initVal = utils::Types<ValueType>::zero);

      /**
       * @brief Constructor for a \b distributed MultiVector with a predefined
       * MultiVector::Storage (i.e., utils::MemoryStorage) and MPIPatternP2P.
       * This constructor transfers the ownership of the input Storage to the
       * MultiVector. This is useful when one does not want to allocate new
       * memory and instead use memory allocated in the input
       * MultiVector::Storage (i.e., utils::MemoryStorage).
       *
       * @param[in] storage unique_ptr to MultiVector::Storage whose ownership
       * is to be transfered to the MultiVector
       * @param[in] mpiPatternP2P A shared_ptr to const MPIPatternP2P
       * based on which the distributed MultiVector will be created.
       * @param[in] linAlgOpContext shared pointer to LinAlgOpContext object
       * @param[in] numVectors number of vectors in the MultiVector
       *
       * @note This Constructor transfers the ownership from the input
       * unique_ptr \p storage to the internal data member of the MultiVector.
       * Thus, after the function call \p storage will point to NULL and any
       * access through \p storage will lead to <b>undefined behavior</b>.
       *
       */
      MultiVector(
        std::unique_ptr<typename MultiVector<ValueType, memorySpace>::Storage>
          &storage,
        std::shared_ptr<const utils::mpi::MPIPatternP2P<memorySpace>>
                                                      mpiPatternP2P,
        std::shared_ptr<LinAlgOpContext<memorySpace>> linAlgOpContext,
        const size_type                               numVectors);

      /**
       * @brief Constructor for a \distributed MultiVector based on locally
       * owned and ghost indices.
       * @note This way of construction is expensive. One should use the other
       * constructor based on an input MPIPatternP2P as far as possible.
       *
       * @param[in] locallyOwnedRange a pair \f$(a,b)\f$ which defines a range
       * of indices (continuous) that are owned by the current processor.
       * @param[in] ghostIndices vector containing an ordered set of ghost
       * indices (ordered in increasing order and non-repeating)
       * @param[in] mpiComm utils::mpi::MPIComm object associated with the group
       * of processors across which the MultiVector is to be distributed
       * @param[in] linAlgOpContext shared pointer to LinAlgOpContext object
       * @param[in] numVectors number of vectors in the MultiVector
       * @param[in] initVal value with which the MultiVector shoud be
       * initialized
       *
       * @note The locallyOwnedRange should be an open interval where the start
       * index is included, but the end index is not included.
       */
      MultiVector(
        const std::pair<global_size_type, global_size_type> locallyOwnedRange,
        const std::vector<global_size_type> &               ghostIndices,
        const utils::mpi::MPIComm &                         mpiComm,
        std::shared_ptr<LinAlgOpContext<memorySpace>>       linAlgOpContext,
        const size_type                                     numVectors,
        ValueType initVal = utils::Types<ValueType>::zero);

      /**
       * @brief Constructor for a special case of \b distributed MultiVector where none
       * none of the processors have any ghost indices.
       * @note This way of construction is expensive. One should use the other
       * constructor based on an input MPIPatternP2P as far as possible.
       *
       * @param[in] locallyOwnedRange a pair \f$(a,b)\f$ which defines a range
       * of indices (continuous) that are owned by the current processor.
       * @param[in] mpiComm utils::mpi::MPIComm object associated with the group
       * of processors across which the MultiVector is to be distributed
       * @param[in] linAlgOpContext shared pointer to LinAlgOpContext object
       * @param[in] numVectors number of vectors in the MultiVector
       * @param[in] initVal value with which the MultiVector shoud be
       * initialized
       *
       * @note The locallyOwnedRange should be an open interval where the start index included,
       * but the end index is not included.
       */
      MultiVector(
        const std::pair<global_size_type, global_size_type> locallyOwnedRange,
        const utils::mpi::MPIComm &                         mpiComm,
        std::shared_ptr<LinAlgOpContext<memorySpace>>       linAlgOpContext,
        const size_type                                     numVectors,
        const ValueType initVal = utils::Types<ValueType>::zero);


      /**
       * @brief Constructor for a \b distributed MultiVector based on total number of global indices.
       * The resulting MultiVector will not contain any ghost indices on any of
       * the processors. Internally, the vector is divided to ensure as much
       * equitable distribution across all the processors much as possible.
       * @note This way of construction is expensive. One should use the other
       * constructor based on an input MPIPatternP2P as far as possible.
       * Further, the decomposition is not compatible with other ways of
       * distributed MultiVector construction.
       * @param[in] globalSize Total number of global indices that is
       * distributed over the processors.
       * @param[in] mpiComm utils::mpi::MPIComm object associated with the group
       * of processors across which the MultiVector is to be distributed
       * @param[in] linAlgOpContext shared pointer to LinAlgOpContext object
       * @param[in] numVectors number of vectors in the MultiVector
       * @param[in] initVal value with which the MultiVector shoud be
       * initialized
       */
      MultiVector(const global_size_type                        globalSize,
                  const utils::mpi::MPIComm &                   mpiComm,
                  std::shared_ptr<LinAlgOpContext<memorySpace>> linAlgOpContext,
                  const size_type                               numVectors,
                  const ValueType initVal = utils::Types<ValueType>::zero);

      /**
       * @brief Copy constructor
       * @param[in] u MultiVector object to copy from
       */
      MultiVector(const MultiVector &u);

      /**
       * @brief Copy constructor with reinitialisation
       * @param[in] u MultiVector object to copy from
       * @param[in] initVal Initial value of the MultiVector
       */
      MultiVector(const MultiVector &u,
                  const ValueType    initVal = utils::Types<ValueType>::zero);

      /**
       * @brief Move constructor
       * @param[in] u MultiVector object to move from
       */
      MultiVector(MultiVector &&u) noexcept;

      /**
       * @brief Copy assignment operator
       * @param[in] u const reference to MultiVector object to copy
       * from
       * @return reference to this object after copying data from u
       */
      MultiVector &
      operator=(const MultiVector &u);

      /**
       * @brief Move assignment operator
       * @param[in] u const reference to MultiVector object to move
       * from
       * @return reference to this object after moving data from u
       */
      MultiVector &
      operator=(MultiVector &&u);

      /**
       * @brief Return iterator pointing to the beginning of MultiVector data.
       *
       * @returns Iterator pointing to the beginning of MultiVector.
       */
      iterator
      begin();

      /**
       * @brief Return iterator pointing to the beginning of MultiVector
       * data.
       *
       * @returns Constant iterator pointing to the beginning of
       * MultiVector.
       */
      const_iterator
      begin() const;

      /**
       * @brief Return iterator pointing to the end of MultiVector data.
       *
       * @returns Iterator pointing to the end of MultiVector.
       */
      iterator
      end();

      /**
       * @brief Return iterator pointing to the end of MultiVector data.
       *
       * @returns Constant iterator pointing to the end of
       * MultiVector.
       */
      const_iterator
      end() const;

      /**
       * @brief Return the raw pointer to the MultiVector data
       * @return pointer to data
       */
      ValueType *
      data();

      /**
       * @brief Return the constant raw pointer to the MultiVector data
       * @return pointer to const data
       */
      const ValueType *
      data() const;


      /**
       * @brief Set all entries of the MultiVector to a given value
       *
       * @param[in] val The value to which the entries are to be set
       */
      void
      setValue(const ValueType val);

      /**
       * @brief Returns \f$ l_2 \f$ norms of all the \f$N\f$ vectors in the  MultiVector
       * @return \f$ l_2 \f$  norms of the various vectors as std::vector<double> type
       */
      std::vector<double>
      l2Norms() const;

      /**
       * @brief Returns \f$ l_{\inf} \f$ norms of all the \f$N\f$ vectors in the  MultiVector
       * @return \f$ l_{\inf} \f$  norms of the various vectors as std::vector<double> type
       */
      std::vector<double>
      lInfNorms() const;

      /**
       * @brief Returns the nunber of vectors in this multi-vector
       * @return d_numVectors
       */
      size_type
      getNumberComponents() const;

      void
      updateGhostValues(const size_type communicationChannel = 0);

      void
      accumulateAddLocallyOwned(const size_type communicationChannel = 0);

      void
      updateGhostValuesBegin(const size_type communicationChannel = 0);

      void
      updateGhostValuesEnd();

      void
      accumulateAddLocallyOwnedBegin(const size_type communicationChannel = 0);

      void
      accumulateAddLocallyOwnedEnd();

      bool
      isCompatible(const MultiVector<ValueType, memorySpace> &rhs) const;

      std::shared_ptr<const utils::mpi::MPIPatternP2P<memorySpace>>
      getMPIPatternP2P() const;

      std::shared_ptr<LinAlgOpContext<memorySpace>>
      getLinAlgOpContext() const;

      global_size_type
      globalSize() const;

      size_type
      localSize() const;

      size_type
      locallyOwnedSize() const;

      size_type
      ghostSize() const;

      size_type
      numVectors() const;

    protected:
      std::unique_ptr<Storage>                      d_storage;
      std::shared_ptr<LinAlgOpContext<memorySpace>> d_linAlgOpContext;
      VectorAttributes                              d_vectorAttributes;
      size_type                                     d_localSize;
      global_size_type                              d_globalSize;
      size_type                                     d_locallyOwnedSize;
      size_type                                     d_ghostSize;
      size_type                                     d_numVectors;
      std::unique_ptr<utils::mpi::MPICommunicatorP2P<ValueType, memorySpace>>
        d_mpiCommunicatorP2P;
      std::shared_ptr<const utils::mpi::MPIPatternP2P<memorySpace>>
        d_mpiPatternP2P;
    };

    //
    // helper functions
    //

    /**
     * @brief Perform \f$ w = au + bv \f$
     * @param[in] a scalar
     * @param[in] u first MultiVector on the right
     * @param[in] b scalar
     * @param[in] v second MultiVector on the right
     * @param[out] w resulting MultiVector
     *
     * @tparam ValueType1 DataType (double, float, complex<double>, etc.) of
     *  u vector
     * @tparam ValueType2 DataType (double, float, complex<double>, etc.) of
     *  v vector
     * @tparam memorySpace defines the MemorySpace (i.e., HOST or
     * DEVICE) in which the vector must reside.
     * @note The datatype of the scalars a, b, and the resultant MultiVector w is
     * decided through a union of ValueType1 and ValueType2
     * (e.g., union of double and complex<double> is complex<double>)
     */
    template <typename ValueType1,
              typename ValueType2,
              utils::MemorySpace memorySpace>
    void
    add(blasLapack::scalar_type<ValueType1, ValueType2> a,
        const MultiVector<ValueType1, memorySpace> &    u,
        blasLapack::scalar_type<ValueType1, ValueType2> b,
        const MultiVector<ValueType2, memorySpace> &    v,
        MultiVector<blasLapack::scalar_type<ValueType1, ValueType2>,
                    memorySpace> &                      w);


    /**
     * @brief Perform \f$ w = au + bv \f$
     * @param[in] a vector
     * @param[in] u first MultiVector on the right
     * @param[in] b vector
     * @param[in] v second MultiVector on the right
     * @param[out] w resulting MultiVector
     *
     * @tparam ValueType1 DataType (double, float, complex<double>, etc.) of
     *  u vector
     * @tparam ValueType2 DataType (double, float, complex<double>, etc.) of
     *  v vector
     * @tparam memorySpace defines the MemorySpace (i.e., HOST or
     * DEVICE) in which the vector must reside.
     * @note The datatype of the scalars a, b, and the resultant MultiVector w is
     * decided through a union of ValueType1 and ValueType2
     * (e.g., union of double and complex<double> is complex<double>)
     */
    template <typename ValueType1,
              typename ValueType2,
              utils::MemorySpace memorySpace>
    void
    add(
        const std::vector<blasLapack::scalar_type<ValueType1, ValueType2>> & a,
        const MultiVector<ValueType1, memorySpace> &    u,
        const std::vector<blasLapack::scalar_type<ValueType1, ValueType2>> & b,
        const MultiVector<ValueType2, memorySpace> &    v,
        MultiVector<blasLapack::scalar_type<ValueType1, ValueType2>,
                    memorySpace> &   w);



    /**
     * @brief Perform \f$d_I$\f = dot product of op(\f$\mathbf{u}_I$\f)
     * and op(\f$\mathbf{v}_I$\f),
     * where \f$\mathbf{u}_I$\f and \f$\mathbf{v}_I$\f are the \f$I$\f-th
     * vector in the MultiVector. In other words,
     * \f$ d_I = \sum_i op(\mathbf{u}_{I,i}) op(\mathbf{v}_{I,i})$\f
     * where op is an operation of a scalar and can be
     * (a) blasLapack::ScalarOp::Identity for op(x) = x (the usual dot product)
     * or (b) blasLapack::ScalarOp::ComplexConjugate for op(x) = complex
     * conjugate of x
     *
     * The output resides on utils::MemorySpace::HOST (i.e., CPU)
     *
     * @param[in] u first MultiVector
     * @param[in] v second MultiVector
     * @param[in] opU blasLapack::ScalarOp for u MultiVector
     * @param[in] opV blasLapack::ScalarOp for v MultiVector
     * @param[out] dotPords STL vector where the I-th element contains the
     * the dot product between opU(u_I) and opV(v_I), where u_I and v_I are
     * I-th vector from u and v, respectively
     *
     * @tparam ValueType1 DataType (double, float, complex<double>, etc.) of
     *  u vector
     * @tparam ValueType2 DataType (double, float, complex<double>, etc.) of
     *  v vector
     * @tparam memorySpace defines the MemorySpace (i.e., HOST or
     * DEVICE) in which the vector must reside.
     * @note The dotProds must be appropriately allocated before calling
     *  this function
     * @note The datatype of the dot product is
     * decided through a union of ValueType1 and ValueType2
     * (e.g., union of double and complex<double> is complex<double>)
     */
    template <typename ValueType1,
              typename ValueType2,
              utils::MemorySpace memorySpace>
    void
    dot(const MultiVector<ValueType1, memorySpace> &                  u,
        const MultiVector<ValueType2, memorySpace> &                  v,
        std::vector<blasLapack::scalar_type<ValueType1, ValueType2>> &dotProds,
        const blasLapack::ScalarOp &opU = blasLapack::ScalarOp::Identity,
        const blasLapack::ScalarOp &opV = blasLapack::ScalarOp::Identity);

    /**
     * @brief Same as the above dot() function but instead of returning the
     * result on utils::MemorySpace:::HOST it returns it in a user-provided
     * memory on the input MemorySpace.
     *
     * @param[in] u first MultiVector
     * @param[in] v second MultiVector
     * @param[in] opU blasLapack::ScalarOp for u MultiVector
     * @param[in] opV blasLapack::ScalarOp for v MultiVector
     * @param[out] dotProd Pointer to dot products between opU(u_I) and
     * opV(v_I), where u_I and v_I are I-th vector from u and v, respectively
     *
     * @note The pointer dotProd must be properly allocated within the memory
     *  space of the MultiVector involved
     *
     * @tparam ValueType1 DataType (double, float, complex<double>, etc.) of
     *  u vector
     * @tparam ValueType2 DataType (double, float, complex<double>, etc.) of
     *  v vector
     * @tparam memorySpace defines the MemorySpace (i.e., HOST or
     * DEVICE) in which the vector must reside.
     * @note The datatype of the dot product is
     * decided through a union of ValueType1 and ValueType2
     * (e.g., union of double and complex<double> is complex<double>)
     *
     */
    template <typename ValueType1,
              typename ValueType2,
              utils::MemorySpace memorySpace>
    void
    dot(const MultiVector<ValueType1, memorySpace> &     u,
        const MultiVector<ValueType2, memorySpace> &     v,
        blasLapack::scalar_type<ValueType1, ValueType2> *dotProd,
        const blasLapack::ScalarOp &opU = blasLapack::ScalarOp::Identity,
        const blasLapack::ScalarOp &opV = blasLapack::ScalarOp::Identity);


  } // end of namespace linearAlgebra
} // end of namespace dftefe
#include <linearAlgebra/MultiVector.t.cpp>
#endif // dftefeMultiVector_h
