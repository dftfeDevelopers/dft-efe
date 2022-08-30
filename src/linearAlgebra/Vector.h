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
 * @author Bikash Kanungo
 */


#ifndef dftefeVector_h
#define dftefeVector_h

#include <utils/MemoryStorage.h>
#include <utils/MPITypes.h>
#include <utils/MPIPatternP2P.h>
#include <utils/MPICommunicatorP2P.h>
#include <utils/TypeConfig.h>
#include <linearAlgebra/VectorAttributes.h>
#include <linearAlgebra/LinAlgOpContext.h>
#include <linearAlgebra/BlasLapackTypedef.h>
#include <memory>
namespace dftefe
{
  namespace linearAlgebra
  {
    /**
     * @brief A class that encapsulates a vector.
     * This is a vector in the mathematical sense and not in the sense of an
     * array or STL container.
     * This class handles both serial and distributed
     * vector in a unfied way. There are different constructors provided for the
     * serial and distributed case.
     *
     * The serial Vector, as the name suggests, resides entirely in a processor.
     *
     * The distributed Vector, on the other hand, is distributed across a set of
     * processors. The storage of the distributed Vector in a processor
     * comprises of two parts:
     *   1. <b>locally owned part</b>: A part of the distribute dVector, defined
     * through a contiguous range of indices \f$[a,b)\f$ (\f$a\f$ is included,
     * but \f$b\f$ is not), for which the current processor is the sole owner.
     *      The size of the locally owned part (i.e., \f$b-a\f$) is termed as \e
     * locallyOwnedSize.
     *   2. <b>ghost part</b>: Part of the distributed Vector, defined through a
     * set of ghost indices, that are owned by other processors. The size of
     * ghost part is termed as \e ghostSize.
     *
     * Both the <b>locally owned part</b> and the <b>ghost part</b> are stored
     * in a contiguous memory inside a MemoryStorage object, with the <b>locally
     * owned part</b> stored first. The global size of the distributed Vector
     * (i.e., the number of unique indices across all the processors) is simply
     * termed as \e size. Additionally, we define \e localSize = \e
     * locallyOwnedSize + \e ghostSize.
     *
     * We handle the serial Vector as a special case of the distributed Vector,
     * wherein \e size = \e locallyOwnedSize and \e ghostSize = 0.
     *
     * @note While typically one would link to an MPI library while compiling this class,
     * care is taken to seamlessly allow usage of this class even while not
     * linking to an MPI library. To do so, we have our own MPI wrappers that
     * defaults to the MPI library's function calls and definitions while
     * linking to an MPI library and provides a serial equivalent of those
     * functions while not linking to an MPI library. This allows the user of
     * this class to seamlessly switch between linking and de-linking to an MPI
     * library without any change in the code and with the expected behavior.
     *
     * @note Note that the case of not linking to an MPI library and the case of
     * creating a serial mult-Vector are two independent things. One can still
     * create a serial Vector while linking to an MPI library and
     * running the code across multipe processors. That is, one can create a
     * serial Vector in one or more than one of the set of processors used when
     * running in parallel. Internally, we handle this by using MPI_COMM_SELF
     * as our MPI_Comm for the serial Vector (i.e., the processor does self
     * communication). However, while not linking to an MPI library (which by
     * definition means running on a single processor), there is no notion of
     * communication (neither with self nor with other processors). In such
     * case, both serial and distributed mult-Vector mean the same thing and the
     * MPI wrappers ensure the expected behavior (i.e., the behavior of a Vector
     * while using just one processor)
     *
     * @note Broadly, there are two ways of constructing a distributed Vector.
     *  1. [<b>Prefered and efficient approach</b>] The first approach takes a
     *     pointer to an MPIPatternP2P as an input argument (along with other
     * arguments). The MPIPatternP2P, in turn, contains all the information
     *     regarding the locally owned and ghost part of the Vector as well as
     * the interaction map between processors. This is the most efficient way of
     *     constructing a distributed Vector as it allows for reusing of an
     *     already constructed MPIPatternP2P.
     *  2. [<b> Expensive approach</b>] The second approach takes in the
     *     locally owned, ghost indices or the total number of indices
     *     across all the processors and internally creates an
     *     MPIPatternP2P object. Given that the creation of an MPIPatternP2P is
     *     expensive, this route of constructing a distributed Vector
     *     <b>should</b> be avoided.
     *
     * @tparam template parameter ValueType defines underlying datatype being stored
     *  in the vector (i.e., int, double, complex<double>, etc.)
     * @tparam template parameter memorySpace defines the MemorySpace (i.e., HOST or
     * DEVICE) in which the vector must reside.
     */
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    class Vector
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
       * @brief Default constructor
       */
      Vector() = default;

      /**
       * @brief Default Destructor
       */
      ~Vector() = default;

      /**
       * @brief Constructor for a <b>serial</b> Vector with size and initial value arguments
       * @param[in] size size of the serial Vector
       * @param[in] linAlgOpContext shared pointer to LinAlgOpContext object
       * @param[in] initVal initial value of elements of the SerialVector
       */
      Vector(size_type                                     size,
             std::shared_ptr<LinAlgOpContext<memorySpace>> linAlgOpContext,
             ValueType initVal = ValueType());

      /**
       * @brief Constructor for a <b>serial</b> Vector with predefined Storage (i.e., utils::MemoryStorage).
       * This Constructor transfers the ownership of input Storage to the
       * Vector. This is useful when one does not want to allocate new memory
       * and instead use memory allocated in the Vector::Storage (i.e.,
       * MemoryStorage). The \e locallyOwnedSize, \e ghostSize, etc., are
       * automatically set using the size of the \p storage.
       *
       * @param[in] storage unique_ptr to Storage whose ownership
       * is to be transfered to the Vector
       * @param[in] linAlgOpContext shared pointer to LinAlgOpContext object
       *
       * @note This Constructor transfers the ownership from the input unique_ptr \p storage to the internal data member of the Vector.
       * Thus, after the function call \p storage will point to \p NULL and any
       * access through \p storage will lead to <b>undefined behavior</b>.
       *
       */
      Vector(std::unique_ptr<typename Vector<ValueType, memorySpace>::Storage>
                                                           storage,
             std::shared_ptr<LinAlgOpContext<memorySpace>> linAlgOpContext);

      /**
       * @brief Constructor for a \b distributed Vector based on an input MPIPatternP2P.
       * This is the \p most prefered and optimal way of constructing a \b
       * distributed Vector, as one can directly use the information already
       * stored in the MPIPatternP2P
       *
       * @param[in] mpiPatternP2P A shared_ptr to const MPIPatternP2P
       * based on which the distributed Vector will be created.
       * @param[in] linAlgOpContext shared pointer to LinAlgOpContext object
       * @param[in] initVal value with which the Vector shoud be
       * initialized
       *
       */
      Vector(std::shared_ptr<const utils::mpi::MPIPatternP2P<memorySpace>>
                                                           mpiPatternP2P,
             std::shared_ptr<LinAlgOpContext<memorySpace>> linAlgOpContext,
             const ValueType initVal = ValueType());

      /**
       * @brief Constructor for a \b distributed Vector with a predefined Storage (i.e., utils::MemoryStorage) and MPIPatternP2P.
       * This Constructor transfers the ownership of input Storage to the
       * Vector. This is useful when one does not want to allocate new memory
       * and instead use memory allocated in the Storage (i.e., MemoryStorage).
       *
       * @param[in] storage unique_ptr to Vector::Storage whose ownership
       * is to be transfered to the Vector
       * @param[in] mpiPatternP2P A shared_ptr to const MPIPatternP2P
       * based on which the Vector will be created.
       * @param[in] linAlgOpContext shared pointer to LinAlgOpContext object
       *
       * @note This Constructor transfers the ownership from the input unique_ptr \p storage to the internal data member of the Vector.
       * Thus, after the function call \p storage will point to NULL and any
       * access through \p storage will lead to <b>undefined behavior</b>.
       *
       */
      Vector(std::unique_ptr<typename Vector<ValueType, memorySpace>::Storage>
               &storage,
             std::shared_ptr<const utils::mpi::MPIPatternP2P<memorySpace>>
                                                           mpiPatternP2P,
             std::shared_ptr<LinAlgOpContext<memorySpace>> linAlgOpContext);

      /**
       * @brief Constructor for a \b distributed Vector based on locally owned and ghost indices.
       * @note This way of construction is \p expensive. One should use the other
       * constructor based on an input MPIPatternP2P as far as possible.
       *
       * @param[in] locallyOwnedRange a pair \f$(a,b)\f$ which defines a range
       * of indices (continuous) that are owned by the current processor.
       * @param[in] ghostIndices vector containing an ordered set of ghost
       * indices (ordered in increasing order and non-repeating)
       * @param[in] mpiComm utils::mpi::MPIComm object associated with the group
       * of processors across which the Vector is to be distributed
       * @param[in] linAlgOpContext shared pointer to LinAlgOpContext object
       * @param[in] initVal value with which the Vector shoud be
       * initialized
       *
       * @note The locallyOwnedRange should be an open interval where the start index included,
       * but the end index is not included.
       */
      Vector(
        const std::pair<global_size_type, global_size_type> locallyOwnedRange,
        const std::vector<dftefe::global_size_type> &       ghostIndices,
        const utils::mpi::MPIComm &                         mpiComm,
        std::shared_ptr<LinAlgOpContext<memorySpace>>       linAlgOpContext,
        const ValueType                                     initVal);

      /**
       * @brief Constructor for a special case of \b distributed Vector where none
       * none of the processors have any ghost indices.
       * @note This way of construction is expensive. One should use the other
       * constructor based on an input MPIPatternP2P as far as possible.
       *
       * @param[in] locallyOwnedRange a pair \f$(a,b)\f$ which defines a range
       * of indices (continuous) that are owned by the current processor.
       * @param[in] mpiComm utils::mpi::MPIComm object associated with the group
       * of processors across which the Vector is to be distributed
       * @param[in] linAlgOpContext shared pointer to LinAlgOpContext object
       * @param[in] initVal value with which the Vector shoud be
       * initialized
       *
       * @note The locallyOwnedRange should be an open interval where the start index included,
       * but the end index is not included.
       */
      Vector(
        const std::pair<global_size_type, global_size_type> locallyOwnedRange,
        const utils::mpi::MPIComm &                         mpiComm,
        std::shared_ptr<LinAlgOpContext<memorySpace>>       linAlgOpContext,
        const ValueType initVal = ValueType());


      /**
       * @brief Constructor for a \b distributed Vector based on total number of global indices.
       * The resulting Vector will not contain any ghost indices on any of the
       * processors. Internally, the vector is divided to ensure as much
       * equitable distribution across all the processors much as possible.
       * @note This way of construction is expensive. One should use the other
       * constructor based on an input MPIPatternP2P as far as possible.
       * Further, the decomposition is not compatible with other ways of
       * distributed vector construction.
       * @param[in] globalSize Total number of global indices that is
       * distributed over the processors.
       * @param[in] mpiComm utils::mpi::MPIComm object associated with the group
       * of processors across which the Vector is to be distributed
       * @param[in] linAlgOpContext shared pointer to LinAlgOpContext object
       * @param[in] initVal value with which the Vector shoud be
       * initialized
       *
       *
       */
      Vector(const global_size_type                        globalSize,
             const utils::mpi::MPIComm &                   mpiComm,
             std::shared_ptr<LinAlgOpContext<memorySpace>> linAlgOpContext,
             const ValueType initVal = ValueType());


      /**
       * @brief Copy constructor
       * @param[in] u Vector object to copy from
       */
      Vector(const Vector &u);

      /**
       * @brief Copy constructor with reinitialisation
       * @param[in] u Vector object to copy from
       * @param[in] initVal Initial value of the vector
       */
      Vector(const Vector &u, ValueType initVal = ValueType());

      /**
       * @brief Move constructor
       * @param[in] u Vector object to move from
       */
      Vector(Vector &&u) noexcept;

      /**
       * @brief Copy assignment operator
       * @param[in] u const reference to Vector object to copy from
       * @return reference to this object after copying data from u
       */
      Vector &
      operator=(const Vector &u);

      /**
       * @brief Move assignment operator
       * @param[in] u const reference to Vector object to move from
       * @return reference to this object after moving data from u
       */
      Vector &
      operator=(Vector &&u);

      /**
       * @brief Return iterator pointing to the beginning of Vector data.
       *
       * @returns Iterator pointing to the beginning of Vector.
       */
      iterator
      begin();

      /**
       * @brief Return iterator pointing to the beginning of Vector
       * data.
       *
       * @returns Constant iterator pointing to the beginning of
       * Vector.
       */
      const_iterator
      begin() const;

      /**
       * @brief Return iterator pointing to the end of Vector data.
       *
       * @returns Iterator pointing to the end of Vector.
       */
      iterator
      end();

      /**
       * @brief Return iterator pointing to the end of Vector data.
       *
       * @returns Constant iterator pointing to the end of
       * Vector.
       */
      const_iterator
      end() const;

      /**
       * @brief Return the raw pointer to the Vector data
       * @return pointer to data
       */
      ValueType *
      data();

      /**
       * @brief Return the constant raw pointer to the Vector data
       * @return pointer to const data
       */
      const ValueType *
      data() const;

      /**
       * @brief Set all the entries of the Vector to a given value
       * @param[in] val The value to which all the entries in the Vector are
       * to be set
       */
      void
      setValue(const ValueType val);

      /**
       * @brief Returns \f$ l_2 \f$ norm of the Vector
       * @return \f$ l_2 \f$  norm of the vector
       */
      double
      l2Norm() const;

      /**
       * @brief Returns \f$ l_{\inf} \f$ norm of the Vector
       * @return \f$ l_{\inf} \f$  norm of the vector
       */
      double
      lInfNorm() const;

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
      isCompatible(const Vector<ValueType, memorySpace> &rhs) const;

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

    private:
      std::unique_ptr<Storage>                      d_storage;
      std::shared_ptr<LinAlgOpContext<memorySpace>> d_linAlgOpContext;
      VectorAttributes                              d_vectorAttributes;
      size_type                                     d_localSize;
      global_size_type                              d_globalSize;
      size_type                                     d_locallyOwnedSize;
      size_type                                     d_ghostSize;
      std::unique_ptr<utils::mpi::MPICommunicatorP2P<ValueType, memorySpace>>
        d_mpiCommunicatorP2P;
      std::shared_ptr<const utils::mpi::MPIPatternP2P<memorySpace>>
        d_mpiPatternP2P;
    };

    // helper functions

    /**
     * @brief Perform \f$ \mathbf{w} = a\mathbf{u} + b\mathbf{v} \f$
     * @param[in] a scalar
     * @param[in] u first Vector on the right
     * @param[in] b scalar
     * @param[in] v second Vector on the right
     * @param[out] w resulting Vector
     *
     * @tparam ValueType1 DataType (double, float, complex<double>, etc.) of
     *  u vector
     * @tparam ValueType2 DataType (double, float, complex<double>, etc.) of
     *  v vector
     * @tparam memorySpace defines the MemorySpace (i.e., HOST or
     * DEVICE) in which the vector must reside.
     * @note The datatype of the scalars a, b, and the resultant Vector w is
     * decided through a union of ValueType1 and ValueType2
     * (e.g., union of double and complex<double> is complex<double>)
     */
    template <typename ValueType1, ValueType2, utils::MemorySpace memorySpace>
    void
    add(
      blasLapack::scalar_type<ValueType1, ValueType2>                       a,
      const Vector<ValueType1, memorySpace> &                               u,
      blasLapack::scalar_type<ValueType1, ValueType2>                       b,
      const Vector<ValueType2, memorySpace> &                               v,
      Vector<blasLapack::scalar_type<ValueType1, ValueType2>, memorySpace> &w);

    /**
     * @brief Perform dot product of op(u) and op(v), i.e.,
     * evaluate \f$ alpha = \sum_i op(\mathbf{u}_i) op(\mathbf{v}_i)$\f,
     * where op is an operation of a scalar and can be
     * (a) blasLapack::ScalarOp::Identity for op(x) = x (the usual dot product)
     * or (b) blasLapack::ScalarOp::ComplexConjugate for op(x) = complex
     * conjugate of x
     *
     * The returned value resides on utils::MemorySpace::HOST (i.e., CPU)
     *
     * @param[in] u first Vector
     * @param[in] v second Vector
     * @param[in] opU blasLapack::ScalarOp for u Vector
     * @param[in] opV blasLapack::ScalarOp for v Vector
     * @return dot product of opU(u) and opV(v)
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
     */
    template <typename ValueType1, ValueType2, utils::MemorySpace memorySpace>
    blasLapack::scalar_type<ValueType1, ValueType2>
    dot(const Vector<ValueType1, memorySpace> &u,
        const Vector<ValueType2, memorySpace> &v,
        const blasLapack::ScalarOp &opU = blasLapack::ScalarOp::Identity,
        const blasLapack::ScalarOp &opV = blasLapack::ScalarOp::Identity);

    /**
     * @brief Same as the above dot() function but instead of returning the
     * result on utils::MemorySpace:::HOST it returns it in a user-provided
     * memory on the input MemorySpace.
     *
     * @param[in] u first Vector
     * @param[in] v second Vector
     * @param[in] opU blasLapack::ScalarOp for u Vector
     * @param[in] opV blasLapack::ScalarOp for v Vector
     * @param[out] dotProd Pointer to dot product of opU(u) and opV(v)
     * @note The pointer dotProd must be properly allocated
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
    template <typename ValueType1, ValueType2, utils::MemorySpace memorySpace>
    void
    dot(const Vector<ValueType1, memorySpace> &          u,
        const Vector<ValueType2, memorySpace> &          v,
        blasLapack::scalar_type<ValueType1, ValueType2> *dotProd,
        const blasLapack::ScalarOp &opU = blasLapack::ScalarOp::Identity,
        const blasLapack::ScalarOp &opV = blasLapack::ScalarOp::Identity);

  } // end of namespace linearAlgebra
} // end of namespace dftefe
#include <linearAlgebra/Vector.t.cpp>
#endif // dftefeVector_h
