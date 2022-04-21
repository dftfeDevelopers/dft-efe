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
 * @author Sambit Das
 */

#ifndef dftefeMultiVector_h
#define dftefeMultiVector_h

#include <linearAlgebra/VectorAttributes.h>
#include <linearAlgebra/BlasLapack.h>
#include <utils/MemoryStorage.h>
#include <utils/TypeConfig.h>
#include <memory>
namespace dftefe
{
  namespace linearAlgebra
  {
    /**
     * @brief An base class template which provides an interface for a multi vector.
     * This is a collection of \f$N\f$ vectors belonging to the same
     * finite-dimensional vector space, where usual notion of vector size
     * denotes the dimension of the vector space. Note that this in the
     * mathematical sense and not in the sense of an multi-dimensional array.The
     * multi vector is stored contiguously with the vector index being the
     * fastest index, or in other words a matrix of size \f$M \times N\f$ in row
     * major format with \f$M \f$ denoting the dimension of the vector space
     * (size of the vector).
     *
     * It provides the interface for two derived classes: SerialMultiVector and
     * DistributedMultiVector.
     *
     * SerialMultiVector, as the name suggests, resides entirely in a processor.
     *
     * DistributedMultiVector, on the other hand, is distributed across a set of
     * processors. The storage of each of the \f$N\f$ vectors in the
     * DistributedMultiVector in a processor follows along similar lines to
     * DistributedVector and comprises of two parts (considering the
     * \f$i^{\textrm{th}}\f$ vector component):
     *   1. <b>locally owned part</b>: A part of the DistributedMultiVector,
     * defined through a strided range of indices
     * \f$\{a*N+i,\,(a+1)*N+i,\,(a+2)*N+i,\,\cdots,b*N+i\}\f$
     * (\f$a+N*i\f$ is included, but \f$b*N+i\f$ is not), for which the current
     * processor is the sole owner. The size of the locally owned indices for
     * each vector is termed as \e locallyOwnedSize.
     *   2. <b>ghost part</b>: Part of the DistributedMultiVector, defined
     * through a set of ghost indices, that are owned by other processors. The
     * size of ghost indices for each vector is termed as \e ghostSize.
     *
     * The global size each vector in of the DistributedMultiVector
     * (i.e., the number of unique indices across all the processors) is simply
     * termed as \e size. Additionally, we define \e localSize = \e
     * locallyOwnedSize + \e ghostSize.
     *
     * @note For a SerialMultiVector, \e size = \e locallyOwnedSize and \e ghostSize = 0.
     *
     * @tparam template parameter ValueType defines underlying datatype being stored
     *  in the multi vector (i.e., int, double, complex<double>, etc.)
     * @tparam template parameter memorySpace defines the MemorySpace (i.e., HOST or
     * DEVICE) in which the multi vector must reside.
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
      virtual ~MultiVector() = default;

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
       * @brief Returns the global size of the vectors in the MultiVector (see top for explanation)
       * @returns global size of each vector
       */
      global_size_type
      size() const;

      /**
       * @brief Returns the size of the part of each vector in the MultiVector that
       * is locally owned in the current processor.
       * For a SerialMultiVector, it is same as the \e globalSize.
       * For a DistributedMultiVector, it is the size of the part of each vector
       * in the DistributedMultiVector, for which the current processor is the
       * sole owner.
       * @returns the \e locallyOwnedSize of each vector
       */
      size_type
      locallyOwnedSize() const;

      /**
       * @brief Returns the size of the \b ghost \b part (i.e., \e ghostSize) of each vector in the
       * MultiVector (see top for an explanation)
       * @returns the \e ghostSize of each vector
       */
      size_type
      ghostSize() const;

      /**
       * @brief Returns the combined size of the locally owned and the ghost part of each vector in
       * the MultiVector in the current processor.
       * For a SerialMultiVector, it is same as the \e globalSize. For a
       * DistributedMultiVector, it is the sum of \e locallyOwnedSize and \e
       * ghostSize.
       * @returns the local size of each vector
       */
      size_type
      localSize() const;

      /**
       * @brief Returns the number of vectors in the MultiVector
       * @returns the number of vectors in the MultiVector
       */
      size_type
      numVectors() const;

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
       * @brief Compound addition for elementwise addition lhs += rhs
       * @param[in] rhs the MultiVector to add
       * @return the original MultiVector
       * @throws exception if the sizes and type (SerialMultiVector or
       * DistributedMultiVector) are incompatible
       */
      MultiVector &
      operator+=(const MultiVector &rhs);

      /**
       * @brief Compound subtraction for elementwise addition lhs -= rhs
       * @param[in] rhs the vector to subtract
       * @return the original vector
       * @throws exception if the sizes and type (SerialMultiVector or
       * DistributedMultiVector) are incompatible
       */
      MultiVector &
      operator-=(const MultiVector &rhs);

      /**
       * @brief Returns a reference to the underlying storage (i.e., MemoryStorage object)
       * of the MultiVector.
       *
       * @return reference to the underlying MemoryStorage.
       */
      Storage &
      getValues();

      /**
       * @brief Returns a const reference to the underlying storage (i.e., MemoryStorage object)
       * of the MultiVector.
       *
       * @return const reference to the underlying MemoryStorage.
       */
      const Storage &
      getValues() const;

      /**
       * @brief Returns a shared pointer to underlyign BlasQueue.
       *
       * @return shared pointer to BlasQueue.
       */
      std::shared_ptr<blasLapack::blasQueueType<memorySpace>>
      getBlasQueue() const;

      /**
       * @brief Set values in the MultiVector using a user provided MultiVector::Storage object (i.e., MemoryStorage object).
       * The MemoryStorage may lie in a different memoryspace (say memSpace2)
       * than the MultiVector's memory space (memSpace). The function internally
       * does a data transfer from memSpace2 to memSpace.
       *
       * @param[in] storage const reference to MemoryStorage object from which
       * to set values into the MultiVector.
       * @throws exception if the size of the input storage is smaller than the
       * \e localSize (\e locallyOwnedSize + \e ghostSize) of the MultiVector
       */
      template <dftefe::utils::MemorySpace memorySpace2>
      void
      setValues(
        const typename MultiVector<ValueType, memorySpace2>::Storage &storage);

      /**
       * @brief Transfer ownership of a user provided MultiVector::Storage object (i.e., MemoryStorage object)
       * to the MultiVector. This is useful when a MemoryStorage has been
       * already been allocated and we need the the MultiVector to claim its
       * ownership. This avoids reallocation of memory.
       *
       * @param[in] storage unique_ptr to MemoryStorage object whose ownership
       * is to be passed to the MultiVector
       *
       * @note Since we are passing the ownership of the input storage to the MultiVector, the
       * storage will point to NULL after a call to this function. Accessing the
       * input storage pointer will lead to undefined behavior.
       *
       */
      void
      setStorage(std::unique_ptr<Storage> &storage);

      /**
       * @brief Returns a VectorAttributes object that stores various attributes
       * (e.g., Serial or Distributed etc)
       *
       * @return const reference to the VectorAttributes
       */
      const VectorAttributes &
      getVectorAttributes() const;

      //
      // virtual functions
      //

      /**
       * @brief Returns \f$ l_2 \f$ norms of all the \f$N\f$ vectors in the  MultiVector
       * @return \f$ l_2 \f$  norms of the various vectors as std::vector<double> type
       */
      virtual std::vector<double>
      l2Norms() const = 0;

      /**
       * @brief Returns \f$ l_{\inf} \f$ norms of all the \f$N\f$ vectors in the  MultiVector
       * @return \f$ l_{\inf} \f$  norms of the various vectors as std::vector<double> type
       */
      virtual std::vector<double>
      lInfNorms() const = 0;

      virtual void
      updateGhostValues(const size_type communicationChannel = 0) = 0;

      virtual void
      accumulateAddLocallyOwned(const size_type communicationChannel = 0) = 0;

      virtual void
      updateGhostValuesBegin(const size_type communicationChannel = 0) = 0;

      virtual void
      updateGhostValuesEnd() = 0;

      virtual void
      accumulateAddLocallyOwnedBegin(
        const size_type communicationChannel = 0) = 0;

      virtual void
      accumulateAddLocallyOwnedEnd() = 0;

    protected:
      /**
       * @brief Constructor
       *
       * @param[in] storage reference to unique_ptr to MultiVector::Storage
       * (i.e., MemoryStorage) from which the MultiVector to transfer ownership.
       * @param[in] globalSize global size of the vector (i.e., the number of
       * unique indices across all processors).
       * @param[in] locallyOwnedSize size of the part of the vector for which
       * the current processor is the sole owner (see top for explanation). For
       * a SerialMultiVector, the locallyOwnedSize and the the globalSize are
       * the same.
       * @param[in] ghostSize size of the part of the vector that is owned by
       * the other processors but required by the current processor. For a
       * SerialMultiVector, the ghostSize is 0.
       * @param[in] numVectors number of vectors in the MultiVector
       * @param[in] blasQueue handle for linear algebra operations on
       * HOST/DEVICE.
       *
       *
       * @note Since we are passing the ownership of the input storage to the MultiVector, the
       * storage will point to NULL after a call to this Constructor. Accessing
       * the input storage pointer will lead to undefined behavior.
       */
      MultiVector(
        std::unique_ptr<Storage> &storage,
        const global_size_type    globalSize,
        const size_type           locallyOwnedSize,
        const size_type           ghostSize,
        const size_type           numVectors,
        std::shared_ptr<blasLapack::blasQueueType<memorySpace>> blasQueue);

      /**
       * @brief Default Constructor
       */
      MultiVector();

    protected:
      std::unique_ptr<Storage>                                d_storage;
      std::shared_ptr<blasLapack::blasQueueType<memorySpace>> d_blasQueue;
      VectorAttributes d_vectorAttributes;
      size_type        d_localSize;
      global_size_type d_globalSize;
      size_type        d_locallyOwnedSize;
      size_type        d_ghostSize;
      size_type        d_numVectors;
    };

    // helper functions

    /**
     * @brief Perform \f$ w = au + bv \f$
     * @param[in] a scalar
     * @param[in] u array
     * @param[in] b scalar
     * @param[in] v array
     * @param[out] w array of the result
     */
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    add(ValueType                                  a,
        const MultiVector<ValueType, memorySpace> &u,
        ValueType                                  b,
        const MultiVector<ValueType, memorySpace> &v,
        MultiVector<ValueType, memorySpace> &      w);

  } // namespace linearAlgebra
} // end of namespace dftefe
#include <linearAlgebra/MultiVector.t.cpp>
#endif
