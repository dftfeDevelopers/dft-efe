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
 * @author Bikash Kanungo, Sambit Das
 */

#ifndef dftefeVector_h
#define dftefeVector_h

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
     * @brief An base class template which provides an interface for a vector.
     * This is a vector in the mathematical sense and not in the sense of an
     * array or STL container.
     *
     * It provides the interface for two derived classes: SerialVector and
     * DistributedVector.
     *
     * SerialVector, as the name suggests, resides entirely in a processor.
     *
     * DistributedVector, on the other hand, is distributed across a set of
     * processors. The storage of the DistributedVector in a processor comprises
     * of two parts:
     *   1. <b>locally owned part</b>: A part of the DistributedVector, defined
     * through a contiguous range of indices \f$[a,b)\f$ (\f$a\f$ is included,
     * but \f$b\f$ is not), for which the current processor is the sole owner.
     *      The size of the locally owned part (i.e., \f$b-a\f$) is termed as \e
     * locallyOwnedSize.
     *   2. <b>ghost part</b>: Part of the DistributedVector, defined through a
     * set of ghost indices, that are owned by other processors. The size of
     * ghost part is termed as \e ghostSize.
     *
     * Both the <b>locally owned part</b> and the <b>ghost part</b> are stored
     * in a contiguous memory inside a MemoryStorage object, with the <b>locally
     * owned part</b> stored first. The global size of the DistributedVector
     * (i.e., the number of unique indices across all the processors) is simply
     * termed as \e size. Additionally, we define \e localSize = \e
     * locallyOwnedSize + \e ghostSize.
     *
     * @note For a SerialVector, \e size = \e locallyOwnedSize and \e ghostSize = 0.
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
      virtual ~Vector() = default;

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
       * @brief Returns the global size of the Vector (see top for explanation)
       * @returns global size of the Vector
       */
      global_size_type
      size() const;

      /**
       * @brief Returns the size of the part of Vector that is locally owned in the current processor.
       * For a SerialVector, it is same as the \e globalSize.
       * For a DistributedVector, it is the size of the part of
       * DistributedVector, defined through a contiguous range of indices
       * \f$[a,b)\f$ (\f$a\f$ is included, but \f$b\f$ is not), for which the
       * current processor is the sole owner.
       * @returns the \e locallyOwnedSize of the Vector
       */
      size_type
      locallyOwnedSize() const;

      /**
       * @brief Returns the size of the \b ghost \b part (i.e., \e ghostSize) of Vector (see top for an explanation)
       * @returns the \e ghostSize of the Vector
       */
      size_type
      ghostSize() const;

      /**
       * @brief Returns the combined size of the locally owned and the ghost part of Vector in the current processor.
       * For a SerialVector, it is same as the \e globalSize. For a
       * DistributedVector, it is the sum of \e locallyOwnedSize and \e
       * ghostSize.
       * @returns the local size of the Vector
       */
      size_type
      localSize() const;

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
       * @brief Compound addition for elementwise addition lhs += rhs
       * @param[in] rhs the Vector to add
       * @return the original Vector
       * @throws exception if the sizes and type (SerialVector or
       * DistributedVector) are incompatible
       */
      Vector &
      operator+=(const Vector &rhs);

      /**
       * @brief Compound subtraction for elementwise addition lhs -= rhs
       * @param[in] rhs the vector to subtract
       * @return the original vector
       * @throws exception if the sizes and type (SerialVector or
       * DistributedVector) are incompatible
       */
      Vector &
      operator-=(const Vector &rhs);

      /**
       * @brief subtraction of elements owned locally in the processor
       * @param[in] rhs the Vector to subtract
       * @throws exception if the sizes and type (SerialVector or
       * DistributedVector) are incompatible
       */
      void
      subLocal(const Vector &rhs);

      /**
       * @brief Returns a reference to the underlying storage (i.e., MemoryStorage object)
       * of the Vector.
       *
       * @return reference to the underlying MemoryStorage.
       */
      Storage &
      getValues();

      /**
       * @brief Returns a const reference to the underlying storage (i.e., MemoryStorage object)
       * of the Vector.
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
      std::shared_ptr<blasLapack::BlasQueue<memorySpace>>
      getBlasQueue() const;

      /**
       * @brief Set values in the Vector using a user provided Vector::Storage object (i.e., MemoryStorage object).
       * The MemoryStorage may lie in a different memoryspace (say memSpace2)
       * than the Vector's memory space (memSpace). The function internally does
       * a data transfer from memSpace2 to memSpace.
       *
       * @param[in] storage const reference to MemoryStorage object from which
       * to set values into the Vector.
       * @throws exception if the size of the input storage is smaller than the
       * \e localSize (\e locallyOwnedSize + \e ghostSize) of the Vector
       */
      template <dftefe::utils::MemorySpace memorySpace2>
      void
      setValues(
        const typename Vector<ValueType, memorySpace2>::Storage &storage);

      /**
       * @brief Transfer ownership of a user provided Vector::Storage object (i.e., MemoryStorage object)
       * to the Vector. This is useful when a MemoryStorage has been already
       * been allocated and we need the the Vector to claim its ownership. This
       * avoids reallocation of memory.
       *
       * @param[in] storage unique_ptr to MemoryStorage object whose ownership
       * is to be passed to the Vector
       *
       * @note Since we are passing the ownership of the input storage to the Vector, the
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
       * @brief Returns \f$ l_2 \f$ norm of the Vector
       * @return \f$ l_2 \f$  norm of the vector as double type
       */
      virtual double
      l2Norm() const = 0;

      /**
       * @brief Returns \f$ l_{\inf} \f$ norm of the Vector
       * @return \f$ l_{\inf} \f$  norm of the vector as double type
       */
      virtual double
      lInfNorm() const = 0;

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
       * @param[in] storage reference to unique_ptr to Vector::Storage (i.e.,
       * MemoryStorage) from which the Vector to transfer ownership.
       * @param[in] globalSize global size of the vector (i.e., the number of
       * unique indices across all processors).
       * @param[in] locallyOwnedSize size of the part of the vector for which
       * the current processor is the sole owner (see top for explanation). For
       * a SerialVector, the locallyOwnedSize and the the globalSize are the
       * same.
       * @param[in] ghostSize size of the part of the vector that is owned by
       * the other processors but required by the current processor. For a
       * SerialVector, the ghostSize is 0.
       * @param[in] BlasQueue handle for linear algebra operations on
       * HOST/DEVICE.
       *
       * @note Since we are passing the ownership of the input storage to the Vector, the
       * storage will point to NULL after a call to this Constructor. Accessing
       * the input storage pointer will lead to undefined behavior.
       */
      Vector(std::unique_ptr<Storage> &storage,
             const global_size_type    globalSize,
             const size_type           locallyOwnedSize,
             const size_type           ghostSize,
             std::shared_ptr<blasLapack::BlasQueue<memorySpace>> BlasQueue);

      /**
       * @brief Default Constructor
       */
      Vector();

    protected:
      std::unique_ptr<Storage>                            d_storage;
      std::shared_ptr<blasLapack::BlasQueue<memorySpace>> d_BlasQueue;
      VectorAttributes                                    d_vectorAttributes;
      size_type                                           d_localSize;
      global_size_type                                    d_globalSize;
      size_type                                           d_locallyOwnedSize;
      size_type                                           d_ghostSize;
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
    add(ValueType                             a,
        const Vector<ValueType, memorySpace> &u,
        ValueType                             b,
        const Vector<ValueType, memorySpace> &v,
        Vector<ValueType, memorySpace> &      w);

  } // namespace linearAlgebra
} // end of namespace dftefe
#include <linearAlgebra/Vector.t.cpp>
#endif
