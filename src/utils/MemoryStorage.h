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
 * @author Ian C. Lin, Sambit Das.
 */

#ifndef dftefeMemoryStorage_h
#define dftefeMemoryStorage_h

#include <utils/MemoryManager.h>
#include <utils/TypeConfig.h>
#include <vector>
namespace dftefe
{
  namespace utils
  {
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    class MemoryStorage
    {
      /**
       * @brief A class template to provide an interface that can act similar to
       * STL vectors but with different MemorySpace---
       * HOST (cpu) , DEVICE (gpu), etc,.
       *
       * @tparam ValueType The underlying value type for the MemoryStorage
       *  (e.g., int, double, complex<double>, etc.)
       * @tparam memorySpace The memory space in which the MemoryStorage needs
       *  to reside
       *
       */

      //
      // typedefs
      //
    public:
      typedef ValueType        value_type;
      typedef ValueType *      pointer;
      typedef ValueType &      reference;
      typedef const ValueType &const_reference;
      typedef ValueType *      iterator;
      typedef const ValueType *const_iterator;

    public:
      MemoryStorage() = default;

      /**
       * @brief Copy constructor for a MemoryStorage
       * @param[in] u MemoryStorage object to copy from
       */
      MemoryStorage(const MemoryStorage &u);

      /**
       * @brief Move constructor for a Vector
       * @param[in] u Vector object to move from
       */
      MemoryStorage(MemoryStorage &&u) noexcept;

      /**
       * @brief Constructor for Vector with size and initial value arguments
       * @param[in] size size of the Vector
       * @param[in] initVal initial value of elements of the Vector
       */
      explicit MemoryStorage(size_type size, ValueType initVal = ValueType());

      /**
       * @brief Destructor
       */
      ~MemoryStorage();

      /**
       * @brief Set all the entries to a given value
       * @param[in] val The value to which the entries are to be set
       */
      void
      setValue(const ValueType val);

      /**
       * @brief Return iterator pointing to the beginning of point
       * data.
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
       * @brief Copy assignment operator
       * @param[in] rhs the rhs Vector from which to copy
       * @returns reference to the lhs Vector
       */
      MemoryStorage &
      operator=(const MemoryStorage &rhs);


      /**
       * @brief Move assignment constructor
       * @param[in] rhs the rhs Vector from which to move
       * @returns reference to the lhs Vector
       */
      MemoryStorage &
      operator=(MemoryStorage &&rhs) noexcept;

      //      // This part does not work for GPU version, will work on this
      //      until
      //      // having cleaner solution.
      //      /**
      //       * @brief Operator to get a reference to a element of the Vector
      //       * @param[in] i is the index to the element of the Vector
      //       * @returns reference to the element of the Vector
      //       * @throws exception if i >= size of the Vector
      //       */
      //      reference
      //      operator[](size_type i);
      //
      //      /**
      //       * @brief Operator to get a const reference to a element of the Vector
      //       * @param[in] i is the index to the element of the Vector
      //       * @returns const reference to the element of the Vector
      //       * @throws exception if i >= size of the Vector
      //       */
      //      const_reference
      //      operator[](size_type i) const;

      /**
       * @brief Deallocates and then resizes Vector with new size
       * and initial value arguments
       * @param[in] size size of the Vector
       * @param[in] initVal initial value of elements of the Vector
       */
      void
      resize(size_type size, ValueType initVal = ValueType());

      /**
       * @brief Returns the dimension of the Vector
       * @returns size of the Vector
       */
      size_type
      size() const;

      /**
       * @brief Return the raw pointer to the Vector
       * @return pointer to data
       */
      ValueType *
      data() noexcept;

      /**
       * @brief Return the raw pointer to the Vector without modifying the values
       * @return pointer to const data
       */
      const ValueType *
      data() const noexcept;

      /**
       * @brief Copies the data to a MemoryStorage object in a different memory space.
       * This provides a seamless interface to copy back and forth between
       * memory spaces , including between the same memory spaces.
       *
       * @note The destination MemoryStorage must be pre-allocated appropriately
       *
       * @tparam memorySpaceDst memory space of the destination MemoryStorage
       * @param[in] dstMemoryStorage reference to the destination
       *  MemoryStorage. It must be pre-allocated appropriately
       * @param[out] dstMemoryStorage reference to the destination
       *  MemoryStorage with the data copied into it
       * @throw utils::LengthError exception if the size of dstMemoryStorage is
       * less than underlying MemoryStorage
       */
      template <dftefe::utils::MemorySpace memorySpaceDst>
      void
      copyTo(MemoryStorage<ValueType, memorySpaceDst> &dstMemoryStorage) const;

      /**
       * @brief Copies the data to a MemoryStorage object in a different memory space.
       * This provides a seamless interface to copy back and forth between
       * memory spaces , including between the same memory spaces. This is a
       * more granular version of the above copyTo function as it provides
       * transfer from a specific portion of the source MemoryStorage to a
       * specific portion of the destination MemoryStorage.
       *
       * @note The destination MemoryStorage must be pre-allocated appropriately
       *
       * @tparam memorySpaceDst memory space of the destination MemoryStorage
       * @param[in] dstMemoryStorage reference to the destination
       *  MemoryStorage. It must be pre-allocated appropriately
       * @param[in] N number of entries of the source MemoryStorage
       *  that needs to be copied to the destination MemoryStorage
       * @param[in] srcOffset offset relative to the start of the source
       *  MemoryStorage from which we need to copy data
       * @param[in] dstOffset offset relative to the start of the destination
       *  MemoryStorage to which we need to copy data
       * @param[out] dstMemoryStorage reference to the destination
       *  MemoryStorage with the data copied into it
       * @throw utils::LengthError exception if the size of dstMemoryStorage is
       * less than N + dstOffset
       * @throw utils::LengthError exception if the size of underlying
       * MemoryStorage is less than N + srcOffset
       */
      template <dftefe::utils::MemorySpace memorySpaceDst>
      void
      copyTo(MemoryStorage<ValueType, memorySpaceDst> &dstMemoryStorage,
             const size_type                           N,
             const size_type                           srcOffset,
             const size_type                           dstOffset) const;

      /**
       * @brief Copies data from a MemoryStorage object in a different memory space.
       * This provides a seamless interface to copy back and forth between
       * memory spaces, including between the same memory spaces.
       *
       * @note The MemoryStorage must be pre-allocated appropriately
       *
       * @tparam memorySpaceSrc memory space of the source MemoryStorage
       *  from which to copy
       * @param[in] srcMemoryStorage reference to the source
       *  MemoryStorage
       * @throw utils::LengthError exception if the size of underlying
       * MemoryStorage is less than size of srcMemoryStorage
       */
      template <dftefe::utils::MemorySpace memorySpaceSrc>
      void
      copyFrom(
        const MemoryStorage<ValueType, memorySpaceSrc> &srcMemoryStorage);

      /**
       * @brief Copies data from a MemoryStorage object in a different memory space.
       * This provides a seamless interface to copy back and forth between
       * memory spaces, including between the same memory spaces.
       * This is a more granular version of the above copyFrom function as it
       * provides transfer from a specific portion of the source MemoryStorage
       * to a specific portion of the destination MemoryStorage.
       *
       * @note The MemoryStorage must be pre-allocated appropriately
       *
       * @tparam memorySpaceSrc memory space of the source MemoryStorage
       *  from which to copy
       * @param[in] srcMemoryStorage reference to the source
       *  MemoryStorage
       * @param[in] N number of entries of the source MemoryStorage
       *  that needs to be copied to the destination MemoryStorage
       * @param[in] srcOffset offset relative to the start of the source
       *  MemoryStorage from which we need to copy data
       * @param[in] dstOffset offset relative to the start of the destination
       *  MemoryStorage to which we need to copy data
       * @throw utils::LengthError exception if the size of srcMemoryStorage is
       * less than N + srcOffset
       * @throw utils::LengthError exception if the size of underlying
       * MemoryStorage is less than N + dstOffset
       */
      template <dftefe::utils::MemorySpace memorySpaceSrc>
      void
      copyFrom(MemoryStorage<ValueType, memorySpaceSrc> &srcMemoryStorage,
               const size_type                           N,
               const size_type                           srcOffset,
               const size_type                           dstOffset);

      /**
       * @brief Copies the data to a memory pointed by a raw pointer
       * This provides a seamless interface to copy back and forth between
       * memory spaces, including between the same memory spaces.
       *
       * @note The destination pointer must be pre-allocated to be
       * at least of the size of the MemoryStorage
       *
       * @tparam memorySpaceDst memory space of the destination pointer
       * @param[in] dst pointer to the destination. It must be pre-allocated
       * appropriately
       * @param[out] dst pointer to the destination with the data copied into it
       */
      template <dftefe::utils::MemorySpace memorySpaceDst>
      void
      copyTo(ValueType *dst) const;

      /**
       * @brief Copies the data to a memory pointer by a raw pointer.
       * This provides a seamless interface to copy back and forth between
       * memory spaces , including between the same memory spaces. This is a
       * more granular version of the above copyTo function as it provides
       * transfer from a specific portion of the source MemoryStorage to a
       * specific portion of the destination pointer.
       *
       * @note The destination pointer must be pre-allocated to be at least
       * of the size N + dstOffset
       *
       * @tparam memorySpaceDst memory space of the destination pointer
       * @param[in] dst pointer to the destination. It must be pre-allocated
       * appropriately
       * @param[in] N number of entries of the source MemoryStorage
       *  that needs to be copied to the destination pointer
       * @param[in] srcOffset offset relative to the start of the source
       *  MemoryStorage from which we need to copy data
       * @param[in] dstOffset offset relative to the start of the destination
       *  pointer to which we need to copy data
       * @param[out] dst pointer to the destination with the data copied into it
       * @throw utils::LengthError exception if the size of the MemoryStorage is
       * less than N + srcOffset
       */
      template <dftefe::utils::MemorySpace memorySpaceDst>
      void
      copyTo(ValueType *     dst,
             const size_type N,
             const size_type srcOffset,
             const size_type dstOffset) const;

      /**
       * @brief Copies data from a memory pointed by a raw pointer into
       * the MemoryStorage object.
       * This provides a seamless interface to copy back and forth between
       * memory spaces, including between the same memory spaces.
       *
       * @note The src pointer must point to a memory chunk that is at least the
       * size of the MemoryStorage
       *
       * @tparam memorySpaceSrc memory space of the source pointer
       *  from which to copy
       * @param[in] src pointer to the source memory
       */
      template <dftefe::utils::MemorySpace memorySpaceSrc>
      void
      copyFrom(const ValueType *src);

      /**
       * @brief Copies data from a memory pointer by a raw pointer into the MemoryStorage object.
       * This provides a seamless interface to copy back and forth between
       * memory spaces, including between the same memory spaces.
       * This is a more granular version of the above copyFrom function as it
       * provides transfer from a specific portion of the source memory
       * to a specific portion of the destination MemoryStorage.
       *
       * @note The src pointer must point to a memory chunk that is at least
       * the size of N + srcOffset
       *
       * @tparam memorySpaceSrc memory space of the source pointer
       *  from which to copy
       * @param[in] src pointer to the source memory
       * @param[in] N number of entries of the source pointer
       *  that needs to be copied to the destination MemoryStorage
       * @param[in] srcOffset offset relative to the start of the source
       *  pointer from which we need to copy data
       * @param[in] dstOffset offset relative to the start of the destination
       *  MemoryStorage to which we need to copy data
       * @throw utils::LengthError exception if the size of the MemoryStorage is
       * less than N + dstOffset
       */
      template <dftefe::utils::MemorySpace memorySpaceSrc>
      void
      copyFrom(const ValueType *src,
               const size_type  N,
               const size_type  srcOffset,
               const size_type  dstOffset);

      /**
       * @brief Copies the data to a C++ STL vector, which always resides in
       * the CPU. This provides a seamless interface to copy from any memory
       * space to a C++ STL vector, including the case where source memory space
       * is HOST (i.e., it resides on the CPU)
       *
       * @param[in] dst reference to the destination C++ STL vector to which
       *  the data needs to be copied.
       * @param[out] dst reference to the destination C++ STL vector with
       * the data copied into it
       * @note If the size of the dst vector is less than the the size of
       * the underlying MemoryStorage, it will be resized. Thus, for performance
       * reasons, it is recommened that the dst STL vector be pre-allocated
       * appropriately.
       */
      void
      copyTo(std::vector<ValueType> &dst) const;

      /**
       * @brief Copies the data to a C++ STL vector, which always resides in
       * the CPU. This provides a seamless interface to copy from any memory
       * space to a C++ STL vector, including the case where source memory space
       * is HOST (i.e., it resides on the CPU).
       * This is a more granular version of the above copyToSTL function as it
       * provides transfer from a specific portion of the MemoryStorage
       * to a specific portion of the destination STL vector.
       *
       * @param[in] dst reference to the destination C++ STL vector to which
       *  the data needs to be copied.
       * @note If the size of the dst vector is less than the the size of
       * the underlying memory storage, it will be resized. Thus, for
       * performance reasons it is recommened to should be allocated
       * appropriately.
       * @param[in] N number of entries of the source MemoryStorage
       *  that needs to be copied to the destination pointer
       * @param[in] srcOffset offset relative to the start of the source
       *  MemoryStorage from which we need to copy data
       * @param[in] dstOffset offset relative to the start of the STL vector
       *  to which we need to copy data
       * @param[out] dst reference to the destination C++ STL vector with
       * the data copied into it
       * @throw utils::LengthError exception if the size of the MemoryStorage is
       * less than N + srcOffset
       * @note If the size of the dst vector is less N + srcOffset, it will be resized.
       *  Thus, for performance reasons, it is recommened that the dst STL
       * vector be allocated appropriately.
       */
      void
      copyTo(std::vector<ValueType> &dst,
             const size_type         N,
             const size_type         srcOffset,
             const size_type         dstOffset) const;

      /**
       * @brief Copies data from a C++ STL vector to the MemoryStorage object,
       * which always resides on a CPU. This provides a seamless interface to
       * copy from any memory space, including the case where the same memory
       * spaces is HOST(i.e., the MemoryStorage is on CPU).
       *
       * @param[in] src const reference to the source C++ STL vector from which
       *  the data needs to be copied.
       *
       * @throw utils::LengthError exception if the size of the MemoryStorage is
       * less than the size of the src
       */
      void
      copyFrom(const std::vector<ValueType> &src);

      /**
       * @brief Copies data from a C++ STL vector to the MemoryStorage object,
       * which always resides on a CPU. This provides a seamless interface to
       * copy from any memory space, including the case where the same memory
       * spaces is HOST(i.e., the MemoryStorage is on CPU). This is a more
       * granular version of the above copyFromSTL function as it provides
       * transfer from a specific portion of the source STL vector to to a
       * specific portion of the destination MemoryStorage.
       *
       * @param[in] src const reference to the source C++ STL vector from which
       *  the data needs to be copied.
       * @param[in] N number of entries of the source pointer
       *  that needs to be copied to the destination MemoryStorage
       * @param[in] srcOffset offset relative to the start of the source STL
       *  vector from which we need to copy data
       * @param[in] dstOffset offset relative to the start of the destination
       *  MemoryStorage to which we need to copy data
       * @throw utils::LengthError exception if the size of src is less than
       * N + srcOffset
       * @throw utils::LengthError exception if the size of the MemoryStorage
       * is less thant N + dstOffset
       *
       */
      void
      copyFrom(const std::vector<ValueType> &src,
               const size_type               N,
               const size_type               srcOffset,
               const size_type               dstOffset);


    private:
      ValueType *d_data = nullptr;
      size_type  d_size = 0;
    };

    //
    // helper functions
    //

    /**
     * @brief Create a MemoryStorage object from an input C++ STL vector
     * @param[in] src Input C++ STL vector from which the MemoryStorage
     *  object is to be created
     * @return A MemoryStorage object containing the data in the input C++
     *  STL vector
     * @tparam ValueType Datatype of the underlying data in MemoryStorage
     *  as well as C++ STL vector (e.g., int, double, float, complex<double>,
     *  etc)
     * @tparam memorySpaceDst MemorySpace (e.g. HOST, DEVICE, HOST_PINNED, etc)
     * where the output MemoryStorage object should reside
     */
    template <typename ValueType, utils::MemorySpace memorySpaceDst>
    MemoryStorage<ValueType, memorySpaceDst>
    memoryStorageFromSTL(const std::vector<ValueType> &src);

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    swap(MemoryStorage<ValueType, memorySpace> &X,
         MemoryStorage<ValueType, memorySpace> &Y);

  } // namespace utils
} // end of namespace dftefe

#include "MemoryStorage.t.cpp"

#endif
