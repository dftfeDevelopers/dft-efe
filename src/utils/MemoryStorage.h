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
      explicit MemoryStorage(size_type size, ValueType initVal = 0);

      /**
       * @brief Destructor
       */
      ~MemoryStorage();


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
       * @note The destination pointer must be pre-allocated appropriately
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
       * @note The destination pointer must be pre-allocated appropriately
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
       * @note The MemoryStorage must be pre-allocated appropriately
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
       * @note The MemoryStorage must be pre-allocated appropriately
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
       */
      template <dftefe::utils::MemorySpace memorySpaceSrc>
      void
      copyFrom(const ValueType *src,
               const size_type  N,
               const size_type  srcOffset,
               const size_type  dstOffset);

    private:
      ValueType *d_data = nullptr;
      size_type  d_size = 0;
    };

  } // namespace utils
} // end of namespace dftefe

#include "MemoryStorage.t.cpp"

#endif
