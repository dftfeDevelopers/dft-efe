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
 * @author Sambit Das.
 */

#ifndef dftefeDistributedVectorStorage_h
#define dftefeDistributedVectorStorage_h

#include <utils/VectorStorage.h>
#include <utils/MPICommunicatorBase.h>
namespace dftefe
{
  namespace utils
  {
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    class DistributedVectorStorage
    {
      typedef ValueType        value_type;
      typedef ValueType *      pointer;
      typedef ValueType &      reference;
      typedef const ValueType &const_reference;
      typedef ValueType *      iterator;
      typedef const ValueType *const_iterator;

    public:
      DistributedVectorStorage() = default;

      /**
       * @brief Copy constructor for a Vector
       * @param[in] u Vector object to copy from
       */
      DistributedVectorStorage(const DistributedVectorStorage &u);

      /**
       * @brief Move constructor for a Vector
       * @param[in] u Vector object to move from
       */
      DistributedVectorStorage(DistributedVectorStorage &&u) noexcept;

      /**
       * @brief Constructor for Vector with size and initial value arguments
       * @param[in] local_size size of the Vector in local MPI task
       * @param[in] initVal initial value of elements of the Vector in local MPI task 
       */
      explicit DistributedVectorStorage(size_type localSize, ValueType initVal = 0);

      /**
       * @brief Destructor
       */
      ~DistributedVectorStorage();


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
      DistributedVectorStorage &
      operator=(const VectorStorage &rhs);


      /**
       * @brief Move assignment constructor
       * @param[in] rhs the rhs Vector from which to move
       * @returns reference to the lhs Vector
       */
      DistributedVectorStorage &
      operator=(VectorStorage &&rhs) noexcept;


      /**
       * @brief Deallocates and then resizes Vector with new size
       * and initial value arguments
       * @param[in] size local size of the Vector
       * @param[in] initVal initial value of elements of the Vector
       */
      void
      resize(size_type localSize, ValueType initVal = 0);

      /**
       * @brief Returns the dimension of the Vector
       * @returns global size of the Vector
       */
      global_size_type
      size() const;

      /**
       * @brief Returns the locally owned size of the vector 
       * @returns size
       */
      size_type
      locallyOwnedSize() const;

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

    private:
      VectorStorage<ValueType,memorySpace> d_data;
      global_size_type  d_globalSize = 0;
      size_type         d_locallyOwnedSize=0;
      size_type         d_ghostSize=0;      
    };

  } // namespace utils
} // end of namespace dftefe

#include "VectorStorage.t.cpp"

#endif
