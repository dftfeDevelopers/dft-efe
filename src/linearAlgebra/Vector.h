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

#ifndef dftefeVector_h
#define dftefeVector_h

#include "MemoryManager.h"
#include "TypeConfig.h"
#include <vector>
namespace dftefe
{
  namespace linearAlgebra
  {
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    class Vector
    {
      typedef ValueType        value_type;
      typedef ValueType *      pointer;
      typedef ValueType &      reference;
      typedef const ValueType &const_reference;
      typedef ValueType *      iterator;
      typedef const ValueType *const_iterator;

    public:
      Vector() = default;

      /**
       * @brief Copy constructor for a Vector
       * @param[in] u Vector object to copy from
       */
      Vector(const Vector &u);

      /**
       * @brief Move constructor for a Vector
       * @param[in] u Vector object to move from
       */
      Vector(Vector &&u) noexcept;

      /**
       * @brief Constructor for Vector with size and initial value arguments
       * @param[in] size size of the Vector
       * @param[in] initVal initial value of elements of the Vector
       */
      explicit Vector(size_type size, ValueType initVal = 0);

      /**
       * @brief Destructor
       */
      ~Vector();


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
      Vector &
      operator=(const Vector &rhs);


      /**
       * @brief Move assignment constructor
       * @param[in] rhs the rhs Vector from which to move
       * @returns reference to the lhs Vector
       */
      Vector &
      operator=(Vector &&rhs) noexcept;

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
      resize(size_type size, ValueType initVal = 0);

      /**
       * @brief Compound addition for elementwise addition lhs += rhs
       * @param[in] rhs the vector to add
       * @return the original vector
       */
      Vector &
      operator+=(const Vector &rhs);

      /**
       * @brief Compound subtraction for elementwise addition lhs -= rhs
       * @param[in] rhs the vector to subtract
       * @return the original vector
       */
      Vector &
      operator-=(const Vector &rhs);


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

    private:
      ValueType *d_data = nullptr;
      size_type  d_size = 0;
    };

    // helper functions


    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    add(ValueType                             a,
        const Vector<ValueType, memorySpace> &u,
        ValueType                             b,
        const Vector<ValueType, memorySpace> &v,
        Vector<ValueType, memorySpace> &      w);

  } // namespace linearAlgebra
} // end of namespace dftefe

#include "Vector.t.cpp"

#endif
