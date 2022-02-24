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

#ifndef dftefeMultiComponentVectorBase_h
#define dftefeMultiComponentVectorBase_h

#include <linearAlgebra/VectorAttributes.h>
#include <utils/MemoryStorage.h>
#include <utils/TypeConfig.h>

namespace dftefe
{
  namespace linearAlgebra
  {
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    class MultiComponentVectorBase {
      
      /**
       * @brief An abstract class template for a multi component vector.
       * Each component is a vector in the mathematical
       * sense and not in the sense of an array or STL container.
       *
       * The actual implemental of the class is provided in the derived
       * class (e.g., SerialMultiComponentVector, DistributedMultiComponentVector).
       *
       * @tparam template parameter ValueType defines underlying datatype being stored
       *  in the vector (i.e., int, double, complex<double>, etc.)
       * @tparam template parameter memorySpace defines the MemorySpace (i.e., HOST or
       * DEVICE) in which the vector must reside.
       */

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
      virtual ~MultiComponentVectorBase() = default;

      /**
       * @brief Return iterator pointing to the beginning of MultiComponentVector data.
       *
       * @returns Iterator pointing to the beginning of MultiComponentVector.
       */
      virtual iterator
      begin() = 0;

      /**
       * @brief Return iterator pointing to the beginning of MultiComponentVector
       * data.
       *
       * @returns Constant iterator pointing to the beginning of
       * MultiComponentVector.
       */
      virtual const_iterator
      begin() const = 0;

      /**
       * @brief Return iterator pointing to the end of MultiComponentVector data.
       *
       * @returns Iterator pointing to the end of MultiComponentVector.
       */
      virtual iterator
      end() = 0;

      /**
       * @brief Return iterator pointing to the end of MultiComponentVector data.
       *
       * @returns Constant iterator pointing to the end of
       * MultiComponentVector.
       */
      virtual const_iterator
      end() const = 0;

      /**
       * @brief Returns the size of the MultiComponentVector
       * @returns size of the MultiComponentVector
       */
      virtual size_type
      size() const = 0;

      /**
       * @brief Return the raw pointer to the MultiComponentVector data
       * @return pointer to data
       */
      virtual ValueType *
      data() = 0;

      /**
       * @brief Return the constant raw pointer to the MultiComponentVector data
       * @return pointer to const data
       */
      virtual const ValueType *
      data() const = 0;

      /**
       * @brief Compound addition for elementwise addition lhs += rhs
       * @param[in] rhs the vector to add
       * @return the original vector
       */
      virtual MultiComponentVectorBase &
      operator+=(const MultiComponentVectorBase &rhs) = 0;

      /**
       * @brief Compound subtraction for elementwise addition lhs -= rhs
       * @param[in] rhs the vector to subtract
       * @return the original vector
       */
      virtual MultiComponentVectorBase &
      operator-=(const MultiComponentVectorBase &rhs) = 0;

      /**
       * @brief Returns \f$ l_2 \f$ norm of the MultiComponentVector
       * @return \f$ l_2 \f$  norm of the vector as double type
       */
      virtual 
      std::vector<double>
      l2Norm() const = 0;

      /**
       * @brief Returns \f$ l_{\inf} \f$ norm of the MultiComponentVector
       * @return \f$ l_{\inf} \f$  norm of the vector as double type
       */
      virtual 
      std::vector<double>
      lInfNorm() const = 0;

      /**
       * @brief Returns a const reference to the underlying storage
       * of the MultiComponentVector.
       *
       * @return const reference to the underlying MemoryStorage.
       */
      virtual const Storage &
      getStorage() const = 0;

      /**
       * @brief Returns a VectorAttributes object that stores various attributes
       * (e.g., Serial or Distributed, number of components, etc)
       *
       * @return const reference to the VectorAttributes
       */
      const VectorAttributes &
      getVectorAttributes() const = 0;
    };

  }
}
#endif // dftefeMultiComponentMultiComponentVectorBase_h
