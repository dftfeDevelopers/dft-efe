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
 * @author Ian C. Lin, Sambit Das, Vishal Subramanian
 */

#ifndef dftefeVector_h
#define dftefeVector_h

#include <utils/MemoryStorage.h>
#include <utils/TypeConfig.h>
#include "QueueManager.h"
#include "blasWrappersTypedef.h"
namespace dftefe
{
  namespace linearAlgebra
  {
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    class Vector : public dftefe::utils::MemoryStorage<ValueType, memorySpace>
    {
    private :
      blasWrapper::blasQueueType<memorySapce> d_blasQueue ;
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
       * @param[in] blasQueueInput Queue handle. For Vector objects stored in HOST
       * this is same as int. For Vector object stored in Device this is
       * blas::Queue
       * @param[in] initVal initial value of elements of the Vector
       */
      explicit Vector(size_type size,
                      blasWrapper::blasQueueType<memorySapce> &blasQueueInput,
                      ValueType initVal = 0);


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
       * @brief Returns \f$ l_2 \f$ norm of the Vector
       * @return \f$ l_2 \f$  norm of the vector as double type
       */
      double
      l2Norm() const;

      /**
       * @brief Returns \f$ l_{\inf} \f$ norm of the Vector
       * @return \f$ l_{\inf} \f$  norm of the vector as double type
       */
      double
      lInfNorm() const;

      /**
       * @brief Returns the dot product of this vector with u
       * @return *this . u  as double type
       */
      double
      dotProduct(const Vector<ValueType, memorySpace> &u) const;

      /**
      * @brief Returns the underlying MemoryStorage object. For Matrix object
       * stored on Host, it is same as int. For Matrix object stored on Device
       * this is same as blas::Queue
      * @returns blasWrapper::blasQueueType<memorySapce> of this class
       */
      blasWrapper::blasQueueType<memorySapce> &
      getQueue();
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

#include "Vector.t.cpp"

#endif
