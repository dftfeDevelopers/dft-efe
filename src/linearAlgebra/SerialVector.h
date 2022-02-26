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
 * @author Ian C. Lin, Sambit Das, Bikash Kanungo.
 */

#ifndef dftefeSerialVector_h
#define dftefeSerialVector_h

#include <linearAlgebra/VectorBase.h>
#include <linearAlgebra/VectorAttributes.h>
#include <utils/MemoryStorage.h>
#include <utils/TypeConfig.h>
#include <memory>
namespace dftefe
{
  namespace linearAlgebra
  {
    /**
     * @brief A derived class of VectorBase for a serial vector
     * (i.e., a vector that resides entirely within a processor)
     *
     * @tparam template parameter ValueType defines underlying datatype being stored
     *  in the vector (i.e., int, double, complex<double>, etc.)
     * @tparam template parameter memorySpace defines the MemorySpace (i.e., HOST or
     * DEVICE) in which the vector must reside.
     */
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    class SerialVector : public VectorBase<ValueType, memorySpace>
    {
    public:
      /**
       * @brief Default constructor
       */
      SerialVector() = default;

      /**
       * @brief Copy constructor
       * @param[in] u SerialVector object to copy from
       */
      SerialVector(const SerialVector &u);

      /**
       * @brief Move constructor
       * @param[in] u SerialVector object to move from
       */
      SerialVector(SerialVector &&u) noexcept;

      /**
       * @brief Copy assignment operator
       * @param[in] u const reference to SerialVector object to copy from
       * @return reference to this object after copying data from u
       */
      SerialVector &
      operator=(const SerialVector &u);

      /**
       * @brief Move assignment operator
       * @param[in] u const reference to SerialVector object to move from
       * @return reference to this object after moving data from u
       */
      SerialVector &
      operator=(SerialVector &&u);

      /**
       * @brief Constructor for SerialVector with size and initial value arguments
       * @param[in] size size of the SerialVector
       * @param[in] initVal initial value of elements of the SerialVector
       */
      explicit SerialVector(size_type size, ValueType initVal = ValueType());

      /**
       * @brief Returns \f$ l_2 \f$ norm of the SerialVector
       * @return \f$ l_2 \f$  norm of the vector as double type
       */
      double
      l2Norm() const override;

      /**
       * @brief Returns \f$ l_{\inf} \f$ norm of the SerialVector
       * @return \f$ l_{\inf} \f$  norm of the vector as double type
       */
      double
      lInfNorm() const override;

      void
      scatterToGhost(const size_type communicationChannel = 0) override;

      void
      gatherFromGhost(const size_type communicationChannel = 0) override;

      void
      scatterToGhostBegin(const size_type communicationChannel = 0) override;

      void
      scatterToGhostEnd() override;

      void
      gatherFromGhostBegin(const size_type communicationChannel = 0) override;

      void
      gatherFromGhostEnd() override;
    };

    //
    // helper functions
    //

    //    /**
    //     * @brief In-place elementwise addition of two vector (i.e., lhs += rhs)
    //     * @param[in] lhs the vector to add to
    //     * @param[in] rhs the vector to add
    //     * @return the original vector (i.e., lhs)
    //     */
    //    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    //      SerialVector<ValueType, memorySpace> &
    //      operator+=(SerialVector<ValueType,memorySpace> & lhs,
    //	  const SerialVector<ValueType,memorySpace> &rhs);
    //
    //    /**
    //     * @brief In-place elementwise subtraction (i.e., lhs -= rhs)
    //     * @param[in] lhs the vector to subtract from
    //     * @param[in] rhs the vector to subtract
    //     * @return the original vector (i.e., lhs)
    //     */
    //    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    //      SerialVector<ValueType, memorySpace> &
    //      operator-=(SerialVector<ValueType,memorySpace> &lhs,
    //	  const SerialVector<ValueType,memorySpace> &rhs);

    ///**
    // * @brief Perform \f$ w = au + bv \f$
    // * @param[in] a scalar
    // * @param[in] u array
    // * @param[in] b scalar
    // * @param[in] v array
    // * @param[out] w array of the result
    // */
    // template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    // void
    // add(ValueType                             a,
    //    const SerialVector<ValueType, memorySpace> &u,
    //    ValueType                             b,
    //    const SerialVector<ValueType, memorySpace> &v,
    //    SerialVector<ValueType, memorySpace> &      w);

  } // namespace linearAlgebra
} // end of namespace dftefe

#include "SerialVector.t.cpp"
#endif // dftefeSerialVector_h
