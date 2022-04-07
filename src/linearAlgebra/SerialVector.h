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

#include <linearAlgebra/Vector.h>
#include <linearAlgebra/VectorAttributes.h>
#include <utils/MemoryStorage.h>
#include <utils/TypeConfig.h>
#include <memory>
namespace dftefe
{
  namespace linearAlgebra
  {
    /**
     * @brief A derived class of Vector for a serial vector
     * (i.e., a vector that resides entirely within a processor)
     *
     * @tparam template parameter ValueType defines underlying datatype being stored
     *  in the vector (i.e., int, double, complex<double>, etc.)
     * @tparam template parameter memorySpace defines the MemorySpace (i.e., HOST or
     * DEVICE) in which the vector must reside.
     */
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    class SerialVector : public Vector<ValueType, memorySpace>
    {
    public:
      //
      // Pulling base class (Vector) protected names here so to avoid full name
      // scoping inside the source file. The other work around is to use
      // this->d_m (where d_m is a protected data member of base class). This is
      // something which is peculiar to inheritance using class templates. The
      // reason why this is so is the fact that C++ does not consider base class
      // templates for name resolution (i.e., they are dependent names and
      // dependent names are not considered)
      //
      using Vector<ValueType, memorySpace>::d_storage;
      using Vector<ValueType, memorySpace>::d_blasQueue;
      using Vector<ValueType, memorySpace>::d_vectorAttributes;
      using Vector<ValueType, memorySpace>::d_globalSize;
      using Vector<ValueType, memorySpace>::d_locallyOwnedSize;
      using Vector<ValueType, memorySpace>::d_ghostSize;
      using Vector<ValueType, memorySpace>::d_localSize;

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
      explicit SerialVector(
        size_type size,
        ValueType initVal,
        std::shared_ptr<const blasLapack::blasQueueType<memorySpace>>
          blasQueue);

      /**
       * @brief Constructor with predefined Vector::Storage (i.e., utils::MemoryStorage).
       * This allows the SerialVector to take ownership of input Vector::Storage
       * (i.e., utils::MemoryStorage) This is useful when one does not want to
       * allocate new memory and instead use memory allocated in the
       * Vector::Storage (i.e., MemoryStorage). The \e locallyOwnedSize, \e
       * ghostSize, etc., are automatically set using the size of the \p
       * storage.
       *
       * @param[in] storage unique_ptr to Vector::Storage whose ownership
       * is to be transfered to the SerialVector
       *
       * @note This Constructor transfers the ownership from the input unique_ptr \p storage to the internal data member of the SerialVector.
       * Thus, after the function call \p storage will point to NULL and any
       * access through \p storage will lead to <b>undefined behavior</b>.
       *
       */
      SerialVector(std::unique_ptr<
                     typename Vector<ValueType, memorySpace>::Storage> storage,
                   std::shared_ptr<const blasLapack::blasQueueType<memorySpace>>
                     blasQueue);

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
      updateGhostValues(const size_type communicationChannel = 0) override;

      void
      accumulateAddLocallyOwned(
        const size_type communicationChannel = 0) override;

      void
      updateGhostValuesBegin(const size_type communicationChannel = 0) override;

      void
      updateGhostValuesEnd() override;

      void
      accumulateAddLocallyOwnedBegin(
        const size_type communicationChannel = 0) override;

      void
      accumulateAddLocallyOwnedEnd() override;
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
