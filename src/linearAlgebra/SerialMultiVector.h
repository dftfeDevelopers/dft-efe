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

#ifndef dftefeSerialMultiVector_h
#define dftefeSerialMultiVector_h

#include <linearAlgebra/MultiVector.h>
#include <linearAlgebra/VectorAttributes.h>
#include <utils/MemoryStorage.h>
#include <utils/TypeConfig.h>
#include <memory>
namespace dftefe
{
  namespace linearAlgebra
  {
    /**
     * @brief A derived class of MultiVector for a serial vector
     * (i.e., a multi vector that resides entirely within a processor)
     *
     * @tparam template parameter ValueType defines underlying datatype being stored
     *  in the multi vector (i.e., int, double, complex<double>, etc.)
     * @tparam template parameter memorySpace defines the MemorySpace (i.e., HOST or
     * DEVICE) in which the multi vector must reside.
     */
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    class SerialMultiVector : public MultiVector<ValueType, memorySpace>
    {
    public:
      //
      // Pulling base class (MultiVector) protected names here so to avoid full
      // name scoping inside the source file. The other work around is to use
      // this->d_m (where d_m is a protected data member of base class). This is
      // something which is peculiar to inheritance using class templates. The
      // reason why this is so is the fact that C++ does not consider base class
      // templates for name resolution (i.e., they are dependent names and
      // dependent names are not considered)
      //
      using MultiVector<ValueType, memorySpace>::d_storage;
      using MultiVector<ValueType, memorySpace>::d_linAlgOpContext;
      using MultiVector<ValueType, memorySpace>::d_vectorAttributes;
      using MultiVector<ValueType, memorySpace>::d_globalSize;
      using MultiVector<ValueType, memorySpace>::d_locallyOwnedSize;
      using MultiVector<ValueType, memorySpace>::d_ghostSize;
      using MultiVector<ValueType, memorySpace>::d_localSize;
      using MultiVector<ValueType, memorySpace>::d_numVectors;

    public:
      /**
       * @brief Default constructor
       */
      SerialMultiVector() = default;

      /**
       * @brief Copy constructor
       * @param[in] u SerialMultiVector object to copy from
       */
      SerialMultiVector(const SerialMultiVector &u);

      /**
       * @brief Move constructor
       * @param[in] u SerialMultiVector object to move from
       */
      SerialMultiVector(SerialMultiVector &&u) noexcept;

      /**
       * @brief Copy assignment operator
       * @param[in] u const reference to SerialMultiVector object to copy from
       * @return reference to this object after copying data from u
       */
      SerialMultiVector &
      operator=(const SerialMultiVector &u);

      /**
       * @brief Move assignment operator
       * @param[in] u const reference to SerialMultiVector object to move from
       * @return reference to this object after moving data from u
       */
      SerialMultiVector &
      operator=(SerialMultiVector &&u);

      /**
       * @brief Constructor for SerialMultiVector with vector size, number of vectors and initial value arguments
       * @param[in] size size of each vector in the SerialMultiVector
       * @param[in] numVectors number of vectors in the SerialMultiVector
       * @param[in] initVal initial value of elements of the SerialMultiVector
       * @param[in] linAlgOpContext handle for linear algebra operations on
       * HOST or DEVICE.
       *
       */
      explicit SerialMultiVector(size_type                     size,
                                 size_type                     numVectors,
                                 ValueType                     initVal,
                                 LinAlgOpContext<memorySpace> *linAlgOpContext);

      /**
       * @brief Constructor with predefined MultiVector::Storage (i.e., utils::MemoryStorage).
       * This allows the SerialMultiVector to take ownership of input
       * Vector::Storage (i.e., utils::MemoryStorage) This is useful when one
       * does not want to allocate new memory and instead use memory allocated
       * in the MultiVector::Storage (i.e., MemoryStorage). The \e
       * locallyOwnedSize, \e ghostSize, etc., are automatically set using the
       * size of the \p storage.
       *
       * @param[in] storage unique_ptr to MultiVector::Storage whose ownership
       * is to be transfered to the SerialMultiVector
       * @param[in] numVectors number of vectors in the SerialMultiVector
       * @note This Constructor transfers the ownership from the input unique_ptr \p storage to the internal data member of the SerialMultiVector.
       * Thus, after the function call \p storage will point to NULL and any
       * access through \p storage will lead to <b>undefined behavior</b>.
       *
       */
      SerialMultiVector(
        std::unique_ptr<typename MultiVector<ValueType, memorySpace>::Storage>
                                      storage,
        size_type                     numVectors,
        LinAlgOpContext<memorySpace> *linAlgOpContext);

      /**
       * @brief Returns \f$ l_2 \f$ norms of all the \f$N\f$ vectors in the  MultiVector
       * @return \f$ l_2 \f$  norms of the various vectors as std::vector<double> type
       */
      std::vector<double>
      l2Norms() const override;

      /**
       * @brief Returns \f$ l_{\inf} \f$ norms of all the \f$N\f$ vectors in the  MultiVector
       * @return \f$ l_{\inf} \f$  norms of the various vectors as std::vector<double> type
       */
      std::vector<double>
      lInfNorms() const override;

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
  } // namespace linearAlgebra
} // end of namespace dftefe

#include "SerialMultiVector.t.cpp"
#endif // dftefeSerialMultiVector_h
