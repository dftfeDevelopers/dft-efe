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

#include <utils/Exceptions.h>

namespace dftefe
{
  namespace linearAlgebra
  {
    //
    // Constructor using size, numVectors and init value
    //
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    SerialMultiVector<ValueType, memorySpace>::SerialMultiVector(
      const size_type                     size,
      const size_type                     numVectors,
      const ValueType                     initVal,
      blasLapack::BlasQueue<memorySpace> *BlasQueue)
    {
      d_storage =
        std::make_unique<typename MultiVector<ValueType, memorySpace>::Storage>(
          size * numVectors, initVal);
      d_BlasQueue = BlasQueue;
      d_vectorAttributes =
        VectorAttributes(VectorAttributes::Distribution::SERIAL);
      d_globalSize       = size;
      d_locallyOwnedSize = size;
      d_ghostSize        = 0;
      d_localSize        = d_locallyOwnedSize + d_ghostSize;
      d_numVectors       = numVectors;
    }

    //
    // Constructor using user provided Vector::Storage (i.e.,
    // utils::MemoryStorage)
    //
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    SerialMultiVector<ValueType, memorySpace>::SerialMultiVector(
      std::unique_ptr<typename MultiVector<ValueType, memorySpace>::Storage>
                                          storage,
      const size_type                     numVectors,
      blasLapack::BlasQueue<memorySpace> *BlasQueue)
    {
      d_storage   = std::move(storage);
      d_BlasQueue = BlasQueue;
      d_vectorAttributes =
        VectorAttributes(VectorAttributes::Distribution::SERIAL);
      d_globalSize       = d_storage.size();
      d_locallyOwnedSize = d_storage.size();
      d_ghostSize        = 0;
      d_localSize        = d_locallyOwnedSize + d_ghostSize;
      d_numVectors       = numVectors;
    }

    //
    // Copy Constructor
    //
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    SerialMultiVector<ValueType, memorySpace>::SerialMultiVector(
      const SerialMultiVector<ValueType, memorySpace> &u)
    {
      d_storage =
        std::make_unique<typename MultiVector<ValueType, memorySpace>::Storage>(
          (u.d_storage)->size());
      *d_storage         = *(u.d_storage);
      d_BlasQueue        = u.d_BlasQueue;
      d_vectorAttributes = u.d_vectorAttributes;
      d_globalSize       = u.d_globalSize;
      d_locallyOwnedSize = u.d_locallyOwnedSize;
      d_ghostSize        = u.d_ghostSize;
      d_localSize        = u.d_localSize;
      d_numVectors       = u.d_numVectors;
      VectorAttributes vectorAttributesSerial(
        VectorAttributes::Distribution::SERIAL);
      bool areCompatible =
        d_vectorAttributes.areDistributionCompatible(vectorAttributesSerial);
      utils::throwException<utils::LogicError>(
        areCompatible,
        "Trying to copy from an incompatible Vector. One is a SerialMultiVector and the "
        " other a DistributedVector.");
    }

    //
    // Move Constructor
    //
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    SerialMultiVector<ValueType, memorySpace>::SerialMultiVector(
      SerialMultiVector<ValueType, memorySpace> &&u) noexcept
    {
      d_storage          = std::move(u.d_storage);
      d_BlasQueue        = std::move(u.d_BlasQueue);
      d_vectorAttributes = std::move(u.d_vectorAttributes);
      d_globalSize       = std::move(u.d_globalSize);
      d_locallyOwnedSize = std::move(u.d_locallyOwnedSize);
      d_ghostSize        = std::move(u.d_ghostSize);
      d_localSize        = std::move(u.d_localSize);
      d_numVectors       = std::move(u.d_numVectors);
      VectorAttributes vectorAttributesSerial(
        VectorAttributes::Distribution::SERIAL);
      bool areCompatible =
        d_vectorAttributes.areDistributionCompatible(vectorAttributesSerial);
      utils::throwException<utils::LogicError>(
        areCompatible,
        "Trying to move from an incompatible Vector. One is a SerialMultiVector and the "
        " other a DistributedVector.");
    }

    //
    // Copy assignment
    //
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    SerialMultiVector<ValueType, memorySpace> &
    SerialMultiVector<ValueType, memorySpace>::operator=(
      const SerialMultiVector<ValueType, memorySpace> &u)
    {
      d_storage =
        std::make_unique<typename MultiVector<ValueType, memorySpace>::Storage>(
          (u.d_storage)->size());
      *d_storage         = *(u.d_storage);
      d_BlasQueue        = u.d_BlasQueue;
      d_vectorAttributes = u.d_vectorAttributes;
      d_globalSize       = u.d_globalSize;
      d_locallyOwnedSize = u.d_locallyOwnedSize;
      d_ghostSize        = u.d_ghostSize;
      d_localSize        = u.d_localSize;
      d_numVectors       = u.d_numVectors;
      VectorAttributes vectorAttributesSerial(
        VectorAttributes::Distribution::SERIAL);
      bool areCompatible =
        d_vectorAttributes.areDistributionCompatible(vectorAttributesSerial);
      utils::throwException<utils::LogicError>(
        areCompatible,
        "Trying to copy assign from an incompatible Vector. One is a SerialMultiVector and the "
        " other a DistributedVector.");
      return *this;
    }

    //
    // Move assignment
    //
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    SerialMultiVector<ValueType, memorySpace> &
    SerialMultiVector<ValueType, memorySpace>::operator=(
      SerialMultiVector<ValueType, memorySpace> &&u)
    {
      d_storage          = std::move(u.d_storage);
      d_BlasQueue        = std::move(u.d_BlasQueue);
      d_vectorAttributes = std::move(u.d_vectorAttributes);
      d_globalSize       = std::move(u.d_globalSize);
      d_locallyOwnedSize = std::move(u.d_locallyOwnedSize);
      d_ghostSize        = std::move(u.d_ghostSize);
      d_localSize        = std::move(u.d_localSize);
      d_numVectors       = std::move(u.d_numVectors);
      VectorAttributes vectorAttributesSerial(
        VectorAttributes::Distribution::SERIAL);
      bool areCompatible =
        d_vectorAttributes.areDistributionCompatible(vectorAttributesSerial);
      utils::throwException<utils::LogicError>(
        areCompatible,
        "Trying to move assign from an incompatible Vector. One is a SerialMultiVector and the "
        " other a DistributedVector.");
      return *this;
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    std::vector<double>
    SerialMultiVector<ValueType, memorySpace>::l2Norms() const
    {
      return blasLapack::nrms2MultiVector<ValueType, memorySpace>(
        this->size(), this->numVectors(), this->data(), *d_BlasQueue);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    std::vector<double>
    SerialMultiVector<ValueType, memorySpace>::lInfNorms() const
    {
      return blasLapack::amaxsMultiVector<ValueType, memorySpace>(
        this->size(), this->numVectors(), this->data(), *d_BlasQueue);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    SerialMultiVector<ValueType, memorySpace>::updateGhostValues(
      const size_type communicationChannel /*= 0*/)
    {}

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    SerialMultiVector<ValueType, memorySpace>::accumulateAddLocallyOwned(
      const size_type communicationChannel /*= 0*/)
    {}

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    SerialMultiVector<ValueType, memorySpace>::updateGhostValuesBegin(
      const size_type communicationChannel /*= 0*/)
    {}

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    SerialMultiVector<ValueType, memorySpace>::updateGhostValuesEnd()
    {}

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    SerialMultiVector<ValueType, memorySpace>::accumulateAddLocallyOwnedBegin(
      const size_type communicationChannel /*= 0*/)
    {}

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    SerialMultiVector<ValueType, memorySpace>::accumulateAddLocallyOwnedEnd()
    {}

  } // namespace linearAlgebra
} // namespace dftefe
