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

#include <utils/Exceptions.h>
#include "BlasWrappers.h"

namespace dftefe
{
  namespace linearAlgebra
  {
    //
    // Constructor
    //
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    SerialVector<ValueType, memorySpace>::SerialVector(const size_type size,
                                                       const ValueType initVal)
    {
      d_storage =
        std::make_shared<typename Vector<ValueType, memorySpace>::Storage>(
          size, initVal);
      d_vectorAttributes = VectorAttributes::Distribution::SERIAL;
      d_globalSize       = size;
      d_locallyOwnedSize = size;
      d_ghostSize        = 0;
      d_localSize        = d_locallyOwnedSize + d_ghostSize;
    }

    //
    // Copy Constructor
    //
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    SerialVector<ValueType, memorySpace>::SerialVector(
      const SerialVector<ValueType, memorySpace> &u)
    {
      d_storage =
        std::make_shared<typename Vector<ValueType, memorySpace>::Storage>(
          (u.d_storage)->size());
      *d_storage         = *(u.d_storage);
      d_vectorAttributes = u.d_vectorAttributes;
      d_globalSize       = u.d_globalSize;
      d_locallyOwnedSize = u.d_locallyOwnedSize;
      d_ghostSize        = u.d_ghostSize;
      d_localSize        = u.d_localSize;
      VectorAttributes vectorAttributesSerial(
        VectorAttributes::Distribution::SERIAL);
      bool areCompatible =
        d_vectorAttributes.areDistributionCompatible(vectorAttributesSerial);
      utils::throwException<utils::LogicError>(
        areCompatible,
        "Trying to copy from an incompatible Vector. One is a SerialVector and the "
        " other a DistributedVector.");
    }

    //
    // Move Constructor
    //
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    SerialVector<ValueType, memorySpace>::SerialVector(
      SerialVector<ValueType, memorySpace> &&u) noexcept
    {
      d_storage          = std::move(u.d_storage);
      d_vectorAttributes = std::move(u.d_vectorAttributes);
      d_globalSize       = std::move(u.d_globalSize);
      d_locallyOwnedSize = std::move(u.d_locallyOwnedSize);
      d_ghostSize        = std::move(u.d_ghostSize);
      d_localSize        = std::move(u.d_localSize);
      VectorAttributes vectorAttributesSerial(
        VectorAttributes::Distribution::SERIAL);
      bool areCompatible =
        d_vectorAttributes.areDistributionCompatible(vectorAttributesSerial);
      utils::throwException<utils::LogicError>(
        areCompatible,
        "Trying to move from an incompatible Vector. One is a SerialVector and the "
        " other a DistributedVector.");
    }

    //
    // Copy assignment
    //
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    SerialVector<ValueType, memorySpace> &
    SerialVector<ValueType, memorySpace>::operator=(
      const SerialVector<ValueType, memorySpace> &u)
    {
      d_storage =
        std::make_shared<typename Vector<ValueType, memorySpace>::Storage>(
          (u.d_storage)->size());
      *d_storage         = *(u.d_storage);
      d_vectorAttributes = u.d_vectorAttributes;
      d_globalSize       = u.d_globalSize;
      d_locallyOwnedSize = u.d_locallyOwnedSize;
      d_ghostSize        = u.d_ghostSize;
      d_localSize        = u.d_localSize;
      VectorAttributes vectorAttributesSerial(
        VectorAttributes::Distribution::SERIAL);
      bool areCompatible =
        d_vectorAttributes.areDistributionCompatible(vectorAttributesSerial);
      utils::throwException<utils::LogicError>(
        areCompatible,
        "Trying to copy assign from an incompatible Vector. One is a SerialVector and the "
        " other a DistributedVector.");
      return *this;
    }

    //
    // Move assignment
    //
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    SerialVector<ValueType, memorySpace> &
    SerialVector<ValueType, memorySpace>::operator=(
      SerialVector<ValueType, memorySpace> &&u)
    {
      d_storage          = std::move(u.d_storage);
      d_vectorAttributes = std::move(u.d_vectorAttributes);
      d_globalSize       = std::move(u.d_globalSize);
      d_locallyOwnedSize = std::move(u.d_locallyOwnedSize);
      d_ghostSize        = std::move(u.d_ghostSize);
      d_localSize        = std::move(u.d_localSize);
      VectorAttributes vectorAttributesSerial(
        VectorAttributes::Distribution::SERIAL);
      bool areCompatible =
        d_vectorAttributes.areDistributionCompatible(vectorAttributesSerial);
      utils::throwException<utils::LogicError>(
        areCompatible,
        "Trying to move assign from an incompatible Vector. One is a SerialVector and the "
        " other a DistributedVector.");
      return *this;
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    double
    SerialVector<ValueType, memorySpace>::l2Norm() const
    {

      return nrm2(d_storage->size(), this->data());
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    double
    SerialVector<ValueType, memorySpace>::lInfNorm() const
    {

      return amax(d_storage->size(), this->data());
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    SerialVector<ValueType, memorySpace>::scatterToGhost(
      const size_type communicationChannel /*= 0*/)
    {}

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    SerialVector<ValueType, memorySpace>::gatherFromGhost(
      const size_type communicationChannel /*= 0*/)
    {}

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    SerialVector<ValueType, memorySpace>::scatterToGhostBegin(
      const size_type communicationChannel /*= 0*/)
    {}

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    SerialVector<ValueType, memorySpace>::scatterToGhostEnd()
    {}

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    SerialVector<ValueType, memorySpace>::gatherFromGhostBegin(
      const size_type communicationChannel /*= 0*/)
    {}

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    SerialVector<ValueType, memorySpace>::gatherFromGhostEnd()
    {}

  } // namespace linearAlgebra
} // namespace dftefe
