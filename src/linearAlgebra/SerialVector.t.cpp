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

#include <utils/MemoryManager.h>
#include <linearAlgebra/VectorKernels.h>
#include <utils/Exceptions.h>
#include <utils/MemoryTransfer.h>

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
      : d_storage(size, initVal)
      , d_vectorAttributes(VectorAttributes::Distribution::SERIAL)
    {}

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    SerialVector<ValueType, memorySpace>::SerialVector(
      const SerialVector<ValueType, memorySpace> &u)
      : d_storage(u.d_storage)
      , d_vectorAttributes(u.d_vectorAttributes)
    {
      VectorAttributes vectorAttributesSerial(
	  VectorAttributes::Distribution::SERIAL);
      bool areCompatible =
        d_vectorAttributes.areDistributionCompatible(vectorAttributesSerial);
      utils::throwException<utils::LogicError>(
        areCompatible,
        "Trying to copy from an incompatible vector. One is a serial vector and the "
        " other a distributed vector.");
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    SerialVector<ValueType, memorySpace>::SerialVector(
      SerialVector<ValueType, memorySpace> &&u) noexcept
      : d_storage(std::move(u.d_storage))
      , d_vectorAttributes(std::move(u.d_vectorAttributes))
    {
      VectorAttributes vectorAttributesSerial(
	  VectorAttributes::Distribution::SERIAL);
      bool areCompatible =
        d_vectorAttributes.areDistributionCompatible(vectorAttributesSerial);
      utils::throwException<utils::LogicError>(
        areCompatible,
        "Trying to move from an incompatible vector. One is a serial vector and the "
        " other a distributed vector.");
    
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    VectorBase<ValueType, memorySpace> &
    SerialVector<ValueType, memorySpace>::operator+=(
      const VectorBase<ValueType, memorySpace> &rhs)
    {
      bool areCompatible =
        d_vectorAttributes.areDistributionCompatible(rhs.getVectorAttributes);
      utils::throwException<utils::LogicError>(
        areCompatible,
        "Trying to add incompatible vectors. One is a serial vector and the "
        " other a distributed vector.");
      utils::throwException<utils::LengthError>(
        rhs.size() == this->size(),
        "Mismatch of sizes of the two vectors that are being added.");
      const size_type rhsStorageSize = (rhs.getStorage()).size();
      utils::throwException<utils::LengthError>(
        d_storage.size() == rhsStorageSize,
        "Mismatch of sizes of the underlying"
        "storage of the two vectors that are being added.");
      VectorKernels<ValueType, memorySpace>::add(d_storage.size(),
                                                 rhs.data(),
                                                 this->data());
      return *this;
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    VectorBase<ValueType, memorySpace> &
    SerialVector<ValueType, memorySpace>::operator-=(
      const VectorBase<ValueType, memorySpace> &rhs)
    {
      bool areCompatible =
        d_vectorAttributes.areDistributionCompatible(rhs.getVectorAttributes);
      utils::throwException<utils::LogicError>(
        areCompatible,
        "Trying to add incompatible vectors. "
        "One is a serial vector and the other a distributed vector.");
      utils::throwException<utils::LengthError>(
        rhs.size() == this->size(),
        "Mismatch of sizes of the two vectors that are being subtracted.");
      const size_type rhsStorageSize = (rhs.getStorage()).size();
      utils::throwException<utils::LengthError>(
        (d_storage.size() == rhsStorageSize),
        "Mismatch of sizes of the underlying"
        "storage of the two vectors that are being subtracted.");
      VectorKernels<ValueType, memorySpace>::sub(d_storage.size(),
                                                 rhs.data(),
                                                 this->data());
      return *this;
    }


    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    double
    SerialVector<ValueType, memorySpace>::l2Norm() const
    {
      return VectorKernels<ValueType, memorySpace>::l2Norm(d_storage.size(),
                                                           this->data());
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    double
    SerialVector<ValueType, memorySpace>::lInfNorm() const
    {
      return VectorKernels<ValueType, memorySpace>::lInfNorm(d_storage.size(),
                                                             this->data());
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    const typename dftefe::linearAlgebra::VectorBase<ValueType,
                                                     memorySpace>::Storage &
    SerialVector<ValueType, memorySpace>::getStorage() const
    {
      return d_storage;
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    typename dftefe::linearAlgebra::VectorBase<ValueType, memorySpace>::iterator
    SerialVector<ValueType, memorySpace>::begin()
    {
      return d_storage.begin();
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    typename dftefe::linearAlgebra::VectorBase<ValueType,
                                               memorySpace>::const_iterator
    SerialVector<ValueType, memorySpace>::begin() const
    {
      return d_storage.begin();
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    typename dftefe::linearAlgebra::VectorBase<ValueType, memorySpace>::iterator
    SerialVector<ValueType, memorySpace>::end()
    {
      return d_storage.end();
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    typename dftefe::linearAlgebra::VectorBase<ValueType,
                                               memorySpace>::const_iterator
    SerialVector<ValueType, memorySpace>::end() const
    {
      return d_storage.end();
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    size_type
    SerialVector<ValueType, memorySpace>::size() const
    {
      return d_storage.size();
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    ValueType *
    SerialVector<ValueType, memorySpace>::data()
    {
      return d_storage.data();
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    const ValueType *
    SerialVector<ValueType, memorySpace>::data() const
    {
      return d_storage.data();
    }
    
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
      void
     SerialVector<ValueType, memorySpace>::scatterToGhost(
	 const size_type communicationChannel /*= 0*/) {}
    
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
      void
     SerialVector<ValueType, memorySpace>::gatherFromGhost(
	 const size_type communicationChannel /*= 0*/) {}

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
      void
     SerialVector<ValueType, memorySpace>::scatterToGhostBegin(
	 const size_type communicationChannel /*= 0*/) {}

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
      void
     SerialVector<ValueType, memorySpace>::scatterToGhostEnd(
	 const size_type communicationChannel /*= 0*/) {}

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
      void
     SerialVector<ValueType, memorySpace>::gatherFromGhostBegin(
	 const size_type communicationChannel /*= 0*/) {}
    
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
      void
     SerialVector<ValueType, memorySpace>::gatherFromGhostEnd(
	 const size_type communicationChannel /*= 0*/) {}

  } // namespace linearAlgebra
} // namespace dftefe
