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
 * @author Sambit Das
 */

#include <linearAlgebra/VectorKernels.h>
#include <utils/Exceptions.h>

namespace dftefe
{
  namespace linearAlgebra
  {
    //
    // Constructor
    //
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    MultiVector<ValueType, memorySpace>::MultiVector(
      std::unique_ptr<Storage> &                              storage,
      const global_size_type                                  globalSize,
      const size_type                                         locallyOwnedSize,
      const size_type                                         ghostSize,
      const size_type                                         numVectors,
      std::shared_ptr<blasLapack::blasQueueType<memorySpace>> blasQueue)
      : d_storage(storage)
      , d_blasQueue(blasQueue)
      , d_vectorAttributes(
          VectorAttributes(VectorAttributes::Distribution::SERIAL, numVectors))
      , d_globalSize(globalSize)
      , d_locallyOwnedSize(locallyOwnedSize)
      , d_ghostSize(ghostSize)
      , d_numVectors(numVectors)
    {
      d_localSize = locallyOwnedSize + ghostSize;
    }

    //
    // Default Constructor
    //
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    MultiVector<ValueType, memorySpace>::MultiVector()
      : d_storage(nullptr)
      , d_blasQueue(nullptr)
      , d_vectorAttributes(
          VectorAttributes(VectorAttributes::Distribution::SERIAL, 0))
      , d_globalSize(0)
      , d_locallyOwnedSize(0)
      , d_ghostSize(0)
      , d_localSize(0)
      , d_numVectors(0)
    {}

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    typename MultiVector<ValueType, memorySpace>::iterator
    MultiVector<ValueType, memorySpace>::begin()
    {
      return d_storage->begin();
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    typename MultiVector<ValueType, memorySpace>::const_iterator
    MultiVector<ValueType, memorySpace>::begin() const
    {
      return d_storage->begin();
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    typename MultiVector<ValueType, memorySpace>::iterator
    MultiVector<ValueType, memorySpace>::end()
    {
      return d_storage->end();
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    typename MultiVector<ValueType, memorySpace>::const_iterator
    MultiVector<ValueType, memorySpace>::end() const
    {
      return d_storage->end();
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    global_size_type
    MultiVector<ValueType, memorySpace>::size() const
    {
      return d_globalSize;
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    size_type
    MultiVector<ValueType, memorySpace>::locallyOwnedSize() const
    {
      return d_locallyOwnedSize;
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    size_type
    MultiVector<ValueType, memorySpace>::ghostSize() const
    {
      return d_ghostSize;
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    size_type
    MultiVector<ValueType, memorySpace>::localSize() const
    {
      return d_localSize;
    }


    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    size_type
    MultiVector<ValueType, memorySpace>::numVectors() const
    {
      return d_numVectors;
    }


    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    ValueType *
    MultiVector<ValueType, memorySpace>::data()
    {
      return d_storage->data();
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    const ValueType *
    MultiVector<ValueType, memorySpace>::data() const
    {
      return d_storage->data();
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    MultiVector<ValueType, memorySpace> &
    MultiVector<ValueType, memorySpace>::operator+=(
      const MultiVector<ValueType, memorySpace> &rhs)
    {
      bool areCompatible =
        d_vectorAttributes.areDistributionCompatible(rhs.getVectorAttributes());
      utils::throwException<utils::LogicError>(
        areCompatible,
        "Trying to add incompatible MultiVectors. One is a serial MultiVector and the "
        " other a distributed MultiVector.");
      utils::throwException<utils::LengthError>(
        rhs.size() == this->size() && rhs.localSize() == this->localSize() &&
          rhs.numVectors() == this->numVectors(),
        "Mismatch of sizes of the two MultiVectors that are being added.");
      const size_type rhsStorageSize = (rhs.getValues()).size();
      utils::throwException<utils::LengthError>(
        d_storage->size() == rhsStorageSize,
        "Mismatch of sizes of the underlying"
        "storage of the two MultiVectors that are being added.");

      blasLapack::axpby<ValueType, memorySpace>(this->localSize() *
                                                  this->numVectors(),
                                                1.0,
                                                this->data(),
                                                1.0,
                                                rhs.data(),
                                                this->data(),
                                                *(this->getBlasQueue()));
      return *this;
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    MultiVector<ValueType, memorySpace> &
    MultiVector<ValueType, memorySpace>::operator-=(
      const MultiVector<ValueType, memorySpace> &rhs)
    {
      bool areCompatible =
        d_vectorAttributes.areDistributionCompatible(rhs.getVectorAttributes());
      utils::throwException<utils::LogicError>(
        areCompatible,
        "Trying to add incompatible MultiVectors. "
        "One is a serial vector and the other a distributed MultiVector.");
      utils::throwException<utils::LengthError>(
        rhs.size() == this->size() && rhs.localSize() == this->localSize() &&
          rhs.numVectors() == this->numVectors(),
        "Mismatch of sizes of the two MultiVectors that are being subtracted.");
      const size_type rhsStorageSize = (rhs.getValues()).size();
      utils::throwException<utils::LengthError>(
        (d_storage->size() == rhsStorageSize),
        "Mismatch of sizes of the underlying"
        "storage of the two MultiVectors that are being subtracted.");
      blasLapack::axpby<ValueType, memorySpace>(this->localSize() *
                                                  this->numVectors(),
                                                1.0,
                                                this->data(),
                                                -1.0,
                                                rhs.data(),
                                                this->data(),
                                                *(this->getBlasQueue()));
      return *this;
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    const typename MultiVector<ValueType, memorySpace>::Storage &
    MultiVector<ValueType, memorySpace>::getValues() const
    {
      return *d_storage;
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    typename MultiVector<ValueType, memorySpace>::Storage &
    MultiVector<ValueType, memorySpace>::getValues()
    {
      return *d_storage;
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    std::shared_ptr<blasLapack::blasQueueType<memorySpace>>
    Vector<ValueType, memorySpace>::getBlasQueue() const
    {
      return d_blasQueue;
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    template <dftefe::utils::MemorySpace memorySpace2>
    void
    MultiVector<ValueType, memorySpace>::setValues(
      const typename MultiVector<ValueType, memorySpace2>::Storage &storage)
    {
      d_storage->copyFrom(storage);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    MultiVector<ValueType, memorySpace>::setStorage(
      std::unique_ptr<typename MultiVector<ValueType, memorySpace>::Storage>
        &storage)
    {
      d_storage = std::move(storage);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    const VectorAttributes &
    MultiVector<ValueType, memorySpace>::getVectorAttributes() const
    {
      return d_vectorAttributes;
    }

    //
    // Helper functions
    //
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    add(ValueType                                  a,
        const MultiVector<ValueType, memorySpace> &u,
        ValueType                                  b,
        const MultiVector<ValueType, memorySpace> &v,
        MultiVector<ValueType, memorySpace> &      w)
    {
      const VectorAttributes &uVectorAttributes = u.getVectorAttributes();
      const VectorAttributes &vVectorAttributes = v.getVectorAttributes();
      const VectorAttributes &wVectorAttributes = w.getVectorAttributes();
      bool                    areCompatible =
        uVectorAttributes.areDistributionCompatible(vVectorAttributes);
      utils::throwException(
        areCompatible,
        "Trying to add incompatible MultiVectors. One is a SerialVector and the other a DistributedVector");
      areCompatible =
        vVectorAttributes.areDistributionCompatible(wVectorAttributes);
      utils::throwException(
        areCompatible,
        "Trying to add incompatible MultiVectors. One is a serialVector and the other a DistributedVector.");
      utils::throwException<utils::LengthError>(
        ((u.size() == v.size()) && (v.size() == w.size()) &&
         &&(u.localSize() == v.localSize()) &&
         (v.localSize() == w.localSize()) &&
         (u.numVectors() == v.numVectors()) &&
         (v.numVectors() == w.numVectors())),
        "Mismatch of sizes of the MultiVectors that are added.");
      const size_type uStorageSize = (u.getValues()).size();
      const size_type vStorageSize = (v.getValues()).size();
      const size_type wStorageSize = (w.getValues()).size();
      utils::throwException<utils::LengthError>(
        (uStorageSize == vStorageSize) && (vStorageSize == wStorageSize),
        "Mismatch of sizes of the underlying storages"
        "of the MultiVectors that are added.");

      blasLapack::axpby<ValueType, memorySpace>(u.localSize() * u.numVectors(),
                                                a,
                                                u.data(),
                                                b,
                                                w.data(),
                                                *(w.getBlasQueue()));
    }
  } // namespace linearAlgebra
} // namespace dftefe
