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

#include <linearAlgebra/BlasWrappers.h>
#include <utils/Exceptions.h>

namespace dftefe
{
  namespace linearAlgebra
  {
    //
    // Constructor
    //
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    Vector<ValueType, memorySpace>::Vector(std::unique_ptr<Storage> &storage,
                                           const global_size_type    globalSize,
                                           const size_type locallyOwnedSize,
                                           const size_type ghostSize)
      : d_storage(storage)
      , d_vectorAttributes(
          VectorAttributes(VectorAttributes::Distribution::SERIAL))
      , d_globalSize(globalSize)
      , d_locallyOwnedSize(locallyOwnedSize)
      , d_ghostSize(ghostSize)
    {
      d_localSize = locallyOwnedSize + ghostSize;
    }

    //
    // Default Constructor
    //
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    Vector<ValueType, memorySpace>::Vector()
      : d_storage(nullptr)
      , d_vectorAttributes(
          VectorAttributes(VectorAttributes::Distribution::SERIAL))
      , d_globalSize(0)
      , d_locallyOwnedSize(0)
      , d_ghostSize(0)
      , d_localSize(0)
    {}

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    typename Vector<ValueType, memorySpace>::iterator
    Vector<ValueType, memorySpace>::begin()
    {
      return d_storage->begin();
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    typename Vector<ValueType, memorySpace>::const_iterator
    Vector<ValueType, memorySpace>::begin() const
    {
      return d_storage->begin();
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    typename Vector<ValueType, memorySpace>::iterator
    Vector<ValueType, memorySpace>::end()
    {
      return d_storage->end();
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    typename Vector<ValueType, memorySpace>::const_iterator
    Vector<ValueType, memorySpace>::end() const
    {
      return d_storage->end();
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    global_size_type
    Vector<ValueType, memorySpace>::size() const
    {
      return d_globalSize;
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    size_type
    Vector<ValueType, memorySpace>::locallyOwnedSize() const
    {
      return d_locallyOwnedSize;
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    size_type
    Vector<ValueType, memorySpace>::ghostSize() const
    {
      return d_ghostSize;
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    size_type
    Vector<ValueType, memorySpace>::localSize() const
    {
      return d_localSize;
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    ValueType *
    Vector<ValueType, memorySpace>::data()
    {
      return d_storage->data();
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    const ValueType *
    Vector<ValueType, memorySpace>::data() const
    {
      return d_storage->data();
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    Vector<ValueType, memorySpace> &
    Vector<ValueType, memorySpace>::operator+=(
      const Vector<ValueType, memorySpace> &rhs)
    {
      bool areCompatible =
        d_vectorAttributes.areDistributionCompatible(rhs.getVectorAttributes());
      utils::throwException<utils::LogicError>(
        areCompatible,
        "Trying to add incompatible Vectors. One is a serial Vector and the "
        " other a distributed Vector.");
      utils::throwException<utils::LengthError>(
        rhs.size() == this->size(),
        "Mismatch of sizes of the two Vectors that are being added.");
      const size_type rhsStorageSize = (rhs.getValues()).size();
      utils::throwException<utils::LengthError>(
        d_storage->size() == rhsStorageSize,
        "Mismatch of sizes of the underlying"
        "storage of the two Vectors that are being added.");
      axpy(d_storage->size(), 1.0, rhs.data(), 1, this->data(), 1);

      return *this;
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    Vector<ValueType, memorySpace>::addLocal(
      const Vector<ValueType, memorySpace> &rhs)
    {
      bool areCompatible =
        d_vectorAttributes.areDistributionCompatible(rhs.getVectorAttributes());
      utils::throwException<utils::LogicError>(
        areCompatible,
        "Trying to add incompatible Vectors. One is a serial Vector and the "
        " other a distributed Vector.");
      utils::throwException<utils::LengthError>(
        rhs.size() == d_globalSize,
        "Mismatch of sizes of the two Vectors that are being added.");
      const size_type rhsStorageSize = (rhs.getValues()).size();
      utils::throwException<utils::LengthError>(
        d_localSize <= rhsStorageSize,
        "Mismatch of sizes of the underlying"
        "storage of the two Vectors that are being added.");
      axpy(d_localSize, 1.0, rhs.data(), 1, this->data(), 1);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    Vector<ValueType, memorySpace> &
    Vector<ValueType, memorySpace>::operator-=(
      const Vector<ValueType, memorySpace> &rhs)
    {
      bool areCompatible =
        d_vectorAttributes.areDistributionCompatible(rhs.getVectorAttributes());
      utils::throwException<utils::LogicError>(
        areCompatible,
        "Trying to subtract incompatible Vectors. "
        "One is a serial vector and the other a distributed Vector.");
      utils::throwException<utils::LengthError>(
        rhs.size() == this->size(),
        "Mismatch of sizes of the two Vectors that are being subtracted.");
      const size_type rhsStorageSize = (rhs.getValues()).size();
      utils::throwException<utils::LengthError>(
        (d_storage->size() == rhsStorageSize),
        "Mismatch of sizes of the underlying"
        "storage of the two Vectors that are being subtracted.");
      axpy(d_storage->size(), -1.0, rhs.data(), 1, this->data(), 1);
      return *this;
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    Vector<ValueType, memorySpace>::subLocal(
      const Vector<ValueType, memorySpace> &rhs)
    {
      bool areCompatible =
        d_vectorAttributes.areDistributionCompatible(rhs.getVectorAttributes());
      utils::throwException<utils::LogicError>(
        areCompatible,
        "Trying to subtract incompatible Vectors. "
        "One is a serial vector and the other a distributed Vector.");
      utils::throwException<utils::LengthError>(
        rhs.size() == d_globalSize,
        "Mismatch of sizes of the two Vectors that are being subtracted.");
      const size_type rhsStorageSize = (rhs.getValues()).size();
      utils::throwException<utils::LengthError>(
        (d_localSize <= rhsStorageSize),
        "Mismatch of sizes of the underlying"
        "storage of the two Vectors that are being subtracted.");
      axpy(d_localSize, -1.0, rhs.data(), 1, this->data(), 1);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    const typename Vector<ValueType, memorySpace>::Storage &
    Vector<ValueType, memorySpace>::getValues() const
    {
      return *d_storage;
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    typename Vector<ValueType, memorySpace>::Storage &
    Vector<ValueType, memorySpace>::getValues()
    {
      return *d_storage;
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    template <dftefe::utils::MemorySpace memorySpace2>
    void
    Vector<ValueType, memorySpace>::setValues(
      const typename Vector<ValueType, memorySpace2>::Storage &storage)
    {
      d_storage->copyFrom(storage);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    Vector<ValueType, memorySpace>::setStorage(
      std::unique_ptr<typename Vector<ValueType, memorySpace>::Storage>
        &storage)
    {
      d_storage = std::move(storage);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    const VectorAttributes &
    Vector<ValueType, memorySpace>::getVectorAttributes() const
    {
      return d_vectorAttributes;
    }

    //
    // Helper functions
    //


    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    add(ValueType                             a,
        const Vector<ValueType, memorySpace> &u,
        ValueType                             b,
        const Vector<ValueType, memorySpace> &v,
        Vector<ValueType, memorySpace> &      w)
    {
      const VectorAttributes &uVectorAttributes = u.getVectorAttributes();
      const VectorAttributes &vVectorAttributes = v.getVectorAttributes();
      const VectorAttributes &wVectorAttributes = w.getVectorAttributes();
      bool                    areCompatible =
        uVectorAttributes.areDistributionCompatible(vVectorAttributes);
      utils::throwException(
        areCompatible,
        "Trying to add incompatible Vectors. One is a SerialVector and the other a DistributedVector.");
      areCompatible =
        vVectorAttributes.areDistributionCompatible(wVectorAttributes);
      utils::throwException(
        areCompatible,
        "Trying to add incompatible vectors. One is a serialVector and the other a DistributedVector.");
      utils::throwException<utils::LengthError>(
        ((u.size() == v.size()) && (v.size() == w.size())),
        "Mismatch of sizes of the Vectors that are added.");
      const size_type uStorageSize = (u.getValues()).size();
      const size_type vStorageSize = (v.getValues()).size();
      const size_type wStorageSize = (w.getValues()).size();
      utils::throwException<utils::LengthError>(
        (uStorageSize == vStorageSize) && (vStorageSize == wStorageSize),
        "Mismatch of sizes of the underlying storages"
        "of the Vectors that are added.");
      axpy(uStorageSize, a, u.data(), 1, w.data(), 1);

      axpy(uStorageSize, b, v.data(), 1, w.data(), 1);
    }
  } // namespace linearAlgebra
} // namespace dftefe
