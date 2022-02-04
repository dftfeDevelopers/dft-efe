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
 * @author Ian C. Lin, Sambit Das.
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
    Vector<ValueType, memorySpace>::Vector(const size_type size,
                                           const ValueType initVal)
      : d_storage(size, initVal)
    {}

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    Vector<ValueType, memorySpace>::Vector(
      const Vector<ValueType, memorySpace> &u)
      : d_storage(u.d_storage)
    {}

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    Vector<ValueType, memorySpace>::Vector(
      Vector<ValueType, memorySpace> &&u) noexcept
      : d_storage(std::move(u.d_storage))
    {}


    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    Vector<ValueType, memorySpace> &
    Vector<ValueType, memorySpace>::operator+=(const Vector &rhs)
    {
      utils::throwException<utils::LengthError>(
        rhs.size() == this->size(),
        "Mismatch of sizes of the two vectors that are being added.");
      const size_type rhsStorageSize = (rhs.d_storage).size();
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
    Vector<ValueType, memorySpace> &
    Vector<ValueType, memorySpace>::operator-=(const Vector &rhs)
    {
      utils::throwException<utils::LengthError>(
        rhs.size() == this->size(),
        "Mismatch of sizes of the two vectors that are being subtracted.");
      const size_type rhsStorageSize = (rhs.d_storage).size();
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
    Vector<ValueType, memorySpace>::l2Norm() const
    {
      return VectorKernels<ValueType, memorySpace>::l2Norm(
        d_vectorStoage.size(), this->data());
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    double
    Vector<ValueType, memorySpace>::lInfNorm() const
    {
      return VectorKernels<ValueType, memorySpace>::lInfNorm(d_storage.size(),
                                                             this->data());
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    const Storage &
    Vector<ValueType, memorySpace>::getStorage() const
    {
      return d_storage;
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    iterator
    Vector<ValueType, memorySpace>::begin()
    {
      return d_storage.begin();
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    const_iterator
    Vector<ValueType, memorySpace>::begin() const
    {
      return d_storage.begin();
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    iterator
    Vector<ValueType, memorySpace>::end()
    {
      return d_storage.end();
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    const_iterator
    Vector<ValueType, memorySpace>::end() const
    {
      return d_storage.end();
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    Vector<ValueType, memorySpace>::resize(size_type size,
                                           ValueType initVal /*= ValueType()*/)
    {
      d_storage.resize(size, initVal);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    size_type
    Vector<ValueType, memorySpace>::size() const
    {
      return d_storage.size();
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    ValueType *
    Vector<ValueType, memorySpace>::data()
    {
      return d_storage.data();
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    const ValueType *
    Vector<ValueType, memorySpace>::data() const
    {
      return d_storage.data();
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    add(ValueType                             a,
        const Vector<ValueType, memorySpace> &u,
        ValueType                             b,
        const Vector<ValueType, memorySpace> &v,
        Vector<ValueType, memorySpace> &      w)
    {
      utils::throwException<utils::LengthError>(
        ((u.size() == v.size()) && (v.size() == w.size())),
        "Mismatch of sizes of the vectors that are added.");
      const size_type uStorageSize = (u.d_storage).size();
      const size_type vStorageSize = (v.d_storage).size();
      const size_type wStorageSize = (w.d_storage).size();
      utils::throwException<utils::LengthError>(
	  ((uStorageSize == vStorageSize) && (vStorageSize == wStorageSize),
                           "Mismatch of sizes of the underlying storages"
			   "of the vectors that are added.");
      VectorKernels<ValueType, memorySpace>::add(
        uStorageSize,
	a,
	u.data(),
	b,
	v.data(),
	w.data());
    }

  } // namespace linearAlgebra
} // namespace dftefe
