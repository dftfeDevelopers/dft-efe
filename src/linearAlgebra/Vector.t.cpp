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

#include "MemoryManager.h"
#include "VectorKernels.h"
#include "Exceptions.h"
#include "MemoryTransfer.h"

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
      : d_size(size)
    {
      dftefe::utils::MemoryManager<ValueType, memorySpace>::allocate(size,
                                                                     &d_data);
      dftefe::utils::MemoryManager<ValueType, memorySpace>::set(size,
                                                                d_data,
                                                                initVal);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    Vector<ValueType, memorySpace>::resize(const size_type size,
                                           const ValueType initVal)
    {
      dftefe::utils::MemoryManager<ValueType, memorySpace>::deallocate(d_data);
      d_size = size;
      if (size > 0)
        {
          d_data =
            dftefe::utils::MemoryManager<ValueType, memorySpace>::allocate(
              size, nullptr);
          dftefe::utils::MemoryManager<ValueType, memorySpace>::set(size,
                                                                    d_data,
                                                                    initVal);
        }
    }

    //
    // Destructor
    //
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    Vector<ValueType, memorySpace>::~Vector()
    {
      dftefe::utils::MemoryManager<ValueType, memorySpace>::deallocate(d_data);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    Vector<ValueType, memorySpace>::Vector(
      const Vector<ValueType, memorySpace> &u)
      : d_size(u.d_size)
    {
      dftefe::utils::MemoryManager<ValueType, memorySpace>::allocate(d_size,
                                                                     &d_data);
      utils::MemoryTransfer<memorySpace, memorySpace>::copy(d_size,
                                                            d_data,
                                                            u.d_data);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    Vector<ValueType, memorySpace>::Vector(
      Vector<ValueType, memorySpace> &&u) noexcept
      : d_size(u.d_size)
      , d_data(nullptr)
    {
      *this = std::move(u);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    size_type
    Vector<ValueType, memorySpace>::size() const
    {
      return d_size;
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    typename Vector<ValueType, memorySpace>::iterator
    Vector<ValueType, memorySpace>::begin()
    {
      return d_data;
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    typename Vector<ValueType, memorySpace>::const_iterator
    Vector<ValueType, memorySpace>::begin() const
    {
      return d_data;
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    typename Vector<ValueType, memorySpace>::iterator
    Vector<ValueType, memorySpace>::end()
    {
      return (d_data + d_size);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    typename Vector<ValueType, memorySpace>::const_iterator
    Vector<ValueType, memorySpace>::end() const
    {
      return (d_data + d_size);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    Vector<ValueType, memorySpace> &
    Vector<ValueType, memorySpace>::operator=(
      const Vector<ValueType, memorySpace> &rhs)
    {
      if (&rhs != this)
        {
          if (rhs.d_size != d_size)
            {
              this->resize(rhs.d_size);
            }
          utils::MemoryTransfer<memorySpace, memorySpace>::copy(rhs.d_size,
                                                                this->d_data,
                                                                rhs.d_data);
        }
      return (*this);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    Vector<ValueType, memorySpace> &
    Vector<ValueType, memorySpace>::operator=(
      Vector<ValueType, memorySpace> &&rhs) noexcept
    {
      if (&rhs != this)
        {
          delete[] d_data;
          d_data     = rhs.d_data;
          d_size     = rhs.d_size;
          rhs.d_size = 0;
          rhs.d_data = nullptr;
        }
      return (*this);
    }

    //    // This part does not work for GPU version, will work on this until
    //    // having cleaner solution.
    //    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    //    typename Vector<ValueType, memorySpace>::reference
    //    Vector<ValueType, memorySpace>::operator[](const size_type i)
    //    {
    //
    //      return d_data[i];
    //    }
    //
    //    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    //    typename Vector<ValueType, memorySpace>::const_reference
    //    Vector<ValueType, memorySpace>::operator[](const size_type i) const
    //    {
    //      return d_data[i];
    //    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    Vector<ValueType, memorySpace> &
    Vector<ValueType, memorySpace>::operator+=(const Vector &rhs)
    {
      DFTEFE_AssertWithMsg(rhs.d_size == d_size,
                           "Size of two vectors should be the same.");
      VectorKernels<ValueType, memorySpace>::add(d_size, rhs.d_data, d_data);
      return *this;
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    Vector<ValueType, memorySpace> &
    Vector<ValueType, memorySpace>::operator-=(const Vector &rhs)
    {
      DFTEFE_AssertWithMsg(rhs.d_size == d_size,
                           "Size of two vectors should be the same.");
      VectorKernels<ValueType, memorySpace>::sub(d_size, rhs.d_data, d_data);
      return *this;
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    ValueType *
    Vector<ValueType, memorySpace>::data() noexcept
    {
      return d_data;
    }
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    const ValueType *
    Vector<ValueType, memorySpace>::data() const noexcept
    {
      return d_data;
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    add(ValueType                             a,
        const Vector<ValueType, memorySpace> &u,
        ValueType                             b,
        const Vector<ValueType, memorySpace> &v,
        Vector<ValueType, memorySpace> &      w)
    {
      DFTEFE_AssertWithMsg(((u.size() == v.size()) && (v.size() == w.size())),
                           "Size of two vectors should be the same.");
      VectorKernels<ValueType, memorySpace>::add(
        u.size(), a, u.data(), b, v.data(), w.data());
    }

  } // namespace linearAlgebra
} // namespace dftefe
