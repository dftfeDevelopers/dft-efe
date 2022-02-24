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
      : dftefe::utils::MemoryStorage<ValueType, memorySpace>(size, initVal)
    {}

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    Vector<ValueType, memorySpace>::Vector(
      const Vector<ValueType, memorySpace> &u)
      : dftefe::utils::MemoryStorage<ValueType, memorySpace>(
          (dftefe::utils::MemoryStorage<ValueType, memorySpace> &)u)
    {}

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    Vector<ValueType, memorySpace>::Vector(
      Vector<ValueType, memorySpace> &&u) noexcept
      : dftefe::utils::MemoryStorage<ValueType, memorySpace>(
          (dftefe::utils::MemoryStorage<ValueType, memorySpace> &&) u)
    {}


    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    Vector<ValueType, memorySpace> &
    Vector<ValueType, memorySpace>::operator+=(const Vector &rhs)
    {
      DFTEFE_AssertWithMsg(rhs.size() == this->size(),
                           "Size of two vectors should be the same.");
      VectorKernels<ValueType, memorySpace>::add(this->size(),
                                                 rhs.data(),
                                                 this->data());
      return *this;
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    Vector<ValueType, memorySpace> &
    Vector<ValueType, memorySpace>::operator-=(const Vector &rhs)
    {
      DFTEFE_AssertWithMsg(rhs.size() == this->size(),
                           "Size of two vectors should be the same.");
      VectorKernels<ValueType, memorySpace>::sub(this->size(),
                                                 rhs.data(),
                                                 this->data());
      return *this;
    }


    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    double
    Vector<ValueType, memorySpace>::l2Norm() const
    {
      return VectorKernels<ValueType, memorySpace>::l2Norm(this->size(),
                                                           this->data());
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    double
    Vector<ValueType, memorySpace>::lInfNorm() const
    {
      return VectorKernels<ValueType, memorySpace>::lInfNorm(this->size(),
                                                             this->data());
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
