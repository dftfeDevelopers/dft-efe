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

#include "DeviceAPICalls.h"
#include <algorithm>
#include <bitset>
#include <utils/Exceptions.h>
#include <climits>
#include <cstring>
#include <climits>


namespace dftefe
{
  namespace utils
  {
    template <typename ValueType>
    void
    MemoryManager<ValueType, MemorySpace::HOST>::allocate(size_type   size,
                                                          ValueType **ptr)
    {
      if (size > std::numeric_limits<size_type>::max())
      {
        utils::throwException(false,
                              "Size to be allocated more than the dftefe::size_type.");
      }
      if (size > 0)
        *ptr = new ValueType[size];
      else
        *ptr = nullptr;
    }

    template <typename ValueType>
    void
    MemoryManager<ValueType, MemorySpace::HOST>::deallocate(ValueType *ptr)
    {
      if (ptr != nullptr)
        delete[] ptr;
    }

    template <typename ValueType>
    void
    MemoryManager<ValueType, MemorySpace::HOST>::set(size_type  size,
                                                     ValueType *ptr,
                                                     ValueType  val)
    {
      if (size != 0)
        std::fill(ptr, ptr + size, val);
    }

    template <typename ValueType>
    void
    MemoryManager<ValueType, MemorySpace::HOST>::setZero(size_type  size,
                                                         ValueType *ptr)
    {
      if (size != 0)
        std::memset(ptr, (ValueType)0, size * sizeof(ValueType));
    }

#ifdef DFTEFE_WITH_DEVICE
    template <typename ValueType>
    void
    MemoryManager<ValueType, MemorySpace::HOST_PINNED>::allocate(
      size_type   size,
      ValueType **ptr)
    {
      if (size > std::numeric_limits<size_type>::max())
      {
        utils::throwException(false,
                              "Size to be allocated more than the dftefe::size_type.");
      }
      if (size > 0)
        hostPinnedMalloc((void **)ptr, size * sizeof(ValueType));
      else
        *ptr = nullptr;
    }

    template <typename ValueType>
    void
    MemoryManager<ValueType, MemorySpace::HOST_PINNED>::deallocate(
      ValueType *ptr)
    {
      if (ptr != nullptr)
        hostPinnedFree(ptr);
    }

    template <typename ValueType>
    void
    MemoryManager<ValueType, MemorySpace::HOST_PINNED>::set(size_type  size,
                                                            ValueType *ptr,
                                                            ValueType  val)
    {
      if (size > 0)
        std::fill(ptr, ptr + size, val);
    }

    template <typename ValueType>
    void
    MemoryManager<ValueType, MemorySpace::HOST_PINNED>::setZero(size_type  size,
                                                                ValueType *ptr)
    {
      if (size > 0)
        std::memset(ptr, (ValueType)0, size * sizeof(ValueType));
    }

    template <typename ValueType>
    void
    MemoryManager<ValueType, MemorySpace::DEVICE>::allocate(size_type   size,
                                                            ValueType **ptr)
    {
      if (size > std::numeric_limits<size_type>::max())
      {
        utils::throwException(false,
                              "Size to be allocated more than the dftefe::size_type.");
      }
      if (size > 0)
        deviceMalloc((void **)ptr, size * sizeof(ValueType));
      else
        *ptr = nullptr;
    }

    template <typename ValueType>
    void
    MemoryManager<ValueType, MemorySpace::DEVICE>::deallocate(ValueType *ptr)
    {
      if (ptr != nullptr)
        deviceFree(ptr);
    }

    template <typename ValueType>
    void
    MemoryManager<ValueType, MemorySpace::DEVICE>::set(size_type  size,
                                                       ValueType *ptr,
                                                       ValueType  val)
    {
      if (size > 0)
        deviceSetValue(ptr, val, size);
    }

    template <typename ValueType>
    void
    MemoryManager<ValueType, MemorySpace::DEVICE>::setZero(size_type  size,
                                                           ValueType *ptr)
    {
      if (size > 0)
        deviceSetValue(ptr, (ValueType)0, size);
    }

#endif // DFTEFE_WITH_DEVICE
  }    // namespace utils

} // namespace dftefe
