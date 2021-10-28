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
#include <cstring>
#include <algorithm>


namespace dftefe
{
  namespace utils
  {
    template <typename NumType>
    void
    MemoryManager<NumType, MemorySpace::HOST>::allocate(size_type size,
                                                        NumType **ptr)
    {
      *ptr = new NumType[size];
    }

    template <typename NumType>
    void
    MemoryManager<NumType, MemorySpace::HOST>::deallocate(NumType *ptr)
    {
      delete[] ptr;
    }

    // todo
    template <typename NumType>
    void
    MemoryManager<NumType, MemorySpace::HOST>::set(size_type size,
                                                   NumType  *ptr,
                                                   NumType   val)
    {
      for (int i = 0; i < size; ++i)
        {
          ptr[i] = val;
        }
    }

    template <typename NumType>
    void
    MemoryManager<NumType, MemorySpace::DEVICE>::allocate(size_type size,
                                                          NumType **ptr)
    {
      deviceMalloc((void **)ptr, size * sizeof(NumType));
      deviceMemset(*ptr, size * sizeof(NumType));
    }

    template <typename NumType>
    void
    MemoryManager<NumType, MemorySpace::DEVICE>::deallocate(NumType *ptr)
    {
      deviceFree(ptr);
    }

    template <typename NumType>
    void
    MemoryManager<NumType, MemorySpace::DEVICE>::set(size_type size,
                                                     NumType  *ptr,
                                                     NumType   val)
    {
      // todo
      deviceMemset(ptr, size * sizeof(NumType));
    }
  } // namespace utils

} // namespace dftefe
