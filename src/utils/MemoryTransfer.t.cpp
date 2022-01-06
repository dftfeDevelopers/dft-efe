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

#include <algorithm>
#include "MemoryTransfer.h"
#include "DeviceAPICalls.h"

namespace dftefe
{
  namespace utils
  {
    template <typename ValueType>
    void
    MemoryTransfer<MemorySpace::HOST, MemorySpace::HOST>::copy(
      size_type        size,
      ValueType *      dst,
      const ValueType *src)
    {
      std::copy(src, src + size, dst);
    }

#ifdef DFTEFE_WITH_DEVICE
    template <typename ValueType>
    void
    MemoryTransfer<MemorySpace::HOST, MemorySpace::HOST_PINNED>::copy(
      size_type        size,
      ValueType *      dst,
      const ValueType *src)
    {
      std::copy(src, src + size, dst);
    }

    template <typename ValueType>
    void
    MemoryTransfer<MemorySpace::HOST, MemorySpace::DEVICE>::copy(
      size_type        size,
      ValueType *      dst,
      const ValueType *src)
    {
      deviceMemcpyD2H(dst, src, size * sizeof(ValueType));
    }

    template <typename ValueType>
    void
    MemoryTransfer<MemorySpace::HOST_PINNED, MemorySpace::HOST>::copy(
      size_type        size,
      ValueType *      dst,
      const ValueType *src)
    {
      std::copy(src, src + size, dst);
    }

    template <typename ValueType>
    void
    MemoryTransfer<MemorySpace::HOST_PINNED, MemorySpace::HOST_PINNED>::copy(
      size_type        size,
      ValueType *      dst,
      const ValueType *src)
    {
      std::copy(src, src + size, dst);
    }

    template <typename ValueType>
    void
    MemoryTransfer<MemorySpace::HOST_PINNED, MemorySpace::DEVICE>::copy(
      size_type        size,
      ValueType *      dst,
      const ValueType *src)
    {
      deviceMemcpyD2H(dst, src, size * sizeof(ValueType));
    }

    template <typename ValueType>
    void
    MemoryTransfer<MemorySpace::DEVICE, MemorySpace::HOST>::copy(
      size_type        size,
      ValueType *      dst,
      const ValueType *src)
    {
      deviceMemcpyH2D(dst, src, size * sizeof(ValueType));
    }

    template <typename ValueType>
    void
    MemoryTransfer<MemorySpace::DEVICE, MemorySpace::HOST_PINNED>::copy(
      size_type        size,
      ValueType *      dst,
      const ValueType *src)
    {
      deviceMemcpyH2D(dst, src, size * sizeof(ValueType));
    }

    template <typename ValueType>
    void
    MemoryTransfer<MemorySpace::DEVICE, MemorySpace::DEVICE>::copy(
      size_type        size,
      ValueType *      dst,
      const ValueType *src)
    {
      deviceMemcpyD2D(dst, src, size * sizeof(ValueType));
    }
#endif // DFTEFE_WITH_DEVICE
  }    // namespace utils
} // namespace dftefe
