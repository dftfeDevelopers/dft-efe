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
 * @author Ian C. Lin.
 */

#ifndef dftefeMemoryTransfer_h
#define dftefeMemoryTransfer_h

#include <utils/MemorySpaceType.h>
#include <utils/TypeConfig.h>

namespace dftefe
{
  namespace utils
  {
    template <MemorySpace memorySpaceDst, MemorySpace memorySpaceSrc>
    class MemoryTransfer
    {
    public:
      /**
       * @brief Copy array from the memory space of source to the memory space of destination
       * @param size the length of the array
       * @param dst pointer to the destination
       * @param src pointer to the source
       */
      template <typename ValueType>
      static void
      copy(size_type size, ValueType *dst, const ValueType *src);
    };

    template <>
    class MemoryTransfer<MemorySpace::HOST, MemorySpace::HOST>
    {
    public:
      template <typename ValueType>
      static void
      copy(size_type size, ValueType *dst, const ValueType *src);
    };

#ifdef DFTEFE_WITH_DEVICE
    template <>
    class MemoryTransfer<MemorySpace::HOST, MemorySpace::HOST_PINNED>
    {
    public:
      template <typename ValueType>
      static void
      copy(size_type size, ValueType *dst, const ValueType *src);
    };

    template <>
    class MemoryTransfer<MemorySpace::HOST, MemorySpace::DEVICE>
    {
    public:
      template <typename ValueType>
      static void
      copy(size_type size, ValueType *dst, const ValueType *src);
    };

    template <>
    class MemoryTransfer<MemorySpace::HOST_PINNED, MemorySpace::HOST>
    {
    public:
      template <typename ValueType>
      static void
      copy(size_type size, ValueType *dst, const ValueType *src);
    };

    template <>
    class MemoryTransfer<MemorySpace::HOST_PINNED, MemorySpace::HOST_PINNED>
    {
    public:
      template <typename ValueType>
      static void
      copy(size_type size, ValueType *dst, const ValueType *src);
    };

    template <>
    class MemoryTransfer<MemorySpace::HOST_PINNED, MemorySpace::DEVICE>
    {
    public:
      template <typename ValueType>
      static void
      copy(size_type size, ValueType *dst, const ValueType *src);
    };

    template <>
    class MemoryTransfer<MemorySpace::DEVICE, MemorySpace::HOST>
    {
    public:
      template <typename ValueType>
      static void
      copy(size_type size, ValueType *dst, const ValueType *src);
    };

    template <>
    class MemoryTransfer<MemorySpace::DEVICE, MemorySpace::HOST_PINNED>
    {
    public:
      template <typename ValueType>
      static void
      copy(size_type size, ValueType *dst, const ValueType *src);
    };

    template <>
    class MemoryTransfer<MemorySpace::DEVICE, MemorySpace::DEVICE>
    {
    public:
      template <typename ValueType>
      static void
      copy(size_type size, ValueType *dst, const ValueType *src);
    };
#endif // DFTEFE_WITH_DEVICE
  }    // namespace utils
} // namespace dftefe

#include "MemoryTransfer.t.cpp"

#endif // dftefeMemoryTransfer_h
