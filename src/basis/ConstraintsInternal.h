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
 * @author Vishal Subramanian
 */


#ifndef dftefeConstraintsInternal_h
#define dftefeConstraintsInternal_h

#include <utils/TypeConfig.h>
#include <utils/MemoryStorage.h>
#include <utils/MemorySpaceType.h>

namespace dftefe
{
  namespace basis
  {
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    class ConstraintsInternal
    {
    public:
      static void
      constraintsDistributeParentToChild(
        Vector<ValueType, memorySpace> &vectorData,
        size_type                       blockSize,
        utils::MemoryStorage<global_size_type, memorySpace>
          &rowConstraintsIdsLocal,
        utils::MemoryStorage<global_size_type, memorySpace>
          &                                        columnConstraintsIdsLocal,
        utils::MemoryStorage<double, memorySpace> &columnConstraintsValues,
        utils::MemoryStorage<ValueType, memorySpace>
          &constraintsInhomogenities);

      static void
      constraintsDistributeChildToParent(
        Vector<ValueType, memorySpace> &vectorData,
        size_type                       blockSize,
        utils::MemoryStorage<global_size_type, memorySpace>
          &rowConstraintsIdsLocal,
        utils::MemoryStorage<global_size_type, memorySpace>
          &                                        columnConstraintsIdsLocal,
        utils::MemoryStorage<double, memorySpace> &columnConstraintsValues,
        utils::MemoryStorage<ValueType, memorySpace>
          &constraintsInhomogenities);

      static void
      constraintsSetConstrainedNodesToZero(
        Vector<ValueType, memorySpace> &vectorData,
        size_type                       blockSize,
        utils::MemoryStorage<global_size_type, memorySpace>
          &rowConstraintsIdsLocal);
    };


#ifdef DFTEFE_WITH_DEVICE
    template <typename ValueType>
    class ConstraintsInternal<ValueType, dftefe::utils::MemorySpace::DEVICE>
    {
    public:
      static void
      constraintsDistributeParentToChild(
        Vector<ValueType, dftefe::utils::MemorySpace::DEVICE> &vectorData,
        size_type                                              blockSize,
        utils::MemoryStorage<global_size_type,
                             dftefe::utils::MemorySpace::DEVICE>
          &rowConstraintsIdsLocal,
        utils::MemoryStorage<global_size_type,
                             dftefe::utils::MemorySpace::DEVICE>
          &columnConstraintsIdsLocal,
        utils::MemoryStorage<double, dftefe::utils::MemorySpace::DEVICE>
          &columnConstraintsValues,
        utils::MemoryStorage<ValueType, dftefe::utils::MemorySpace::DEVICE>
          &constraintsInhomogenities);

      static void
      constraintsDistributeChildToParent(
        Vector<ValueType, dftefe::utils::MemorySpace::DEVICE> &vectorData,
        size_type                                              blockSize,
        utils::MemoryStorage<global_size_type,
                             dftefe::utils::MemorySpace::DEVICE>
          &rowConstraintsIdsLocal,
        utils::MemoryStorage<global_size_type,
                             dftefe::utils::MemorySpace::DEVICE>
          &columnConstraintsIdsLocal,
        utils::MemoryStorage<double, dftefe::utils::MemorySpace::DEVICE>
          &columnConstraintsValues,
        utils::MemoryStorage<ValueType, dftefe::utils::MemorySpace::DEVICE>
          &constraintsInhomogenities);

      static void
      constraintsSetConstrainedNodesToZero(
        Vector<ValueType, dftefe::utils::MemorySpace::DEVICE> &vectorData,
        size_type                                              blockSize,
        utils::MemoryStorage<global_size_type,
                             dftefe::utils::MemorySpace::DEVICE>
          &rowConstraintsIdsLocal);
    };
#endif

  } // namespace basis
} // namespace dftefe


#endif // dftefeConstraintsInternal_h
