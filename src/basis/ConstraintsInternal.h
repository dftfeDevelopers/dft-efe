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

#include <linearAlgebra/MultiVector.h>

namespace dftefe
{
  namespace basis
  {
    template <typename ValueTypeBasisCoeff,
              dftefe::utils::MemorySpace memorySpace>
    class ConstraintsInternal
    {
    public:
      static void
      constraintsDistributeParentToChild(
        linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace> &vectorData,
        const size_type                                          blockSize,
        const utils::MemoryStorage<size_type, memorySpace>
          &rowConstraintsIdsLocal,
        const utils::MemoryStorage<size_type, memorySpace> &rowConstraintsSizes,
        const utils::MemoryStorage<size_type, memorySpace>
          &columnConstraintsIdsLocal,
        const utils::MemoryStorage<size_type, memorySpace>
          &columnConstraintsAccumulated,
        const utils::MemoryStorage<double, memorySpace>
          &columnConstraintsValues,
        const utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace>
          &constraintsInhomogenities);

      static void
      constraintsDistributeChildToParent(
        linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace> &vectorData,
        const size_type                                          blockSize,
        const utils::MemoryStorage<size_type, memorySpace>
          &rowConstraintsIdsLocal,
        const utils::MemoryStorage<size_type, memorySpace> &rowConstraintsSizes,
        const utils::MemoryStorage<size_type, memorySpace>
          &columnConstraintsIdsLocal,
        const utils::MemoryStorage<size_type, memorySpace>
          &columnConstraintsAccumulated,
        const utils::MemoryStorage<double, memorySpace>
          &columnConstraintsValues);

      static void
      constraintsSetConstrainedNodesToZero(
        linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace> &vectorData,
        const size_type                                          blockSize,
        const utils::MemoryStorage<size_type, memorySpace>
          &rowConstraintsIdsLocal);
    };


#ifdef DFTEFE_WITH_DEVICE
    template <typename ValueTypeBasisCoeff>
    class ConstraintsInternal<ValueTypeBasisCoeff,
                              dftefe::utils::MemorySpace::DEVICE>
    {
    public:
      static void
      constraintsDistributeParentToChild(
        linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                              dftefe::utils::MemorySpace::DEVICE> &vectorData,
        const size_type                                            blockSize,
        const utils::MemoryStorage<size_type,
                                   dftefe::utils::MemorySpace::DEVICE>
          &rowConstraintsIdsLocal,
        const utils::MemoryStorage<size_type,
                                   dftefe::utils::MemorySpace::DEVICE>
          &rowConstraintsSizes,
        const utils::MemoryStorage<size_type,
                                   dftefe::utils::MemorySpace::DEVICE>
          &columnConstraintsIdsLocal,
        const utils::MemoryStorage<size_type,
                                   dftefe::utils::MemorySpace::DEVICE>
          &columnConstraintsAccumulated,
        const utils::MemoryStorage<double, dftefe::utils::MemorySpace::DEVICE>
          &columnConstraintsValues,
        const utils::MemoryStorage<ValueTypeBasisCoeff,
                                   dftefe::utils::MemorySpace::DEVICE>
          &constraintsInhomogenities);

      static void
      constraintsDistributeChildToParent(
        linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                              dftefe::utils::MemorySpace::DEVICE> &vectorData,
        const size_type                                            blockSize,
        const utils::MemoryStorage<size_type,
                                   dftefe::utils::MemorySpace::DEVICE>
          &rowConstraintsIdsLocal,
        const utils::MemoryStorage<size_type,
                                   dftefe::utils::MemorySpace::DEVICE>
          &rowConstraintsSizes,
        const utils::MemoryStorage<size_type,
                                   dftefe::utils::MemorySpace::DEVICE>
          &columnConstraintsIdsLocal,
        const utils::MemoryStorage<size_type,
                                   dftefe::utils::MemorySpace::DEVICE>
          &columnConstraintsAccumulated,
        const utils::MemoryStorage<double, dftefe::utils::MemorySpace::DEVICE>
          &columnConstraintsValues);

      static void
      constraintsSetConstrainedNodesToZero(
        linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                              dftefe::utils::MemorySpace::DEVICE> &vectorData,
        const size_type                                            blockSize,
        const utils::MemoryStorage<size_type,
                                   dftefe::utils::MemorySpace::DEVICE>
          &rowConstraintsIdsLocal);
    };
#endif

  } // namespace basis
} // namespace dftefe


#endif // dftefeConstraintsInternal_h
