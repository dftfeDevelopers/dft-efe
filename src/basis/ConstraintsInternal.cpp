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


#include "ConstraintsInternal.h"


namespace dftefe
{
  namespace basis
  {
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    ConstraintsInternal<ValueType, memorySpace>::
      constraintsDistributeParentToChild(
        linearAlgebra::Vector<ValueType, memorySpace> &vectorData,
        const size_type                                blockSize,
        const utils::MemoryStorage<size_type, memorySpace>
                                                           &rowConstraintsIdsLocal,
        const utils::MemoryStorage<size_type, memorySpace> &rowConstraintsSizes,
        const utils::MemoryStorage<size_type, memorySpace>
          &columnConstraintsIdsLocal,
        const utils::MemoryStorage<size_type, memorySpace>
          &columnConstraintsAccumulated,
        const utils::MemoryStorage<double, memorySpace>
          &columnConstraintsValues,
        const utils::MemoryStorage<ValueType, memorySpace>
          &constraintsInhomogenities)
    {
      std::vector<ValueType> newValuesBlock(blockSize, 0.0);
      for (size_type i = 0; i < rowConstraintsIdsLocal.size(); ++i)
        {
          std::fill(newValuesBlock.begin(),
                    newValuesBlock.end(),
                    *(constraintsInhomogenities.begin() + i));

          const size_type startingLocalDofIndexRow =
            (*(rowConstraintsIdsLocal.begin() + i)) * blockSize;

          size_type columnIndexStart = columnConstraintsAccumulated[i];
          for (size_type j = 0; j < *(rowConstraintsSizes.begin() + i); ++j)
            {
              utils::throwException(
                count < columnConstraintsValues.size(),
                "Array out of bounds in ConstraintsInternal::constraintsDistributeParentToChild");


              const size_type startingLocalDofIndexColumn =
                ((*(columnConstraintsIdsLocal.begin() + columnIndexStart + j )) * blockSize);

              ValueType alpha = *(columnConstraintsValues.begin() + columnIndexStart + j);

              // TODO check if this performance efficient
              for (size_type iBlock = 0; iBlock < blockSize; iBlock++)
                {
                  newValuesBlock[iBlock] +=
                    (*(vectorData.begin() + startingLocalDofIndexColumn +
                       iBlock)) *
                    alpha;
                }
            }

          std::copy(&newValuesBlock[0],
                    &newValuesBlock[0] + blockSize,
                    vectorData.begin() + startingLocalDofIndexRow);
        }
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    ConstraintsInternal<ValueType, memorySpace>::
      constraintsDistributeChildToParent(
        linearAlgebra::Vector<ValueType, memorySpace> &vectorData,
        const size_type                                blockSize,
        const utils::MemoryStorage<size_type, memorySpace>
          &rowConstraintsIdsLocal,
        const utils::MemoryStorage<size_type, memorySpace> &rowConstraintsSizes,
        const utils::MemoryStorage<size_type, memorySpace>
          &columnConstraintsIdsLocal,
        const utils::MemoryStorage<size_type, memorySpace>
          &columnConstraintsAccumulated,
        const utils::MemoryStorage<double, memorySpace>
          &columnConstraintsValues)
    {
      for (size_type i = 0; i < rowConstraintsIdsLocal.size(); ++i)
        {
          const size_type startingLocalDofIndexRow =
            (*(rowConstraintsIdsLocal.begin() + i)) * blockSize;

          size_type columnIndexStart = columnConstraintsAccumulated[i];
          for (unsigned int j = 0; j < *(rowConstraintsSizes.begin() + i); ++j)
            {
              const size_type startingLocalDofIndexColumn =
                (*(columnConstraintsIdsLocal.begin() + columnIndexStart + j)) * blockSize;

              ValueType alpha = (*(columnConstraintsValues.begin() + columnIndexStart + j));
              for (size_type iBlock = 0; iBlock < blockSize; iBlock++)
                {
                  *(vectorData.begin() + startingLocalDofIndexColumn + iBlock) +=
                    (*(vectorData.begin() + startingLocalDofIndexRow +
                       iBlock)) *
                    alpha;
                }
            }

          //
          // set constraint nodes to zero
          //
          std::fill(vectorData.begin() + startingLocalDofIndexRow,
                    vectorData.begin() + startingLocalDofIndexRow + blockSize,
                    0.0);
        }
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    ConstraintsInternal<ValueType, memorySpace>::
      constraintsSetConstrainedNodesToZero(
        linearAlgebra::Vector<ValueType, memorySpace> &vectorData,
        const size_type                                blockSize,
        const utils::MemoryStorage<size_type, memorySpace>
          &rowConstraintsIdsLocal)
    {
      for (unsigned int i = 0; i < rowConstraintsIdsLocal.size(); ++i)
        {
          const global_size_type startingLocalDofIndexRow =
            *(rowConstraintsIdsLocal.begin() + i) * blockSize;

          // set constrained nodes to zero
          std::fill(vectorData.begin() + startingLocalDofIndexRow,
                    vectorData.begin() + startingLocalDofIndexRow + blockSize,
                    0.0);
        }
    }

    template class ConstraintsInternal<double,
                                       dftefe::utils::MemorySpace::HOST>;
    template class ConstraintsInternal<float, dftefe::utils::MemorySpace::HOST>;
    template class ConstraintsInternal<std::complex<double>,
                                       dftefe::utils::MemorySpace::HOST>;
    template class ConstraintsInternal<std::complex<float>,
                                       dftefe::utils::MemorySpace::HOST>;

#ifdef DFTEFE_WITH_DEVICE
    template class ConstraintsInternal<double,
                                       dftefe::utils::MemorySpace::HOST_PINNED>;
    template class ConstraintsInternal<float,
                                       dftefe::utils::MemorySpace::HOST_PINNED>;
    template class ConstraintsInternal<std::complex<double>,
                                       dftefe::utils::MemorySpace::HOST_PINNED>;
    template class ConstraintsInternal<std::complex<float>,
                                       dftefe::utils::MemorySpace::HOST_PINNED>;
#endif

  } // namespace basis
} // namespace dftefe
