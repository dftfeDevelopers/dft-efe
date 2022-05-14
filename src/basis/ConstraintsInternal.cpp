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
    void ConstraintsInternal<ValueType, memorySpace>::constraintsDistributeParentToChild (Vector<ValueType, memorySpace> &vectorData,
                                       size_type blockSize,
                                       utils::MemoryStorage<global_size_type, memorySpace> & rowConstraintsIdsLocal,
                                       utils::MemoryStorage<global_size_type, memorySpace> & columnConstraintsIdsLocal,
                                       utils::MemoryStorage<double, memorySpace> & columnConstraintsValues,
                                       utils::MemoryStorage<ValueType, memorySpace> & constraintsInhomogenities)
      {
        size_type      count = 0;
        const size_type inc   = 1;
        std::vector<ValueType>     newValuesBlock(blockSize, 0.0);
        for (size_type i = 0; i < rowConstraintsIdsLocal.size(); ++i)
          {
            std::fill(newValuesBlock.begin(),
                      newValuesBlock.end(),
                      *(constraintsInhomogenities.data() + i));

            const global_size_type startingLocalDofIndexRow =
              d_localIndexMapUnflattenedToFlattened[d_rowIdsLocal[i]];

            for (unsigned int j = 0; j < d_rowSizes[i]; ++j)
              {
                Assert(
                  count < d_columnIdsGlobal.size(),
                  dealii::ExcMessage(
                    "Overloaded distribute for flattened array has indices out of bounds"));

                const dealii::types::global_dof_index
                  startingLocalDofIndexColumn =
                    d_localIndexMapUnflattenedToFlattened
                      [d_columnIdsLocal[count]];

                T alpha = d_columnValues[count];

                callaxpy(&blockSize,
                         &alpha,
                         fieldVector.begin() + startingLocalDofIndexColumn,
                         &inc,
                         &newValuesBlock[0],
                         &inc);
                count++;
              }

            std::copy(&newValuesBlock[0],
                      &newValuesBlock[0] + blockSize,
                      fieldVector.begin() + startingLocalDofIndexRow);
          }
      }

      template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
      void ConstraintsInternal<ValueType, memorySpace>::constraintsSetConstrainedNodesToZero (Vector<ValueType, memorySpace> &vectorData,
                                           size_type blockSize,
                                           utils::MemoryStorage<global_size_type, memorySpace> & rowConstraintsIdsLocal)
      {
        for (unsigned int i = 0; i < rowConstraintsIdsLocal.size(); ++i)
          {
            const global_size_type startingLocalDofIndexRow =
              *(rowConstraintsIdsLocal.data() + i)*blockSize;

            // set constrained nodes to zero
            std::fill(vectorData.begin() + startingLocalDofIndexRow,
                      vectorData.begin() + startingLocalDofIndexRow + blockSize,
                      0.0);
          }

      }


      template <typename ValueType>
      void ConstraintsInternal<ValueType,  dftefe::utils::MemorySpace::DEVICE>::constraintsSetConstrainedNodesToZero (Vector<ValueType, dftefe::utils::MemorySpace::DEVICE> &vectorData,
                                                                                        size_type blockSize,
                                                                                        utils::MemoryStorage<global_size_type, dftefe::utils::MemorySpace::DEVICE> & rowConstraintsIdsLocal)
      {
        const unsigned int numConstrainedDofs = rowConstraintsIdsLocal.size();

        if (numConstrainedDofs == 0)
          return;

        setzeroKernel<<<min((blockSize + 255) / 256 * numConstrainedDofs, 30000),
                        256>>>(
          blockSize,
          vectorData.begin(),
          d_rowIdsLocalDevice.begin(),
          numConstrainedDofs,
          blockSize);

      }

      template <typename ValueType>
      __global__ void
      setzeroKernel(const size_type  contiguousBlockSize,
                    ValueType *            xVec,
                    const size_type * constraintLocalRowIdsUnflattened,
                    const size_type  numConstraints,
                    const size_type  blockSize)
      {
        const global_size_type globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const global_size_type numberEntries =
          numConstraints * contiguousBlockSize;

        for (global_size_type index = globalThreadId;
             index < numberEntries;
             index += blockDim.x * gridDim.x)
          {
            const unsigned int blockIndex      = index / contiguousBlockSize;
            const unsigned int intraBlockIndex = index % contiguousBlockSize;
            xVec[constraintLocalRowIdsUnflattened[blockIndex]*blockSize +
                 intraBlockIndex]              = 0;
          }
      }

  }
}
