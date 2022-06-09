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
 * @author Bikash Kanungo, Vishal Subramanian
 */
namespace dftefe
{
  namespace basis
  {
    template <typename ValueType, utils::MemorySpace memorySpace>
    void
    FEBasisOperationsInternal<ValueType, memorySpace>::copyFieldToCellWiseData(
      const ValueType *                             data,
      const size_type                               numComponents,
      const size_type *                             cellLocalIdsStartPtr,
      const std::vector<size_type> &                numCellDofs,
      utils::MemoryStorage<ValueType, memorySpace> &cellWiseStorage)
    {
      auto            itCellWiseStorageBegin = cellWiseStorage.begin();
      const size_type numCells               = numCellDofs.size();
      size_type       cumulativeCellDofs     = 0;
      for (size_type iCell = 0; iCell < numCells; ++iCell)
        {
          const size_type cellDofs = numCellDofs[iCell];
          for (size_type iDof = 0; iDof < cellDofs; ++iDof)
            {
              const size_type localId =
                *(cellLocalIdsStartPtr + cumulativeCellDofs + iDof);
              auto srcPtr = data + localId * numComponents;
              auto dstPtr = itCellWiseStorageBegin +
                            (cumulativeCellDofs + iDof) * numComponents;
              std::copy(srcPtr, srcPtr + numComponents, dstPtr);
            }
          cumulativeCellDofs += cellDofs;
        }
    }

  } // end of namespace basis
} // end of namespace dftefe
