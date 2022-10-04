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
    template <typename ValueTypeBasisCoeff, utils::MemorySpace memorySpace>
    void
    FECellWiseDataOperations<ValueTypeBasisCoeff, memorySpace>::
      copyFieldToCellWiseData(
        const ValueTypeBasisCoeff *data,
        const size_type            numComponents,
        const size_type *          cellLocalIdsStartPtr,
        const typename BasisHandler<ValueTypeBasisCoeff,
                                    memorySpace>::SizeTypeVector &numCellDofs,
        utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace> &cellWiseStorage)
    {
      auto            itCellWiseStorageBegin = cellWiseStorage.begin();
      const size_type numCells               = numCellDofs.size();
      size_type       cumulativeCellDofs     = 0;
      for (size_type iCell = 0; iCell < numCells; ++iCell)
        {
          const size_type cellDofs = *(numCellDofs.data() + iCell);
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

    template <typename ValueTypeBasisCoeff, utils::MemorySpace memorySpace>
    void
    FECellWiseDataOperations<ValueTypeBasisCoeff, memorySpace>::
      addCellWiseDataToFieldData(
        const utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace>
          &              cellWiseStorage,
        const size_type  numComponents,
        const size_type *cellLocalIdsStartPtr,
        const typename BasisHandler<ValueTypeBasisCoeff,
                                    memorySpace>::SizeTypeVector &numCellDofs,
        ValueTypeBasisCoeff *                                     data)
    {
      auto            itCellWiseStorageBegin = cellWiseStorage.begin();
      const size_type numCells               = numCellDofs.size();
      size_type       cumulativeCellDofs     = 0;
      for (size_type iCell = 0; iCell < numCells; ++iCell)
        {
          const size_type cellDofs = *(numCellDofs.data() + iCell);
          for (size_type iDof = 0; iDof < cellDofs; ++iDof)
            {
              const size_type localId =
                *(cellLocalIdsStartPtr + cumulativeCellDofs + iDof);
              auto srcPtr = itCellWiseStorageBegin +
                            (cumulativeCellDofs + iDof) * numComponents;
              auto dstPtr = data + localId * numComponents;

              for (size_type iComp = 0; iComp < numComponents; iComp++)
                {
                  *(dstPtr + iComp) += *(srcPtr + iComp);
                }
            }
          cumulativeCellDofs += cellDofs;
        }
    }

  } // end of namespace basis
} // end of namespace dftefe
