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
 * @author Bikash Kanungo, Vishal Subramanian, Avirup Sircar
 */

#ifndef dftefeFECellWiseDataOperations_h
#define dftefeFECellWiseDataOperations_h

#include <utils/TypeConfig.h>
#include <utils/MemorySpaceType.h>
#include <utils/MemoryStorage.h>
#include <basis/Field.h>
#include <basis/BasisDataStorage.h>
namespace dftefe
{
  namespace basis
  {
    template <typename ValueType, utils::MemorySpace memorySpace>
    class FECellWiseDataOperations
    {
    public:
      // TODO: Add numStrideCellWiseStorageDofs (max of numCellDofs over all
      // cells) This also takes the case where numCellDofs = 0 Appropriately
      // change src and dst ptrs
      static void
      copyFieldToCellWiseData(
        const ValueType *data,
        const size_type  numComponents,
        const size_type *cellLocalIdsStartPtr,
        const typename BasisManager<ValueType, memorySpace>::SizeTypeVector
          &                                                   numCellDofs,
        dftefe::utils::MemoryStorage<ValueType, memorySpace> &cellWiseStorage);

      static void
      copyFieldToCellWiseData(
        const ValueType *data,
        const size_type  numComponents,
        const size_type *cellLocalIdsStartPtr,
        const typename BasisManager<ValueType, memorySpace>::SizeTypeVector
          &                                                   numCellDofs,
        ValueType *itCellWiseStorageBegin);

      // TODO: Add numStrideCellWiseStorageDofs (max of numCellDofs over all
      // cells) This also takes the case where numCellDofs = 0
      static void
      addCellWiseDataToFieldData(
        const dftefe::utils::MemoryStorage<ValueType, memorySpace>
          &              cellWiseStorage,
        const size_type  numComponents,
        const size_type *cellLocalIdsStartPtr,
        const typename BasisManager<ValueType, memorySpace>::SizeTypeVector
          &        numCellDofs,
        ValueType *data);

      static void
      addCellWiseDataToFieldData(
        const ValueType *itCellWiseStorageBegin,
        const size_type  numComponents,
        const size_type *cellLocalIdsStartPtr,
        const typename BasisManager<ValueType, memorySpace>::SizeTypeVector
          &        numCellDofs,
        ValueType *data);        

      static void
      addCellWiseBasisDataToDiagonalData(
        const ValueType *cellWiseBasisData,
        const size_type *cellLocalIdsStartPtr,
        const utils::MemoryStorage<size_type, memorySpace> &numCellDofs,
        ValueType *                                         data);

      static void
      reshapeCellWiseData(
        const dftefe::utils::MemoryStorage<ValueType, memorySpace>
          &                                                 cellWiseStorage,
        const size_type                                     numComponents,
        const utils::MemoryStorage<size_type, memorySpace> &numCellVecs,
        ValueType *                                         data);


    }; // end of class FECellWiseDataOperations


#ifdef DFTEFE_WITH_DEVICE
    template <typename ValueType>
    class FECellWiseDataOperations
    {
    public:
      static void
      copyFieldToCellWiseData(
        const ValueType *data,
        const size_type  numComponents,
        const size_type *cellLocalIdsStartPtr,
        const BasisManager<ValueType,
                           dftefe::utils::MemorySpace::DEVICE>::SizeTypeVector
          &numCellDofs,
        dftefe::utils::MemoryStorage<ValueType,
                                     dftefe::utils::MemorySpace::DEVICE>
          &cellWiseStorage);

      static void
      addCellWiseDataToFieldData(
        const dftefe::utils::MemoryStorage<ValueType,
                                           dftefe::utils::MemorySpace::DEVICE>
          &              cellWiseStorage,
        const size_type  numComponents,
        const size_type *cellLocalIdsStartPtr,
        const BasisManager<ValueType,
                           dftefe::utils::MemorySpace::DEVICE>::SizeTypeVector
          &        numCellDofs,
        ValueType *data);

      static void
      addCellWiseBasisDataToDiagonalData(
        const ValueType *cellWiseBasisData,
        const size_type *cellLocalIdsStartPtr,
        const utils::MemoryStorage<size_type, memorySpace> &numCellDofs,
        ValueType *                                         data);

      static void
      reshapeCellWiseData(const ValueType *cellWiseData,
                          const size_type  numVec,
                          const size_type  vecSize,
                          const size_type  numCells,
                          ValueType *      data);


    }; // end of class FECellWiseDataOperations
#endif
  } // end of namespace basis
} // end of namespace dftefe
#include <basis/FECellWiseDataOperations.t.cpp>
#endif // dftefeFECellWiseDataOperations_h
