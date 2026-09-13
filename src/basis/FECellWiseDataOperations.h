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
#include <basis/BasisManager.h>
#include <basis/BasisDataStorage.h>
#include <linearAlgebra/LinAlgOpContext.h>
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
        // const typename BasisManager<ValueType, memorySpace>::SizeTypeVector
        //   &                                                   numCellDofs,
        const size_type                                       totalCellDofs,
        dftefe::utils::MemoryStorage<ValueType, memorySpace> &cellWiseStorage,
        linearAlgebra::LinAlgOpContext<memorySpace> &         linAlgOpContext);

      static void
      copyFieldToCellWiseData(
        const ValueType *data,
        const size_type  numComponents,
        const size_type *cellLocalIdsStartPtr,
        // const typename BasisManager<ValueType, memorySpace>::SizeTypeVector
        //   &        numCellDofs,
        const size_type                              totalCellDofs,
        ValueType *                                  itCellWiseStorageBegin,
        linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext);

      // TODO: Add numStrideCellWiseStorageDofs (max of numCellDofs over all
      // cells) This also takes the case where numCellDofs = 0
      static void
      addCellWiseDataToFieldData(
        const dftefe::utils::MemoryStorage<ValueType, memorySpace>
          &              cellWiseStorage,
        const size_type  numComponents,
        const size_type *cellLocalIdsStartPtr,
        // const typename BasisManager<ValueType, memorySpace>::SizeTypeVector
        //   &        numCellDofs,
        const size_type                              totalCellDofs,
        ValueType *                                  data,
        linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext);

      static void
      addCellWiseDataToFieldData(
        const ValueType *itCellWiseStorageBegin,
        const size_type  numComponents,
        const size_type *cellLocalIdsStartPtr,
        // const typename BasisManager<ValueType, memorySpace>::SizeTypeVector
        //   &        numCellDofs,
        const size_type                              totalCellDofs,
        ValueType *                                  data,
        linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext);

      static void
      addCellWiseBasisDataToDiagonalData(
        const ValueType *cellWiseBasisData,
        const size_type *cellLocalIdsStartPtr,
        const utils::MemoryStorage<size_type, memorySpace> &numCellDofs,
        const size_type                                     totalCellDofs,
        ValueType *                                         data,
        linearAlgebra::LinAlgOpContext<memorySpace> &       linAlgOpContext);

      static void
      reshapeCellWiseData(
        const dftefe::utils::MemoryStorage<ValueType, memorySpace>
          &                                                 cellWiseStorage,
        const size_type                                     numComponents,
        const utils::MemoryStorage<size_type, memorySpace> &numCellVecs,
        ValueType *                                         data,
        linearAlgebra::LinAlgOpContext<memorySpace> &       linAlgOpContext);


    }; // end of class FECellWiseDataOperations


#ifdef DFTEFE_WITH_DEVICE
    template <typename ValueType>
    class FECellWiseDataOperations<ValueType,
                                   dftefe::utils::MemorySpace::DEVICE>
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
        const size_type  totalCellDofs,
        dftefe::utils::MemoryStorage<ValueType,
                                     dftefe::utils::MemorySpace::DEVICE>
          &cellWiseStorage,
        linearAlgebra::LinAlgOpContext<dftefe::utils::MemorySpace::DEVICE>
          &linAlgOpContext);

      static void
      copyFieldToCellWiseData(
        const ValueType *data,
        const size_type  numComponents,
        const size_type *cellLocalIdsStartPtr,
        const size_type  totalCellDofs,
        ValueType *      itCellWiseStorageBegin,
        linearAlgebra::LinAlgOpContext<dftefe::utils::MemorySpace::DEVICE>
          &linAlgOpContext);

      static void
      addCellWiseDataToFieldData(
        const ValueType *itCellWiseStorageBegin,
        const size_type  numComponents,
        const size_type *cellLocalIdsStartPtr,
        const size_type  totalCellDofs,
        ValueType *      data,
        linearAlgebra::LinAlgOpContext<dftefe::utils::MemorySpace::DEVICE>
          &linAlgOpContext);

      // TODO: Add numStrideCellWiseStorageDofs (max of numCellDofs over all
      // cells) This also takes the case where numCellDofs = 0
      static void
      addCellWiseDataToFieldData(
        const dftefe::utils::MemoryStorage<ValueType,
                                           dftefe::utils::MemorySpace::DEVICE>
          &              cellWiseStorage,
        const size_type  numComponents,
        const size_type *cellLocalIdsStartPtr,
        const size_type  totalCellDofs,
        ValueType *      data,
        linearAlgebra::LinAlgOpContext<dftefe::utils::MemorySpace::DEVICE>
          &linAlgOpContext);

      static void
      addCellWiseBasisDataToDiagonalData(
        const ValueType *cellWiseBasisData,
        const size_type *cellLocalIdsStartPtr,
        const utils::MemoryStorage<size_type,
                                   dftefe::utils::MemorySpace::DEVICE>
          &             numCellDofs,
        const size_type totalCellDofs,
        ValueType *     data,
        linearAlgebra::LinAlgOpContext<dftefe::utils::MemorySpace::DEVICE>
          &linAlgOpContext);

      static void
      reshapeCellWiseData(
        const dftefe::utils::MemoryStorage<ValueType,
                                           utils::MemorySpace::DEVICE>
          &             cellWiseStorage,
        const size_type numComponents,
        const utils::MemoryStorage<size_type, utils::MemorySpace::DEVICE>
          &        numCellVecs,
        ValueType *data,
        linearAlgebra::LinAlgOpContext<dftefe::utils::MemorySpace::DEVICE>
          &linAlgOpContext);


    }; // end of class FECellWiseDataOperations
#endif
  } // end of namespace basis
} // end of namespace dftefe
#endif // dftefeFECellWiseDataOperations_h
