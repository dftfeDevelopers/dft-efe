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


#ifdef DFTEFE_WITH_DEVICE

#  include <utils/DeviceKernelLauncherHelpers.h>
#  include <utils/DeviceDataTypeOverloads.h>
#  include <utils/DeviceTypeConfigHalfPrec.h>
#  include <utils/Exceptions.h>
#  include <complex>
#  include <algorithm>
#  include "FECellWiseDataOperations.h"

namespace dftefe
{
  namespace basis
  {
    namespace
    {
      template <typename ValueType>
      DFTEFE_CREATE_KERNEL(
        void,
        copyFieldToCellWiseDataDeviceKernel,
        {
          size_type totalEntries = totalCellDofs * numComponents;

          for (size_type index = globalThreadId; index < totalEntries;
               index += nThreadsPerBlock * nThreadBlock)
            {
              size_type dofId     = index / numComponents;
              size_type component = index - dofId * numComponents;

              size_type localId = cellLocalIdsStartPtr[dofId];

              dftefe::utils::copyValue(
                itCellWiseStorageBegin + index,
                data[localId * numComponents + component]);
            }
        },
        const ValueType *data,
        const size_type  numComponents,
        const size_type *cellLocalIdsStartPtr,
        size_type        totalCellDofs,
        ValueType *      itCellWiseStorageBegin);

      template <typename ValueType>
      DFTEFE_CREATE_KERNEL(
        void,
        addCellWiseDataToFieldDataDeviceKernel,
        {
          size_type totalEntries = totalCellDofs * numComponents;
          for (size_type index = globalThreadId; index < totalEntries;
               index += nThreadsPerBlock * nThreadBlock)
            {
              size_type dofId     = index / numComponents;
              size_type component = index - dofId * numComponents;

              size_type localId = cellLocalIdsStartPtr[dofId];

              dftefe::utils::atomicAddWrapper(data + localId * numComponents +
                                                component,
                                              itCellWiseStorageBegin[index]);
            }
        },
        const ValueType *itCellWiseStorageBegin,
        const size_type  numComponents,
        const size_type *cellLocalIdsStartPtr,
        size_type        totalCellDofs,
        ValueType *      data);

      template <>
      DFTEFE_CREATE_KERNEL(
        void,
        addCellWiseDataToFieldDataDeviceKernel,
        {
          size_type totalEntries = totalCellDofs * numComponents;
          for (size_type index = globalThreadId; index < totalEntries;
               index += nThreadsPerBlock * nThreadBlock)
            {
              size_type dofId     = index / numComponents;
              size_type component = index - dofId * numComponents;

              size_type localId = cellLocalIdsStartPtr[dofId];

              auto *add_real = reinterpret_cast<float *>(
                data + localId * numComponents + component);
              auto *add_imag = add_real + 1;

              dftefe::utils::atomicAddWrapper(add_real,
                                              dftefe::utils::realPartDevice(
                                                itCellWiseStorageBegin[index]));
              dftefe::utils::atomicAddWrapper(add_imag,
                                              dftefe::utils::imagPartDevice(
                                                itCellWiseStorageBegin[index]));
            }
        },
        const dftefe::utils::deviceFloatComplex *itCellWiseStorageBegin,
        const size_type                          numComponents,
        const size_type *                        cellLocalIdsStartPtr,
        size_type                                totalCellDofs,
        dftefe::utils::deviceFloatComplex *      data);

      template <>
      DFTEFE_CREATE_KERNEL(
        void,
        addCellWiseDataToFieldDataDeviceKernel,
        {
          size_type totalEntries = totalCellDofs * numComponents;
          for (size_type index = globalThreadId; index < totalEntries;
               index += nThreadsPerBlock * nThreadBlock)
            {
              size_type dofId     = index / numComponents;
              size_type component = index - dofId * numComponents;

              size_type localId = cellLocalIdsStartPtr[dofId];

              auto *add_real = reinterpret_cast<double *>(
                data + localId * numComponents + component);
              auto *add_imag = add_real + 1;

              dftefe::utils::atomicAddWrapper(add_real,
                                              dftefe::utils::realPartDevice(
                                                itCellWiseStorageBegin[index]));
              dftefe::utils::atomicAddWrapper(add_imag,
                                              dftefe::utils::imagPartDevice(
                                                itCellWiseStorageBegin[index]));
            }
        },
        const dftefe::utils::deviceDoubleComplex *itCellWiseStorageBegin,
        const size_type                           numComponents,
        const size_type *                         cellLocalIdsStartPtr,
        size_type                                 totalCellDofs,
        dftefe::utils::deviceDoubleComplex *      data);

      template <typename ValueType>
      DFTEFE_CREATE_KERNEL(
        void,
        addCellWiseBasisDataToDiagonalDataDeviceKernel,
        {
          for (size_type index = globalThreadId; index < totalCellDofs;
               index += nThreadsPerBlock * nThreadBlock)
            {
              size_type cumulativeDofs       = 0;
              size_type cumulativeDofsSquare = 0;
              size_type iCell                = 0;
              for (; iCell < numCells; ++iCell)
                {
                  size_type cellDofs = numCellDofs[iCell];
                  if (index < cumulativeDofs + cellDofs)
                    break;
                  cumulativeDofs += cellDofs;
                  cumulativeDofsSquare += cellDofs * cellDofs;
                }
              size_type cellDofs = numCellDofs[iCell];
              size_type iDof     = index - cumulativeDofs;
              size_type diagIndex =
                cumulativeDofsSquare + iDof * cellDofs + iDof;
              size_type localId = cellLocalIdsStartPtr[index];
              dftefe::utils::atomicAddWrapper(data + localId, cellWiseBasisDataBegin[diagIndex]);
            }
        },
        const ValueType *cellWiseBasisDataBegin,
        const size_type *cellLocalIdsStartPtr,
        const size_type *numCellDofs,
        size_type        numCells,
        size_type        totalCellDofs,
        ValueType *      data);

      template <>
      DFTEFE_CREATE_KERNEL(
        void,
        addCellWiseBasisDataToDiagonalDataDeviceKernel,
        {
          for (size_type index = globalThreadId; index < totalCellDofs;
               index += nThreadsPerBlock * nThreadBlock)
            {
              size_type cumulativeDofs       = 0;
              size_type cumulativeDofsSquare = 0;
              size_type iCell                = 0;
              for (; iCell < numCells; ++iCell)
                {
                  size_type cellDofs = numCellDofs[iCell];
                  if (index < cumulativeDofs + cellDofs)
                    break;
                  cumulativeDofs += cellDofs;
                  cumulativeDofsSquare += cellDofs * cellDofs;
                }
              size_type cellDofs = numCellDofs[iCell];
              size_type iDof     = index - cumulativeDofs;
              size_type diagIndex =
                cumulativeDofsSquare + iDof * cellDofs + iDof;
              size_type localId  = cellLocalIdsStartPtr[index];
              auto *    add_real = reinterpret_cast<float *>(data + localId);
              auto *    add_imag = add_real + 1;

              dftefe::utils::atomicAddWrapper(
                add_real,
                dftefe::utils::realPartDevice(
                  cellWiseBasisDataBegin[diagIndex]));
              dftefe::utils::atomicAddWrapper(
                add_imag,
                dftefe::utils::imagPartDevice(
                  cellWiseBasisDataBegin[diagIndex]));
            }
        },
        const dftefe::utils::deviceFloatComplex *cellWiseBasisDataBegin,
        const size_type *                        cellLocalIdsStartPtr,
        const size_type *                        numCellDofs,
        size_type                                numCells,
        size_type                                totalCellDofs,
        dftefe::utils::deviceFloatComplex *      data);

      template <>
      DFTEFE_CREATE_KERNEL(
        void,
        addCellWiseBasisDataToDiagonalDataDeviceKernel,
        {
          for (size_type index = globalThreadId; index < totalCellDofs;
               index += nThreadsPerBlock * nThreadBlock)
            {
              size_type cumulativeDofs       = 0;
              size_type cumulativeDofsSquare = 0;
              size_type iCell                = 0;
              for (; iCell < numCells; ++iCell)
                {
                  size_type cellDofs = numCellDofs[iCell];
                  if (index < cumulativeDofs + cellDofs)
                    break;
                  cumulativeDofs += cellDofs;
                  cumulativeDofsSquare += cellDofs * cellDofs;
                }
              size_type cellDofs = numCellDofs[iCell];
              size_type iDof     = index - cumulativeDofs;
              size_type diagIndex =
                cumulativeDofsSquare + iDof * cellDofs + iDof;
              size_type localId  = cellLocalIdsStartPtr[index];
              auto *    add_real = reinterpret_cast<double *>(data + localId);
              auto *    add_imag = add_real + 1;

              dftefe::utils::atomicAddWrapper(
                add_real,
                dftefe::utils::realPartDevice(
                  cellWiseBasisDataBegin[diagIndex]));
              dftefe::utils::atomicAddWrapper(
                add_imag,
                dftefe::utils::imagPartDevice(
                  cellWiseBasisDataBegin[diagIndex]));
            }
        },
        const dftefe::utils::deviceDoubleComplex *cellWiseBasisDataBegin,
        const size_type *                         cellLocalIdsStartPtr,
        const size_type *                         numCellDofs,
        size_type                                 numCells,
        size_type                                 totalCellDofs,
        dftefe::utils::deviceDoubleComplex *      data);

    } // namespace

    template <typename ValueType>
    void
    FECellWiseDataOperations<ValueType, utils::MemorySpace::DEVICE>::
      copyFieldToCellWiseData(
        const ValueType *data,
        const size_type  numComponents,
        const size_type *cellLocalIdsStartPtr,
        const size_type totalCellDofs,
        ValueType *itCellWiseStorageBegin)
    {
      DFTEFE_LAUNCH_KERNEL(
        copyFieldToCellWiseDataDeviceKernel,
        (totalCellDofs * numComponents) / dftefe::utils::DEVICE_BLOCK_SIZE + 1,
        dftefe::utils::DEVICE_BLOCK_SIZE,
        dftefe::utils::defaultStream,
        dftefe::utils::makeDataTypeDeviceCompatible(data),
        numComponents,
        cellLocalIdsStartPtr,
        totalCellDofs,
        dftefe::utils::makeDataTypeDeviceCompatible(itCellWiseStorageBegin));
    }

    template <typename ValueType>
    void
    FECellWiseDataOperations<ValueType, utils::MemorySpace::DEVICE>::
      addCellWiseDataToFieldData(
        const ValueType *itCellWiseStorageBegin,
        const size_type  numComponents,
        const size_type *cellLocalIdsStartPtr,
        const size_type totalCellDofs,
        ValueType *data)
    {
      DFTEFE_LAUNCH_KERNEL(
        addCellWiseDataToFieldDataDeviceKernel,
        (totalCellDofs * numComponents) / dftefe::utils::DEVICE_BLOCK_SIZE + 1,
        dftefe::utils::DEVICE_BLOCK_SIZE,
        dftefe::utils::defaultStream,
        dftefe::utils::makeDataTypeDeviceCompatible(itCellWiseStorageBegin),
        numComponents,
        cellLocalIdsStartPtr,
        totalCellDofs,
        dftefe::utils::makeDataTypeDeviceCompatible(data));
    }


    template <typename ValueType>
    void
    FECellWiseDataOperations<ValueType, utils::MemorySpace::DEVICE>::
      copyFieldToCellWiseData(
        const ValueType *data,
        const size_type  numComponents,
        const size_type *cellLocalIdsStartPtr,
        const size_type totalCellDofs,
        utils::MemoryStorage<ValueType, utils::MemorySpace::DEVICE>
          &cellWiseStorage)
    {
      auto itCellWiseStorageBegin = cellWiseStorage.begin();
      copyFieldToCellWiseData(data,
                              numComponents,
                              cellLocalIdsStartPtr,
                              totalCellDofs,
                              itCellWiseStorageBegin);
    }

    template <typename ValueType>
    void
    FECellWiseDataOperations<ValueType, utils::MemorySpace::DEVICE>::
      addCellWiseDataToFieldData(
        const utils::MemoryStorage<ValueType, utils::MemorySpace::DEVICE>
          &              cellWiseStorage,
        const size_type  numComponents,
        const size_type *cellLocalIdsStartPtr,
        const size_type totalCellDofs,
        ValueType *data)
    {
      auto itCellWiseStorageBegin = cellWiseStorage.begin();
      addCellWiseDataToFieldData(itCellWiseStorageBegin,
                                 numComponents,
                                 cellLocalIdsStartPtr,
                                 totalCellDofs,
                                 data);
    }


    template <typename ValueType>
    void
    FECellWiseDataOperations<ValueType, utils::MemorySpace::DEVICE>::
      addCellWiseBasisDataToDiagonalData(
        const ValueType *cellWiseBasisDataBegin,
        const size_type *cellLocalIdsStartPtr,
        const utils::MemoryStorage<size_type, utils::MemorySpace::DEVICE>
          &        numCellDofs,
        const size_type totalCellDofs,
        ValueType *data)
    {
      DFTEFE_LAUNCH_KERNEL(
        addCellWiseBasisDataToDiagonalDataDeviceKernel,
        (totalCellDofs) / dftefe::utils::DEVICE_BLOCK_SIZE + 1,
        dftefe::utils::DEVICE_BLOCK_SIZE,
        dftefe::utils::defaultStream,
        dftefe::utils::makeDataTypeDeviceCompatible(cellWiseBasisDataBegin),
        cellLocalIdsStartPtr,
        numCellDofs.begin(),
        numCellDofs.size(),
        totalCellDofs,
        dftefe::utils::makeDataTypeDeviceCompatible(data));
    }

    template <typename ValueType>
    void
    FECellWiseDataOperations<ValueType, utils::MemorySpace::DEVICE>::
      reshapeCellWiseData(
        const dftefe::utils::MemoryStorage<ValueType,
                                           utils::MemorySpace::DEVICE>
          &             cellWiseStorage,
        const size_type numComponents,
        const utils::MemoryStorage<size_type, utils::MemorySpace::DEVICE>
          &        numCellVecs,
        ValueType *data)
    {
      utils::throwException(
        false,
        "reshapeCellWiseData() is not implemented for utils::MemorySpace::DEVICE .... ");
    }

    template class FECellWiseDataOperations<double,
                                            dftefe::utils::MemorySpace::DEVICE>;
    template class FECellWiseDataOperations<float,
                                            dftefe::utils::MemorySpace::DEVICE>;
    template class FECellWiseDataOperations<std::complex<double>,
                                            dftefe::utils::MemorySpace::DEVICE>;
    template class FECellWiseDataOperations<std::complex<float>,
                                            dftefe::utils::MemorySpace::DEVICE>;
  } // namespace basis
} // namespace dftefe
#endif
