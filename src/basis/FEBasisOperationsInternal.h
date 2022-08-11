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

#ifndef dftefeFEBasisOperationsInternal_h
#define dftefeFEBasisOperationsInternal_h

#include <utils/TypeConfig.h>
#include <utils/MemorySpaceType.h>
#include <utils/MemoryStorage.h>
#include <basis/Field.h>
#include <basis/BasisDataStorage.h>
namespace dftefe
{
  namespace basis
  {
    template <typename ValueTypeBasisCoeff, utils::MemorySpace memorySpace>
    class FEBasisOperationsInternal
    {
    public:
      static void
      copyFieldToCellWiseData(
        const ValueTypeBasisCoeff *data,
        const size_type            numComponents,
        const size_type *          cellLocalIdsStartPtr,
        const BasisHandler<ValueTypeBasisCoeff, memorySpace>::SizeTypeVector
          &numCellDofs,
        dftefe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace>
          &cellWiseStorage);

      static void
      addCellWiseDataToFieldData(
        const dftefe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace>
          &              cellWiseStorage,
        const size_type  numComponents,
        const size_type *cellLocalIdsStartPtr,
        const BasisHandler<ValueTypeBasisCoeff, memorySpace>::SizeTypeVector
          &                  numCellDofs,
        ValueTypeBasisCoeff *data);


    }; // end of class FEBasisOperationsInternal


#ifdef DFTEFE_WITH_DEVICE
    template <typename ValueTypeBasisCoeff>
    class FEBasisOperationsInternal
    {
    public:
      static void
      copyFieldToCellWiseData(
        const ValueTypeBasisCoeff *data,
        const size_type            numComponents,
        const size_type *          cellLocalIdsStartPtr,
        const BasisHandler<ValueTypeBasisCoeff,
                           dftefe::utils::MemorySpace::DEVICE>::SizeTypeVector
          &numCellDofs,
        dftefe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                     dftefe::utils::MemorySpace::DEVICE>
          &cellWiseStorage);

      static void
      addCellWiseDataToFieldData(
        const dftefe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                           dftefe::utils::MemorySpace::DEVICE>
          &              cellWiseStorage,
        const size_type  numComponents,
        const size_type *cellLocalIdsStartPtr,
        const BasisHandler<ValueTypeBasisCoeff,
                           dftefe::utils::MemorySpace::DEVICE>::SizeTypeVector
          &                  numCellDofs,
        ValueTypeBasisCoeff *data);


    }; // end of class FEBasisOperationsInternal
#endif
  } // end of namespace basis
} // end of namespace dftefe
#include <basis/FEBasisOperationsInternal.t.cpp>
#endif // dftefeFEBasisOperationsInternal_h
