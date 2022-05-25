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
namespace dftefe
{
  namespace basis
  {
    template<typename ValueType, utils::MemorySpace memorySpace>
    class FEBasisOperationsInternal
    {
      public:
	static void
	  copyFieldToCellWiseData(const ValueType * data,
	      const size_type numComponents,
	      const size_type * cellLocalIdsStartPtr,
	      const std::vector<size_type> & numCellDofs,
	      MemoryStorage<ValueType, memorySpace> & cellWiseStorage);

    }; // end of class FEBasisOperationsInternal
  } // end of namespace basis
}// end of namespace dftefe
#endif //dftefeFEBasisOperationsInternal_h
