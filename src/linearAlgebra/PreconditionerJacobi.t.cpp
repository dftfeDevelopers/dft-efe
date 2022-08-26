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
 * @author Bikash Kanungo
 */


#include <utils/MemoryTransfer.h>
#include <BlasLapack.h>
namespace dftefe
{
  namespace linearAlgebra
  {
    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    template <utils::MemorySpace memorySpaceSrc>
    PreconditionerJacobi<ValueTypeOperator, ValueTypeOperand, memorySpace>::
      PreconditionerJacobi(
        const utils::MemoryStorage<ValueTypeOperator, memoryStorageSrc>
          &diagonal,
	  LinAlgOpContext<memorySpace> & linAlgContext)
      : d_digonalInv(digonal.size())
    {
      const size_type                              N = diagonal.size();
      if (memorySpaceSrc != memorySpace)
        {
          utils::MemoryStorage<ValueType, memorySpace> diagonalCopy(N);
          diagonalCopy.copyFrom(diagonal);
	  blasLapack::reciprocalX(N, 1.0, diagonalCopy.data(), d_digonalInv.data(), linAlgContext); 
        }
      else
        {
	  blasLapack::reciprocalX(N, 1.0, diagonal.data(), d_digonalInv.data(), linAlgContext); 
        }
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    template <utils::MemorySpace memorySpaceSrc>
    PreconditionerJacobi<ValueTypeOperator, ValueTypeOperand, memorySpace>::
      PreconditionerJacobi(const ValueTypeOperator *diagonal, const size_type N,
	  LinAlgOpContext<memorySpace> & linAlgContext)
      : d_digonalInv(N)
    {
      if (memorySpaceSrc != memorySpace)
        {
          utils::MemoryStorage<ValueType, memorySpace> diagonalCopy(N);
          diagonalCopy.copyFrom(diagonal);
	  blasLapack::reciprocalX(N, 1.0, diagonalCopy.data(), d_digonalInv.data(), linAlgContext); 
        }
      else
        {
	  blasLapack::reciprocalX(N, 1.0, diagonal.data(), d_digonalInv.data(), linAlgContext); 
        }
    }

  } // end of namespace linearAlgebra
} // end of namespace dftefe
