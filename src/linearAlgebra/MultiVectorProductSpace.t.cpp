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
 * @author Avirup Sircar
 */

#include <utils/Exceptions.h>

namespace dftefe
{
  namespace linearAlgebra
  {
    template <typename ValueType, utils::MemorySpace memorySpace>
    MultiVectorProductSpace<ValueType, memorySpace>::MultiVectorProductSpace(
      std::shared_ptr<const utils::mpi::MPIPatternP2P<memorySpace>>
                                                     mpiPatternP2P,
      std::shared_ptr<LinAlgOpContext<memorySpace>>  linAlgOpContext,
      size_type                                      numSpaces,
      size_type                                      numVectorsPerSpace,
      ValueType                                      initVal)
      : MultiVector<ValueType, memorySpace>(mpiPatternP2P,
                                           linAlgOpContext,
                                           numSpaces * numVectorsPerSpace,
                                           initVal)
      , d_numSpaces(numSpaces)
      , d_numVectorsPerSpace(numVectorsPerSpace)
    {
      utils::throwException(numSpaces >= 1,
                            "MultiVectorProductSpace: numSpaces must be >= 1.");
      utils::throwException(
        numVectorsPerSpace >= 1,
        "MultiVectorProductSpace: numVectorsPerSpace must be >= 1.");
    }

    template <typename ValueType, utils::MemorySpace memorySpace>
    size_type
    MultiVectorProductSpace<ValueType, memorySpace>::numSpaces() const
    {
      return d_numSpaces;
    }

    template <typename ValueType, utils::MemorySpace memorySpace>
    size_type
    MultiVectorProductSpace<ValueType, memorySpace>::numVectorsPerSpace() const
    {
      return d_numVectorsPerSpace;
    }

  } // namespace linearAlgebra
} // namespace dftefe
