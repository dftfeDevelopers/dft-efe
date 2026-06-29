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

#ifndef dftefeMultiVectorProductSpace_h
#define dftefeMultiVectorProductSpace_h

#include <utils/TypeConfig.h>
#include <utils/MemorySpaceType.h>
#include <utils/MPIPatternP2P.h>
#include <linearAlgebra/LinAlgOpContext.h>
#include <linearAlgebra/MultiVector.h>
#include <memory>

namespace dftefe
{
  namespace linearAlgebra
  {
    /**
     * @brief MultiVector for product spaces: flat layout M × S × N
     *        where S = numSpaces, N = numVectorsPerSpace.
     *
     * The base MultiVector sees numVectors = S*N (Nflattened).
     * Flat offset for DOF i, space s, orbital n: i*S*N + s*N + n.
     *
     * Covers unpolarized (S=1) and non-collinear (S=2).
     * For collinear (S=2) use MultiVectorProductSpaceBlocked instead.
     */
    template <typename ValueType, utils::MemorySpace memorySpace>
    class MultiVectorProductSpace : public MultiVector<ValueType, memorySpace>
    {
    public:
      MultiVectorProductSpace(
        std::shared_ptr<const utils::mpi::MPIPatternP2P<memorySpace>>
                                                       mpiPatternP2P,
        std::shared_ptr<LinAlgOpContext<memorySpace>>  linAlgOpContext,
        size_type                                      numSpaces,
        size_type                                      numVectorsPerSpace,
        ValueType initVal = utils::Types<ValueType>::zero);

      ~MultiVectorProductSpace() = default;

      size_type
      numSpaces() const;

      size_type
      numVectorsPerSpace() const;

    private:
      size_type d_numSpaces;
      size_type d_numVectorsPerSpace;
    };

  } // namespace linearAlgebra
} // namespace dftefe

#include <linearAlgebra/MultiVectorProductSpace.t.cpp>
#endif // dftefeMultiVectorProductSpace_h
