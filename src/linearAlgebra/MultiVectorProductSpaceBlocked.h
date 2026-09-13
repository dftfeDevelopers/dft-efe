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

#ifndef dftefeMultiVectorProductSpaceBlocked_h
#define dftefeMultiVectorProductSpaceBlocked_h

#include <utils/MemorySpaceType.h>
#include <linearAlgebra/MultiVectorProductSpace.h>

namespace dftefe
{
  namespace linearAlgebra
  {
    /**
     * @brief Type-tag subclass of MultiVectorProductSpace for collinear spin (S=2).
     *
     * Identical memory layout and interface to MultiVectorProductSpace.
     * The distinct type causes dynamic_cast dispatch in RayleighRitzEigenSolver
     * and OrthonormalizationFunctions to select per-spin-space (blocked) paths
     * instead of the coupled single-ELPA path.
     */
    template <typename ValueType, utils::MemorySpace memorySpace>
    class MultiVectorProductSpaceBlocked
      : public MultiVectorProductSpace<ValueType, memorySpace>
    {
    public:
      using MultiVectorProductSpace<ValueType,
                                    memorySpace>::MultiVectorProductSpace;

      ~MultiVectorProductSpaceBlocked() = default;
    };

  } // namespace linearAlgebra
} // namespace dftefe

#include <linearAlgebra/MultiVectorProductSpaceBlocked.t.cpp>
#endif // dftefeMultiVectorProductSpaceBlocked_h
