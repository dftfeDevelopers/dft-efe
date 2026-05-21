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
#include <basis/Defaults.h>
#include <limits.h>
namespace dftefe
{
  namespace basis
  {
    /**
     * @brief Setting all the L2ProjectionDefaults
     */
    template <utils::MemorySpace memorySpace>
    const linearAlgebra::PreconditionerType
      L2ProjectionDefaults<memorySpace>::PC_TYPE =
        linearAlgebra::PreconditionerType::JACOBI;
    template <utils::MemorySpace memorySpace>
    const size_type L2ProjectionDefaults<memorySpace>::MAX_ITER = 1e8;
    template <utils::MemorySpace memorySpace>
    const double L2ProjectionDefaults<memorySpace>::ABSOLUTE_TOL = 1e-13;
    template <utils::MemorySpace memorySpace>
    const double L2ProjectionDefaults<memorySpace>::RELATIVE_TOL = 1e-14;
    template <utils::MemorySpace memorySpace>
    const double L2ProjectionDefaults<memorySpace>::DIVERGENCE_TOL = 1e6;

    template <>
    const size_type
      L2ProjectionDefaults<utils::MemorySpace::HOST>::CELL_BATCH_SIZE = 1;
    template <>
    const size_type
      L2ProjectionDefaults<utils::MemorySpace::DEVICE>::CELL_BATCH_SIZE = 50;

    template class L2ProjectionDefaults<utils::MemorySpace::HOST>;
    template class L2ProjectionDefaults<utils::MemorySpace::DEVICE>;
    const size_type GenerateMeshDefaults::MAX_REFINEMENT_STEPS = 40;
    const size_type ECIDefaults::ENRICHMENT_BATCH_SIZE         = 400;
    const double ECIDefaults::ENRICHMENT_ORTHO_COEFF_TOL    = 1e-8;
    template <>
    const size_type
      BasisDataStorageDefaults<utils::MemorySpace::HOST>::CELL_BATCH_SIZE = 1;
    template <>
    const size_type
      BasisDataStorageDefaults<utils::MemorySpace::DEVICE>::CELL_BATCH_SIZE =
        50;

    template class BasisDataStorageDefaults<utils::MemorySpace::HOST>;
    template class BasisDataStorageDefaults<utils::MemorySpace::DEVICE>;
    const size_type MaxSizeDefaults::SIZE_TYPE_MAX = std::numeric_limits<size_type>::max();
    const global_size_type MaxSizeDefaults::GLOBAL_SIZE_TYPE_MAX = std::numeric_limits<global_size_type>::max();
  } // end of namespace basis
} // end of namespace dftefe
