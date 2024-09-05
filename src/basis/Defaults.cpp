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
    const linearAlgebra::PreconditionerType L2ProjectionDefaults::PC_TYPE =
      linearAlgebra::PreconditionerType::JACOBI;
    const size_type L2ProjectionDefaults::CELL_BATCH_SIZE      = 50;
    const size_type L2ProjectionDefaults::MAX_ITER             = 1e8;
    const double    L2ProjectionDefaults::ABSOLUTE_TOL         = 1e-13;
    const double    L2ProjectionDefaults::RELATIVE_TOL         = 1e-14;
    const double    L2ProjectionDefaults::DIVERGENCE_TOL       = 1e6;
    const size_type GenerateMeshDefaults::MAX_REFINEMENT_STEPS = 40;
  } // end of namespace basis
} // end of namespace dftefe
