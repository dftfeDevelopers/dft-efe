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
#include <ksdft/Defaults.h>
#include <limits.h>
namespace dftefe
{
  namespace ksdft
  {
    /**
     * @brief Setting all the PoissonProblemDefaults
     */
    const linearAlgebra::PreconditionerType PoissonProblemDefaults::PC_TYPE =
      linearAlgebra::PreconditionerType::JACOBI;
    const size_type PoissonProblemDefaults::MAX_CELL_TIMES_NUMVECS = 50;
    const size_type PoissonProblemDefaults::MAX_ITER               = 1e8;
    const double    PoissonProblemDefaults::ABSOLUTE_TOL           = 1e-13;
    const double    PoissonProblemDefaults::RELATIVE_TOL           = 1e-14;
    const double    PoissonProblemDefaults::DIVERGENCE_TOL         = 1e6;

    /**
     * @brief Setting all the LinearEigenSolverDefaults
     */
    const double LinearEigenSolverDefaults::ILL_COND_TOL = 1e-14;
    const double LinearEigenSolverDefaults::LANCZOS_EXTREME_EIGENVAL_TOL = 1e-6;

  } // end of namespace ksdft
} // end of namespace dftefe
