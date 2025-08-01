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
#include <linearAlgebra/Defaults.h>
#include <limits.h>
namespace dftefe
{
  namespace linearAlgebra
  {
    /**
     * @brief Setting all the LinearSolverDefaults
     */
    const double LinearSolverDefaults::ABS_TOL        = 1e-12;
    const double LinearSolverDefaults::REL_TOL        = 1e-12;
    const double LinearSolverDefaults::DIVERGENCE_TOL = 1e6;

    /**
     * @brief Setting all the MultiPassLowdinDefaults
     */
    const size_type MultiPassLowdinDefaults::MAX_PASS     = 50;
    const double    MultiPassLowdinDefaults::SHIFT_TOL    = 1e-12;
    const double    MultiPassLowdinDefaults::IDENTITY_TOL = 1e-12;

    /**
     * @brief Setting all the PrintControlDefaults
     */

    const size_type         PrintControlDefaults::WALL_TIME_FREQ        = 0;
    const size_type         PrintControlDefaults::PRINT_FREQ            = 0;
    const bool              PrintControlDefaults::PRINT_FINAL           = true;
    const bool              PrintControlDefaults::PRINT_TOTAL_WALL_TIME = false;
    const ParallelPrintType PrintControlDefaults::PARALLEL_PRINT_TYPE =
      ParallelPrintType::ALL;
    const size_type   PrintControlDefaults::PRECISION = 15;
    const std::string PrintControlDefaults::DELIMITER = "\t";

    /**
     * @brief Setting all the LinAlgOpContextDefaults
     */
    std::shared_ptr<linearAlgebra::LinAlgOpContext<utils::MemorySpace::HOST>>
      LINALG_OP_CONTXT_HOST = std::make_shared<
        linearAlgebra::LinAlgOpContext<utils::MemorySpace::HOST>>(
        std::make_shared<
          linearAlgebra::blasLapack::BlasQueue<utils::MemorySpace::HOST>>(0),
        std::make_shared<
          linearAlgebra::blasLapack::LapackQueue<utils::MemorySpace::HOST>>(0));

    /**
     * @brief Setting all the RayleighRitzRDefaults
     */
    const size_type RayleighRitzDefaults::SUBSPACE_ROT_DOF_BATCH = 20000;
    const size_type RayleighRitzDefaults::WAVE_FN_BATCH          = 2000;

  } // end of namespace linearAlgebra
} // end of namespace dftefe
