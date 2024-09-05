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
    const size_type PoissonProblemDefaults::CELL_BATCH_SIZE = 50;
    const size_type PoissonProblemDefaults::MAX_ITER        = 2e7;
    const double    PoissonProblemDefaults::ABSOLUTE_TOL    = 1e-12;
    const double    PoissonProblemDefaults::RELATIVE_TOL    = 1e-14;
    const double    PoissonProblemDefaults::DIVERGENCE_TOL  = 1e10;

    /**
     * @brief Setting all the LinearEigenSolverDefaults
     */
    const double LinearEigenSolverDefaults::ILL_COND_TOL = 1e-14;
    const double LinearEigenSolverDefaults::LANCZOS_EXTREME_EIGENVAL_TOL = 1e-6;
    const double LinearEigenSolverDefaults::LANCZOS_BETA_TOL = 1e-14;
    const size_type LinearEigenSolverDefaults::LANCZOS_MAX_KRYLOV_SUBSPACE = 40;
    /**
     * @brief Setting all the ChebyshevPolynomialDegreeDefaults
     */
    const std::map<size_type, size_type>
      LinearEigenSolverDefaults::CHEBY_ORDER_LOOKUP{
        {500, 24}, // <= 500 ~> chebyshevOrder = 24
        {750, 30},
        {1000, 39},
        {1500, 50},
        {2000, 53},
        {3000, 57},
        {4000, 62},
        {5000, 69},
        {9000, 77},
        {14000, 104},
        {20000, 119},
        {30000, 162},
        {50000, 300},
        {80000, 450},
        {100000, 550},
        {200000, 700},
        {500000, 1000}};

    /**
     * @brief Setting all the NewtonRaphsonSolverDefaults
     */
    const size_type NewtonRaphsonSolverDefaults::MAX_ITER  = 2e7;
    const double    NewtonRaphsonSolverDefaults::FORCE_TOL = 1e-14;

    /**
     * @brief Setting all the constants
     */
    const double Constants::BOLTZMANN_CONST_HARTREE = 3.166811429e-06;
    const double Constants::LDA_EXCHANGE_ENERGY_CONST =
      (-3.0 / 4) * std::pow((3 / utils::mathConstants::pi), (1.0 / 3));

    /**
     * @brief Setting all the LibxcDefaults
     */
    const double LibxcDefaults::DENSITY_ZERO_TOL = 1e-10;

    /**
     * @brief Setting all the KSDFTDefaults
     */
    const size_type KSDFTDefaults::MAX_WAVEFN_BATCH_SIZE = 400;
    const size_type KSDFTDefaults::CELL_BATCH_SIZE       = 50;

  } // end of namespace ksdft
} // end of namespace dftefe
