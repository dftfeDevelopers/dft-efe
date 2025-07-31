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
 * @author Bikash Kanungo , Avirup Sircar
 */

#ifndef dftefeLinearAlgebraDefaults_h
#define dftefeLinearAlgebraDefaults_h

#include <utils/TypeConfig.h>
#include <linearAlgebra/LinearAlgebraTypes.h>
#include <string>
#include <memory>
#include "LinAlgOpContext.h"
#include <utils/MemoryStorage.h>

namespace dftefe
{
  namespace linearAlgebra
  {
    class PrintControlDefaults
    {
    public:
      //
      // frequency at which to measure wall time of each iteration
      // of the linear solve
      //
      static const size_type WALL_TIME_FREQ;

      //
      // frequency at which to print residual info
      //
      static const size_type PRINT_FREQ;

      //
      // print final flag. It sets whether to print the final info
      // (e.g., number of iterations to convergence, ErrorCode, etc)
      //
      static const bool PRINT_FINAL;

      //
      // print total solver wall wall time flag
      //
      static const bool PRINT_TOTAL_WALL_TIME;

      //
      // parallel print type
      // (e.g., print on all ranks, or just root rank, or all, etc)
      //
      static const ParallelPrintType PARALLEL_PRINT_TYPE;

      //
      // floating point precision to which one should print
      //
      static const size_type PRECISION;

      //
      // delimiter
      //
      static const std::string DELIMITER;

    }; // end of class PrintControlDefaults

    class LinearSolverDefaults
    {
    public:
      //
      // absolute tolerance on the residual
      //
      static const double ABS_TOL;

      //
      // relative tolerance on the residual (i.e. residual norm divided by
      // norm of the right hand side vector)
      //
      static const double REL_TOL;

      //
      // Tolerance to abort the linear solver if the
      // L2 norm of the residual exceeds it
      static const double DIVERGENCE_TOL;
    }; // end of class LinearSolverDefaults

    class MultiPassLowdinDefaults
    {
    public:
      //
      // maximum pass of lowdin orthonormalizations
      //
      static const size_type MAX_PASS;

      //
      // Min tolerance of min eigenvalue below which to shift the eigenvalue by
      // this tolerance.
      //
      static const double SHIFT_TOL;

      //
      // calculate frobenus norm |S - I| < this tolerance
      static const double IDENTITY_TOL;
    }; // end of class MultiPassLowdinDefaults

    class LinAlgOpContextDefaults
    {
    public:
      static const std::shared_ptr<
        linearAlgebra::LinAlgOpContext<utils::MemorySpace::HOST>>
        LINALG_OP_CONTXT_HOST;
    }; // end of class LinAlgOpContextHost

    class RayleighRitzDefaults
    {
    public:
      static const size_type SUBSPACE_ROT_DOF_BATCH;
    }; // end of class LinAlgOpContextHost

  } // end of namespace linearAlgebra
} // end of namespace dftefe
#endif // dftefeLinearAlgebraDefaults_h
