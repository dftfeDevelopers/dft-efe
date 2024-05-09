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

#ifndef dftefeDFTDefaults_h
#define dftefeDFTDefaults_h

#include <utils/TypeConfig.h>
#include <string>
#include <linearAlgebra/LinearAlgebraTypes.h>

namespace dftefe
{
  namespace ksdft
  {
    class PoissonProblemDefaults
    {
    public:
      //
      // The CG Preconditioner Type chosen for L2 Projection
      //
      static const linearAlgebra::PreconditionerType PC_TYPE;

      //
      // The number of batched gemms done
      //
      static const size_type MAX_CELL_TIMES_NUMVECS;

      //
      // Maximum iteration for CG
      //
      static const size_type MAX_ITER;

      //
      // Absolute tolerance of the residual |AX-b|
      //
      static const double ABSOLUTE_TOL;

      //
      // Relative tolerance of the residual |AX-b|/|b|
      //
      static const double RELATIVE_TOL;

      //
      // Maximum residual tolerance for divergence
      //
      static const double DIVERGENCE_TOL;

    }; // end of class PoissonProblemDefaults

    class LinearEigenSolverDefaults
    {
    public:
      //
      // Tolerance of ill conditioning for the
      // orthogonalization step in CHFSi
      //
      static const double ILL_COND_TOL;

      //
      // Tolerance for lanczos extreme eigenvalues
      //
      static const double LANCZOS_EXTREME_EIGENVAL_TOL;

    }; // end of class PoissonProblemDefaults

  } // end of namespace ksdft
} // end of namespace dftefe
#endif // dftefeDFTDefaults_h
