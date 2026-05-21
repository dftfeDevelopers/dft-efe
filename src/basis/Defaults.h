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

#ifndef dftefeBasisDefaults_h
#define dftefeBasisDefaults_h

#include <utils/TypeConfig.h>
#include <utils/MemorySpaceType.h>
#include <string>
#include <linearAlgebra/LinearAlgebraTypes.h>

namespace dftefe
{
  namespace basis
  {
    template <utils::MemorySpace memorySpace>
    class L2ProjectionDefaults
    {
    public:
      //
      // The CG Preconditioner Type chosen for L2 Projection
      //
      static const linearAlgebra::PreconditionerType PC_TYPE;

      //
      // The number of batched gemms done: 1 for HOST, 50 for DEVICE
      //
      static const size_type CELL_BATCH_SIZE;

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

    }; // end of class L2ProjectionDefaults

    class GenerateMeshDefaults
    {
    public:
      //
      // The maximum refinement steps used for adaptive mesh generation
      // algorithm.
      //
      static const size_type MAX_REFINEMENT_STEPS;

    }; // end of class GenerateMeshDefaults

    class ECIDefaults
    {
    public:
      //
      // The maximum refinement steps used for adaptive mesh generation
      // algorithm.
      //
      static const size_type ENRICHMENT_BATCH_SIZE;
      static const double ENRICHMENT_ORTHO_COEFF_TOL;

    }; // end of class ECIDefaults

    template <utils::MemorySpace memorySpace>
    class BasisDataStorageDefaults
    {
    public:
      //
      // The number of batched gemms done: 1 for HOST, 50 for DEVICE
      //
      static const size_type CELL_BATCH_SIZE;

    }; // end of class BasisDataStorageDefaults

    class MaxSizeDefaults
    {
    public:
      static const size_type SIZE_TYPE_MAX;
      static const global_size_type GLOBAL_SIZE_TYPE_MAX;
    }; // end of class MaxSizeDefaults
  } // end of namespace basis
} // end of namespace dftefe
#endif // dftefebasisDefaults_h
