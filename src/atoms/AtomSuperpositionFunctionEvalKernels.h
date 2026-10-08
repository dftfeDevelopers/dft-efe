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

#ifndef dftefeAtomSuperpositionFunctionEvalKernels_h
#define dftefeAtomSuperpositionFunctionEvalKernels_h

#include <utils/TypeConfig.h>
#include <utils/MemorySpaceType.h>
#include <linearAlgebra/LinAlgOpContext.h>
#include <atoms/SphericalData.h>
#include <memory>
#include <vector>

namespace dftefe
{
  namespace atoms
  {
    /**
     * @brief Real valued evaluation of spherical data at arbitrary points.
     * Kept here so atoms need not depend on basis.
     */
    template <utils::MemorySpace memorySpace>
    class AtomSuperpositionFunctionEvalKernels
    {
    public:
      static void
      getEnrichmentValues(
        const size_type               numEnrichmentFunc,
        const std::vector<size_type> &pointsPerEnrichId,
        const std::vector<std::shared_ptr<atoms::SphericalData>>
          &                                          sphericalDataVec,
        const double *                               points,
        const double *                               origin,
        double *                                     values,
        linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext);

      static void
      getEnrichmentGradients(
        const size_type               numEnrichmentFunc,
        const std::vector<size_type> &pointsPerEnrichId,
        const std::vector<std::shared_ptr<atoms::SphericalData>>
          &                                          sphericalDataVec,
        const double *                               points,
        const double *                               origin,
        double *                                     values,
        linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext);
    }; // end of class AtomSuperpositionFunctionEvalKernels

#ifdef DFTEFE_WITH_DEVICE
    template <>
    class AtomSuperpositionFunctionEvalKernels<
      dftefe::utils::MemorySpace::DEVICE>
    {
    public:
      static void
      getEnrichmentValues(
        const size_type               numEnrichmentFunc,
        const std::vector<size_type> &pointsPerEnrichId,
        const std::vector<std::shared_ptr<atoms::SphericalData>>
          &           sphericalDataVec,
        const double *points,
        const double *origin,
        double *      values,
        linearAlgebra::LinAlgOpContext<dftefe::utils::MemorySpace::DEVICE>
          &linAlgOpContext);

      static void
      getEnrichmentGradients(
        const size_type               numEnrichmentFunc,
        const std::vector<size_type> &pointsPerEnrichId,
        const std::vector<std::shared_ptr<atoms::SphericalData>>
          &           sphericalDataVec,
        const double *points,
        const double *origin,
        double *      values,
        linearAlgebra::LinAlgOpContext<dftefe::utils::MemorySpace::DEVICE>
          &linAlgOpContext);
    }; // end of class AtomSuperpositionFunctionEvalKernels
#endif

  } // end of namespace atoms
} // end of namespace dftefe
#endif // dftefeAtomSuperpositionFunctionEvalKernels_h
