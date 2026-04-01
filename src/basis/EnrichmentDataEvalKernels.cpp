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

#include <basis/EnrichmentDataEvalKernels.h>

namespace dftefe
{
  namespace basis
  {
    template <utils::MemorySpace memorySpace>
    void
    EnrichmentDataEvalKernels<memorySpace>::getEnrichmentValues(
      const size_type                                            numEnrichmentFunc,
      const std::vector<size_type> &                            pointsPerEnrichId,
      const std::vector<std::shared_ptr<atoms::SphericalData>> &sphericalDataVec,
      const double *                                            points,
      const double *                                            origin,
      double *                                                  values,
      linearAlgebra::LinAlgOpContext<memorySpace> &             linAlgOpContext)
    {
      size_type cumulativeValuesOffset = 0;
      size_type cumulativeCoordsOffset = 0;
      for (int i = 0; i < numEnrichmentFunc; i++)
        {
          sphericalDataVec[i]->getValue(pointsPerEnrichId[i],
                                        points + cumulativeCoordsOffset,
                                        origin + i * 3,
                                        values + cumulativeValuesOffset);
          cumulativeValuesOffset += pointsPerEnrichId[i];
          cumulativeCoordsOffset += pointsPerEnrichId[i] * 3;
        }
    }

    template <utils::MemorySpace memorySpace>
    void
    EnrichmentDataEvalKernels<memorySpace>::getEnrichmentGradients(
      const size_type                                            numEnrichmentFunc,
      const std::vector<size_type> &                            pointsPerEnrichId,
      const std::vector<std::shared_ptr<atoms::SphericalData>> &sphericalDataVec,
      const double *                                            points,
      const double *                                            origin,
      double *                                                  values,
      linearAlgebra::LinAlgOpContext<memorySpace> &             linAlgOpContext)
    {
      size_type cumulativeValuesOffset = 0;
      size_type cumulativeCoordsOffset = 0;
      for (int i = 0; i < numEnrichmentFunc; i++)
        {
          sphericalDataVec[i]->getGradientValue(pointsPerEnrichId[i],
                                                points + cumulativeCoordsOffset,
                                                origin + i * 3,
                                                values + cumulativeValuesOffset);
          cumulativeValuesOffset += pointsPerEnrichId[i] * 3;
          cumulativeCoordsOffset += pointsPerEnrichId[i] * 3;
        }
    }

    template class EnrichmentDataEvalKernels<utils::MemorySpace::HOST>;
  } // end of namespace basis
} // end of namespace dftefe
