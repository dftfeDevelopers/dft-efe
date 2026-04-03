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
    // EnrichmentDataEvalKernels<DEVICE> is a full class specialization in the
    // header, so its member definitions must NOT use "template <>".
    void
    EnrichmentDataEvalKernels<utils::MemorySpace::DEVICE>::getEnrichmentValues(
      const size_type  numEnrichmentFunc,
      const std::vector<size_type> &pointsPerEnrichId,
      const std::vector<std::shared_ptr<atoms::SphericalData>> &sphericalDataVec,
      const double *points,
      const double *origin,
      double * values,
      linearAlgebra::LinAlgOpContext<utils::MemorySpace::DEVICE> &linAlgOpContext)
    {
      const size_type numStreams  = linAlgOpContext.numBlasStreams();
      auto *          streams     = linAlgOpContext.getBlasStreamsVec();

      size_type cumulativeValuesOffset = 0;
      size_type cumulativeCoordsOffset = 0;
      for (int i = 0; i < numEnrichmentFunc; i++)
        {
          size_type sid = i % numStreams;
          sphericalDataVec[i]->getValueDevice(pointsPerEnrichId[i],
                                              points + cumulativeCoordsOffset,
                                              origin + i * 3,
                                              values + cumulativeValuesOffset,
                                              streams[sid]);
          cumulativeValuesOffset += pointsPerEnrichId[i];
          cumulativeCoordsOffset += pointsPerEnrichId[i] * 3;
        }
        for (int s = 0; s < numStreams; ++s)
          utils::deviceStreamSynchronize(streams[s]);
    }

    void
    EnrichmentDataEvalKernels<utils::MemorySpace::DEVICE>::getEnrichmentGradients(
      const size_type  numEnrichmentFunc,
      const std::vector<size_type> &pointsPerEnrichId,
      const std::vector<std::shared_ptr<atoms::SphericalData>> &sphericalDataVec,
      const double *points,
      const double *origin,
      double * values,
      linearAlgebra::LinAlgOpContext<utils::MemorySpace::DEVICE> &linAlgOpContext)
    {
      const size_type numStreams  = linAlgOpContext.numBlasStreams();
      auto *          streams     = linAlgOpContext.getBlasStreamsVec();

      size_type cumulativeValuesOffset = 0;
      size_type cumulativeCoordsOffset = 0;
      for (int i = 0; i < numEnrichmentFunc; i++)
        {
          size_type sid = i % numStreams;
          sphericalDataVec[i]->getGradientValueDevice(pointsPerEnrichId[i],
                                                      points + cumulativeCoordsOffset,
                                                      origin + i * 3,
                                                      values + cumulativeValuesOffset,
                                                      streams[sid]);
          cumulativeValuesOffset += pointsPerEnrichId[i] * 3;
          cumulativeCoordsOffset += pointsPerEnrichId[i] * 3;
        }
        for (int s = 0; s < numStreams; ++s)
          utils::deviceStreamSynchronize(streams[s]);
    }

  } // end of namespace basis
} // end of namespace dftefe
