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

#ifdef DFTEFE_WITH_DEVICE
#  include <utils/DeviceKernelLauncherHelpers.h>
#  include <utils/DeviceAPICalls.h>
#  include <atoms/AtomSuperpositionFunction.h>
#  include <cmath>

namespace dftefe
{
  namespace atoms
  {
    namespace
    {
      // Contracts nComp values per point: q[iPoint] += constant * sum_k
      // pow(values[iPoint*nComp+k], power)
      DFTEFE_CREATE_KERNEL(
        void,
        addPowSumKernel,
        {
          for (size_type iPoint = globalThreadId; iPoint < numPoints;
               iPoint += nThreadsPerBlock * nThreadBlock)
            {
              double s = 0.0;
              for (size_type k = 0; k < nComp; ++k)
                s +=
                  pow(values[iPoint * nComp + k], static_cast<double>(power));
              q[iPoint] += constant * s;
            }
        },
        const size_type numPoints,
        const size_type nComp,
        const size_type power,
        const double    constant,
        const double *  values,
        double *        q);

    } // namespace

    template <>
    void
    AtomSuperpositionFunction<utils::MemorySpace::HOST>::evalDevice(
      size_type                       numPoints,
      const AtomSuperpositionFuncType atomSupType,
      const double                    constant,
      const double *                  t,
      double *                        q,
      const std::vector<double> &     atomWeights) const
    {
      utils::throwException(
        false,
        "AtomSuperpositionFunction<HOST>::evalDevice should not be called.");
    }

    template <>
    void
    AtomSuperpositionFunction<utils::MemorySpace::DEVICE>::evalDevice(
      size_type                       numPoints,
      const AtomSuperpositionFuncType atomSupType,
      const double                    constant,
      const double *                  t,
      double *                        q,
      const std::vector<double> &     atomWeights) const
    {
      utils::throwException(
        d_linAlgOpContext != nullptr,
        "AtomSuperpositionFunction::evalDevice requires a non-null LinAlgOpContext.");

      const size_type E         = d_numEnrichmentFuncTotal;
      const size_type blockSize = utils::DEVICE_BLOCK_SIZE;

      utils::MemoryStorage<double, utils::MemorySpace::DEVICE> d_values;

      if (atomSupType == AtomSuperpositionFuncType::Identity ||
          atomSupType == AtomSuperpositionFuncType::IdentitySq)
        {
          const size_type nComp = 1;
          const size_type power =
            (atomSupType == AtomSuperpositionFuncType::IdentitySq) ? 2 : 1;
          const size_type addGrid = (numPoints + blockSize - 1) / blockSize;

          if (d_values.size() != numPoints * nComp)
            d_values.resize(numPoints * nComp);

          utils::deviceSetValue(q, 0.0, numPoints);

          const std::vector<size_type> pointsPerEnrich(1, numPoints);
          for (size_type e = 0; e < E; ++e)
            {
              const double wt =
                (atomWeights.empty() ||
                 atomSupType == AtomSuperpositionFuncType::IdentitySq) ?
                  1.0 :
                  atomWeights[d_enrichmentToAtomId[e]];
              const std::vector<std::shared_ptr<atoms::SphericalData>>
                singleVec = {d_sphericalDataVecAll[e]};
              basis::EnrichmentDataEvalKernels<utils::MemorySpace::DEVICE>::
                getEnrichmentValues(1,
                                    pointsPerEnrich,
                                    singleVec,
                                    t,
                                    d_originsFlat.data() + e * d_dim,
                                    d_values.data(),
                                    *d_linAlgOpContext);
              DFTEFE_LAUNCH_KERNEL(addPowSumKernel,
                                   addGrid,
                                   blockSize,
                                   0,
                                   numPoints,
                                   nComp,
                                   power,
                                   constant * wt,
                                   d_values.data(),
                                   q);
            }
        }
      else if (atomSupType == AtomSuperpositionFuncType::GradDotGradSq)
        {
          const size_type nComp   = d_dim;
          const size_type power   = 2;
          const size_type addGrid = (numPoints + blockSize - 1) / blockSize;

          if (d_values.size() != numPoints * nComp)
            d_values.resize(numPoints * nComp);

          utils::deviceSetValue(q, 0.0, numPoints);

          const std::vector<size_type> pointsPerEnrich(1, numPoints);
          for (size_type e = 0; e < E; ++e)
            {
              const std::vector<std::shared_ptr<atoms::SphericalData>>
                singleVec = {d_sphericalDataVecAll[e]};
              basis::EnrichmentDataEvalKernels<utils::MemorySpace::DEVICE>::
                getEnrichmentGradients(1,
                                       pointsPerEnrich,
                                       singleVec,
                                       t,
                                       d_originsFlat.data() + e * d_dim,
                                       d_values.data(),
                                       *d_linAlgOpContext);
              DFTEFE_LAUNCH_KERNEL(addPowSumKernel,
                                   addGrid,
                                   blockSize,
                                   0,
                                   numPoints,
                                   nComp,
                                   power,
                                   constant,
                                   d_values.data(),
                                   q);
            }
        }
      else if (atomSupType == AtomSuperpositionFuncType::Grad)
        {
          const size_type nComp       = d_dim;
          const size_type numElements = numPoints * d_dim;
          const size_type addGrid = (numElements + blockSize - 1) / blockSize;

          if (d_values.size() != numElements)
            d_values.resize(numElements);

          utils::deviceSetValue(q, 0.0, numElements);

          const std::vector<size_type> pointsPerEnrich(1, numPoints);
          for (size_type e = 0; e < E; ++e)
            {
              const double wt = atomWeights.empty() ?
                                  1.0 :
                                  atomWeights[d_enrichmentToAtomId[e]];
              const std::vector<std::shared_ptr<atoms::SphericalData>>
                singleVec = {d_sphericalDataVecAll[e]};
              basis::EnrichmentDataEvalKernels<utils::MemorySpace::DEVICE>::
                getEnrichmentGradients(1,
                                       pointsPerEnrich,
                                       singleVec,
                                       t,
                                       d_originsFlat.data() + e * d_dim,
                                       d_values.data(),
                                       *d_linAlgOpContext);
              DFTEFE_LAUNCH_KERNEL(addPowSumKernel,
                                   addGrid,
                                   blockSize,
                                   0,
                                   numElements,
                                   1,
                                   1,
                                   constant * wt,
                                   d_values.data(),
                                   q);
            }
        }
    }

  } // namespace atoms
} // namespace dftefe
#endif // DFTEFE_WITH_DEVICE
