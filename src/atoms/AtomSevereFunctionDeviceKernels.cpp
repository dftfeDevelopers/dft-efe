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
#  include <atoms/AtomSevereFunction.h>
#  include <cmath>

namespace dftefe
{
  namespace atoms
  {
    namespace
    {
      DFTEFE_CREATE_KERNEL(
        void,
        addPowKernel,
        {
          for (size_type iPoint = globalThreadId; iPoint < numPoints;
               iPoint += nThreadsPerBlock * nThreadBlock)
            {
              double s = 0.0;
              for (size_type k = 0; k < nComp; ++k)
                s += pow(values[iPoint * nComp + k], static_cast<double>(power));
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
    AtomSevereFunction<utils::MemorySpace::HOST>::evalDevice(
      size_type      numPoints,
      const double * t,
      double *       q) const
    {
      utils::throwException(
        false,
        "AtomSevereFunction<HOST>::evalDevice should not be called.");
    }

    template <>
    void
    AtomSevereFunction<utils::MemorySpace::DEVICE>::evalDevice(
      size_type      numPoints,
      const double * t,
      double *       q) const
    {
      const size_type nComp = (d_derivativeType == 1) ? d_dim : 1;
      const size_type E     = d_numEnrichmentFuncTotal;

      utils::MemoryStorage<double, utils::MemorySpace::DEVICE> d_values;
      if (d_values.size() != numPoints * nComp)
        d_values.resize(numPoints * nComp);

      utils::deviceSetValue(q, 0.0, numPoints);

      utils::throwException(
        d_linAlgOpContext != nullptr,
        "AtomSevereFunction::evalDevice requires a non-null LinAlgOpContext.");

      const std::vector<size_type> pointsPerEnrich(1, numPoints);
      const size_type              blockSize = utils::DEVICE_BLOCK_SIZE;
      const size_type addGrid = (numPoints + blockSize - 1) / blockSize;

      for (size_type e = 0; e < E; ++e)
        {
          const std::vector<std::shared_ptr<atoms::SphericalData>> singleVec = {
            d_sphericalDataVecAll[e]};
          if (d_derivativeType == 0)
            basis::EnrichmentDataEvalKernels<
              utils::MemorySpace::DEVICE>::getEnrichmentValues(1,
                                                               pointsPerEnrich,
                                                               singleVec,
                                                               t,
                                                               d_originsFlat
                                                                   .data() +
                                                                 e * d_dim,
                                                               d_values.data(),
                                                               *d_linAlgOpContext);
          else
            basis::EnrichmentDataEvalKernels<
              utils::MemorySpace::DEVICE>::getEnrichmentGradients(
              1,
              pointsPerEnrich,
              singleVec,
              t,
              d_originsFlat.data() + e * d_dim,
              d_values.data(),
              *d_linAlgOpContext);

          DFTEFE_LAUNCH_KERNEL(addPowKernel,
                               addGrid,
                               blockSize,
                               0,
                               numPoints,
                               nComp,
                               d_sphericalValPower,
                               d_constant,
                               d_values.data(),
                               q);
        }
    }

  } // namespace atoms
} // namespace dftefe
#endif // DFTEFE_WITH_DEVICE
