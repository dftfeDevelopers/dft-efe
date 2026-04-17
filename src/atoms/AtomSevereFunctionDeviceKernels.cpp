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
        tileCoordsKernel,
        {
          const size_type total = numEnrich * numPoints * dim;
          for (size_type i = globalThreadId; i < total;
               i += nThreadsPerBlock * nThreadBlock)
            tiled[i] = src[i % (numPoints * dim)];
        },
        const size_type numEnrich,
        const size_type numPoints,
        const size_type dim,
        const double *  src,
        double *        tiled);

      DFTEFE_CREATE_KERNEL(
        void,
        accumPowKernel,
        {
          for (size_type iPoint = globalThreadId; iPoint < numPoints;
               iPoint += nThreadsPerBlock * nThreadBlock)
            {
              double s = 0.0;
              for (size_type e = 0; e < numEnrich; ++e)
                for (size_type k = 0; k < nComp; ++k)
                  s += pow(values[e * numPoints * nComp + iPoint * nComp + k],
                           static_cast<double>(power));
              q[iPoint] = constant * s;
            }
        },
        const size_type numPoints,
        const size_type numEnrich,
        const size_type nComp,
        const size_type power,
        const double    constant,
        const double *  values,
        double *        q);
    } // namespace

    void
    AtomSevereFunction::evalDevice(size_type      numPoints,
                                   const double * t,
                                   double *       q) const
    {
      const size_type nComp = (d_derivativeType == 1) ? d_dim : 1;
      const size_type E     = d_numEnrichmentFuncTotal;

      if(d_pointsTiledDevice.size() != E * numPoints * d_dim)
        d_pointsTiledDevice.resize(E * numPoints * d_dim);
      if(d_valuesDevice.size() != E * numPoints * nComp)
        d_valuesDevice.resize(E * numPoints * nComp);

      const size_type blockSize = utils::DEVICE_BLOCK_SIZE;
      const size_type tileGrid =
        (E * numPoints * d_dim + blockSize - 1) / blockSize;
      DFTEFE_LAUNCH_KERNEL(tileCoordsKernel,
                           tileGrid,
                           blockSize,
                           0,
                           E,
                           numPoints,
                           d_dim,
                           t,
                           d_pointsTiledDevice.data());

      std::vector<size_type> pointsPerEnrichId(E, numPoints);

      utils::throwException(
        d_linAlgOpContext != nullptr,
        "AtomSevereFunction::evalDevice requires a non-null LinAlgOpContext.");
      if (d_derivativeType == 0)
        basis::EnrichmentDataEvalKernels<
          utils::MemorySpace::DEVICE>::getEnrichmentValues(E,
                                                           pointsPerEnrichId,
                                                           d_sphericalDataVecAll,
                                                           d_pointsTiledDevice
                                                             .data(),
                                                           d_originsFlat.data(),
                                                           d_valuesDevice.data(),
                                                           *d_linAlgOpContext);
      else
        basis::EnrichmentDataEvalKernels<
          utils::MemorySpace::DEVICE>::getEnrichmentGradients(
          E,
          pointsPerEnrichId,
          d_sphericalDataVecAll,
          d_pointsTiledDevice.data(),
          d_originsFlat.data(),
          d_valuesDevice.data(),
          *d_linAlgOpContext);

      const size_type accumGrid = (numPoints + blockSize - 1) / blockSize;
      DFTEFE_LAUNCH_KERNEL(accumPowKernel,
                           accumGrid,
                           blockSize,
                           0,
                           numPoints,
                           E,
                           nComp,
                           d_sphericalValPower,
                           d_constant,
                           d_valuesDevice.data(),
                           q);
    }

  } // namespace atoms
} // namespace dftefe
#endif // DFTEFE_WITH_DEVICE
