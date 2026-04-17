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
#  include <utils/SmearChargePotentialFunction.h>
#  include <cmath>

namespace dftefe
{
  namespace utils
  {
    namespace
    {
      DFTEFE_CREATE_KERNEL(
        void,
        smearChargePotentialKernel,
        {
          for (size_type idx = globalThreadId; idx < numPoints * numAtoms;
               idx += nThreadsPerBlock * nThreadBlock)
            {
              const size_type iPoint = idx % numPoints;
              const size_type iAtom  = idx / numPoints;

              double r = 0.0;
              for (size_type j = 0; j < dim; ++j)
                {
                  double diff =
                    t[iPoint * dim + j] - atomCoordsFlat[iAtom * dim + j];
                  r += diff * diff;
                }
              r = sqrt(r);
              double val = 0.0;
              if (r > rc[iAtom])
                val = z[iAtom] / r;
              else
                {
                  double rci = rc[iAtom];
                  val = z[iAtom] *
                        (9.0 * r * r * r * r * r * r * r -
                         30.0 * r * r * r * r * r * r * rci +
                         28.0 * r * r * r * r * r * rci * rci -
                         14.0 * r * r * rci * rci * rci * rci * rci +
                         12.0 * rci * rci * rci * rci * rci * rci * rci) /
                        (5.0 * rci * rci * rci * rci * rci * rci * rci * rci);
                }
              atomicAdd(&q[iPoint], val);
            }
        },
        const size_type numPoints,
        const size_type numAtoms,
        const size_type dim,
        const double *  t,
        const double *  atomCoordsFlat,
        const double *  rc,
        const double *  z,
        double *        q);
    } // namespace

    void
    SmearChargePotentialFunction::evalDevice(size_type      numPoints,
                                             const double * t,
                                             double *       q) const
    {
      deviceMemset(q, 0, numPoints * sizeof(double));
      const size_type total     = numPoints * d_numAtoms;
      const size_type blockSize = DEVICE_BLOCK_SIZE;
      const size_type gridSize  = (total + blockSize - 1) / blockSize;
      DFTEFE_LAUNCH_KERNEL(smearChargePotentialKernel,
                           gridSize,
                           blockSize,
                           0,
                           numPoints,
                           d_numAtoms,
                           d_dim,
                           t,
                           d_atomCoordsFlatDevice.data(),
                           d_rcDevice.data(),
                           d_zDevice.data(),
                           q);
    }

  } // namespace utils
} // namespace dftefe
#endif // DFTEFE_WITH_DEVICE
