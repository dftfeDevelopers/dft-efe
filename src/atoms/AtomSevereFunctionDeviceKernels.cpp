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
        absBPlusRhoTimesVTotalKernel,
        {
          for (size_type iPoint = globalThreadId; iPoint < numPoints;
               iPoint += nThreadsPerBlock * nThreadBlock)
            {
              atomicAdd(&q[iPoint],
                        fabs(vtotal[iPoint] * (b[iPoint] + rho[iPoint])));
            }
        },
        const size_type numPoints,
        const double *  vtotal,
        const double *  b,
        const double *  rho,
        double *        q);

      DFTEFE_CREATE_KERNEL(
        void,
        absBTimesVNuclearKernel,
        {
          for (size_type iPoint = globalThreadId; iPoint < numPoints;
               iPoint += nThreadsPerBlock * nThreadBlock)
            {
              atomicAdd(&q[iPoint], fabs(vnuclear[iPoint] * b[iPoint]));
            }
        },
        const size_type numPoints,
        const double *  vnuclear,
        const double *  b,
        double *        q);

      DFTEFE_CREATE_KERNEL(
        void,
        absVExtTimesOrbitalSqKernel,
        {
          for (size_type iPoint = globalThreadId; iPoint < numPoints;
               iPoint += nThreadsPerBlock * nThreadBlock)
            {
              atomicAdd(&q[iPoint],
                        fabs(orbital[iPoint] * orbital[iPoint] * vext[iPoint]));
            }
        },
        const size_type numPoints,
        const double *  orbital,
        const double *  vext,
        double *        q);

    } // namespace

    template <>
    void
    AtomSevereFunction<utils::MemorySpace::HOST>::evalDevice(
      size_type     numPoints,
      const double *t,
      double *      q) const
    {
      utils::throwException(
        false, "AtomSevereFunction<HOST>::evalDevice should not be called.");
    }

    template <>
    void
    AtomSevereFunction<utils::MemorySpace::DEVICE>::evalDevice(
      size_type     numPoints,
      const double *t,
      double *      q) const
    {
      if (!d_isComposite)
        {
          AtomSuperpositionFunction<utils::MemorySpace::DEVICE>::evalDevice(
            numPoints, d_atomSupType, d_constant, t, q);
          return;
        }

      utils::deviceSetValue(q, 0.0, numPoints);

      const size_type E         = this->d_numEnrichmentFuncTotal;
      const size_type blockSize = utils::DEVICE_BLOCK_SIZE;
      const size_type grid      = (numPoints + blockSize - 1) / blockSize;

      utils::MemoryStorage<double, utils::MemorySpace::DEVICE> sphericalStorage(
        numPoints);

      if (d_atomicType == AtomSevereFuncType::Atomic::bPlusRhoTimesVTotal)
        {
          utils::throwException(
            this->d_linAlgOpContext != nullptr,
            "AtomSevereFunction::bPlusRhoTimesVTotal evalDevice requires "
            "non-null LinAlgOpContext.");

          utils::MemoryStorage<double, utils::MemorySpace::DEVICE> bStorage(
            numPoints),
            rhoStorage(numPoints);

          d_b->template eval<utils::MemorySpace::DEVICE>(numPoints,
                                                         t,
                                                         bStorage.data());
          d_rho->template eval<utils::MemorySpace::DEVICE>(numPoints,
                                                           t,
                                                           rhoStorage.data());

          for (size_type e = 0; e < E; ++e)
            {
              this->d_sphericalDataVecAll[e]->getValueDevice(
                numPoints,
                t,
                this->d_originsFlat.data() + e * this->d_dim,
                sphericalStorage.data());
              DFTEFE_LAUNCH_KERNEL(absBPlusRhoTimesVTotalKernel,
                                   grid,
                                   blockSize,
                                   0,
                                   numPoints,
                                   sphericalStorage.data(),
                                   bStorage.data(),
                                   rhoStorage.data(),
                                   q);
            }
        }
      else if (d_atomicType == AtomSevereFuncType::Atomic::bTimesVNuclear)
        {
          utils::MemoryStorage<double, utils::MemorySpace::DEVICE> bStorage(
            numPoints);

          d_b->template eval<utils::MemorySpace::DEVICE>(numPoints,
                                                         t,
                                                         bStorage.data());

          for (size_type e = 0; e < E; ++e)
            {
              this->d_sphericalDataVecAll[e]->getValueDevice(
                numPoints,
                t,
                this->d_originsFlat.data() + e * this->d_dim,
                sphericalStorage.data());
              DFTEFE_LAUNCH_KERNEL(absBTimesVNuclearKernel,
                                   grid,
                                   blockSize,
                                   0,
                                   numPoints,
                                   sphericalStorage.data(),
                                   bStorage.data(),
                                   q);
            }
        }
      else if (d_atomicType == AtomSevereFuncType::Atomic::vExtTimesOrbitalSq)
        {
          utils::MemoryStorage<double, utils::MemorySpace::DEVICE> vextStorage(
            numPoints);

          d_vext->template eval<utils::MemorySpace::DEVICE>(numPoints,
                                                            t,
                                                            vextStorage.data());

          for (size_type e = 0; e < E; ++e)
            {
              this->d_sphericalDataVecAll[e]->getValueDevice(
                numPoints,
                t,
                this->d_originsFlat.data() + e * this->d_dim,
                sphericalStorage.data());
              DFTEFE_LAUNCH_KERNEL(absVExtTimesOrbitalSqKernel,
                                   grid,
                                   blockSize,
                                   0,
                                   numPoints,
                                   sphericalStorage.data(),
                                   vextStorage.data(),
                                   q);
            }
        }
    }

  } // namespace atoms
} // namespace dftefe
#endif // DFTEFE_WITH_DEVICE
