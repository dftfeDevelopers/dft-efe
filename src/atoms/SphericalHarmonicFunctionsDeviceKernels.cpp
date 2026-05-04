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
 *
 * Batch (all-points) kernel launchers for SphericalHarmonicFunctions on DEVICE.
 * Scalar per-point device helpers (plm, dplmDTheta, etc.) and
 * the scalar member functions (PlmDevice, dPlmDThetaDevice, etc.) live in
 * SphericalHarmonicFunctionsDeviceKernels.h, pulled in via
 * SphericalHarmonicFunctions.h.
 */

#ifdef DFTEFE_WITH_DEVICE
#  include <utils/DeviceKernelLauncherHelpers.h>
#  include <utils/DeviceTypeConfig.h>
#  include <utils/DeviceAPICalls.h>
#  include <utils/Exceptions.h>
#  include <atoms/SphericalHarmonicFunctions.h>  // also pulls in DeviceKernels header
#  include <cmath>

namespace dftefe
{
  namespace atoms
  {
    namespace
    {
      //-----------------------------------------------------------------------
      // Batch kernels
      //-----------------------------------------------------------------------
      DFTEFE_CREATE_KERNEL(
        void,
        FillConstantKernel,
        {
          for (size_type i = globalThreadId; i < nPoints;
               i += nThreadsPerBlock * nThreadBlock)
            out[i] = value;
        },
        const size_type nPoints,
        const double    value,
        double *        out);

      DFTEFE_CREATE_KERNEL(
        void,
        ScaleArrayKernel,
        {
          for (size_type i = globalThreadId; i < nPoints;
               i += nThreadsPerBlock * nThreadBlock)
            out[i] *= scale;
        },
        const size_type nPoints,
        const double    scale,
        double *        out);

      DFTEFE_CREATE_KERNEL(
        void,
        CartesianToSphericalKernel,
        {
          for (size_type i = globalThreadId; i < nPoints;
               i += nThreadsPerBlock * nThreadBlock)
            {
              double pt[3];
              pt[0] = points[3 * i];
              pt[1] = points[3 * i + 1];
              pt[2] = points[3 * i + 2];
              convertCartesianToSpherical(
                pt, r[i], theta[i], phi[i], polarAngleTolerance);
            }
        },
        const size_type nPoints,
        const double *  points,
        double *        r,
        double *        theta,
        double *        phi,
        const double    polarAngleTolerance);

      DFTEFE_CREATE_KERNEL(
        void,
        QmKernel,
        {
          for (size_type i = globalThreadId; i < nPoints;
               i += nThreadsPerBlock * nThreadBlock)
            out[i] = Qm(m, phi[i]);
        },
        const size_type nPoints,
        const int       m,
        const double *  phi,
        double *        out);

      DFTEFE_CREATE_KERNEL(
        void,
        DQmDPhiKernel,
        {
          for (size_type i = globalThreadId; i < nPoints;
               i += nThreadsPerBlock * nThreadBlock)
            out[i] = dQmDPhi(m, phi[i]);
        },
        const size_type nPoints,
        const int       m,
        const double *  phi,
        double *        out);

      DFTEFE_CREATE_KERNEL(
        void,
        PlmAnalyticalKernel,
        {
          for (size_type i = globalThreadId; i < nPoints;
               i += nThreadsPerBlock * nThreadBlock)
            out[i] = factor * plm(l, absm, cos(theta[i]));
        },
        const size_type nPoints,
        const int       l,
        const int       absm,
        const double    factor,
        const double *  theta,
        double *        out);

      DFTEFE_CREATE_KERNEL(
        void,
        DPlmDThetaAnalyticalKernel,
        {
          for (size_type i = globalThreadId; i < nPoints;
               i += nThreadsPerBlock * nThreadBlock)
            out[i] = factor * dplmDTheta(l, absm, cos(theta[i]));
        },
        const size_type nPoints,
        const int       l,
        const int       absm,
        const double    factor,
        const double *  theta,
        double *        out);

      DFTEFE_CREATE_KERNEL(
        void,
        D2PlmDTheta2AnalyticalKernel,
        {
          for (size_type i = globalThreadId; i < nPoints;
               i += nThreadsPerBlock * nThreadBlock)
            out[i] = factor * d2plmDTheta2(l, absm, cos(theta[i]));
        },
        const size_type nPoints,
        const int       l,
        const int       absm,
        const double    factor,
        const double *  theta,
        double *        out);

    } // anonymous namespace

    template <>
    void
    convertCartesianToSpherical<utils::MemorySpace::DEVICE>(
      size_type             numPoints,
      const double *        points,
      double *              r,
      double *              theta,
      double *              phi,
      double                polarAngleTolerance,
      utils::deviceStream_t streamId)
    {
      DFTEFE_LAUNCH_KERNEL(CartesianToSphericalKernel,
                           numPoints / dftefe::utils::DEVICE_BLOCK_SIZE + 1,
                           dftefe::utils::DEVICE_BLOCK_SIZE,
                           streamId,
                           numPoints,
                           points,
                           r,
                           theta,
                           phi,
                           polarAngleTolerance);
    }

    template <>
    void
    Qm<utils::MemorySpace::DEVICE>(size_type             numPoints,
                                   const int             m,
                                   const double *        phi,
                                   double *              out,
                                   utils::deviceStream_t streamId)
    {
      DFTEFE_LAUNCH_KERNEL(QmKernel,
                           numPoints / dftefe::utils::DEVICE_BLOCK_SIZE + 1,
                           dftefe::utils::DEVICE_BLOCK_SIZE,
                           streamId,
                           numPoints,
                           m,
                           phi,
                           out);
    }

    template <>
    void
    dQmDPhi<utils::MemorySpace::DEVICE>(size_type             numPoints,
                                        const int             m,
                                        const double *        phi,
                                        double *              out,
                                        utils::deviceStream_t streamId)
    {
      DFTEFE_LAUNCH_KERNEL(DQmDPhiKernel,
                           numPoints / dftefe::utils::DEVICE_BLOCK_SIZE + 1,
                           dftefe::utils::DEVICE_BLOCK_SIZE,
                           streamId,
                           numPoints,
                           m,
                           phi,
                           out);
    }

    template <>
    void
    SphericalHarmonicFunctions::Plm<utils::MemorySpace::DEVICE>(
      size_type             numPoints,
      const int             l,
      const int             m,
      const double *        theta,
      double *              out,
      utils::deviceStream_t streamId) const
    {
      const size_type grid  = numPoints / dftefe::utils::DEVICE_BLOCK_SIZE + 1;
      const size_type block = dftefe::utils::DEVICE_BLOCK_SIZE;
      const int       absm  = std::abs(m);
      const double factor   = (m < 0) ? pow(-1.0, m) * Rlm(l, absm) : 1.0;

      if (d_isAssocLegendreSplineEval)
        {
          if (l == 0)
            {
              DFTEFE_LAUNCH_KERNEL(
                FillConstantKernel, grid, block, streamId, numPoints, 1.0, out);
            }
          else
            {
              d_assocLegendreSpline[l][absm]->evalAll<utils::MemorySpace::DEVICE>(
                numPoints, theta, out, streamId);
              if (m < 0)
                DFTEFE_LAUNCH_KERNEL(
                  ScaleArrayKernel, grid, block, streamId, numPoints, factor, out);
            }
        }
      else
        {
          if (absm > l)
            DFTEFE_LAUNCH_KERNEL(
              FillConstantKernel, grid, block, streamId, numPoints, 0.0, out);
          else
            DFTEFE_LAUNCH_KERNEL(PlmAnalyticalKernel,
                                 grid,
                                 block,
                                 streamId,
                                 numPoints,
                                 l,
                                 absm,
                                 factor,
                                 theta,
                                 out);
        }
    }

    template <>
    void
    SphericalHarmonicFunctions::dPlmDTheta<utils::MemorySpace::DEVICE>(
      size_type             numPoints,
      const int             l,
      const int             m,
      const double *        theta,
      double *              out,
      utils::deviceStream_t streamId) const
    {
      const size_type grid  = numPoints / dftefe::utils::DEVICE_BLOCK_SIZE + 1;
      const size_type block = dftefe::utils::DEVICE_BLOCK_SIZE;

      if (std::abs(m) > l || l == 0)
        {
          DFTEFE_LAUNCH_KERNEL(
            FillConstantKernel, grid, block, streamId, numPoints, 0.0, out);
        }
      else
        {
          const int    absm  = std::abs(m);
          const double factor = (m < 0) ? pow(-1.0, m) * Rlm(l, absm) : 1.0;

          if (d_isAssocLegendreSplineEval)
            {
              d_assocLegendreSpline[l][absm]->derivAll<utils::MemorySpace::DEVICE>(
                numPoints, 1, theta, out, streamId);
              if (m < 0)
                DFTEFE_LAUNCH_KERNEL(
                  ScaleArrayKernel, grid, block, streamId, numPoints, factor, out);
            }
          else
            {
              DFTEFE_LAUNCH_KERNEL(DPlmDThetaAnalyticalKernel,
                                   grid,
                                   block,
                                   streamId,
                                   numPoints,
                                   l,
                                   absm,
                                   factor,
                                   theta,
                                   out);
            }
        }
    }

    template <>
    void
    SphericalHarmonicFunctions::d2PlmDTheta2<utils::MemorySpace::DEVICE>(
      size_type             numPoints,
      const int             l,
      const int             m,
      const double *        theta,
      double *              out,
      utils::deviceStream_t streamId) const
    {
      const size_type grid  = numPoints / dftefe::utils::DEVICE_BLOCK_SIZE + 1;
      const size_type block = dftefe::utils::DEVICE_BLOCK_SIZE;

      if (std::abs(m) > l || l == 0)
        {
          DFTEFE_LAUNCH_KERNEL(
            FillConstantKernel, grid, block, streamId, numPoints, 0.0, out);
        }
      else
        {
          const int    absm  = std::abs(m);
          const double factor = (m < 0) ? pow(-1.0, m) * Rlm(l, absm) : 1.0;

          if (d_isAssocLegendreSplineEval)
            {
              d_assocLegendreSpline[l][absm]->derivAll<utils::MemorySpace::DEVICE>(
                numPoints, 2, theta, out, streamId);
              if (m < 0)
                DFTEFE_LAUNCH_KERNEL(
                  ScaleArrayKernel, grid, block, streamId, numPoints, factor, out);
            }
          else
            {
              DFTEFE_LAUNCH_KERNEL(D2PlmDTheta2AnalyticalKernel,
                                   grid,
                                   block,
                                   streamId,
                                   numPoints,
                                   l,
                                   absm,
                                   factor,
                                   theta,
                                   out);
            }
        }
    }

  } // namespace atoms
} // namespace dftefe
#endif // DFTEFE_WITH_DEVICE
