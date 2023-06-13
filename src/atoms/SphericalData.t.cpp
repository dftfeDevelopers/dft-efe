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

#include <vector>
#include <utils/Point.h>
#include <utils/TypeConfig.h>
#include "SphericalData.h"
#include "SphericalHarmonicFunctions.h"
#include "SmoothCutoffFunctions.h"
#include <utils/Spline.h>
#include "BoostAutoDiff.h"
#include <cmath>

namespace dftefe
{
  namespace atoms
  {
    namespace SphericalDataInternal
    {
      template <size_type dim>
      void
      getValueAnalytical(const utils::Point &                 point,
                         const utils::Point &                 origin,
                         const double                         cutoff,
                         const double                         smoothness,
                         const std::vector<int> &             qNumbers,
                         std::shared_ptr<const utils::Spline> spline,
                         const double polarAngleTolerance,
                         double &     value)
      {
        // do the spline interpolation in the radial points
        std::vector<double> atomCenteredPoint;
        atomCenteredPoint.resize(dim, 0.);
        double r, theta, phi;
        for (unsigned int i = 0; i < dim; i++)
          {
            atomCenteredPoint[i] = point[i] - origin[i];
          }
        convertCartesianToSpherical(
          atomCenteredPoint, r, theta, phi, polarAngleTolerance);
        double radialValue = (*spline)(r);
        int    n = qNumbers[0], l = qNumbers[1], m = qNumbers[2];
        auto   Ylm = Clm(l, m) * Dm(m) * Plm(l, m, cos(theta)) * Qm(m, phi);
        value = radialValue * Ylm * smoothCutoffValue(r, cutoff, smoothness);
      }

      template <size_type dim>
      void
      getGradientValueAnalytical(const utils::Point &    point,
                                 const utils::Point &    origin,
                                 const double            cutoff,
                                 const double            smoothness,
                                 const std::vector<int> &qNumbers,
                                 std::shared_ptr<const utils::Spline> spline,
                                 const double         polarAngleTolerance,
                                 const double         cutoffTolerance,
                                 std::vector<double> &gradient)
      {
        // do the spline interpolation in the radial points
        std::vector<double> atomCenteredPoint;
        atomCenteredPoint.resize(dim, 0.);
        double r, theta, phi;
        for (unsigned int i = 0; i < dim; i++)
          {
            atomCenteredPoint[i] = point[i] - origin[i];
          }
        convertCartesianToSpherical(
          atomCenteredPoint, r, theta, phi, polarAngleTolerance);
        utils::throwException(r != 0, "Value undefined at nucleus");
        double radialValue           = (*spline)(r);
        double radialDerivativeValue = spline->deriv(1, r);
        double cutoffValue           = smoothCutoffValue(r, cutoff, smoothness);
        double cutoffDerv =
          smoothCutoffDerivative(r, cutoff, smoothness, cutoffTolerance);

        int n = qNumbers[0], l = qNumbers[1], m = qNumbers[2];

        auto Ylm = Clm(l, m) * Dm(m) * Plm(l, m, cos(theta)) * Qm(m, phi);
        auto dYlmDTheta =
          Clm(l, m) * Dm(m) * dPlmDTheta(l, m, theta) * Qm(m, phi);

        // Here used the Legendre differential equation for calculating
        // P_lm/sin(theta) given in the paper
        // https://doi.org/10.1016/S1464-1895(00)00101-0 Pt. no. 3 of
        // verification.

        double dYlmDPhiBysinTheta = 0.;
        if (m != 0)
          {
            dYlmDPhiBysinTheta =
              Clm(l, m) * Dm(m) *
              (sin(theta) * d2PlmDTheta2(l, m, theta) +
               cos(theta) * dPlmDTheta(l, m, theta) +
               sin(theta) * l * (l + 1) * Plm(l, m, cos(theta))) *
              (1 / (m * m)) * dQmDPhi(m, phi);
          }

        auto dValueDR =
          (radialDerivativeValue * cutoffValue + cutoffDerv * radialValue) *
          Ylm;
        double dValueDThetaByr = 0.;
        dValueDThetaByr        = (radialValue / r) * cutoffValue * dYlmDTheta;
        double dValueDPhiByrsinTheta = 0.;
        dValueDPhiByrsinTheta =
          (radialValue / r) * cutoffValue * dYlmDPhiBysinTheta;

        gradient.resize(dim, 0.);
        gradient[0] = dValueDR * (sin(theta) * cos(phi)) +
                      dValueDThetaByr * (cos(theta) * cos(phi)) -
                      sin(phi) * dValueDPhiByrsinTheta;
        gradient[1] = dValueDR * (sin(theta) * sin(phi)) +
                      dValueDThetaByr * (cos(theta) * sin(phi)) +
                      cos(phi) * dValueDPhiByrsinTheta;
        gradient[2] = dValueDR * (cos(theta)) - dValueDThetaByr * (sin(theta));
      }

      template <size_type dim>
      void
      getHessianValueAnalytical(const utils::Point &                 point,
                                const utils::Point &                 origin,
                                const double                         cutoff,
                                const double                         smoothness,
                                const std::vector<int> &             qNumbers,
                                std::shared_ptr<const utils::Spline> spline,
                                const double         polarAngleTolerance,
                                const double         cutoffTolerance,
                                std::vector<double> &gradient)
      {
        utils::throwException(
          false,
          "Hessian matirx using analytical expressions is Not Yet Implemented");
      }

      template <size_type dim>
      void
      getValueAutoDiff(const utils::Point &                 point,
                       const utils::Point &                 origin,
                       const double                         cutoff,
                       const double                         smoothness,
                       const std::vector<int> &             qNumbers,
                       std::shared_ptr<const utils::Spline> spline,
                       const double                         polarAngleTolerance,
                       double &                             value)
      {
        double              r, theta, phi;
        std::vector<double> atomCenteredPoint;
        atomCenteredPoint.resize(dim, 0.);
        for (unsigned int i = 0; i < dim; i++)
          {
            atomCenteredPoint[i] = point[i] - origin[i];
          }
        convertCartesianToSpherical(
          atomCenteredPoint, r, theta, phi, polarAngleTolerance);
        std::vector<double> coeffVec(0);
        coeffVec = spline->coefficients(r);
        int l = qNumbers[1], m = qNumbers[2];
        value = getValueBoostAutoDiff(point,
                                      origin,
                                      coeffVec,
                                      smoothness,
                                      cutoff,
                                      l,
                                      m,
                                      polarAngleTolerance);
      }

      template <size_type dim>
      void
      getGradientValueAutoDiff(const utils::Point &                 point,
                               const utils::Point &                 origin,
                               const double                         cutoff,
                               const double                         smoothness,
                               const std::vector<int> &             qNumbers,
                               std::shared_ptr<const utils::Spline> spline,
                               const double         polarAngleTolerance,
                               std::vector<double> &gradient)
      {
        double              r, theta, phi;
        std::vector<double> atomCenteredPoint;
        atomCenteredPoint.resize(dim, 0.);
        for (unsigned int i = 0; i < dim; i++)
          {
            atomCenteredPoint[i] = point[i] - origin[i];
          }
        convertCartesianToSpherical(
          atomCenteredPoint, r, theta, phi, polarAngleTolerance);
        utils::throwException(r != 0, "Value undefined at nucleus");
        std::vector<double> coeffVec(0);
        coeffVec = spline->coefficients(r);
        int l = qNumbers[1], m = qNumbers[2];
        gradient = getGradientValueBoostAutoDiff(point,
                                                 origin,
                                                 coeffVec,
                                                 smoothness,
                                                 cutoff,
                                                 l,
                                                 m,
                                                 polarAngleTolerance);
      }

      template <size_type dim>
      void
      getHessianValueAutoDiff(const utils::Point &                 point,
                              const utils::Point &                 origin,
                              const double                         cutoff,
                              const double                         smoothness,
                              const std::vector<int> &             qNumbers,
                              std::shared_ptr<const utils::Spline> spline,
                              const double         polarAngleTolerance,
                              std::vector<double> &hessian)
      {
        double              r, theta, phi;
        std::vector<double> atomCenteredPoint;
        atomCenteredPoint.resize(dim, 0.);
        for (unsigned int i = 0; i < dim; i++)
          {
            atomCenteredPoint[i] = point[i] - origin[i];
          }
        convertCartesianToSpherical(
          atomCenteredPoint, r, theta, phi, polarAngleTolerance);
        utils::throwException(r != 0, "Value undefined at nucleus");
        std::vector<double> coeffVec(0);
        coeffVec = spline->coefficients(r);
        int l = qNumbers[1], m = qNumbers[2];
        hessian = getHessianValueBoostAutoDiff(point,
                                               origin,
                                               coeffVec,
                                               smoothness,
                                               cutoff,
                                               l,
                                               m,
                                               polarAngleTolerance);
      }
    } // namespace SphericalDataInternal

    template <size_type dim>
    double
    SphericalData::getValue(const utils::Point &point,
                            const utils::Point &origin,
                            const double        polarAngleTolerance)
    {
      utils::throwException(
        dim == 3, "getDerivativeValue() defined only for 3 dimensional case");
      DFTEFE_AssertWithMsg(point.size() == dim && origin.size() == dim,
                           "Dimension mismatch between the point and origin.");
      DFTEFE_AssertWithMsg(qNumbers.size() == 3,
                           "All quantum numbers not given");
      // SphericalDataInternal::getValueAnalytical<dim>(point,
      //                       origin,
      //                       cutoff,
      //                       smoothness,
      //                       qNumbers,
      //                       d_spline,
      //                       polarAngleTolerance,
      //                       d_value);

      SphericalDataInternal::getValueAutoDiff<dim>(point,
                                                   origin,
                                                   cutoff,
                                                   smoothness,
                                                   qNumbers,
                                                   d_spline,
                                                   polarAngleTolerance,
                                                   d_value);
      return d_value;
    }

    template <size_type dim>
    std::vector<double>
    SphericalData::getGradientValue(const utils::Point &point,
                                    const utils::Point &origin,
                                    const double        polarAngleTolerance,
                                    const double        cutoffTolerance)
    {
      utils::throwException(
        dim == 3, "getDerivativeValue() defined only for 3 dimensional case");
      DFTEFE_AssertWithMsg(point.size() == dim && origin.size() == dim,
                           "Dimension mismatch between the point and origin.");
      DFTEFE_AssertWithMsg(qNumbers.size() == 3,
                           "All quantum numbers not given");
      // SphericalDataInternal::getGradientValueAnalytical<dim>(point,
      //                       origin,
      //                       cutoff,
      //                       smoothness,
      //                       qNumbers,
      //                       d_spline,
      //                       polarAngleTolerance,
      //                       cutoffTolerance,
      //                       d_gradient);

      SphericalDataInternal::getGradientValueAutoDiff<dim>(point,
                                                           origin,
                                                           cutoff,
                                                           smoothness,
                                                           qNumbers,
                                                           d_spline,
                                                           polarAngleTolerance,
                                                           d_gradient);
      DFTEFE_AssertWithMsg(d_gradient.size() == dim,
                           "Gradient vector should be of length dim");
      return d_gradient;
    }

    template <size_type dim>
    std::vector<double>
    SphericalData::getHessianValue(const utils::Point &point,
                                   const utils::Point &origin,
                                   const double        polarAngleTolerance,
                                   const double        cutoffTolerance)
    {
      utils::throwException(
        dim == 3, "getDerivativeValue() defined only for 3 dimensional case");
      DFTEFE_AssertWithMsg(point.size() == dim && origin.size() == dim,
                           "Dimension mismatch between the point and origin.");
      DFTEFE_AssertWithMsg(qNumbers.size() == 3,
                           "All quantum numbers not given");
      // SphericalDataInternal::getHessianValueAnalytical<dim>(point,
      //                       origin,
      //                       cutoff,
      //                       smoothness,
      //                       qNumbers,
      //                       d_spline,
      //                       polarAngleTolerance,
      //                       cutoffTolerance,
      //                       d_hessian);

      SphericalDataInternal::getHessianValueAutoDiff<dim>(point,
                                                          origin,
                                                          cutoff,
                                                          smoothness,
                                                          qNumbers,
                                                          d_spline,
                                                          polarAngleTolerance,
                                                          d_hessian);
      DFTEFE_AssertWithMsg(d_hessian.size() == dim * dim,
                           "Hessian vector should be of length dim*dim");
      return d_hessian;
    }
  } // namespace atoms
} // namespace dftefe
