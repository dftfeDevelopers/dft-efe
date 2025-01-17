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
#include "SphericalHarmonicFunctions.h"
#include "SmoothCutoffFunctions.h"
#include <utils/Spline.h>
#include "BoostAutoDiff.h"
#include <cmath>
#include <atoms/SphericalDataNumerical.h>

namespace dftefe
{
  namespace atoms
  {
    namespace SphericalDataInternal
    {
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
        size_type dim = point.size();
        // do the spline interpolation in the radial points
        std::vector<double> atomCenteredPoint(dim, 0.);
        double              r, theta, phi;
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

      void
      getGradientValueAnalytical(const utils::Point &    point,
                                 const utils::Point &    origin,
                                 const double            cutoff,
                                 const double            smoothness,
                                 const std::vector<int> &qNumbers,
                                 std::shared_ptr<const utils::Spline> spline,
                                 const double         polarAngleTolerance,
                                 const double         cutoffTolerance,
                                 const double         radiusTolerance,
                                 std::vector<double> &gradient)
      {
        size_type dim = point.size();
        // do the spline interpolation in the radial points
        std::vector<double> atomCenteredPoint(dim, 0.);
        double              r, theta, phi;
        for (unsigned int i = 0; i < dim; i++)
          {
            atomCenteredPoint[i] = point[i] - origin[i];
          }
        convertCartesianToSpherical(
          atomCenteredPoint, r, theta, phi, polarAngleTolerance);
        utils::throwException(std::abs(r) >= radiusTolerance,
                              "Value undefined at nucleus");
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

        gradient[0] = dValueDR * (sin(theta) * cos(phi)) +
                      dValueDThetaByr * (cos(theta) * cos(phi)) -
                      sin(phi) * dValueDPhiByrsinTheta;
        gradient[1] = dValueDR * (sin(theta) * sin(phi)) +
                      dValueDThetaByr * (cos(theta) * sin(phi)) +
                      cos(phi) * dValueDPhiByrsinTheta;
        gradient[2] = dValueDR * (cos(theta)) - dValueDThetaByr * (sin(theta));
      }

      void
      getHessianValueAnalytical(const utils::Point &                 point,
                                const utils::Point &                 origin,
                                const double                         cutoff,
                                const double                         smoothness,
                                const std::vector<int> &             qNumbers,
                                std::shared_ptr<const utils::Spline> spline,
                                const double         polarAngleTolerance,
                                const double         cutoffTolerance,
                                const double         radiusTolerance,
                                std::vector<double> &gradient)
      {
        utils::throwException(
          false,
          "Hessian matrix using analytical expressions is Not Yet Implemented");
      }

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
        size_type           dim = point.size();
        double              r, theta, phi;
        std::vector<double> atomCenteredPoint(dim, 0.);
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

      void
      getGradientValueAutoDiff(const utils::Point &                 point,
                               const utils::Point &                 origin,
                               const double                         cutoff,
                               const double                         smoothness,
                               const std::vector<int> &             qNumbers,
                               std::shared_ptr<const utils::Spline> spline,
                               const double         polarAngleTolerance,
                               const double         radiusTolerance,
                               std::vector<double> &gradient)
      {
        size_type           dim = point.size();
        double              r, theta, phi;
        std::vector<double> atomCenteredPoint(dim, 0.);
        for (unsigned int i = 0; i < dim; i++)
          {
            atomCenteredPoint[i] = point[i] - origin[i];
          }
        convertCartesianToSpherical(
          atomCenteredPoint, r, theta, phi, polarAngleTolerance);
        utils::throwException(std::abs(r) >= radiusTolerance,
                              "Value undefined at nucleus");
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

      void
      getHessianValueAutoDiff(const utils::Point &                 point,
                              const utils::Point &                 origin,
                              const double                         cutoff,
                              const double                         smoothness,
                              const std::vector<int> &             qNumbers,
                              std::shared_ptr<const utils::Spline> spline,
                              const double         polarAngleTolerance,
                              const double         radiusTolerance,
                              std::vector<double> &hessian)
      {
        size_type           dim = point.size();
        double              r, theta, phi;
        std::vector<double> atomCenteredPoint(dim, 0.);
        for (unsigned int i = 0; i < dim; i++)
          {
            atomCenteredPoint[i] = point[i] - origin[i];
          }
        convertCartesianToSpherical(
          atomCenteredPoint, r, theta, phi, polarAngleTolerance);
        utils::throwException(std::abs(r) >= radiusTolerance,
                              "Value undefined at nucleus");
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

    SphericalDataNumerical::SphericalDataNumerical(
      const std::vector<int>    qNumbers,
      const std::vector<double> radialPoints,
      const std::vector<double> radialValues,
      const double              cutoff,
      const double              smoothness,
      const double              polarAngleTolerance,
      const double              cutoffTolerance,
      const double              radiusTolerance,
      const size_type           dim)
      : d_qNumbers(qNumbers)
      , d_radialPoints(radialPoints)
      , d_radialValues(radialValues)
      , d_cutoff(cutoff)
      , d_smoothness(smoothness)
      , d_polarAngleTolerance(polarAngleTolerance)
      , d_cutoffTolerance(cutoffTolerance)
      , d_radiusTolerance(radiusTolerance)
      , d_dim(dim)
    {
      utils::throwException<utils::InvalidArgument>(d_dim == 3,
                                                    "Dimension has to be 3.");
      initSpline();
    }

    void
    SphericalDataNumerical::initSpline()
    {
      d_spline = std::make_shared<const utils::Spline>(this->d_radialPoints,
                                                       this->d_radialValues);
    }

    double
    SphericalDataNumerical::getValue(const utils::Point &point,
                                     const utils::Point &origin)
    {
      double value;
      DFTEFE_AssertWithMsg(point.size() == d_dim && origin.size() == d_dim,
                           "getValue() has a dimension mismatch");
      DFTEFE_AssertWithMsg(d_qNumbers.size() == 3,
                           "All quantum numbers not given");
      SphericalDataInternal::getValueAnalytical(point,
                                                origin,
                                                d_cutoff,
                                                d_smoothness,
                                                d_qNumbers,
                                                d_spline,
                                                d_polarAngleTolerance,
                                                value);

      return value;
    }

    std::vector<double>
    SphericalDataNumerical::getGradientValue(const utils::Point &point,
                                             const utils::Point &origin)
    {
      std::vector<double> gradient(d_dim, 0.);
      DFTEFE_AssertWithMsg(point.size() == d_dim && origin.size() == d_dim,
                           "getDerivativeValue() has a dimension mismatch");
      DFTEFE_AssertWithMsg(d_qNumbers.size() == 3,
                           "All quantum numbers not given");
      SphericalDataInternal::getGradientValueAnalytical(point,
                                                        origin,
                                                        d_cutoff,
                                                        d_smoothness,
                                                        d_qNumbers,
                                                        d_spline,
                                                        d_polarAngleTolerance,
                                                        d_cutoffTolerance,
                                                        d_radiusTolerance,
                                                        gradient);

      DFTEFE_AssertWithMsg(gradient.size() == d_dim,
                           "Gradient vector should be of length dim");
      return gradient;
    }

    std::vector<double>
    SphericalDataNumerical::getHessianValue(const utils::Point &point,
                                            const utils::Point &origin)
    {
      std::vector<double> hessian(d_dim * d_dim, 0.);
      DFTEFE_AssertWithMsg(point.size() == d_dim && origin.size() == d_dim,
                           "getHessianValue() has a dimension mismatch");
      DFTEFE_AssertWithMsg(d_qNumbers.size() == 3,
                           "All quantum numbers not given");

      SphericalDataInternal::getHessianValueAutoDiff(point,
                                                     origin,
                                                     d_cutoff,
                                                     d_smoothness,
                                                     d_qNumbers,
                                                     d_spline,
                                                     d_polarAngleTolerance,
                                                     d_radiusTolerance,
                                                     hessian);
      DFTEFE_AssertWithMsg(hessian.size() == d_dim * d_dim,
                           "Hessian vector should be of length dim*dim");
      return hessian;
    }

    std::vector<int>
    SphericalDataNumerical::getQNumbers() const
    {
      return d_qNumbers;
    }

    double
    SphericalDataNumerical::getCutoff() const
    {
      return d_cutoff;
    }

    double
    SphericalDataNumerical::getSmoothness() const
    {
      return d_smoothness;
    }

  } // namespace atoms
} // namespace dftefe
