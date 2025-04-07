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
    namespace SphericalDataNumericalInternal
    {
      void
      getValueAnalytical(
        const std::vector<utils::Point> &    point,
        const utils::Point &                 origin,
        const double                         cutoff,
        const double                         smoothness,
        const SphericalHarmonicFunctions &   sphericalHarmonicFunc,
        const std::vector<int> &             qNumbers,
        std::shared_ptr<const utils::Spline> spline,
        const double                         polarAngleTolerance,
        std::vector<double> &                value)
      {
        for (int i = 0; i < point.size(); i++)
          {
            // do the spline interpolation in the radial points
            double r, theta, phi;
            convertCartesianToSpherical(
              point[i] - origin, r, theta, phi, polarAngleTolerance);
            if (r <= cutoff + cutoff / smoothness)
              {
                int  n = qNumbers[0], l = qNumbers[1], m = qNumbers[2];
                auto Ylm = Clm(l, m) * Dm(m) *
                           sphericalHarmonicFunc.Plm(l, m, theta) * Qm(m, phi);
                value[i] =
                  (*spline)(r)*Ylm * smoothCutoffValue(r, cutoff, smoothness);
              }
            else
              value[i] = 0.0;
          }

        // std::vector<double> atomCenteredPoint(dim, 0.);
        // double              r, theta, phi;
        // for (unsigned int i = 0; i < dim; i++)
        //   {
        //     atomCenteredPoint[i] = point[i] - origin[i];
        //   }
        // convertCartesianToSpherical(
        //   atomCenteredPoint, r, theta, phi, polarAngleTolerance);
        // if (r <= cutoff + cutoff / smoothness)
        //   {
        //     double radialValue = (*spline)(r);
        //     int    n = qNumbers[0], l = qNumbers[1], m = qNumbers[2];
        //     auto   Ylm = Clm(l, m) * Dm(m) * Plm(l, m, cos(theta)) * Qm(m,
        //     phi); value =
        //       radialValue * Ylm * smoothCutoffValue(r, cutoff, smoothness);
        //   }
        // else
        //   value = 0.0;
      }

      void
      getGradientValueAnalytical(
        const std::vector<utils::Point> &    point,
        const utils::Point &                 origin,
        const double                         cutoff,
        const double                         smoothness,
        const SphericalHarmonicFunctions &   sphericalHarmonicFunc,
        const std::vector<int> &             qNumbers,
        std::shared_ptr<const utils::Spline> spline,
        const double                         polarAngleTolerance,
        const double                         cutoffTolerance,
        const double                         radiusTolerance,
        std::vector<double> &                gradient)
      {
        for (int i = 0; i < point.size(); i++)
          {
            // do the spline interpolation in the radial points
            double r, theta, phi;
            convertCartesianToSpherical(
              point[i] - origin, r, theta, phi, polarAngleTolerance);

            if (r <= cutoff + cutoff / smoothness)
              {
                DFTEFE_AssertWithMsg(std::abs(r) >= radiusTolerance,
                                     "Value undefined at nucleus");
                double radialValue           = (*spline)(r);
                double radialDerivativeValue = spline->deriv(1, r);

                double cutoffValue = smoothCutoffValue(r, cutoff, smoothness);
                double cutoffDerv  = smoothCutoffDerivative(r,
                                                           cutoff,
                                                           smoothness,
                                                           cutoffTolerance);

                int n = qNumbers[0], l = qNumbers[1], m = qNumbers[2];

                double constant  = Clm(l, m) * Dm(m);
                double plm_theta = sphericalHarmonicFunc.Plm(l, m, theta);
                double dPlmDTheta_theta =
                  sphericalHarmonicFunc.dPlmDTheta(l, m, theta);

                double qm_mphi    = Qm(m, phi);
                auto   Ylm        = constant * plm_theta * qm_mphi;
                auto   dYlmDTheta = constant * dPlmDTheta_theta * qm_mphi;

                // Here used the Legendre differential equation for calculating
                // P_lm/sin(theta) given in the paper
                // https://doi.org/10.1016/S1464-1895(00)00101-0 Pt. no. 3 of
                // verification.

                double dYlmDPhiBysinTheta = 0.;
                if (m != 0)
                  {
                    dYlmDPhiBysinTheta = /*(fabs(theta - 0.0) >=
                    polarAngleTolerance && fabs(theta - M_PI) >=
                    polarAngleTolerance) ? constant * plm_theta * dQmDPhi(m,
                    phi)/sin(theta) :*/
                      constant *
                      (sin(theta) *
                         sphericalHarmonicFunc.d2PlmDTheta2(l, m, theta) +
                       cos(theta) * dPlmDTheta_theta +
                       sin(theta) * l * (l + 1) * plm_theta) *
                      (1 / (m * m)) * dQmDPhi(m, phi);
                  }

                auto dValueDR = (radialDerivativeValue * cutoffValue +
                                 cutoffDerv * radialValue) *
                                Ylm;
                double dValueDThetaByr = 0.;
                dValueDThetaByr = (radialValue / r) * cutoffValue * dYlmDTheta;
                double dValueDPhiByrsinTheta = 0.;
                dValueDPhiByrsinTheta =
                  (radialValue / r) * cutoffValue * dYlmDPhiBysinTheta;

                gradient[3 * i + 0] =
                  dValueDR * (sin(theta) * cos(phi)) +
                  dValueDThetaByr * (cos(theta) * cos(phi)) -
                  sin(phi) * dValueDPhiByrsinTheta;
                gradient[3 * i + 1] =
                  dValueDR * (sin(theta) * sin(phi)) +
                  dValueDThetaByr * (cos(theta) * sin(phi)) +
                  cos(phi) * dValueDPhiByrsinTheta;
                gradient[3 * i + 2] =
                  dValueDR * (cos(theta)) - dValueDThetaByr * (sin(theta));
              }
            else
              {
                gradient[3 * i + 0] = 0.0;
                gradient[3 * i + 1] = 0.0;
                gradient[3 * i + 2] = 0.0;
              }
          }
      }

      void
      getHessianValueAnalytical(
        const utils::Point &                 point,
        const utils::Point &                 origin,
        const double                         cutoff,
        const double                         smoothness,
        const SphericalHarmonicFunctions &   sphericalHarmonicFunc,
        const std::vector<int> &             qNumbers,
        std::shared_ptr<const utils::Spline> spline,
        const double                         polarAngleTolerance,
        const double                         cutoffTolerance,
        const double                         radiusTolerance,
        std::vector<double> &                gradient)
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
        // do the spline interpolation in the radial points
        double r, theta, phi;
        convertCartesianToSpherical(
          point - origin, r, theta, phi, polarAngleTolerance);
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
        // do the spline interpolation in the radial points
        double r, theta, phi;
        convertCartesianToSpherical(
          point - origin, r, theta, phi, polarAngleTolerance);
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
        // do the spline interpolation in the radial points
        double r, theta, phi;
        convertCartesianToSpherical(
          point - origin, r, theta, phi, polarAngleTolerance);
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
    } // namespace SphericalDataNumericalInternal

    SphericalDataNumerical::SphericalDataNumerical(
      const std::vector<int>            qNumbers,
      const std::vector<double>         radialPoints,
      const std::vector<double>         radialValues,
      const double                      cutoff,
      const double                      smoothness,
      const SphericalHarmonicFunctions &sphericalHarmonicFunc,
      const double                      polarAngleTolerance,
      const double                      cutoffTolerance,
      const double                      radiusTolerance,
      const size_type                   dim)
      : d_qNumbers(qNumbers)
      , d_radialPoints(radialPoints)
      , d_radialValues(radialValues)
      , d_cutoff(cutoff)
      , d_smoothness(smoothness)
      , d_polarAngleTolerance(polarAngleTolerance)
      , d_cutoffTolerance(cutoffTolerance)
      , d_radiusTolerance(radiusTolerance)
      , d_dim(dim)
      , d_sphericalHarmonicFunc(sphericalHarmonicFunc)
    {
      utils::throwException<utils::InvalidArgument>(d_dim == 3,
                                                    "Dimension has to be 3.");
      initSpline();
    }

    void
    SphericalDataNumerical::initSpline()
    {
      d_spline = std::make_shared<const utils::Spline>(this->d_radialPoints,
                                                       this->d_radialValues,
                                                       true);
    }

    std::vector<double>
    SphericalDataNumerical::getValue(const std::vector<utils::Point> &point,
                                     const utils::Point &             origin)
    {
      std::vector<double> value(point.size(), 0.);
      DFTEFE_AssertWithMsg(point[0].size() == d_dim && origin.size() == d_dim,
                           "getValue() has a dimension mismatch");
      DFTEFE_AssertWithMsg(d_qNumbers.size() == 3,
                           "All quantum numbers not given");
      SphericalDataNumericalInternal::getValueAnalytical(
        point,
        origin,
        d_cutoff,
        d_smoothness,
        d_sphericalHarmonicFunc,
        d_qNumbers,
        d_spline,
        d_polarAngleTolerance,
        value);

      return value;
    }

    std::vector<double>
    SphericalDataNumerical::getGradientValue(
      const std::vector<utils::Point> &point,
      const utils::Point &             origin)
    {
      std::vector<double> gradient(d_dim * point.size(), 0.);
      DFTEFE_AssertWithMsg(point[0].size() == d_dim && origin.size() == d_dim,
                           "getDerivativeValue() has a dimension mismatch");
      DFTEFE_AssertWithMsg(d_qNumbers.size() == 3,
                           "All quantum numbers not given");
      SphericalDataNumericalInternal::getGradientValueAnalytical(
        point,
        origin,
        d_cutoff,
        d_smoothness,
        d_sphericalHarmonicFunc,
        d_qNumbers,
        d_spline,
        d_polarAngleTolerance,
        d_cutoffTolerance,
        d_radiusTolerance,
        gradient);

      DFTEFE_AssertWithMsg(gradient.size() == d_dim * point.size(),
                           "Gradient vector should be of length dim");
      return gradient;
    }

    std::vector<double>
    SphericalDataNumerical::getHessianValue(
      const std::vector<utils::Point> &point,
      const utils::Point &             origin)
    {
      std::vector<double> hessian(d_dim * d_dim, 0.),
        ret(d_dim * d_dim * point.size(), 0.);
      DFTEFE_AssertWithMsg(point[0].size() == d_dim && origin.size() == d_dim,
                           "getHessianValue() has a dimension mismatch");
      DFTEFE_AssertWithMsg(d_qNumbers.size() == 3,
                           "All quantum numbers not given");

      for (int i = 0; i < point.size(); i++)
        {
          SphericalDataNumericalInternal::getHessianValueAutoDiff(
            point[i],
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
          std::copy(hessian.begin(),
                    hessian.end(),
                    ret.begin() + d_dim * d_dim * i);
        }
      return ret;
    }

    double
    SphericalDataNumerical::getValue(const utils::Point &point,
                                     const utils::Point &origin)
    {
      std::vector<double> value(1, 0);
      DFTEFE_AssertWithMsg(point.size() == d_dim && origin.size() == d_dim,
                           "getValue() has a dimension mismatch");
      DFTEFE_AssertWithMsg(d_qNumbers.size() == 3,
                           "All quantum numbers not given");
      SphericalDataNumericalInternal::getValueAnalytical(
        std::vector<utils::Point>{point},
        origin,
        d_cutoff,
        d_smoothness,
        d_sphericalHarmonicFunc,
        d_qNumbers,
        d_spline,
        d_polarAngleTolerance,
        value);

      return value[0];
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
      SphericalDataNumericalInternal::getGradientValueAnalytical(
        std::vector<utils::Point>{point},
        origin,
        d_cutoff,
        d_smoothness,
        d_sphericalHarmonicFunc,
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

      SphericalDataNumericalInternal::getHessianValueAutoDiff(
        point,
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

    std::vector<double>
    SphericalDataNumerical::getRadialValue(const std::vector<double> &r)
    {
      std::vector<double> retVal(r.size(), 0.);
      for (int i = 0; i < r.size(); i++)
        {
          double radius = r[i];
          retVal[i]     = (radius <= d_cutoff + d_cutoff / d_smoothness) ?
                            (*d_spline)(radius)*smoothCutoffValue(radius,
                                                              d_cutoff,
                                                              d_smoothness) :
                            0.;
        }
      return retVal;
    }

    std::vector<double>
    SphericalDataNumerical::getAngularValue(const std::vector<double> &r,
                                            const std::vector<double> &theta,
                                            const std::vector<double> &phi)
    {
      int                 l        = d_qNumbers[1];
      int                 m        = d_qNumbers[2];
      double              constant = Clm(l, m) * Dm(m);
      std::vector<double> retVal(r.size(), 0.);
      for (int i = 0; i < r.size(); i++)
        {
          retVal[i] = (r[i] <= d_cutoff + d_cutoff / d_smoothness) ?
                        constant * d_sphericalHarmonicFunc.Plm(l, m, theta[i]) *
                          Qm(m, phi[i]) :
                        0.;
        }
      return retVal;
    }

    std::vector<double>
    SphericalDataNumerical::getRadialDerivative(const std::vector<double> &r)
    {
      std::vector<double> retVal(r.size(), 0.);
      for (int i = 0; i < r.size(); i++)
        {
          double radius = r[i];
          if (radius <= d_cutoff + d_cutoff / d_smoothness)
            {
              double radialValue           = (*d_spline)(radius);
              double radialDerivativeValue = d_spline->deriv(1, radius);

              double cutoffValue =
                smoothCutoffValue(radius, d_cutoff, d_smoothness);
              double cutoffDerv = smoothCutoffDerivative(radius,
                                                         d_cutoff,
                                                         d_smoothness,
                                                         d_cutoffTolerance);

              retVal[i] =
                radialDerivativeValue * cutoffValue + cutoffDerv * radialValue;
            }
          else
            retVal[i] = 0.;
        }
      return retVal;
    }

    std::vector<std::vector<double>>
    SphericalDataNumerical::getAngularDerivative(
      const std::vector<double> &r,
      const std::vector<double> &thetaVec,
      const std::vector<double> &phiVec)
    {
      std::vector<std::vector<double>> retVal(2,
                                              std::vector<double>(r.size(),
                                                                  0.));
      int                              l        = d_qNumbers[1];
      int                              m        = d_qNumbers[2];
      double                           constant = Clm(l, m) * Dm(m);
      for (int i = 0; i < r.size(); i++)
        {
          if (r[i] <= d_cutoff + d_cutoff / d_smoothness)
            {
              DFTEFE_AssertWithMsg(
                std::abs(r[i]) >= d_radiusTolerance,
                "Value undefined at nucleus while calling SphericalData::getAngularDerivative()");

              double theta     = thetaVec[i];
              double phi       = phiVec[i];
              double plm_theta = d_sphericalHarmonicFunc.Plm(l, m, theta);
              double dPlmDTheta_theta =
                d_sphericalHarmonicFunc.dPlmDTheta(l, m, theta);
              double qm_mphi    = Qm(m, phi);
              auto   Ylm        = constant * plm_theta * qm_mphi;
              auto   dYlmDTheta = constant * dPlmDTheta_theta * qm_mphi;

              double dYlmDPhiBysinTheta = 0.;
              if (m != 0)
                {
                  dYlmDPhiBysinTheta =
                    constant *
                    (sin(theta) *
                       d_sphericalHarmonicFunc.d2PlmDTheta2(l, m, theta) +
                     cos(theta) * dPlmDTheta_theta +
                     sin(theta) * l * (l + 1) * plm_theta) *
                    (1 / (m * m)) * dQmDPhi(m, phi);
                }

              retVal[0][i] = dYlmDTheta * (1 / r[i]);
              retVal[1][i] = dYlmDPhiBysinTheta * (1 / r[i]);
            }
          else
            {
              retVal[0][i] = 0.;
              retVal[1][i] = 0.;
            }
        }
      return retVal;
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
