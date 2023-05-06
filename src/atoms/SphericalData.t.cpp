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
#include <cmath>

namespace dftefe
{
  namespace atoms
  {
    template <size_type dim>
    double 
    SphericalData::getValue(const utils::Point &point, 
                            const utils::Point &origin,
                            const double polarAngleTolerance) const
    {
      utils::throwException(dim == 3 ,
                      "getValue() defined only for 3 dimensional case");
      // do the spline interpolation in the radial points
      std::vector<double> atomCenteredPoint;
      atomCenteredPoint.resize(dim,0.);
      DFTEFE_AssertWithMsg(point.size() == dim && origin.size() == dim,
                          "Dimension mismatch between the point and origin.");
      double r, theta, phi;
      for (unsigned int i=0 ; i<dim ; i++)
      {
          atomCenteredPoint[i] = point[i] - origin[i];
      }
      convertCartesianToSpherical(atomCenteredPoint, r, theta, phi, polarAngleTolerance);
      double radialValue = d_spline->operator()(r);
      DFTEFE_AssertWithMsg(qNumbers.size() == 3,
                  "All quantum numbers not given");
      int n = qNumbers[0], l = qNumbers[1], m = qNumbers[2];
      auto Ylm = Clm(l,m) * Dm(m) * Plm(l,m,cos(theta)) * Qm(m,phi);
      double retValue = radialValue * Ylm * smoothCutoffValue(r, cutoff, smoothness);
      return retValue;
    }

    template <size_type dim>
    std::vector<double>
    SphericalData::getGradientValue(const utils::Point &point, 
                            const utils::Point &origin,
                            const double polarAngleTolerance, 
                            const double cutoffTolerance) const
    {
      utils::throwException(dim == 3 ,
                "getDerivativeValue() defined only for 3 dimensional case");
      // do the spline interpolation in the radial points
      std::vector<double> atomCenteredPoint;
      atomCenteredPoint.resize(dim,0.);
      DFTEFE_AssertWithMsg(point.size() == dim && origin.size() == dim,
                          "Dimension mismatch between the point and origin.");
      double r, theta, phi;
      for (unsigned int i=0 ; i<dim ; i++)
      {
          atomCenteredPoint[i] = point[i] - origin[i];
      }
      convertCartesianToSpherical(atomCenteredPoint, r, theta, phi, polarAngleTolerance);
      double radialValue = d_spline->operator()(r);
      double radialDerivativeValue = d_spline->deriv(1 , r);
      double cutoffValue = smoothCutoffValue(r, cutoff, smoothness);
      double cutoffDerv = smoothCutoffDerivative(r, cutoff, smoothness, cutoffTolerance);

      DFTEFE_AssertWithMsg(qNumbers.size() == 3,
                  "All quantum numbers not given");
      int n = qNumbers[0], l = qNumbers[1], m = qNumbers[2];

      auto Ylm = Clm(l,m) * Dm(m) * Plm(l,m,cos(theta)) * Qm(m,phi);
      auto dYlmDTheta = Clm(l,m) * Dm(m) * dPlmDTheta(l,m,theta) * Qm(m,phi);

      // Here used the Legendre differential equation for calculating P_lm/sin(theta)
      // given in the paper https://doi.org/10.1016/S1464-1895(00)00101-0 
      // Pt. no. 3 of verification.

      double dYlmDPhiBysinTheta = 0.;
      if( m != 0 )
      {
        dYlmDPhiBysinTheta = Clm(l,m) * Dm(m) * (sin(theta) * d2PlmDTheta2(l, m, theta) +
                                                  cos(theta) * dPlmDTheta(l,m,theta) +
                                                  sin(theta) * l * (l+1) * Plm(l,m,cos(theta)))
                                                * (1/(m*m)) * dQmDPhi(m,phi);
      }

      auto dValueDR = (radialDerivativeValue * cutoffValue + cutoffDerv * radialValue) * Ylm;
      double dValueDThetaByr = 0.;
      if (r != 0)
      {
        dValueDThetaByr = (radialValue/r) * cutoffValue * dYlmDTheta;
      }
      double dValueDPhiByrsinTheta = 0.;
      if (r != 0)
      {
          dValueDPhiByrsinTheta = (radialValue/r) * cutoffValue * dYlmDPhiBysinTheta;
      }

      std::vector<double> retValue;
      retValue.resize(dim,0.);
      retValue[0] = dValueDR * (sin(theta)*cos(phi)) + dValueDThetaByr * (cos(theta)*cos(phi)) - sin(phi) * dValueDPhiByrsinTheta;
      retValue[1] = dValueDR * (sin(theta)*sin(phi)) + dValueDThetaByr * (cos(theta)*sin(phi)) + cos(phi) * dValueDPhiByrsinTheta;
      retValue[2] = dValueDR * (cos(theta)) - dValueDThetaByr * (sin(theta));

      return retValue;
    }

    template <size_type dim>
    std::vector<double>
    SphericalData::getHessianValue(const utils::Point &point, 
                            const utils::Point &origin,
                            const double polarAngleTolerance, 
                            const double cutoffTolerance) const
    {
      utils::throwException(dim == 3 ,
                "getDerivativeValue() defined only for 3 dimensional case");
      // do the spline interpolation in the radial points
      std::vector<double> atomCenteredPoint;
      atomCenteredPoint.resize(dim,0.);
      DFTEFE_AssertWithMsg(point.size() == dim && origin.size() == dim,
                          "Dimension mismatch between the point and origin.");
      double r, theta, phi;
      for (unsigned int i=0 ; i<dim ; i++)
      {
          atomCenteredPoint[i] = point[i] - origin[i];
      }
      convertCartesianToSpherical(atomCenteredPoint, r, theta, phi, polarAngleTolerance);
      double radialValue = d_spline->operator()(r);
      double radialDerivativeValue = d_spline->deriv(1 , r);
      double cutoffValue = smoothCutoffValue(r, cutoff, smoothness);
      double cutoffDerv = smoothCutoffDerivative(r, cutoff, smoothness, cutoffTolerance);
      double cutoffDerv2 = smoothCutoffDerivative2(r, cutoff, smoothness, cutoffTolerance);

      DFTEFE_AssertWithMsg(qNumbers.size() == 3,
                  "All quantum numbers not given");
      int n = qNumbers[0], l = qNumbers[1], m = qNumbers[2];

      auto Ylm = Clm(l,m) * Dm(m) * Plm(l,m,cos(theta)) * Qm(m,phi);
      auto dYlmDTheta = Clm(l,m) * Dm(m) * dPlmDTheta(l,m,theta) * Qm(m,phi);

      // Here used the Legendre differential equation for calculating P_lm/sin(theta)
      // given in the paper https://doi.org/10.1016/S1464-1895(00)00101-0 
      // Pt. no. 3 of verification.

      double dYlmDPhiBysinTheta = 0.;
      if( m != 0 )
      {
        dYlmDPhiBysinTheta = Clm(l,m) * Dm(m) * (sin(theta) * d2PlmDTheta2(l, m, theta) +
                                                  cos(theta) * dPlmDTheta(l,m,theta) +
                                                  sin(theta) * l * (l+1) * Plm(l,m,cos(theta)))
                                                * (1/(m*m)) * dQmDPhi(m,phi);
      }

      auto dValueDR = (radialDerivativeValue * cutoffValue + cutoffDerv * radialValue) * Ylm;
      double dValueDThetaByr = 0.;
      if (r != 0)
      {
        dValueDThetaByr = (radialValue/r) * cutoffValue * dYlmDTheta;
      }
      double dValueDPhiByrsinTheta = 0.;
      if (r != 0)
      {
          dValueDPhiByrsinTheta = (radialValue/r) * cutoffValue * dYlmDPhiBysinTheta;
      }

      std::vector<double> retValue;
      retValue.resize(dim*dim,0.);
      retValue[0] = dValueDR * (sin(theta)*cos(phi)) + dValueDThetaByr * (cos(theta)*cos(phi)) - sin(phi) * dValueDPhiByrsinTheta;
      retValue[1] = dValueDR * (sin(theta)*sin(phi)) + dValueDThetaByr * (cos(theta)*sin(phi)) + cos(phi) * dValueDPhiByrsinTheta;
      retValue[2] = dValueDR * (cos(theta)) - dValueDThetaByr * (sin(theta));

      return retValue;
    }

  }
}