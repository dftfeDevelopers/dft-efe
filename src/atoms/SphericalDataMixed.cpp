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
#include <atoms/SphericalDataMixed.h>

namespace dftefe
{
  namespace atoms
  {
    namespace SphericalDataMixedInternal
    {
      void
      getValueAnalytical(
        const std::vector<utils::Point> &       point,
        const utils::Point &                    origin,
        const double                            lastRadialGridPoint,
        const utils::ScalarSpatialFunctionReal &funcAfterRadialGrid,
        const SphericalHarmonicFunctions &      sphericalHarmonicFunc,
        const std::vector<int> &                qNumbers,
        std::shared_ptr<const utils::Spline>    spline,
        const double                            polarAngleTolerance,
        std::vector<double> &                   value)
      {
        for (int i = 0; i < point.size(); i++)
          {
            // do the spline interpolation in the radial points
            double r, theta, phi;
            convertCartesianToSpherical(
              point[i] - origin, r, theta, phi, polarAngleTolerance);
            int  n = qNumbers[0], l = qNumbers[1], m = qNumbers[2];
            auto Ylm = Clm(l, m) * Dm(m) *
                       sphericalHarmonicFunc.Plm(l, std::abs(m), theta) *
                       Qm(m, phi);
            if (r <= lastRadialGridPoint)
              value[i] = (*spline)(r)*Ylm;
            else
              value[i] = funcAfterRadialGrid(point[i] - origin) * Ylm;
          }
      }
    } // namespace SphericalDataMixedInternal

    SphericalDataMixed::SphericalDataMixed(
      const std::vector<int>                  qNumbers,
      const std::vector<double>               radialPoints,
      const std::vector<double>               radialValues,
      utils::Spline::bd_type                  left,
      double                                  leftValue,
      utils::Spline::bd_type                  right,
      double                                  rightValue,
      const utils::ScalarSpatialFunctionReal &funcAfterRadialGrid,
      const SphericalHarmonicFunctions &      sphericalHarmonicFunc,
      const double                            polarAngleTolerance,
      const size_type                         dim)
      : d_qNumbers(qNumbers)
      , d_radialPoints(radialPoints)
      , d_radialValues(radialValues)
      , d_polarAngleTolerance(polarAngleTolerance)
      , d_funcAfterRadialGrid(funcAfterRadialGrid)
      , d_dim(dim)
      , d_sphericalHarmonicFunc(sphericalHarmonicFunc)
    {
      utils::throwException<utils::InvalidArgument>(d_dim == 3,
                                                    "Dimension has to be 3.");
      initSpline(left, leftValue, right, rightValue);
    }

    void
    SphericalDataMixed::initSpline(utils::Spline::bd_type left,
                                   double                 leftValue,
                                   utils::Spline::bd_type right,
                                   double                 rightValue)
    {
      d_spline = std::make_shared<const utils::Spline>(
        this->d_radialPoints,
        this->d_radialValues,
        true,
        utils::Spline::spline_type::cspline,
        false,
        left,
        leftValue,
        right,
        rightValue);
    }

    std::vector<double>
    SphericalDataMixed::getValue(const std::vector<utils::Point> &point,
                                 const utils::Point &             origin)
    {
      std::vector<double> value(point.size(), 0.);
      DFTEFE_AssertWithMsg(point[0].size() == d_dim && origin.size() == d_dim,
                           "getValue() has a dimension mismatch");
      DFTEFE_AssertWithMsg(d_qNumbers.size() == 3,
                           "All quantum numbers not given");
      SphericalDataMixedInternal::getValueAnalytical(point,
                                                     origin,
                                                     d_radialPoints.back(),
                                                     d_funcAfterRadialGrid,
                                                     d_sphericalHarmonicFunc,
                                                     d_qNumbers,
                                                     d_spline,
                                                     d_polarAngleTolerance,
                                                     value);

      return value;
    }

    std::vector<double>
    SphericalDataMixed::getGradientValue(const std::vector<utils::Point> &point,
                                         const utils::Point &origin)
    {
      std::vector<double> gradient(d_dim * point.size(), 0.);
      utils::throwException(
        false,
        "getGradientValue() function in SphericalDataMixed is not yet implemented.");
      return gradient;
    }

    std::vector<double>
    SphericalDataMixed::getHessianValue(const std::vector<utils::Point> &point,
                                        const utils::Point &             origin)
    {
      std::vector<double> hessian(d_dim * d_dim, 0.),
        ret(d_dim * d_dim * point.size(), 0.);
      utils::throwException(
        false,
        "getHessianValue() function in SphericalDataMixed is not yet implemented.");
      return ret;
    }

    double
    SphericalDataMixed::getValue(const utils::Point &point,
                                 const utils::Point &origin)
    {
      std::vector<double> value(1, 0);
      DFTEFE_AssertWithMsg(point.size() == d_dim && origin.size() == d_dim,
                           "getValue() has a dimension mismatch");
      DFTEFE_AssertWithMsg(d_qNumbers.size() == 3,
                           "All quantum numbers not given");
      SphericalDataMixedInternal::getValueAnalytical(
        std::vector<utils::Point>{point},
        origin,
        d_radialPoints.back(),
        d_funcAfterRadialGrid,
        d_sphericalHarmonicFunc,
        d_qNumbers,
        d_spline,
        d_polarAngleTolerance,
        value);

      return value[0];
    }

    std::vector<double>
    SphericalDataMixed::getGradientValue(const utils::Point &point,
                                         const utils::Point &origin)
    {
      std::vector<double> gradient(d_dim, 0.);
      utils::throwException(
        false,
        "getGradientValue() function in SphericalDataMixed is not yet implemented.");
      return gradient;
    }

    std::vector<double>
    SphericalDataMixed::getHessianValue(const utils::Point &point,
                                        const utils::Point &origin)
    {
      std::vector<double> hessian(d_dim * d_dim, 0.);
      utils::throwException(
        false,
        "hessian() function in SphericalDataMixed is not yet implemented.");
      return hessian;
    }

    std::vector<double>
    SphericalDataMixed::getRadialValue(const std::vector<double> &r)
    {
      std::vector<double> retVal(r.size(), 0.);
      utils::throwException(
        false,
        "getRadialValue() function in SphericalDataMixed is not yet implemented.");
      return retVal;
    }

    std::vector<double>
    SphericalDataMixed::getAngularValue(const std::vector<double> &r,
                                        const std::vector<double> &theta,
                                        const std::vector<double> &phi)
    {
      std::vector<double> retVal(r.size(), 0.);
      utils::throwException(
        false,
        "getRadialDerivative() function in SphericalDataMixed is not yet implemented.");
      return retVal;
    }

    std::vector<double>
    SphericalDataMixed::getRadialDerivative(const std::vector<double> &r)
    {
      std::vector<double> retVal(r.size(), 0.);
      utils::throwException(
        false,
        "getRadialDerivative() function in SphericalDataMixed is not yet implemented.");
      return retVal;
    }

    std::vector<std::vector<double>>
    SphericalDataMixed::getAngularDerivative(
      const std::vector<double> &r,
      const std::vector<double> &thetaVec,
      const std::vector<double> &phiVec)
    {
      std::vector<std::vector<double>> retVal(2,
                                              std::vector<double>(r.size(),
                                                                  0.));
      utils::throwException(
        false,
        "getAngularDerivative() function in SphericalDataMixed is not yet implemented.");
      return retVal;
    }

    std::vector<int>
    SphericalDataMixed::getQNumbers() const
    {
      return d_qNumbers;
    }

    double
    SphericalDataMixed::getCutoff() const
    {
      utils::throwException(
        false, "Cannot call getCutoff() function in SphericalDataMixed.");
      return 0;
    }

    double
    SphericalDataMixed::getSmoothness() const
    {
      utils::throwException(
        false, "Cannot call getSmoothness() function in SphericalDataMixed.");
      return 0;
    }

  } // namespace atoms
} // namespace dftefe
